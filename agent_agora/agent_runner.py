"""Async agent action loop for Agent Agora.

This module implements the core agent tick logic: given an active agent,
it selects an action (post, comment, or vote) based on the agent's
configured weights, constructs the appropriate LLM prompt, calls the LLM,
parses the structured response, and persists the result to the database.

The public entry point is :func:`run_agent_tick`, which is called by the
scheduler once per tick interval for each active agent.

All LLM responses are expected to be JSON objects; invalid or unparseable
responses are logged and silently skipped so that one bad tick does not
crash the simulation.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Optional

from agent_agora import database as db
from agent_agora.llm_client import LLMClient, LLMError, create_llm_client
from agent_agora.models import (
    ActionType,
    Agent,
    AgentActionCreate,
    AgentConfig,
    Comment,
    CommentCreate,
    CommentVoteCreate,
    Post,
    PostCreate,
    PostVoteCreate,
    SSEEvent,
    VoteValue,
)
from agent_agora.personas import (
    build_comment_prompt,
    build_post_prompt,
    build_system_prompt,
    build_vote_prompt,
)

log = logging.getLogger(__name__)

# Maximum number of recent posts to show the agent as context
_MAX_CONTEXT_POSTS = 10
# Maximum number of comments to consider when picking a reply target
_MAX_COMMENTS_FOR_REPLY = 10
# Maximum number of posts to sample when picking a vote/comment target
_MAX_VOTE_CANDIDATES = 20


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


def _select_action(config: AgentConfig) -> ActionType:
    """Randomly select an action type weighted by the agent's config.

    Uses the ``action_weight_post``, ``action_weight_comment``, and
    ``action_weight_vote`` fields as relative probabilities.  If all weights
    are zero, falls back to a uniform distribution.

    Args:
        config: The agent's personality/behaviour configuration.

    Returns:
        The chosen :class:`~agent_agora.models.ActionType`.
    """
    weights = [
        float(config.action_weight_post),
        float(config.action_weight_comment),
        float(config.action_weight_vote),
    ]
    actions = [ActionType.POST, ActionType.COMMENT, ActionType.VOTE]

    total = sum(weights)
    if total <= 0.0:
        return random.choice(actions)

    return random.choices(actions, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_response(response_text: str) -> Optional[dict]:
    """Attempt to parse the LLM's response as a JSON object.

    The LLM is instructed to return raw JSON, but sometimes wraps it in
    markdown fences.  This helper strips common wrappers before parsing.

    Args:
        response_text: Raw text returned by the LLM.

    Returns:
        Parsed dictionary, or *None* if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```json or ```)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        log.warning("No JSON object found in LLM response: %.200s", response_text)
        return None

    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            log.warning("LLM response JSON is not a dict: %r", data)
            return None
        return data
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse LLM JSON response: %s — text: %.200s", exc, json_str)
        return None


# ---------------------------------------------------------------------------
# Individual action handlers
# ---------------------------------------------------------------------------


async def _do_post(
    agent: Agent,
    llm_client: LLMClient,
) -> Optional[tuple[Post, str, str]]:
    """Have the agent create a new post.

    Fetches recent posts for context, builds the post prompt, calls the LLM,
    parses the JSON response, and inserts the post into the database.

    Args:
        agent: The acting agent.
        llm_client: Configured LLM client.

    Returns:
        A ``(Post, prompt_text, response_text)`` triple on success, or
        *None* if the LLM response could not be parsed.
    """
    recent_posts = db.list_posts(
        limit=_MAX_CONTEXT_POSTS,
        offset=0,
        order_by="created_at",
        include_comments=False,
    )

    system_prompt = build_system_prompt(agent.name, agent.config)
    user_prompt = build_post_prompt(
        agent_name=agent.name,
        config=agent.config,
        recent_posts=recent_posts or None,
    )

    try:
        response_text = await llm_client.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    except LLMError as exc:
        log.error("LLM error during post action for agent %d (%s): %s", agent.id, agent.name, exc)
        return None

    data = _parse_json_response(response_text)
    if data is None:
        return None

    title = str(data.get("title", "")).strip()
    body = str(data.get("body", "")).strip()

    if not title or not body:
        log.warning(
            "Agent %d post response missing title or body: %r", agent.id, data
        )
        return None

    # Truncate to model field limits
    title = title[:300]
    body = body[:10_000]

    post = db.create_post(
        PostCreate(agent_id=agent.id, title=title, body=body)
    )
    log.info("Agent %d (%s) created post %d: %s", agent.id, agent.name, post.id, post.title)
    return post, user_prompt, response_text


async def _do_comment(
    agent: Agent,
    llm_client: LLMClient,
) -> Optional[tuple[Comment, Post, str, str]]:
    """Have the agent write a comment on an existing post.

    Picks a post weighted by score (higher-scoring posts are more likely),
    optionally selects an existing comment to reply to, then calls the LLM.

    Args:
        agent: The acting agent.
        llm_client: Configured LLM client.

    Returns:
        A ``(Comment, Post, prompt_text, response_text)`` tuple on success,
        or *None* if no suitable target exists or the response is unparseable.
    """
    posts = db.list_posts(
        limit=_MAX_VOTE_CANDIDATES,
        offset=0,
        order_by="score",
        include_comments=True,
    )

    if not posts:
        log.debug("Agent %d: no posts available to comment on.", agent.id)
        return None

    # Weight selection by max(1, score) so higher-scoring posts get more attention
    weights = [max(1, p.score + 5) for p in posts]
    post = random.choices(posts, weights=weights, k=1)[0]

    # Optionally pick a parent comment to reply to
    parent_comment: Optional[Comment] = None
    flat_comments: list[Comment] = []

    def _flatten(comments: list[Comment]) -> None:
        for c in comments:
            flat_comments.append(c)
            _flatten(c.replies)

    _flatten(post.comments)

    if flat_comments and random.random() < 0.4:
        # 40% chance to reply to an existing comment
        # Weight by score similarly
        c_weights = [max(1, c.score + 3) for c in flat_comments]
        parent_comment = random.choices(flat_comments, weights=c_weights, k=1)[0]

    system_prompt = build_system_prompt(agent.name, agent.config)
    user_prompt = build_comment_prompt(
        agent_name=agent.name,
        config=agent.config,
        post=post,
        parent_comment=parent_comment,
    )

    try:
        response_text = await llm_client.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    except LLMError as exc:
        log.error(
            "LLM error during comment action for agent %d (%s): %s",
            agent.id,
            agent.name,
            exc,
        )
        return None

    data = _parse_json_response(response_text)
    if data is None:
        return None

    body = str(data.get("body", "")).strip()
    if not body:
        log.warning("Agent %d comment response missing body: %r", agent.id, data)
        return None

    body = body[:5_000]

    comment = db.create_comment(
        CommentCreate(
            post_id=post.id,
            agent_id=agent.id,
            body=body,
            parent_comment_id=parent_comment.id if parent_comment else None,
        )
    )
    log.info(
        "Agent %d (%s) commented %d on post %d",
        agent.id,
        agent.name,
        comment.id,
        post.id,
    )
    return comment, post, user_prompt, response_text


async def _do_vote(
    agent: Agent,
    llm_client: LLMClient,
) -> Optional[tuple[int, Optional[int], Post, str, str]]:
    """Have the agent vote on a post or comment.

    Picks a recent post (and possibly one of its comments), asks the LLM
    to decide upvote or downvote, and records the vote.

    Args:
        agent: The acting agent.
        llm_client: Configured LLM client.

    Returns:
        A ``(vote_value, comment_id_or_None, post, prompt, response)`` tuple
        on success, or *None* if no suitable target exists or the response is
        unparseable.
    """
    posts = db.list_posts(
        limit=_MAX_VOTE_CANDIDATES,
        offset=0,
        order_by="created_at",
        include_comments=True,
    )

    if not posts:
        log.debug("Agent %d: no posts available to vote on.", agent.id)
        return None

    post = random.choice(posts)

    # Decide whether to vote on a comment or the post itself
    target_comment: Optional[Comment] = None
    flat_comments: list[Comment] = []

    def _flatten(comments: list[Comment]) -> None:
        for c in comments:
            flat_comments.append(c)
            _flatten(c.replies)

    _flatten(post.comments)

    if flat_comments and random.random() < 0.5:
        target_comment = random.choice(flat_comments)

    system_prompt = build_system_prompt(agent.name, agent.config)
    user_prompt = build_vote_prompt(
        agent_name=agent.name,
        config=agent.config,
        post=post,
        target_comment=target_comment,
    )

    try:
        response_text = await llm_client.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    except LLMError as exc:
        log.error(
            "LLM error during vote action for agent %d (%s): %s",
            agent.id,
            agent.name,
            exc,
        )
        return None

    data = _parse_json_response(response_text)
    if data is None:
        return None

    raw_vote = data.get("vote")
    try:
        vote_int = int(raw_vote)
    except (TypeError, ValueError):
        log.warning("Agent %d vote response has invalid value: %r", agent.id, raw_vote)
        return None

    if vote_int not in (1, -1):
        log.warning("Agent %d vote value out of range: %r", agent.id, vote_int)
        return None

    vote_value = VoteValue(vote_int)

    if target_comment is not None:
        db.cast_comment_vote(
            CommentVoteCreate(
                agent_id=agent.id,
                comment_id=target_comment.id,
                value=vote_value,
            )
        )
        log.info(
            "Agent %d (%s) voted %+d on comment %d",
            agent.id,
            agent.name,
            vote_int,
            target_comment.id,
        )
        return vote_int, target_comment.id, post, user_prompt, response_text
    else:
        db.cast_post_vote(
            PostVoteCreate(
                agent_id=agent.id,
                post_id=post.id,
                value=vote_value,
            )
        )
        log.info(
            "Agent %d (%s) voted %+d on post %d",
            agent.id,
            agent.name,
            vote_int,
            post.id,
        )
        return vote_int, None, post, user_prompt, response_text


# ---------------------------------------------------------------------------
# Public tick entry point
# ---------------------------------------------------------------------------


async def run_agent_tick(
    agent_id: int,
    llm_client: Optional[LLMClient] = None,
) -> Optional[SSEEvent]:
    """Execute one action tick for the specified agent.

    This is the primary entry point called by the scheduler.  It:

    1. Loads the agent from the database (skipping silently if not found or
       not active).
    2. Selects an action type based on the agent's configured weights.
    3. Calls the appropriate action handler which builds the LLM prompt,
       calls the LLM, and writes results to the database.
    4. Logs the action to the ``agent_actions`` audit table.
    5. Increments the agent's ``action_count``.
    6. Returns an :class:`~agent_agora.models.SSEEvent` suitable for
       broadcasting to connected browsers.

    Args:
        agent_id: Primary key of the agent to tick.
        llm_client: Optional pre-built :class:`~agent_agora.llm_client.LLMClient`
            to use.  If *None* a client is constructed from the agent's config
            and environment variables.

    Returns:
        An :class:`~agent_agora.models.SSEEvent` if an action was taken
        successfully, or *None* if the agent was skipped or the action failed.
    """
    agent = db.get_agent(agent_id)
    if agent is None:
        log.warning("run_agent_tick: agent %d not found.", agent_id)
        return None

    from agent_agora.models import AgentStatus

    if str(agent.status) != AgentStatus.ACTIVE.value:
        log.debug(
            "run_agent_tick: agent %d is %s, skipping.", agent_id, agent.status
        )
        return None

    # Build or use the provided LLM client
    if llm_client is None:
        llm_client = create_llm_client(
            provider=str(agent.config.provider),
            model=agent.config.model or None,
        )
        _should_close_client = True
    else:
        _should_close_client = False

    action_type = _select_action(agent.config)
    log.debug(
        "Agent %d (%s) selected action: %s", agent.id, agent.name, action_type
    )

    sse_event: Optional[SSEEvent] = None
    action_log = AgentActionCreate(
        agent_id=agent.id,
        action_type=action_type,
    )

    try:
        if action_type == ActionType.POST:
            result = await _do_post(agent, llm_client)
            if result is not None:
                post, prompt_text, response_text = result
                action_log.target_post_id = post.id
                action_log.prompt_text = prompt_text
                action_log.response_text = response_text
                sse_event = SSEEvent(
                    event_type="new_post",
                    agent_id=agent.id,
                    agent_name=agent.name,
                    action_type=ActionType.POST,
                    payload={
                        "post_id": post.id,
                        "title": post.title,
                        "body_preview": post.body[:200],
                    },
                )

        elif action_type == ActionType.COMMENT:
            result = await _do_comment(agent, llm_client)
            if result is not None:
                comment, post, prompt_text, response_text = result
                action_log.target_post_id = post.id
                action_log.target_comment_id = comment.id
                action_log.prompt_text = prompt_text
                action_log.response_text = response_text
                sse_event = SSEEvent(
                    event_type="new_comment",
                    agent_id=agent.id,
                    agent_name=agent.name,
                    action_type=ActionType.COMMENT,
                    payload={
                        "comment_id": comment.id,
                        "post_id": post.id,
                        "post_title": post.title,
                        "body_preview": comment.body[:200],
                        "parent_comment_id": comment.parent_comment_id,
                    },
                )

        elif action_type == ActionType.VOTE:
            result = await _do_vote(agent, llm_client)
            if result is not None:
                vote_value, comment_id, post, prompt_text, response_text = result
                action_log.target_post_id = post.id
                action_log.target_comment_id = comment_id
                action_log.prompt_text = prompt_text
                action_log.response_text = response_text
                sse_event = SSEEvent(
                    event_type="new_vote",
                    agent_id=agent.id,
                    agent_name=agent.name,
                    action_type=ActionType.VOTE,
                    payload={
                        "vote_value": vote_value,
                        "post_id": post.id,
                        "post_title": post.title,
                        "comment_id": comment_id,
                    },
                )

    except Exception as exc:  # noqa: BLE001
        log.exception(
            "Unexpected error during agent %d tick (action=%s): %s",
            agent.id,
            action_type,
            exc,
        )
        return None
    finally:
        if _should_close_client:
            await llm_client.close()

    # Persist the audit log entry regardless of success/failure
    try:
        db.log_agent_action(action_log)
        db.increment_agent_action_count(agent.id)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to log agent action for agent %d: %s", agent.id, exc)

    return sse_event
