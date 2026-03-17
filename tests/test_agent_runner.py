"""Integration tests for agent_agora.agent_runner with a mocked LLM client.

All tests use an in-memory SQLite database and a mock LLM client so no
real API calls are made.  The tests exercise:
- Action selection weighting
- Post / comment / vote action handlers
- JSON response parsing (including edge cases)
- The main run_agent_tick orchestration function
- SSEEvent output shape
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_agora import database as db
from agent_agora.agent_runner import (
    _do_comment,
    _do_post,
    _do_vote,
    _parse_json_response,
    _select_action,
    run_agent_tick,
)
from agent_agora.models import (
    ActionType,
    AgentConfig,
    AgentCreate,
    AgentStatus,
    CommentCreate,
    LLMProvider,
    PostCreate,
    Tone,
    VoteValue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all DB calls to a fresh in-memory database for each test."""
    monkeypatch.setattr(db, "_db_path", ":memory:")
    db.init_db(path=":memory:")


@pytest.fixture
def agent() -> db.Agent:
    """Create and return a minimal active agent."""
    return db.create_agent(
        AgentCreate(
            name="TestBot",
            config=AgentConfig(
                tone=Tone.NEUTRAL,
                interests=["technology", "science"],
                contrarianism=0.1,
                verbosity=0.5,
                action_weight_post=0.34,
                action_weight_comment=0.33,
                action_weight_vote=0.33,
            ),
        )
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock LLMClient whose complete() coroutine can be configured."""
    client = MagicMock()
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# _parse_json_response tests
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Unit tests for the JSON response parsing helper."""

    def test_valid_json_object(self) -> None:
        """A valid JSON object string should be parsed correctly."""
        result = _parse_json_response('{"title": "Hello", "body": "World"}')
        assert result == {"title": "Hello", "body": "World"}

    def test_json_with_surrounding_whitespace(self) -> None:
        """Leading/trailing whitespace should be stripped before parsing."""
        result = _parse_json_response('   {"vote": 1}   ')
        assert result == {"vote": 1}

    def test_json_in_markdown_fence(self) -> None:
        """JSON wrapped in markdown code fences should be extracted."""
        text = '```json\n{"body": "Nice post!"}\n```'
        result = _parse_json_response(text)
        assert result == {"body": "Nice post!"}

    def test_json_in_plain_fence(self) -> None:
        """JSON wrapped in plain ``` fences should be extracted."""
        text = '```\n{"vote": -1}\n```'
        result = _parse_json_response(text)
        assert result == {"vote": -1}

    def test_no_json_returns_none(self) -> None:
        """Text with no JSON object should return None."""
        result = _parse_json_response("I have no curly braces here.")
        assert result is None

    def test_invalid_json_returns_none(self) -> None:
        """Malformed JSON should return None."""
        result = _parse_json_response("{not valid json}")
        assert result is None

    def test_json_array_returns_none(self) -> None:
        """A JSON array (not object) should return None."""
        # The helper looks for { ... } so an array won't be found as a dict
        result = _parse_json_response('["a", "b"]')
        assert result is None

    def test_json_with_preamble(self) -> None:
        """JSON preceded by prose text should still be parsed."""
        text = 'Here is my response: {"body": "Great post!"}'
        result = _parse_json_response(text)
        assert result == {"body": "Great post!"}

    def test_empty_string_returns_none(self) -> None:
        """An empty string should return None."""
        result = _parse_json_response("")
        assert result is None


# ---------------------------------------------------------------------------
# _select_action tests
# ---------------------------------------------------------------------------


class TestSelectAction:
    """Unit tests for action selection weighting."""

    def test_returns_valid_action_type(self) -> None:
        """_select_action should return an ActionType."""
        config = AgentConfig()
        action = _select_action(config)
        assert action in ActionType.__members__.values()

    def test_post_only_weight(self) -> None:
        """With post weight=1 and others=0, should always select POST."""
        config = AgentConfig(
            action_weight_post=1.0,
            action_weight_comment=0.0,
            action_weight_vote=0.0,
        )
        for _ in range(20):
            assert _select_action(config) == ActionType.POST

    def test_comment_only_weight(self) -> None:
        """With comment weight=1 and others=0, should always select COMMENT."""
        config = AgentConfig(
            action_weight_post=0.0,
            action_weight_comment=1.0,
            action_weight_vote=0.0,
        )
        for _ in range(20):
            assert _select_action(config) == ActionType.COMMENT

    def test_vote_only_weight(self) -> None:
        """With vote weight=1 and others=0, should always select VOTE."""
        config = AgentConfig(
            action_weight_post=0.0,
            action_weight_comment=0.0,
            action_weight_vote=1.0,
        )
        for _ in range(20):
            assert _select_action(config) == ActionType.VOTE

    def test_all_zero_weights_still_returns_action(self) -> None:
        """Zero weights should fall back to a uniform choice without error."""
        config = AgentConfig(
            action_weight_post=0.0,
            action_weight_comment=0.0,
            action_weight_vote=0.0,
        )
        action = _select_action(config)
        assert action in ActionType.__members__.values()

    def test_distribution_is_roughly_correct(self) -> None:
        """50/50 post/comment should produce roughly equal distribution."""
        config = AgentConfig(
            action_weight_post=0.5,
            action_weight_comment=0.5,
            action_weight_vote=0.0,
        )
        counts = {ActionType.POST: 0, ActionType.COMMENT: 0, ActionType.VOTE: 0}
        for _ in range(400):
            counts[_select_action(config)] += 1
        # Both should be between 150 and 250 out of 400 trials
        assert 150 <= counts[ActionType.POST] <= 250
        assert 150 <= counts[ActionType.COMMENT] <= 250
        assert counts[ActionType.VOTE] == 0


# ---------------------------------------------------------------------------
# _do_post tests
# ---------------------------------------------------------------------------


class TestDoPost:
    """Integration tests for the _do_post action handler."""

    @pytest.mark.asyncio
    async def test_creates_post_on_valid_response(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """A valid LLM JSON response should create a post and return it."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "My Great Post", "body": "Post body here."})
        )
        result = await _do_post(agent, mock_llm)
        assert result is not None
        post, prompt_text, response_text = result
        assert post.id > 0
        assert post.title == "My Great Post"
        assert post.body == "Post body here."
        assert post.agent_id == agent.id

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """An unparseable LLM response should return None without crashing."""
        mock_llm.complete = AsyncMock(return_value="Sorry, I cannot do that.")
        result = await _do_post(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_title(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """A response missing the title key should return None."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"body": "Body without title."})
        )
        result = await _do_post(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_body(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """A response missing the body key should return None."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "Title without body."})
        )
        result = await _do_post(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """An LLMError should be caught and None returned."""
        from agent_agora.llm_client import LLMError
        mock_llm.complete = AsyncMock(side_effect=LLMError("API down"))
        result = await _do_post(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_prompt_text_returned(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """The prompt text used for the LLM call should be returned in the tuple."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "T", "body": "B"})
        )
        result = await _do_post(agent, mock_llm)
        assert result is not None
        _, prompt_text, _ = result
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0

    @pytest.mark.asyncio
    async def test_title_truncated_to_300_chars(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """Titles longer than 300 characters should be truncated."""
        long_title = "A" * 500
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": long_title, "body": "Body"})
        )
        result = await _do_post(agent, mock_llm)
        assert result is not None
        post, _, _ = result
        assert len(post.title) <= 300


# ---------------------------------------------------------------------------
# _do_comment tests
# ---------------------------------------------------------------------------


class TestDoComment:
    """Integration tests for the _do_comment action handler."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_posts(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """Without any posts, _do_comment should return None."""
        mock_llm.complete = AsyncMock(return_value=json.dumps({"body": "Nice!"}))
        result = await _do_comment(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_comment_on_valid_response(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A valid LLM response should create a top-level comment."""
        post = db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"body": "Interesting take!"}))

        # Force top-level comment (no reply) by patching random
        with patch("agent_agora.agent_runner.random.random", return_value=0.9):
            result = await _do_comment(agent, mock_llm)

        assert result is not None
        comment, target_post, prompt_text, response_text = result
        assert comment.id > 0
        assert comment.body == "Interesting take!"
        assert comment.post_id == post.id
        assert comment.parent_comment_id is None

    @pytest.mark.asyncio
    async def test_creates_reply_to_existing_comment(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """With existing comments and random forcing reply, should create a reply."""
        post = db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        parent = db.create_comment(
            CommentCreate(post_id=post.id, agent_id=agent.id, body="Parent comment")
        )
        mock_llm.complete = AsyncMock(return_value=json.dumps({"body": "I agree!"}))

        # Force reply path
        with patch("agent_agora.agent_runner.random.random", return_value=0.1):
            result = await _do_comment(agent, mock_llm)

        assert result is not None
        comment, target_post, prompt_text, response_text = result
        assert comment.parent_comment_id == parent.id

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """An unparseable response should return None."""
        db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(return_value="not json")
        result = await _do_comment(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """An LLMError should be caught and None returned."""
        from agent_agora.llm_client import LLMError
        db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(side_effect=LLMError("timeout"))
        result = await _do_comment(agent, mock_llm)
        assert result is None


# ---------------------------------------------------------------------------
# _do_vote tests
# ---------------------------------------------------------------------------


class TestDoVote:
    """Integration tests for the _do_vote action handler."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_posts(self, agent: db.Agent, mock_llm: MagicMock) -> None:
        """Without any posts, _do_vote should return None."""
        mock_llm.complete = AsyncMock(return_value=json.dumps({"vote": 1}))
        result = await _do_vote(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_upvote_on_post(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """Voting +1 on a post should increment its score."""
        post = db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"vote": 1}))

        # Force post vote (not comment vote)
        with patch("agent_agora.agent_runner.random.random", return_value=0.9):
            result = await _do_vote(agent, mock_llm)

        assert result is not None
        vote_value, comment_id, target_post, prompt_text, response_text = result
        assert vote_value == 1
        assert comment_id is None

        updated_post = db.get_post(post.id)
        assert updated_post is not None
        assert updated_post.score == 1

    @pytest.mark.asyncio
    async def test_downvote_on_post(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """Voting -1 on a post should decrement its score."""
        post = db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"vote": -1}))

        with patch("agent_agora.agent_runner.random.random", return_value=0.9):
            result = await _do_vote(agent, mock_llm)

        assert result is not None
        vote_value, *_ = result
        assert vote_value == -1

        updated_post = db.get_post(post.id)
        assert updated_post is not None
        assert updated_post.score == -1

    @pytest.mark.asyncio
    async def test_vote_on_comment(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """Voting on a comment should update the comment score."""
        post = db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        comment = db.create_comment(
            CommentCreate(post_id=post.id, agent_id=agent.id, body="A comment")
        )
        mock_llm.complete = AsyncMock(return_value=json.dumps({"vote": 1}))

        # Force comment vote path
        with patch("agent_agora.agent_runner.random.random", return_value=0.1):
            result = await _do_vote(agent, mock_llm)

        assert result is not None
        vote_value, comment_id, target_post, prompt_text, response_text = result
        assert comment_id == comment.id

        updated_comment = db.get_comment(comment.id)
        assert updated_comment is not None
        assert updated_comment.score == 1

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_vote_value(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A vote value other than 1 or -1 should return None."""
        db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"vote": 99}))
        result = await _do_vote(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_vote_key(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A JSON response without the 'vote' key should return None."""
        db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"body": "oops"}))
        result = await _do_vote(agent, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """An LLMError during vote should be caught and None returned."""
        from agent_agora.llm_client import LLMError
        db.create_post(PostCreate(agent_id=agent.id, title="Test", body="Body"))
        mock_llm.complete = AsyncMock(side_effect=LLMError("rate limited"))
        result = await _do_vote(agent, mock_llm)
        assert result is None


# ---------------------------------------------------------------------------
# run_agent_tick tests
# ---------------------------------------------------------------------------


class TestRunAgentTick:
    """Integration tests for the main run_agent_tick orchestrator."""

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_agent(self) -> None:
        """Ticking a non-existent agent should return None."""
        result = await run_agent_tick(99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_paused_agent(self, agent: db.Agent) -> None:
        """Ticking a paused agent should return None without calling the LLM."""
        db.update_agent_status(agent.id, AgentStatus.PAUSED)
        result = await run_agent_tick(agent.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_stopped_agent(self, agent: db.Agent) -> None:
        """Ticking a stopped agent should return None."""
        db.update_agent_status(agent.id, AgentStatus.STOPPED)
        result = await run_agent_tick(agent.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_post_action_returns_sse_event(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A successful post action should return an SSEEvent with event_type='new_post'."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "SSE Test Post", "body": "Body text."})
        )

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.POST), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm):
            event = await run_agent_tick(agent.id)

        assert event is not None
        assert event.event_type == "new_post"
        assert event.agent_id == agent.id
        assert event.agent_name == agent.name
        assert event.action_type == ActionType.POST
        assert "post_id" in event.payload
        assert "title" in event.payload

    @pytest.mark.asyncio
    async def test_comment_action_returns_sse_event(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A successful comment action should return an SSEEvent with event_type='new_comment'."""
        db.create_post(PostCreate(agent_id=agent.id, title="Post", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"body": "Great post!"}))

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.COMMENT), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm), \
             patch("agent_agora.agent_runner.random.random", return_value=0.9):
            event = await run_agent_tick(agent.id)

        assert event is not None
        assert event.event_type == "new_comment"
        assert event.agent_id == agent.id
        assert "comment_id" in event.payload
        assert "post_id" in event.payload

    @pytest.mark.asyncio
    async def test_vote_action_returns_sse_event(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A successful vote action should return an SSEEvent with event_type='new_vote'."""
        db.create_post(PostCreate(agent_id=agent.id, title="Post", body="Body"))
        mock_llm.complete = AsyncMock(return_value=json.dumps({"vote": 1}))

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.VOTE), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm), \
             patch("agent_agora.agent_runner.random.random", return_value=0.9):
            event = await run_agent_tick(agent.id)

        assert event is not None
        assert event.event_type == "new_vote"
        assert "vote_value" in event.payload

    @pytest.mark.asyncio
    async def test_increments_action_count_on_success(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A successful tick should increment the agent's action_count by 1."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "Count Test", "body": "Body."})
        )

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.POST), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm):
            await run_agent_tick(agent.id)

        updated = db.get_agent(agent.id)
        assert updated is not None
        assert updated.action_count == 1

    @pytest.mark.asyncio
    async def test_logs_agent_action_on_success(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """A successful tick should persist an AgentAction audit log entry."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "Log Test", "body": "Body."})
        )

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.POST), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm):
            await run_agent_tick(agent.id)

        actions = db.list_agent_actions(agent.id)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.POST.value

    @pytest.mark.asyncio
    async def test_returns_none_when_action_fails(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """When the action handler returns None the tick should return None."""
        # No posts exist, so comment action will find nothing to comment on
        mock_llm.complete = AsyncMock(return_value=json.dumps({"body": "No posts!"}))

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.COMMENT), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm):
            event = await run_agent_tick(agent.id)

        assert event is None

    @pytest.mark.asyncio
    async def test_sse_event_has_timestamp(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """The SSE event should carry a timestamp."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "Timestamp Test", "body": "Body."})
        )

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.POST), \
             patch("agent_agora.agent_runner.create_llm_client", return_value=mock_llm):
            event = await run_agent_tick(agent.id)

        assert event is not None
        assert isinstance(event.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_uses_provided_llm_client(
        self, agent: db.Agent, mock_llm: MagicMock
    ) -> None:
        """When a client is explicitly passed, create_llm_client should not be called."""
        mock_llm.complete = AsyncMock(
            return_value=json.dumps({"title": "Provided client", "body": "Body."})
        )

        with patch("agent_agora.agent_runner._select_action", return_value=ActionType.POST), \
             patch("agent_agora.agent_runner.create_llm_client") as mock_factory:
            await run_agent_tick(agent.id, llm_client=mock_llm)

        mock_factory.assert_not_called()
        # Provided client should NOT be closed by the runner
        mock_llm.close.assert_not_called()
