"""Unit tests for agent_agora.database CRUD operations.

All tests run against a fresh in-memory SQLite database so there is no
disk I/O and no state leaks between test runs.
"""

from __future__ import annotations

import pytest

from agent_agora import database as db
from agent_agora.models import (
    ActionType,
    AgentConfig,
    AgentCreate,
    AgentStatus,
    CommentCreate,
    CommentVoteCreate,
    LLMProvider,
    PostCreate,
    PostVoteCreate,
    Tone,
    VoteValue,
    AgentActionCreate,
)


# ---------------------------------------------------------------------------
# Fixture: isolated in-memory database
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all DB calls to a fresh in-memory database for each test."""
    monkeypatch.setattr(db, "_db_path", ":memory:")
    db.init_db(path=":memory:")


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_agent_create(
    name: str = "TestAgent",
    tone: Tone = Tone.NEUTRAL,
    interests: list[str] | None = None,
) -> AgentCreate:
    """Return a minimal AgentCreate fixture."""
    return AgentCreate(
        name=name,
        config=AgentConfig(
            tone=tone,
            interests=interests or ["technology"],
            provider=LLMProvider.OPENAI,
        ),
    )


def _make_post_create(agent_id: int, title: str = "Test Post", body: str = "Post body.") -> PostCreate:
    """Return a minimal PostCreate fixture."""
    return PostCreate(agent_id=agent_id, title=title, body=body)


def _make_comment_create(
    post_id: int,
    agent_id: int,
    body: str = "Test comment.",
    parent_comment_id: int | None = None,
) -> CommentCreate:
    """Return a minimal CommentCreate fixture."""
    return CommentCreate(
        post_id=post_id,
        agent_id=agent_id,
        body=body,
        parent_comment_id=parent_comment_id,
    )


# ---------------------------------------------------------------------------
# Agent CRUD
# ---------------------------------------------------------------------------


class TestAgentCRUD:
    """Tests for agent create / read / update / delete helpers."""

    def test_create_agent_returns_agent(self) -> None:
        """create_agent should return an Agent with a positive id."""
        agent = db.create_agent(_make_agent_create())
        assert agent.id > 0
        assert agent.name == "TestAgent"

    def test_create_agent_default_status_is_active(self) -> None:
        """Newly created agents should have status=active."""
        agent = db.create_agent(_make_agent_create())
        assert str(agent.status) == AgentStatus.ACTIVE.value

    def test_create_agent_preserves_config(self) -> None:
        """Agent config should be round-tripped correctly."""
        agent = db.create_agent(
            _make_agent_create(tone=Tone.SARCASTIC, interests=["chess", "art"])
        )
        assert str(agent.config.tone) == Tone.SARCASTIC.value
        assert "chess" in agent.config.interests
        assert "art" in agent.config.interests

    def test_create_agent_increments_ids(self) -> None:
        """Each new agent should receive a unique, incrementing id."""
        a1 = db.create_agent(_make_agent_create("AgentA"))
        a2 = db.create_agent(_make_agent_create("AgentB"))
        assert a1.id != a2.id
        assert a2.id > a1.id

    def test_get_agent_returns_existing(self) -> None:
        """get_agent should return the agent when it exists."""
        agent = db.create_agent(_make_agent_create())
        fetched = db.get_agent(agent.id)
        assert fetched is not None
        assert fetched.id == agent.id
        assert fetched.name == agent.name

    def test_get_agent_returns_none_for_missing(self) -> None:
        """get_agent should return None for a non-existent id."""
        result = db.get_agent(99999)
        assert result is None

    def test_list_agents_returns_all(self) -> None:
        """list_agents with no filter should return all agents."""
        db.create_agent(_make_agent_create("A"))
        db.create_agent(_make_agent_create("B"))
        agents = db.list_agents()
        assert len(agents) == 2

    def test_list_agents_filters_by_status(self) -> None:
        """list_agents with a status filter should return only matching agents."""
        a1 = db.create_agent(_make_agent_create("Active"))
        a2 = db.create_agent(_make_agent_create("Paused"))
        db.update_agent_status(a2.id, AgentStatus.PAUSED)

        active = db.list_agents(status=AgentStatus.ACTIVE)
        paused = db.list_agents(status=AgentStatus.PAUSED)

        assert any(a.id == a1.id for a in active)
        assert not any(a.id == a2.id for a in active)
        assert any(a.id == a2.id for a in paused)

    def test_update_agent_status(self) -> None:
        """update_agent_status should change the agent's status."""
        agent = db.create_agent(_make_agent_create())
        updated = db.update_agent_status(agent.id, AgentStatus.PAUSED)
        assert updated is not None
        assert str(updated.status) == AgentStatus.PAUSED.value

    def test_update_agent_status_returns_none_for_missing(self) -> None:
        """update_agent_status should return None when the agent does not exist."""
        result = db.update_agent_status(99999, AgentStatus.PAUSED)
        assert result is None

    def test_update_agent_config(self) -> None:
        """update_agent_config should replace the agent's configuration."""
        agent = db.create_agent(_make_agent_create(tone=Tone.NEUTRAL))
        new_config = AgentConfig(tone=Tone.ACADEMIC, interests=["math"])
        updated = db.update_agent_config(agent.id, new_config)
        assert updated is not None
        assert str(updated.config.tone) == Tone.ACADEMIC.value
        assert "math" in updated.config.interests

    def test_update_agent_config_returns_none_for_missing(self) -> None:
        """update_agent_config should return None when the agent does not exist."""
        config = AgentConfig()
        result = db.update_agent_config(99999, config)
        assert result is None

    def test_delete_agent_returns_true(self) -> None:
        """delete_agent should return True when the agent exists."""
        agent = db.create_agent(_make_agent_create())
        result = db.delete_agent(agent.id)
        assert result is True

    def test_delete_agent_returns_false_for_missing(self) -> None:
        """delete_agent should return False when the agent does not exist."""
        result = db.delete_agent(99999)
        assert result is False

    def test_delete_agent_removes_from_db(self) -> None:
        """After deletion, get_agent should return None."""
        agent = db.create_agent(_make_agent_create())
        db.delete_agent(agent.id)
        assert db.get_agent(agent.id) is None

    def test_increment_agent_action_count(self) -> None:
        """increment_agent_action_count should increase action_count by 1."""
        agent = db.create_agent(_make_agent_create())
        assert agent.action_count == 0
        db.increment_agent_action_count(agent.id)
        updated = db.get_agent(agent.id)
        assert updated is not None
        assert updated.action_count == 1

    def test_increment_agent_action_count_multiple_times(self) -> None:
        """Multiple increments should accumulate correctly."""
        agent = db.create_agent(_make_agent_create())
        for _ in range(5):
            db.increment_agent_action_count(agent.id)
        updated = db.get_agent(agent.id)
        assert updated is not None
        assert updated.action_count == 5

    def test_list_agents_empty_initially(self) -> None:
        """A fresh database should have no agents."""
        agents = db.list_agents()
        assert agents == []

    def test_agent_action_count_starts_at_zero(self) -> None:
        """Newly created agents should have action_count = 0."""
        agent = db.create_agent(_make_agent_create())
        assert agent.action_count == 0


# ---------------------------------------------------------------------------
# Post CRUD
# ---------------------------------------------------------------------------


class TestPostCRUD:
    """Tests for post create / read / list / delete helpers."""

    def test_create_post_returns_post(self) -> None:
        """create_post should return a Post with a positive id."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        assert post.id > 0
        assert post.title == "Test Post"
        assert post.body == "Post body."
        assert post.agent_id == agent.id

    def test_create_post_initial_score_is_zero(self) -> None:
        """Newly created posts should have score = 0."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        assert post.score == 0

    def test_get_post_returns_existing(self) -> None:
        """get_post should return the post when it exists."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        fetched = db.get_post(post.id)
        assert fetched is not None
        assert fetched.id == post.id
        assert fetched.title == post.title

    def test_get_post_returns_none_for_missing(self) -> None:
        """get_post should return None for a non-existent id."""
        result = db.get_post(99999)
        assert result is None

    def test_get_post_includes_comments_by_default(self) -> None:
        """get_post should include comments when include_comments=True."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.create_comment(_make_comment_create(post.id, agent.id, "Comment body"))
        fetched = db.get_post(post.id, include_comments=True)
        assert fetched is not None
        assert len(fetched.comments) == 1

    def test_get_post_excludes_comments_when_requested(self) -> None:
        """get_post should return an empty comments list when include_comments=False."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.create_comment(_make_comment_create(post.id, agent.id, "Comment body"))
        fetched = db.get_post(post.id, include_comments=False)
        assert fetched is not None
        assert fetched.comments == []

    def test_list_posts_returns_all(self) -> None:
        """list_posts should return all posts."""
        agent = db.create_agent(_make_agent_create())
        db.create_post(_make_post_create(agent.id, title="P1"))
        db.create_post(_make_post_create(agent.id, title="P2"))
        posts = db.list_posts(limit=10, offset=0, order_by="created_at", include_comments=False)
        assert len(posts) == 2

    def test_list_posts_pagination(self) -> None:
        """list_posts should respect limit and offset."""
        agent = db.create_agent(_make_agent_create())
        for i in range(5):
            db.create_post(_make_post_create(agent.id, title=f"Post {i}"))
        page1 = db.list_posts(limit=2, offset=0, order_by="id", include_comments=False)
        page2 = db.list_posts(limit=2, offset=2, order_by="id", include_comments=False)
        assert len(page1) == 2
        assert len(page2) == 2
        ids1 = {p.id for p in page1}
        ids2 = {p.id for p in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_posts_order_by_score(self) -> None:
        """list_posts ordered by score should return highest score first."""
        agent = db.create_agent(_make_agent_create())
        p_low  = db.create_post(_make_post_create(agent.id, title="Low"))
        p_high = db.create_post(_make_post_create(agent.id, title="High"))
        # Give high post a better score by upvoting
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=p_high.id, value=VoteValue.UP))
        posts = db.list_posts(limit=10, offset=0, order_by="score", include_comments=False)
        assert posts[0].id == p_high.id

    def test_delete_post_returns_true(self) -> None:
        """delete_post should return True when the post exists."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        result = db.delete_post(post.id)
        assert result is True

    def test_delete_post_returns_false_for_missing(self) -> None:
        """delete_post should return False when the post does not exist."""
        result = db.delete_post(99999)
        assert result is False

    def test_delete_post_removes_from_db(self) -> None:
        """After deletion, get_post should return None."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.delete_post(post.id)
        assert db.get_post(post.id) is None

    def test_list_posts_empty_initially(self) -> None:
        """A fresh database should have no posts."""
        posts = db.list_posts(limit=10, offset=0, order_by="created_at", include_comments=False)
        assert posts == []


# ---------------------------------------------------------------------------
# Comment CRUD
# ---------------------------------------------------------------------------


class TestCommentCRUD:
    """Tests for comment create / read / list helpers."""

    def test_create_comment_returns_comment(self) -> None:
        """create_comment should return a Comment with a positive id."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        assert comment.id > 0
        assert comment.body == "Test comment."
        assert comment.post_id == post.id
        assert comment.agent_id == agent.id
        assert comment.parent_comment_id is None

    def test_create_comment_initial_score_is_zero(self) -> None:
        """Newly created comments should have score = 0."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        assert comment.score == 0

    def test_create_reply_has_parent_comment_id(self) -> None:
        """A reply comment should store the parent_comment_id."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        parent = db.create_comment(_make_comment_create(post.id, agent.id, "Parent"))
        reply = db.create_comment(
            _make_comment_create(post.id, agent.id, "Reply", parent_comment_id=parent.id)
        )
        assert reply.parent_comment_id == parent.id

    def test_get_comment_returns_existing(self) -> None:
        """get_comment should return the comment when it exists."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        fetched = db.get_comment(comment.id)
        assert fetched is not None
        assert fetched.id == comment.id

    def test_get_comment_returns_none_for_missing(self) -> None:
        """get_comment should return None for a non-existent id."""
        result = db.get_comment(99999)
        assert result is None

    def test_list_comments_for_post_returns_top_level(self) -> None:
        """list_comments_for_post should return top-level comments."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.create_comment(_make_comment_create(post.id, agent.id, "C1"))
        db.create_comment(_make_comment_create(post.id, agent.id, "C2"))
        comments = db.list_comments_for_post(post.id)
        assert len(comments) == 2

    def test_list_comments_for_post_nests_replies(self) -> None:
        """Replies should appear nested inside their parent comment."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        parent = db.create_comment(_make_comment_create(post.id, agent.id, "Parent"))
        db.create_comment(
            _make_comment_create(post.id, agent.id, "Reply", parent_comment_id=parent.id)
        )
        comments = db.list_comments_for_post(post.id)
        # Only the parent should be at top level
        assert len(comments) == 1
        assert len(comments[0].replies) == 1
        assert comments[0].replies[0].body == "Reply"

    def test_list_comments_for_post_returns_empty_for_no_comments(self) -> None:
        """list_comments_for_post should return [] when there are no comments."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comments = db.list_comments_for_post(post.id)
        assert comments == []

    def test_create_multiple_top_level_comments(self) -> None:
        """Multiple top-level comments should all be listed."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        for i in range(4):
            db.create_comment(_make_comment_create(post.id, agent.id, f"Comment {i}"))
        comments = db.list_comments_for_post(post.id)
        assert len(comments) == 4


# ---------------------------------------------------------------------------
# Post votes
# ---------------------------------------------------------------------------


class TestPostVotes:
    """Tests for post vote casting and score computation."""

    def test_upvote_increases_post_score(self) -> None:
        """An upvote should increment post score by 1."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UP))
        updated = db.get_post(post.id)
        assert updated is not None
        assert updated.score == 1

    def test_downvote_decreases_post_score(self) -> None:
        """A downvote should decrement post score by 1."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.DOWN))
        updated = db.get_post(post.id)
        assert updated is not None
        assert updated.score == -1

    def test_multiple_upvotes_accumulate(self) -> None:
        """Multiple upvotes from different agents should accumulate."""
        poster = db.create_agent(_make_agent_create("Poster"))
        voter1 = db.create_agent(_make_agent_create("Voter1"))
        voter2 = db.create_agent(_make_agent_create("Voter2"))
        post = db.create_post(_make_post_create(poster.id))
        db.cast_post_vote(PostVoteCreate(agent_id=voter1.id, post_id=post.id, value=VoteValue.UP))
        db.cast_post_vote(PostVoteCreate(agent_id=voter2.id, post_id=post.id, value=VoteValue.UP))
        updated = db.get_post(post.id)
        assert updated is not None
        assert updated.score == 2

    def test_duplicate_vote_same_agent_does_not_double_count(self) -> None:
        """The same agent voting twice on the same post should not double the score."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UP))
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UP))
        updated = db.get_post(post.id)
        assert updated is not None
        assert updated.score == 1

    def test_vote_returns_vote_object(self) -> None:
        """cast_post_vote should return a Vote with the correct fields."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        vote = db.cast_post_vote(
            PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UP)
        )
        assert vote is not None
        assert vote.agent_id == agent.id
        assert vote.target_id == post.id

    def test_changing_vote_adjusts_score(self) -> None:
        """Changing a vote from up to down should adjust the post score accordingly."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UP))
        db.cast_post_vote(PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.DOWN))
        updated = db.get_post(post.id)
        assert updated is not None
        assert updated.score == -1


# ---------------------------------------------------------------------------
# Comment votes
# ---------------------------------------------------------------------------


class TestCommentVotes:
    """Tests for comment vote casting and score computation."""

    def test_upvote_increases_comment_score(self) -> None:
        """An upvote should increment comment score by 1."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        db.cast_comment_vote(
            CommentVoteCreate(agent_id=agent.id, comment_id=comment.id, value=VoteValue.UP)
        )
        updated = db.get_comment(comment.id)
        assert updated is not None
        assert updated.score == 1

    def test_downvote_decreases_comment_score(self) -> None:
        """A downvote should decrement comment score by 1."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        db.cast_comment_vote(
            CommentVoteCreate(agent_id=agent.id, comment_id=comment.id, value=VoteValue.DOWN)
        )
        updated = db.get_comment(comment.id)
        assert updated is not None
        assert updated.score == -1

    def test_multiple_comment_votes_accumulate(self) -> None:
        """Multiple upvotes from different agents should accumulate."""
        poster = db.create_agent(_make_agent_create("Poster"))
        voter1 = db.create_agent(_make_agent_create("Voter1"))
        voter2 = db.create_agent(_make_agent_create("Voter2"))
        post = db.create_post(_make_post_create(poster.id))
        comment = db.create_comment(_make_comment_create(post.id, poster.id))
        db.cast_comment_vote(
            CommentVoteCreate(agent_id=voter1.id, comment_id=comment.id, value=VoteValue.UP)
        )
        db.cast_comment_vote(
            CommentVoteCreate(agent_id=voter2.id, comment_id=comment.id, value=VoteValue.UP)
        )
        updated = db.get_comment(comment.id)
        assert updated is not None
        assert updated.score == 2

    def test_duplicate_comment_vote_does_not_double_count(self) -> None:
        """The same agent voting twice on the same comment should not double the score."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        db.cast_comment_vote(
            CommentVoteCreate(agent_id=agent.id, comment_id=comment.id, value=VoteValue.UP)
        )
        db.cast_comment_vote(
            CommentVoteCreate(agent_id=agent.id, comment_id=comment.id, value=VoteValue.UP)
        )
        updated = db.get_comment(comment.id)
        assert updated is not None
        assert updated.score == 1

    def test_comment_vote_returns_vote_object(self) -> None:
        """cast_comment_vote should return a Vote object."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        vote = db.cast_comment_vote(
            CommentVoteCreate(agent_id=agent.id, comment_id=comment.id, value=VoteValue.UP)
        )
        assert vote is not None
        assert vote.agent_id == agent.id
        assert vote.target_id == comment.id


# ---------------------------------------------------------------------------
# Agent actions
# ---------------------------------------------------------------------------


class TestAgentActions:
    """Tests for agent action logging and retrieval."""

    def test_log_agent_action_creates_entry(self) -> None:
        """log_agent_action should persist an action to the database."""
        agent = db.create_agent(_make_agent_create())
        db.log_agent_action(
            AgentActionCreate(agent_id=agent.id, action_type=ActionType.POST)
        )
        actions = db.list_agent_actions(agent.id)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.POST.value
        assert actions[0].agent_id == agent.id

    def test_log_agent_action_stores_prompt_text(self) -> None:
        """Prompt text should be stored and retrievable."""
        agent = db.create_agent(_make_agent_create())
        db.log_agent_action(
            AgentActionCreate(
                agent_id=agent.id,
                action_type=ActionType.COMMENT,
                prompt_text="User prompt",
                response_text="LLM response",
            )
        )
        actions = db.list_agent_actions(agent.id)
        assert len(actions) == 1
        assert actions[0].prompt_text == "User prompt"
        assert actions[0].response_text == "LLM response"

    def test_log_agent_action_stores_target_post_id(self) -> None:
        """target_post_id should be stored correctly."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.log_agent_action(
            AgentActionCreate(
                agent_id=agent.id,
                action_type=ActionType.POST,
                target_post_id=post.id,
            )
        )
        actions = db.list_agent_actions(agent.id)
        assert actions[0].target_post_id == post.id

    def test_list_agent_actions_respects_limit(self) -> None:
        """list_agent_actions should respect the limit parameter."""
        agent = db.create_agent(_make_agent_create())
        for _ in range(10):
            db.log_agent_action(
                AgentActionCreate(agent_id=agent.id, action_type=ActionType.VOTE)
            )
        actions = db.list_agent_actions(agent.id, limit=5)
        assert len(actions) == 5

    def test_list_agent_actions_respects_offset(self) -> None:
        """list_agent_actions should respect the offset parameter."""
        agent = db.create_agent(_make_agent_create())
        for _ in range(6):
            db.log_agent_action(
                AgentActionCreate(agent_id=agent.id, action_type=ActionType.POST)
            )
        page1 = db.list_agent_actions(agent.id, limit=3, offset=0)
        page2 = db.list_agent_actions(agent.id, limit=3, offset=3)
        ids1 = {a.id for a in page1}
        ids2 = {a.id for a in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_agent_actions_returns_empty_for_no_actions(self) -> None:
        """An agent with no actions should return an empty list."""
        agent = db.create_agent(_make_agent_create())
        actions = db.list_agent_actions(agent.id)
        assert actions == []

    def test_list_recent_actions_returns_across_agents(self) -> None:
        """list_recent_actions should return actions from multiple agents."""
        a1 = db.create_agent(_make_agent_create("A1"))
        a2 = db.create_agent(_make_agent_create("A2"))
        db.log_agent_action(AgentActionCreate(agent_id=a1.id, action_type=ActionType.POST))
        db.log_agent_action(AgentActionCreate(agent_id=a2.id, action_type=ActionType.COMMENT))
        actions = db.list_recent_actions(limit=10)
        agent_ids = {a.agent_id for a in actions}
        assert a1.id in agent_ids
        assert a2.id in agent_ids

    def test_list_recent_actions_respects_limit(self) -> None:
        """list_recent_actions should respect the limit."""
        agent = db.create_agent(_make_agent_create())
        for _ in range(15):
            db.log_agent_action(
                AgentActionCreate(agent_id=agent.id, action_type=ActionType.VOTE)
            )
        actions = db.list_recent_actions(limit=5)
        assert len(actions) == 5


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """Tests for the get_stats aggregate query."""

    def test_stats_empty_database(self) -> None:
        """A fresh database should return zero counts."""
        stats = db.get_stats()
        assert stats["agent_count"] == 0
        assert stats["active_agent_count"] == 0
        assert stats["post_count"] == 0
        assert stats["comment_count"] == 0
        assert stats["vote_count"] == 0
        assert stats["action_count"] == 0

    def test_stats_counts_agents(self) -> None:
        """agent_count should reflect the total number of agents."""
        db.create_agent(_make_agent_create("A"))
        db.create_agent(_make_agent_create("B"))
        stats = db.get_stats()
        assert stats["agent_count"] == 2

    def test_stats_counts_active_agents(self) -> None:
        """active_agent_count should reflect only active agents."""
        a1 = db.create_agent(_make_agent_create("A"))
        a2 = db.create_agent(_make_agent_create("B"))
        db.update_agent_status(a2.id, AgentStatus.PAUSED)
        stats = db.get_stats()
        assert stats["agent_count"] == 2
        assert stats["active_agent_count"] == 1

    def test_stats_counts_posts(self) -> None:
        """post_count should reflect the number of posts."""
        agent = db.create_agent(_make_agent_create())
        db.create_post(_make_post_create(agent.id, title="P1"))
        db.create_post(_make_post_create(agent.id, title="P2"))
        stats = db.get_stats()
        assert stats["post_count"] == 2

    def test_stats_counts_comments(self) -> None:
        """comment_count should reflect the number of comments."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.create_comment(_make_comment_create(post.id, agent.id, "C1"))
        db.create_comment(_make_comment_create(post.id, agent.id, "C2"))
        stats = db.get_stats()
        assert stats["comment_count"] == 2

    def test_stats_counts_votes(self) -> None:
        """vote_count should reflect the number of votes cast."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.cast_post_vote(
            PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UP)
        )
        stats = db.get_stats()
        assert stats["vote_count"] >= 1

    def test_stats_counts_actions(self) -> None:
        """action_count should reflect the number of logged agent actions."""
        agent = db.create_agent(_make_agent_create())
        db.log_agent_action(
            AgentActionCreate(agent_id=agent.id, action_type=ActionType.POST)
        )
        stats = db.get_stats()
        assert stats["action_count"] == 1

    def test_stats_returns_dict(self) -> None:
        """get_stats should return a dictionary."""
        stats = db.get_stats()
        assert isinstance(stats, dict)
        for key in ("agent_count", "active_agent_count", "post_count", "comment_count", "vote_count", "action_count"):
            assert key in stats


# ---------------------------------------------------------------------------
# Cascade deletion
# ---------------------------------------------------------------------------


class TestCascadeDeletion:
    """Tests for cascaded deletes (agent → posts → comments → votes)."""

    def test_delete_post_removes_its_comments(self) -> None:
        """Deleting a post should also delete its comments."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        comment = db.create_comment(_make_comment_create(post.id, agent.id))
        db.delete_post(post.id)
        assert db.get_comment(comment.id) is None

    def test_delete_agent_removes_associated_posts(self) -> None:
        """Deleting an agent should also cascade to their posts."""
        agent = db.create_agent(_make_agent_create())
        post = db.create_post(_make_post_create(agent.id))
        db.delete_agent(agent.id)
        assert db.get_post(post.id) is None

    def test_delete_agent_removes_associated_actions(self) -> None:
        """Deleting an agent should also remove their action log entries."""
        agent = db.create_agent(_make_agent_create())
        db.log_agent_action(
            AgentActionCreate(agent_id=agent.id, action_type=ActionType.POST)
        )
        db.delete_agent(agent.id)
        # After agent deletion, listing actions for that agent_id should be empty
        actions = db.list_agent_actions(agent.id)
        assert actions == []
