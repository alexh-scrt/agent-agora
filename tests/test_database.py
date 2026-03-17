"""Unit tests for agent_agora.database CRUD helpers.

All tests run against a fresh in-memory SQLite database so there is no
disk I/O and no state leaks between test runs.
"""

from __future__ import annotations

import pytest

from agent_agora import database as db
from agent_agora.models import (
    AgentConfig,
    AgentCreate,
    AgentStatus,
    CommentCreate,
    CommentVoteCreate,
    PostCreate,
    PostVoteCreate,
    VoteValue,
    AgentActionCreate,
    ActionType,
    LLMProvider,
    Tone,
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
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str = "TestAgent") -> db.Agent:
    """Create a minimal agent for use in tests."""
    return db.create_agent(AgentCreate(name=name, config=AgentConfig()))


def _make_post(agent_id: int, title: str = "Hello World") -> db.Post:
    """Create a minimal post for use in tests."""
    return db.create_post(PostCreate(agent_id=agent_id, title=title, body="Body text."))


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


def test_create_agent_returns_agent_with_id() -> None:
    """Creating an agent should return an Agent with a positive id."""
    agent = _make_agent("Alice")
    assert agent.id > 0
    assert agent.name == "Alice"
    assert agent.status == AgentStatus.ACTIVE.value
    assert agent.action_count == 0


def test_create_agent_preserves_config() -> None:
    """AgentConfig fields should round-trip through the database."""
    config = AgentConfig(
        provider=LLMProvider.ANTHROPIC,
        tone=Tone.SARCASTIC,
        interests=["philosophy", "chess"],
        contrarianism=0.8,
    )
    agent = db.create_agent(AgentCreate(name="Bob", config=config))
    assert agent.config.provider == LLMProvider.ANTHROPIC.value
    assert agent.config.tone == Tone.SARCASTIC.value
    assert "philosophy" in agent.config.interests
    assert agent.config.contrarianism == pytest.approx(0.8)


def test_get_agent_returns_none_for_unknown_id() -> None:
    """get_agent should return None when the id does not exist."""
    result = db.get_agent(99999)
    assert result is None


def test_get_agent_roundtrip() -> None:
    """An agent created then fetched should have identical data."""
    created = _make_agent("Carol")
    fetched = db.get_agent(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == created.name


def test_list_agents_empty() -> None:
    """list_agents on an empty database should return an empty list."""
    assert db.list_agents() == []


def test_list_agents_multiple() -> None:
    """list_agents should return all created agents."""
    _make_agent("A")
    _make_agent("B")
    _make_agent("C")
    agents = db.list_agents()
    assert len(agents) == 3


def test_list_agents_filter_by_status() -> None:
    """list_agents with a status filter should only return matching agents."""
    a1 = _make_agent("Active1")
    a2 = _make_agent("Active2")
    db.update_agent_status(a2.id, AgentStatus.PAUSED)
    active = db.list_agents(status=AgentStatus.ACTIVE)
    paused = db.list_agents(status=AgentStatus.PAUSED)
    assert len(active) == 1
    assert active[0].id == a1.id
    assert len(paused) == 1
    assert paused[0].id == a2.id


def test_update_agent_status() -> None:
    """update_agent_status should change the status field."""
    agent = _make_agent()
    updated = db.update_agent_status(agent.id, AgentStatus.PAUSED)
    assert updated is not None
    assert updated.status == AgentStatus.PAUSED.value


def test_update_agent_status_unknown_id() -> None:
    """update_agent_status on a missing id should return None."""
    result = db.update_agent_status(99999, AgentStatus.PAUSED)
    assert result is None


def test_update_agent_config() -> None:
    """update_agent_config should persist the new configuration."""
    agent = _make_agent()
    new_config = AgentConfig(tone=Tone.HUMOROUS, contrarianism=0.9)
    updated = db.update_agent_config(agent.id, new_config)
    assert updated is not None
    assert updated.config.tone == Tone.HUMOROUS.value
    assert updated.config.contrarianism == pytest.approx(0.9)


def test_increment_agent_action_count() -> None:
    """increment_agent_action_count should increase action_count by one."""
    agent = _make_agent()
    assert agent.action_count == 0
    db.increment_agent_action_count(agent.id)
    db.increment_agent_action_count(agent.id)
    updated = db.get_agent(agent.id)
    assert updated is not None
    assert updated.action_count == 2


def test_delete_agent() -> None:
    """delete_agent should remove the agent from the database."""
    agent = _make_agent()
    result = db.delete_agent(agent.id)
    assert result is True
    assert db.get_agent(agent.id) is None


def test_delete_agent_not_found() -> None:
    """delete_agent on a missing id should return False."""
    assert db.delete_agent(99999) is False


# ---------------------------------------------------------------------------
# Post tests
# ---------------------------------------------------------------------------


def test_create_post() -> None:
    """create_post should return a Post with id and zero score."""
    agent = _make_agent()
    post = _make_post(agent.id, title="First Post")
    assert post.id > 0
    assert post.title == "First Post"
    assert post.body == "Body text."
    assert post.score == 0
    assert post.agent_id == agent.id


def test_get_post_returns_none_for_unknown_id() -> None:
    """get_post should return None when the id does not exist."""
    assert db.get_post(99999) is None


def test_get_post_roundtrip() -> None:
    """A post created then fetched should have identical data."""
    agent = _make_agent()
    created = _make_post(agent.id)
    fetched = db.get_post(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == created.title


def test_list_posts_empty() -> None:
    """list_posts on an empty database should return an empty list."""
    assert db.list_posts() == []


def test_list_posts_pagination() -> None:
    """list_posts should respect limit and offset."""
    agent = _make_agent()
    for i in range(5):
        _make_post(agent.id, title=f"Post {i}")
    page1 = db.list_posts(limit=3, offset=0)
    page2 = db.list_posts(limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 2


def test_list_posts_invalid_order_by() -> None:
    """list_posts should raise ValueError for unsafe column names."""
    with pytest.raises(ValueError, match="order_by"):
        db.list_posts(order_by="DROP TABLE agents")


def test_update_post_score() -> None:
    """update_post_score should adjust the score by the given delta."""
    agent = _make_agent()
    post = _make_post(agent.id)
    db.update_post_score(post.id, 5)
    updated = db.get_post(post.id)
    assert updated is not None
    assert updated.score == 5
    db.update_post_score(post.id, -2)
    updated2 = db.get_post(post.id)
    assert updated2 is not None
    assert updated2.score == 3


def test_delete_post() -> None:
    """delete_post should remove the post and return True."""
    agent = _make_agent()
    post = _make_post(agent.id)
    assert db.delete_post(post.id) is True
    assert db.get_post(post.id) is None


def test_delete_post_cascades_comments() -> None:
    """Deleting a post should also remove its comments."""
    agent = _make_agent()
    post = _make_post(agent.id)
    db.create_comment(CommentCreate(post_id=post.id, agent_id=agent.id, body="hi"))
    db.delete_post(post.id)
    assert db.list_comments_for_post(post.id) == []


# ---------------------------------------------------------------------------
# Comment tests
# ---------------------------------------------------------------------------


def test_create_comment() -> None:
    """create_comment should return a Comment with a positive id."""
    agent = _make_agent()
    post = _make_post(agent.id)
    comment = db.create_comment(
        CommentCreate(post_id=post.id, agent_id=agent.id, body="Great post!")
    )
    assert comment.id > 0
    assert comment.post_id == post.id
    assert comment.agent_id == agent.id
    assert comment.body == "Great post!"
    assert comment.score == 0
    assert comment.parent_comment_id is None


def test_create_nested_comment() -> None:
    """Nested replies should have parent_comment_id set correctly."""
    agent = _make_agent()
    post = _make_post(agent.id)
    parent = db.create_comment(
        CommentCreate(post_id=post.id, agent_id=agent.id, body="Parent")
    )
    child = db.create_comment(
        CommentCreate(
            post_id=post.id,
            agent_id=agent.id,
            body="Child",
            parent_comment_id=parent.id,
        )
    )
    assert child.parent_comment_id == parent.id


def test_list_comments_tree_structure() -> None:
    """list_comments_for_post should nest replies under their parents."""
    agent = _make_agent()
    post = _make_post(agent.id)
    root = db.create_comment(
        CommentCreate(post_id=post.id, agent_id=agent.id, body="Root")
    )
    db.create_comment(
        CommentCreate(
            post_id=post.id,
            agent_id=agent.id,
            body="Reply",
            parent_comment_id=root.id,
        )
    )
    tree = db.list_comments_for_post(post.id)
    assert len(tree) == 1  # Only one root-level comment
    assert len(tree[0].replies) == 1
    assert tree[0].replies[0].body == "Reply"


def test_update_comment_score() -> None:
    """update_comment_score should adjust the score."""
    agent = _make_agent()
    post = _make_post(agent.id)
    comment = db.create_comment(
        CommentCreate(post_id=post.id, agent_id=agent.id, body="Test")
    )
    db.update_comment_score(comment.id, 3)
    fetched = db.get_comment(comment.id)
    assert fetched is not None
    assert fetched.score == 3


# ---------------------------------------------------------------------------
# Vote tests
# ---------------------------------------------------------------------------


def test_cast_post_vote_upvote() -> None:
    """Casting an upvote on a post should increment the post score."""
    agent = _make_agent()
    post = _make_post(agent.id)
    vote = db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UPVOTE)
    )
    assert vote.value == VoteValue.UPVOTE.value
    updated_post = db.get_post(post.id)
    assert updated_post is not None
    assert updated_post.score == 1


def test_cast_post_vote_downvote() -> None:
    """Casting a downvote on a post should decrement the post score."""
    agent = _make_agent()
    post = _make_post(agent.id)
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.DOWNVOTE)
    )
    updated_post = db.get_post(post.id)
    assert updated_post is not None
    assert updated_post.score == -1


def test_cast_post_vote_replace() -> None:
    """Replacing an upvote with a downvote should adjust score by -2."""
    agent = _make_agent()
    post = _make_post(agent.id)
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UPVOTE)
    )
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.DOWNVOTE)
    )
    updated_post = db.get_post(post.id)
    assert updated_post is not None
    assert updated_post.score == -1


def test_cast_comment_vote() -> None:
    """Casting a vote on a comment should update the comment score."""
    agent = _make_agent()
    post = _make_post(agent.id)
    comment = db.create_comment(
        CommentCreate(post_id=post.id, agent_id=agent.id, body="Nice")
    )
    db.cast_comment_vote(
        CommentVoteCreate(
            agent_id=agent.id, comment_id=comment.id, value=VoteValue.UPVOTE
        )
    )
    fetched = db.get_comment(comment.id)
    assert fetched is not None
    assert fetched.score == 1


def test_list_votes_by_agent() -> None:
    """list_votes_by_agent should return all votes cast by the agent."""
    agent = _make_agent()
    post1 = _make_post(agent.id, "P1")
    post2 = _make_post(agent.id, "P2")
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post1.id, value=VoteValue.UPVOTE)
    )
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post2.id, value=VoteValue.DOWNVOTE)
    )
    votes = db.list_votes_by_agent(agent.id)
    assert len(votes) == 2


# ---------------------------------------------------------------------------
# AgentAction tests
# ---------------------------------------------------------------------------


def test_log_agent_action() -> None:
    """log_agent_action should persist an action and return it with an id."""
    agent = _make_agent()
    post = _make_post(agent.id)
    action = db.log_agent_action(
        AgentActionCreate(
            agent_id=agent.id,
            action_type=ActionType.POST,
            target_post_id=post.id,
            prompt_text="What should I post?",
            response_text="Here is my post.",
        )
    )
    assert action.id > 0
    assert action.agent_id == agent.id
    assert action.action_type == ActionType.POST.value
    assert action.target_post_id == post.id


def test_list_agent_actions_pagination() -> None:
    """list_agent_actions should paginate correctly."""
    agent = _make_agent()
    for _ in range(5):
        db.log_agent_action(
            AgentActionCreate(agent_id=agent.id, action_type=ActionType.VOTE)
        )
    page1 = db.list_agent_actions(agent.id, limit=3, offset=0)
    page2 = db.list_agent_actions(agent.id, limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 2


def test_list_recent_actions() -> None:
    """list_recent_actions should return actions across all agents."""
    a1 = _make_agent("Alpha")
    a2 = _make_agent("Beta")
    db.log_agent_action(AgentActionCreate(agent_id=a1.id, action_type=ActionType.POST))
    db.log_agent_action(AgentActionCreate(agent_id=a2.id, action_type=ActionType.COMMENT))
    actions = db.list_recent_actions(limit=10)
    agent_ids = {a.agent_id for a in actions}
    assert a1.id in agent_ids
    assert a2.id in agent_ids


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


def test_get_stats_empty() -> None:
    """Stats on an empty database should all be zero."""
    stats = db.get_stats()
    assert stats["agent_count"] == 0
    assert stats["post_count"] == 0
    assert stats["comment_count"] == 0
    assert stats["vote_count"] == 0


def test_get_stats_populated() -> None:
    """Stats should reflect created entities."""
    agent = _make_agent()
    post = _make_post(agent.id)
    db.create_comment(CommentCreate(post_id=post.id, agent_id=agent.id, body="Hi"))
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UPVOTE)
    )
    stats = db.get_stats()
    assert stats["agent_count"] == 1
    assert stats["active_agent_count"] == 1
    assert stats["post_count"] == 1
    assert stats["comment_count"] == 1
    assert stats["vote_count"] == 1


def test_delete_agent_cascades_posts_and_votes() -> None:
    """Deleting an agent should cascade-delete posts, comments and votes."""
    agent = _make_agent()
    post = _make_post(agent.id)
    db.create_comment(CommentCreate(post_id=post.id, agent_id=agent.id, body="reply"))
    db.cast_post_vote(
        PostVoteCreate(agent_id=agent.id, post_id=post.id, value=VoteValue.UPVOTE)
    )
    db.delete_agent(agent.id)
    stats = db.get_stats()
    assert stats["agent_count"] == 0
    assert stats["post_count"] == 0
    assert stats["comment_count"] == 0
    assert stats["vote_count"] == 0
