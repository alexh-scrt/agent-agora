"""Pydantic data models for Agent Agora.

Defines all domain models used throughout the application:
Agent, Post, Comment, Vote, and their configuration counterparts.
All models use Pydantic v2 with strict typing and validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM backend providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentStatus(str, Enum):
    """Lifecycle status of an agent."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"


class ActionType(str, Enum):
    """Types of actions an agent can take."""

    POST = "post"
    COMMENT = "comment"
    VOTE = "vote"


class VoteValue(int, Enum):
    """Valid vote values."""

    UPVOTE = 1
    DOWNVOTE = -1


class Tone(str, Enum):
    """Agent tone / communication style."""

    FRIENDLY = "friendly"
    SARCASTIC = "sarcastic"
    ACADEMIC = "academic"
    CASUAL = "casual"
    AGGRESSIVE = "aggressive"
    HUMOROUS = "humorous"
    NEUTRAL = "neutral"


class PoliticalLean(str, Enum):
    """Rough political orientation used to colour agent opinions."""

    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    FAR_RIGHT = "far_right"
    APOLITICAL = "apolitical"


# ---------------------------------------------------------------------------
# AgentConfig — configures personality traits for a single agent
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Configuration describing the personality and behaviour of an AI agent.

    This is stored as a JSON blob inside the ``agents`` table and is also
    used at runtime to construct the system prompt sent to the LLM.
    """

    model_config = ConfigDict(use_enum_values=True)

    # Provider / model selection
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM backend to use for this agent.",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Model identifier override. If None the application default is used."
        ),
    )

    # Personality traits
    tone: Tone = Field(
        default=Tone.NEUTRAL,
        description="Overall communication tone / style.",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="List of topic keywords the agent is interested in.",
    )
    political_lean: PoliticalLean = Field(
        default=PoliticalLean.APOLITICAL,
        description="Political orientation that colours the agent's opinions.",
    )
    contrarianism: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Probability [0, 1] that the agent takes the opposite stance "
            "to prevailing opinion."
        ),
    )
    verbosity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How long / detailed the agent's responses tend to be. "
            "0 = terse, 1 = very detailed."
        ),
    )
    custom_backstory: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text backstory injected into the system prompt.",
    )

    # Action weights — relative probabilities for each action type
    action_weight_post: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Relative weight for choosing to create a new post.",
    )
    action_weight_comment: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Relative weight for choosing to write a comment.",
    )
    action_weight_vote: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Relative weight for choosing to vote.",
    )

    @field_validator("interests", mode="before")
    @classmethod
    def _coerce_interests(cls, v: object) -> list[str]:
        """Accept a comma-separated string as well as a list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentBase(BaseModel):
    """Fields shared by agent creation and retrieval."""

    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(
        min_length=1,
        max_length=64,
        description="Human-readable display name for the agent.",
    )
    config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Personality and provider configuration for this agent.",
    )


class AgentCreate(AgentBase):
    """Payload for creating a new agent via the API."""


class Agent(AgentBase):
    """Full agent record as stored in and retrieved from the database."""

    model_config = ConfigDict(use_enum_values=True, from_attributes=True)

    id: int = Field(description="Auto-incrementing primary key.")
    status: AgentStatus = Field(
        default=AgentStatus.ACTIVE,
        description="Current lifecycle state of the agent.",
    )
    created_at: datetime = Field(description="UTC timestamp of agent creation.")
    updated_at: datetime = Field(description="UTC timestamp of last status change.")
    action_count: int = Field(
        default=0,
        description="Total number of actions this agent has taken.",
    )


# ---------------------------------------------------------------------------
# Post
# ---------------------------------------------------------------------------


class PostBase(BaseModel):
    """Fields shared by post creation and retrieval."""

    title: str = Field(
        min_length=1,
        max_length=300,
        description="Post headline / title.",
    )
    body: str = Field(
        min_length=1,
        max_length=10_000,
        description="Full body text of the post.",
    )


class PostCreate(PostBase):
    """Payload for inserting a new post."""

    agent_id: int = Field(description="ID of the agent authoring this post.")


class Post(PostBase):
    """Full post record retrieved from the database."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    score: int = Field(
        default=0,
        description="Net vote score (upvotes minus downvotes).",
    )
    created_at: datetime
    comments: list["Comment"] = Field(
        default_factory=list,
        description="Nested list of comments on this post.",
    )


# ---------------------------------------------------------------------------
# Comment
# ---------------------------------------------------------------------------


class CommentBase(BaseModel):
    """Fields shared by comment creation and retrieval."""

    body: str = Field(
        min_length=1,
        max_length=5_000,
        description="Comment text.",
    )


class CommentCreate(CommentBase):
    """Payload for inserting a new comment."""

    post_id: int = Field(description="ID of the post this comment belongs to.")
    agent_id: int = Field(description="ID of the agent authoring this comment.")
    parent_comment_id: Optional[int] = Field(
        default=None,
        description="If set, this comment is a reply to another comment.",
    )


class Comment(CommentBase):
    """Full comment record retrieved from the database."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    post_id: int
    agent_id: int
    parent_comment_id: Optional[int] = None
    score: int = Field(default=0)
    created_at: datetime
    replies: list["Comment"] = Field(
        default_factory=list,
        description="Nested replies to this comment.",
    )


# Resolve forward references for nested models
Post.model_rebuild()
Comment.model_rebuild()


# ---------------------------------------------------------------------------
# Vote
# ---------------------------------------------------------------------------


class VoteBase(BaseModel):
    """Fields shared by vote creation and retrieval."""

    model_config = ConfigDict(use_enum_values=True)

    agent_id: int = Field(description="ID of the agent casting this vote.")
    value: VoteValue = Field(description="+1 for upvote, -1 for downvote.")


class PostVoteCreate(VoteBase):
    """Payload for casting a vote on a post."""

    post_id: int


class CommentVoteCreate(VoteBase):
    """Payload for casting a vote on a comment."""

    comment_id: int


class Vote(VoteBase):
    """Full vote record retrieved from the database."""

    model_config = ConfigDict(use_enum_values=True, from_attributes=True)

    id: int
    post_id: Optional[int] = None
    comment_id: Optional[int] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# AgentAction — an audit-log entry for a single agent action
# ---------------------------------------------------------------------------


class AgentActionCreate(BaseModel):
    """Payload for logging an agent action."""

    model_config = ConfigDict(use_enum_values=True)

    agent_id: int
    action_type: ActionType
    target_post_id: Optional[int] = None
    target_comment_id: Optional[int] = None
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    metadata: Optional[str] = None  # JSON string for extra context


class AgentAction(AgentActionCreate):
    """Full agent action record retrieved from the database."""

    model_config = ConfigDict(use_enum_values=True, from_attributes=True)

    id: int
    created_at: datetime


# ---------------------------------------------------------------------------
# SSE event envelope
# ---------------------------------------------------------------------------


class SSEEvent(BaseModel):
    """Envelope for a Server-Sent Event broadcast to connected browsers."""

    event_type: str = Field(description="Discriminator string, e.g. 'new_post'.")
    agent_id: int
    agent_name: str
    action_type: ActionType
    payload: dict = Field(
        default_factory=dict,
        description="Action-specific data (post id, comment id, etc.).",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
