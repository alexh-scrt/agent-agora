"""Unit tests for agent_agora.personas — system-prompt and action-prompt generation.

All tests are purely functional: no LLM calls are made and no database is needed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent_agora.models import (
    AgentConfig,
    Comment,
    LLMProvider,
    PoliticalLean,
    Post,
    Tone,
)
from agent_agora.personas import (
    BUILT_IN_PERSONAS,
    build_comment_prompt,
    build_post_prompt,
    build_system_prompt,
    build_vote_prompt,
    get_persona,
    get_persona_names,
)

from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: object) -> AgentConfig:
    """Return an AgentConfig with optional overrides."""
    return AgentConfig(**kwargs)  # type: ignore[arg-type]


def _make_post(
    title: str = "Test Post",
    body: str = "This is the post body.",
    post_id: int = 1,
    agent_id: int = 1,
    score: int = 0,
) -> Post:
    """Return a minimal Post fixture."""
    return Post(
        id=post_id,
        agent_id=agent_id,
        title=title,
        body=body,
        score=score,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        comments=[],
    )


def _make_comment(
    body: str = "Test comment body.",
    comment_id: int = 1,
    post_id: int = 1,
    agent_id: int = 2,
) -> Comment:
    """Return a minimal Comment fixture."""
    return Comment(
        id=comment_id,
        post_id=post_id,
        agent_id=agent_id,
        body=body,
        score=0,
        created_at=datetime(2024, 1, 1, 12, 5, 0),
        replies=[],
    )


# ---------------------------------------------------------------------------
# build_system_prompt tests
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for build_system_prompt."""

    def test_contains_agent_name(self) -> None:
        """The system prompt must reference the agent's name."""
        prompt = build_system_prompt("Alice", _make_config())
        assert "Alice" in prompt

    def test_contains_tone_description(self) -> None:
        """Tone information should appear in the prompt."""
        prompt = build_system_prompt("Bob", _make_config(tone=Tone.SARCASTIC))
        # The tone description for sarcastic mentions irony or sarcasm
        lower = prompt.lower()
        assert "sarcasm" in lower or "sardonic" in lower or "irony" in lower

    def test_contains_interests(self) -> None:
        """Listed interests should appear in the prompt."""
        config = _make_config(interests=["chess", "philosophy"])
        prompt = build_system_prompt("Carol", config)
        assert "chess" in prompt
        assert "philosophy" in prompt

    def test_no_interests_uses_fallback(self) -> None:
        """When there are no interests the fallback text is used."""
        config = _make_config(interests=[])
        prompt = build_system_prompt("Dana", config)
        assert "range of topics" in prompt or "anything" in prompt or "wide" in prompt

    def test_contains_political_lean(self) -> None:
        """Political lean description should appear in the prompt."""
        config = _make_config(political_lean=PoliticalLean.FAR_LEFT)
        prompt = build_system_prompt("Eve", config)
        assert "progressive" in prompt.lower() or "socialist" in prompt.lower()

    def test_apolitical_lean(self) -> None:
        """Apolitical lean should include apolitical description."""
        config = _make_config(political_lean=PoliticalLean.APOLITICAL)
        prompt = build_system_prompt("Frank", config)
        assert "uninterested" in prompt.lower() or "apolitical" in prompt.lower() or "avoids" in prompt.lower()

    def test_custom_backstory_included(self) -> None:
        """Custom backstory should appear verbatim in the system prompt."""
        backstory = "You grew up in a lighthouse and love maritime history."
        config = _make_config(custom_backstory=backstory)
        prompt = build_system_prompt("Grace", config)
        assert backstory in prompt

    def test_no_custom_backstory(self) -> None:
        """When backstory is None the prompt should not contain the Background label."""
        config = _make_config(custom_backstory=None)
        prompt = build_system_prompt("Hank", config)
        assert "Background:" not in prompt

    def test_high_contrarianism(self) -> None:
        """High contrarianism should produce strongly contrarian language."""
        config = _make_config(contrarianism=0.95)
        prompt = build_system_prompt("Ivan", config)
        lower = prompt.lower()
        assert "contrarian" in lower or "oppose" in lower or "disagree" in lower

    def test_low_contrarianism(self) -> None:
        """Low contrarianism should produce agreeable language."""
        config = _make_config(contrarianism=0.05)
        prompt = build_system_prompt("Jane", config)
        lower = prompt.lower()
        assert "agree" in lower or "rarely" in lower

    def test_high_verbosity(self) -> None:
        """High verbosity should include instruction for detailed responses."""
        config = _make_config(verbosity=0.95)
        prompt = build_system_prompt("Karl", config)
        lower = prompt.lower()
        assert "verbose" in lower or "detailed" in lower or "long" in lower

    def test_low_verbosity(self) -> None:
        """Low verbosity should include instruction for short responses."""
        config = _make_config(verbosity=0.05)
        prompt = build_system_prompt("Lily", config)
        lower = prompt.lower()
        assert "short" in lower or "concise" in lower or "brief" in lower

    def test_no_ai_revelation_instruction(self) -> None:
        """The prompt should instruct the agent not to reveal it is an AI."""
        prompt = build_system_prompt("Max", _make_config())
        lower = prompt.lower()
        assert "never reveal" in lower or "do not reveal" in lower or "ai" in lower

    def test_prompt_is_nonempty_string(self) -> None:
        """The returned value should be a non-empty string."""
        prompt = build_system_prompt("Nina", _make_config())
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_all_tones_produce_nonempty_prompts(self) -> None:
        """Every Tone value should produce a valid non-empty system prompt."""
        for tone in Tone:
            config = _make_config(tone=tone)
            prompt = build_system_prompt("Olive", config)
            assert len(prompt) > 50, f"Empty prompt for tone {tone}"

    def test_all_political_leans_produce_nonempty_prompts(self) -> None:
        """Every PoliticalLean value should produce a valid non-empty system prompt."""
        for lean in PoliticalLean:
            config = _make_config(political_lean=lean)
            prompt = build_system_prompt("Pete", config)
            assert len(prompt) > 50, f"Empty prompt for lean {lean}"


# ---------------------------------------------------------------------------
# build_post_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPostPrompt:
    """Tests for build_post_prompt."""

    def test_returns_string(self) -> None:
        """build_post_prompt should return a non-empty string."""
        prompt = build_post_prompt("Alice", _make_config())
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_contains_json_instruction(self) -> None:
        """The prompt must instruct the agent to return a JSON object."""
        prompt = build_post_prompt("Bob", _make_config())
        assert "JSON" in prompt
        assert "title" in prompt
        assert "body" in prompt

    def test_includes_interests(self) -> None:
        """Agent interests should appear in the post prompt."""
        config = _make_config(interests=["robotics", "space"])
        prompt = build_post_prompt("Carol", config)
        assert "robotics" in prompt
        assert "space" in prompt

    def test_recent_posts_context(self) -> None:
        """When recent posts are supplied their titles should appear in the prompt."""
        posts = [
            _make_post(title="How to brew kombucha"),
            _make_post(title="The fall of Rome revisited", post_id=2),
        ]
        prompt = build_post_prompt("Dana", _make_config(), recent_posts=posts)
        assert "How to brew kombucha" in prompt
        assert "The fall of Rome revisited" in prompt

    def test_no_recent_posts(self) -> None:
        """Absence of recent posts should not cause an error."""
        prompt = build_post_prompt("Eve", _make_config(), recent_posts=None)
        assert isinstance(prompt, str)

    def test_empty_recent_posts_list(self) -> None:
        """Empty recent_posts list should not include a context section."""
        prompt = build_post_prompt("Frank", _make_config(), recent_posts=[])
        assert "Recent posts" not in prompt


# ---------------------------------------------------------------------------
# build_comment_prompt tests
# ---------------------------------------------------------------------------


class TestBuildCommentPrompt:
    """Tests for build_comment_prompt."""

    def test_returns_string(self) -> None:
        """build_comment_prompt should return a non-empty string."""
        post = _make_post()
        prompt = build_comment_prompt("Alice", _make_config(), post)
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_contains_post_title(self) -> None:
        """The post title should appear in the comment prompt."""
        post = _make_post(title="A fascinating article about bees")
        prompt = build_comment_prompt("Bob", _make_config(), post)
        assert "A fascinating article about bees" in prompt

    def test_contains_post_body(self) -> None:
        """Post body content should appear in the comment prompt."""
        post = _make_post(body="Bees are incredible creatures.")
        prompt = build_comment_prompt("Carol", _make_config(), post)
        assert "Bees are incredible creatures." in prompt

    def test_post_body_truncated(self) -> None:
        """Very long post bodies should be truncated to avoid huge prompts."""
        long_body = "x" * 2000
        post = _make_post(body=long_body)
        prompt = build_comment_prompt("Dana", _make_config(), post)
        # The full body should not appear; the prompt should be much shorter than 2000 chars for body
        assert long_body not in prompt
        assert "..." in prompt

    def test_contains_json_instruction(self) -> None:
        """The prompt must request a JSON object with a 'body' key."""
        post = _make_post()
        prompt = build_comment_prompt("Eve", _make_config(), post)
        assert "JSON" in prompt
        assert "body" in prompt

    def test_reply_includes_parent_comment(self) -> None:
        """When a parent comment is given it should appear in the prompt."""
        post = _make_post()
        parent = _make_comment(body="I think this is a great point.")
        prompt = build_comment_prompt("Frank", _make_config(), post, parent_comment=parent)
        assert "I think this is a great point." in prompt

    def test_reply_uses_reply_verb(self) -> None:
        """Prompt for a reply should use 'reply' rather than 'comment'."""
        post = _make_post()
        parent = _make_comment()
        prompt = build_comment_prompt("Grace", _make_config(), post, parent_comment=parent)
        assert "reply" in prompt.lower()

    def test_top_level_comment_uses_comment_verb(self) -> None:
        """Prompt for a top-level comment should use 'comment'."""
        post = _make_post()
        prompt = build_comment_prompt("Hank", _make_config(), post, parent_comment=None)
        assert "comment" in prompt.lower()

    def test_contrarian_hint_injected_with_high_contrarianism(self) -> None:
        """With contrarianism=1.0 the contrarian hint should always appear."""
        config = _make_config(contrarianism=1.0)
        post = _make_post()
        # Patch random.random to always return 0.0 so the branch is taken
        with patch("agent_agora.personas.random.random", return_value=0.0):
            prompt = build_comment_prompt("Ivan", config, post)
        assert "contrarian" in prompt.lower() or "push back" in prompt.lower() or "disagree" in prompt.lower()


# ---------------------------------------------------------------------------
# build_vote_prompt tests
# ---------------------------------------------------------------------------


class TestBuildVotePrompt:
    """Tests for build_vote_prompt."""

    def test_returns_string(self) -> None:
        """build_vote_prompt should return a non-empty string."""
        post = _make_post()
        prompt = build_vote_prompt("Alice", _make_config(), post)
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_contains_json_instruction(self) -> None:
        """The prompt must request a JSON object with a 'vote' key."""
        post = _make_post()
        prompt = build_vote_prompt("Bob", _make_config(), post)
        assert "JSON" in prompt
        assert "vote" in prompt
        assert "1" in prompt
        assert "-1" in prompt

    def test_post_vote_contains_post_title(self) -> None:
        """A post-level vote prompt should reference the post title."""
        post = _make_post(title="An essay on stoicism")
        prompt = build_vote_prompt("Carol", _make_config(), post)
        assert "An essay on stoicism" in prompt

    def test_comment_vote_contains_comment_body(self) -> None:
        """A comment-level vote prompt should include the comment body."""
        post = _make_post()
        comment = _make_comment(body="Stoicism is overrated.")
        prompt = build_vote_prompt("Dana", _make_config(), post, target_comment=comment)
        assert "Stoicism is overrated." in prompt

    def test_high_contrarianism_label(self) -> None:
        """High contrarianism should be labelled 'high' in the vote prompt."""
        config = _make_config(contrarianism=0.9)
        post = _make_post()
        prompt = build_vote_prompt("Eve", config, post)
        assert "high" in prompt.lower()

    def test_low_contrarianism_label(self) -> None:
        """Low contrarianism should be labelled 'low' in the vote prompt."""
        config = _make_config(contrarianism=0.05)
        post = _make_post()
        prompt = build_vote_prompt("Frank", config, post)
        assert "low" in prompt.lower()


# ---------------------------------------------------------------------------
# Built-in personas tests
# ---------------------------------------------------------------------------


class TestBuiltInPersonas:
    """Tests for the BUILT_IN_PERSONAS registry."""

    def test_at_least_five_personas(self) -> None:
        """There should be at least five built-in personas."""
        assert len(BUILT_IN_PERSONAS) >= 5

    def test_all_personas_are_agent_configs(self) -> None:
        """Every entry in BUILT_IN_PERSONAS must be an AgentConfig instance."""
        for slug, config in BUILT_IN_PERSONAS.items():
            assert isinstance(config, AgentConfig), f"{slug!r} is not an AgentConfig"

    def test_all_personas_have_valid_contrarianism(self) -> None:
        """All personas must have contrarianism in [0, 1]."""
        for slug, config in BUILT_IN_PERSONAS.items():
            assert 0.0 <= config.contrarianism <= 1.0, (
                f"{slug!r} has out-of-range contrarianism {config.contrarianism}"
            )

    def test_all_personas_have_valid_verbosity(self) -> None:
        """All personas must have verbosity in [0, 1]."""
        for slug, config in BUILT_IN_PERSONAS.items():
            assert 0.0 <= config.verbosity <= 1.0, (
                f"{slug!r} has out-of-range verbosity {config.verbosity}"
            )

    def test_all_personas_have_valid_action_weights(self) -> None:
        """All action weight fields must be in [0, 1]."""
        for slug, config in BUILT_IN_PERSONAS.items():
            for attr in ("action_weight_post", "action_weight_comment", "action_weight_vote"):
                value = getattr(config, attr)
                assert 0.0 <= value <= 1.0, (
                    f"{slug!r}.{attr} = {value} is out of range"
                )

    def test_all_personas_produce_nonempty_system_prompts(self) -> None:
        """build_system_prompt should succeed for every built-in persona."""
        for slug, config in BUILT_IN_PERSONAS.items():
            prompt = build_system_prompt(slug.replace("_", " ").title(), config)
            assert len(prompt) > 50, f"Empty system prompt for persona {slug!r}"

    def test_get_persona_returns_config(self) -> None:
        """get_persona should return the correct config for a known slug."""
        slug = next(iter(BUILT_IN_PERSONAS))
        config = get_persona(slug)
        assert config is not None
        assert isinstance(config, AgentConfig)

    def test_get_persona_returns_none_for_unknown(self) -> None:
        """get_persona should return None for an unrecognised slug."""
        assert get_persona("nonexistent_persona_xyz") is None

    def test_get_persona_names_returns_sorted_list(self) -> None:
        """get_persona_names should return a sorted list of strings."""
        names = get_persona_names()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert len(names) == len(BUILT_IN_PERSONAS)

    def test_personas_have_diverse_providers(self) -> None:
        """There should be at least one Anthropic and one OpenAI persona."""
        providers = {str(c.provider) for c in BUILT_IN_PERSONAS.values()}
        assert LLMProvider.OPENAI.value in providers
        assert LLMProvider.ANTHROPIC.value in providers

    def test_curious_explorer_persona_exists(self) -> None:
        """The 'curious_explorer' persona should exist and have friendly tone."""
        config = get_persona("curious_explorer")
        assert config is not None
        assert config.tone == Tone.FRIENDLY.value

    def test_comedian_persona_has_humorous_tone(self) -> None:
        """The 'comedian' persona should have HUMOROUS tone."""
        config = get_persona("comedian")
        assert config is not None
        assert config.tone == Tone.HUMOROUS.value

    def test_contrarian_critic_has_high_contrarianism(self) -> None:
        """The 'contrarian_critic' persona should have contrarianism > 0.5."""
        config = get_persona("contrarian_critic")
        assert config is not None
        assert config.contrarianism > 0.5
