"""Unified LLM client abstraction for Agent Agora.

Provides a single ``LLMClient`` class that routes completion requests to
either the OpenAI or Anthropic backend based on the agent's configuration.
Exponential backoff with jitter is applied on transient API errors so that
transient rate-limit or network hiccups do not crash agent actions.

Usage::

    client = LLMClient(provider=LLMProvider.OPENAI, model="gpt-4o")
    response = await client.complete(
        user_prompt="What should I post today?",
        system_prompt="You are a curious AI named Alice.",
        max_tokens=256,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Optional

import anthropic
import openai

from agent_agora.models import LLMProvider

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model identifiers (overridable via environment variables)
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_MODEL: str = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o")
DEFAULT_ANTHROPIC_MODEL: str = os.getenv(
    "DEFAULT_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
)

# Retry / backoff configuration
DEFAULT_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
DEFAULT_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("MAX_TOKENS_PER_ACTION", "512"))

# Transient error types that warrant a retry
_OPENAI_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_ANTHROPIC_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for LLM client failures."""


class LLMProviderNotConfiguredError(LLMError):
    """Raised when the required API key is not set in the environment."""


class LLMMaxRetriesExceededError(LLMError):
    """Raised when all retry attempts have been exhausted."""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified LLM completion client supporting OpenAI and Anthropic.

    Args:
        provider: Which backend to use.  Defaults to the value of
            ``DEFAULT_LLM_PROVIDER`` environment variable, falling back to
            ``LLMProvider.OPENAI``.
        model: Model identifier.  If *None* the per-provider default is used.
        max_retries: Maximum number of retry attempts on transient errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
        openai_api_key: Override the OpenAI API key (default: env var).
        anthropic_api_key: Override the Anthropic API key (default: env var).
    """

    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.OPENAI,
        model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        # Normalise provider to enum
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        self.provider: LLMProvider = provider
        self.model: str = model or self._default_model(provider)
        self.max_retries: int = max_retries
        self.retry_base_delay: float = retry_base_delay

        # Lazily-initialised SDK clients
        self._openai_client: Optional[openai.AsyncOpenAI] = None
        self._anthropic_client: Optional[anthropic.AsyncAnthropic] = None

        # Store key overrides
        self._openai_api_key: Optional[str] = openai_api_key
        self._anthropic_api_key: Optional[str] = anthropic_api_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_model(provider: LLMProvider) -> str:
        """Return the default model identifier for the given provider."""
        if provider == LLMProvider.OPENAI:
            return DEFAULT_OPENAI_MODEL
        if provider == LLMProvider.ANTHROPIC:
            return DEFAULT_ANTHROPIC_MODEL
        raise LLMError(f"Unknown provider: {provider!r}")

    def _get_openai_client(self) -> openai.AsyncOpenAI:
        """Return (or create) the async OpenAI client."""
        if self._openai_client is None:
            api_key = self._openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMProviderNotConfiguredError(
                    "OPENAI_API_KEY is not set. "
                    "Please add it to your .env file or pass it explicitly."
                )
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)
        return self._openai_client

    def _get_anthropic_client(self) -> anthropic.AsyncAnthropic:
        """Return (or create) the async Anthropic client."""
        if self._anthropic_client is None:
            api_key = self._anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMProviderNotConfiguredError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Please add it to your .env file or pass it explicitly."
                )
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    async def _complete_openai(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Execute a completion request against the OpenAI API.

        Args:
            user_prompt: The user-turn message.
            system_prompt: The system instruction.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text content.

        Raises:
            LLMError: On unexpected API errors.
        """
        client = self._get_openai_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("OpenAI returned an empty response content.")
        return content.strip()

    async def _complete_anthropic(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Execute a completion request against the Anthropic API.

        Args:
            user_prompt: The user-turn message.
            system_prompt: The system instruction.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text content.

        Raises:
            LLMError: On unexpected API errors.
        """
        client = self._get_anthropic_client()
        response = await client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Anthropic returns a list of content blocks
        text_blocks = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text
        ]
        if not text_blocks:
            raise LLMError("Anthropic returned an empty response content.")
        return " ".join(text_blocks).strip()

    @staticmethod
    def _jitter(attempt: int, base_delay: float) -> float:
        """Compute the sleep duration for a given retry attempt.

        Uses full-jitter exponential backoff: ``sleep = random(0, base * 2^attempt)``.

        Args:
            attempt: Zero-based attempt index (0 = first retry).
            base_delay: Base delay in seconds.

        Returns:
            Seconds to sleep before the next attempt.
        """
        cap = base_delay * (2 ** attempt)
        return random.uniform(0.0, cap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.85,
    ) -> str:
        """Send a completion request and return the generated text.

        Transient API errors are retried with exponential backoff and
        full jitter up to ``self.max_retries`` times.

        Args:
            user_prompt: The user-turn message to send to the LLM.
            system_prompt: Optional system instruction.  Defaults to an
                empty string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.  Higher values produce more
                varied output.

        Returns:
            The generated text, stripped of leading/trailing whitespace.

        Raises:
            LLMProviderNotConfiguredError: If the required API key is missing.
            LLMMaxRetriesExceededError: If all retry attempts fail.
            LLMError: For non-retryable API errors.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == LLMProvider.OPENAI:
                    return await self._complete_openai(
                        user_prompt, system_prompt, max_tokens, temperature
                    )
                elif self.provider == LLMProvider.ANTHROPIC:
                    return await self._complete_anthropic(
                        user_prompt, system_prompt, max_tokens, temperature
                    )
                else:
                    raise LLMError(f"Unsupported provider: {self.provider!r}")

            except _OPENAI_RETRYABLE as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self._jitter(attempt, self.retry_base_delay)
                    log.warning(
                        "OpenAI transient error (attempt %d/%d): %s — retrying in %.2fs",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        "OpenAI error after %d attempts: %s", self.max_retries + 1, exc
                    )

            except _ANTHROPIC_RETRYABLE as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self._jitter(attempt, self.retry_base_delay)
                    log.warning(
                        "Anthropic transient error (attempt %d/%d): %s — retrying in %.2fs",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        "Anthropic error after %d attempts: %s", self.max_retries + 1, exc
                    )

            except (openai.OpenAIError, anthropic.APIError) as exc:
                # Non-retryable API errors
                raise LLMError(f"Non-retryable LLM API error: {exc}") from exc

        raise LLMMaxRetriesExceededError(
            f"LLM request failed after {self.max_retries + 1} attempt(s). "
            f"Last error: {last_exc}"
        ) from last_exc

    async def close(self) -> None:
        """Close underlying HTTP connections held by SDK clients."""
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LLMClient(provider={self.provider!r}, model={self.model!r}, "
            f"max_retries={self.max_retries})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> LLMClient:
    """Convenience factory that reads provider from the environment if not given.

    Args:
        provider: Provider name string (``"openai"`` or ``"anthropic"``).
            Falls back to the ``DEFAULT_LLM_PROVIDER`` environment variable,
            then to ``"openai"``.
        model: Model identifier override.
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for backoff.

    Returns:
        A configured :class:`LLMClient` instance.
    """
    resolved_provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    return LLMClient(
        provider=LLMProvider(resolved_provider),
        model=model,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
