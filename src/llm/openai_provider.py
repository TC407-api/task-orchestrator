"""OpenAI LLM provider for Graphiti memory integration."""
import os
from typing import AsyncIterator, Optional

from .base import LLMProvider, LLMResponse, Message, ModelCapability, ModelInfo
from ..core.cost_tracker import Provider, get_cost_tracker


# OpenAI models for Graphiti/memory operations
OPENAI_MODELS = {
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
        ],
        max_tokens=4096,
        context_window=128000,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.FAST_RESPONSE,
            ModelCapability.FUNCTION_CALLING,
        ],
        max_tokens=4096,
        context_window=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "text-embedding-3-small": ModelInfo(
        id="text-embedding-3-small",
        name="Text Embedding 3 Small",
        provider="openai",
        capabilities=[ModelCapability.EMBEDDINGS],
        max_tokens=8191,
        context_window=8191,
        cost_per_1k_input=0.00002,
        cost_per_1k_output=0.0,
    ),
    "text-embedding-3-large": ModelInfo(
        id="text-embedding-3-large",
        name="Text Embedding 3 Large",
        provider="openai",
        capabilities=[ModelCapability.EMBEDDINGS],
        max_tokens=8191,
        context_window=8191,
        cost_per_1k_input=0.00013,
        cost_per_1k_output=0.0,
    ),
}


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider primarily for Graphiti memory operations.

    Uses gpt-4o-mini by default for cost efficiency.
    Includes automatic cost tracking with budget enforcement.
    """

    provider_name = "openai"

    def __init__(
        self,
        default_model: str = "gpt-4o-mini",
    ):
        """
        Initialize OpenAI provider.

        Args:
            default_model: Default model (gpt-4o-mini for cost efficiency)
        """
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var."
            )

        self.default_model = default_model
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required. Run: pip install openai")
        return self._client

    async def _track_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Track API usage for cost monitoring."""
        tracker = get_cost_tracker()
        await tracker.record_usage(
            provider=Provider.OPENAI,
            operation="generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    def _check_budget(self):
        """Check if we can proceed with API call."""
        tracker = get_cost_tracker()
        can_proceed, msg = tracker.check_can_proceed(Provider.OPENAI)
        if not can_proceed:
            raise RuntimeError(f"Budget exceeded: {msg}")

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from OpenAI."""
        # Check budget before making API call
        self._check_budget()

        model_id = model or self.default_model
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract usage
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        # Track usage for cost monitoring
        await self._track_usage(
            model_id,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.choices[0].finish_reason or "unknown",
            raw_response=response,
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens from OpenAI."""
        self._check_budget()

        model_id = model or self.default_model
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Multi-turn chat with OpenAI."""
        self._check_budget()

        model_id = model or self.default_model
        client = self._get_client()

        # Convert to OpenAI format
        oai_messages = []
        for msg in messages:
            oai_messages.append({"role": msg.role, "content": msg.content})

        response = await client.chat.completions.create(
            model=model_id,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        await self._track_usage(
            model_id,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.choices[0].finish_reason or "unknown",
            raw_response=response,
        )

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        """Create embedding for Graphiti memory."""
        self._check_budget()

        client = self._get_client()

        response = await client.embeddings.create(
            model=model,
            input=text,
        )

        # Track embedding usage
        await self._track_usage(
            model,
            response.usage.prompt_tokens,
            0,  # No output tokens for embeddings
        )

        return response.data[0].embedding

    def list_models(self) -> list[ModelInfo]:
        """List available OpenAI models."""
        return list(OPENAI_MODELS.values())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info about an OpenAI model."""
        return OPENAI_MODELS.get(model_id)
