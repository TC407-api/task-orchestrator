"""Google Gemini LLM provider."""
import os
from typing import AsyncIterator, Optional

import google.generativeai as genai  # type: ignore[import]
from google.generativeai.types import GenerationConfig  # type: ignore[import]

from .base import LLMProvider, LLMResponse, Message, ModelCapability, ModelInfo
from ..core.cost_tracker import Provider, get_cost_tracker
from ..core.rate_limiter import RateLimiter, CircuitBreaker
from ..core.config import settings

# Grade 5 Langfuse Tracing
try:
    from ..core.tracing import observe, flush_traces, langfuse
    TRACING_ENABLED = langfuse is not None
except ImportError:
    TRACING_ENABLED = False
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    def flush_traces():
        pass


# Available Gemini models with their capabilities
GEMINI_MODELS = {
    # Gemini 3.0 (Preview)
    "gemini-3-flash-preview": ModelInfo(
        id="gemini-3-flash-preview",
        name="Gemini 3.0 Flash Preview",
        provider="google",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.FAST_RESPONSE,
        ],
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.0,  # Preview pricing
        cost_per_1k_output=0.0,
        is_preview=True,
    ),
    "gemini-3-pro-preview": ModelInfo(
        id="gemini-3-pro-preview",
        name="Gemini 3.0 Pro Preview",
        provider="google",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.LONG_CONTEXT,
        ],
        max_tokens=8192,
        context_window=2000000,
        is_preview=True,
    ),
    # Gemini 2.5 Flash (Stable)
    "gemini-2.5-flash": ModelInfo(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.FAST_RESPONSE,
        ],
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "gemini-2.5-flash-lite": ModelInfo(
        id="gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash Lite",
        provider="google",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.FAST_RESPONSE,
        ],
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
    # Gemini 2.0 Flash
    "gemini-2.0-flash": ModelInfo(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.FAST_RESPONSE,
        ],
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
    ),
    "gemini-2.0-flash-lite": ModelInfo(
        id="gemini-2.0-flash-lite",
        name="Gemini 2.0 Flash Lite",
        provider="google",
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.FAST_RESPONSE,
        ],
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
}

# Aliases for convenience
MODEL_ALIASES = {
    "flash": "gemini-2.5-flash",
    "flash-lite": "gemini-2.5-flash-lite",
    "flash-3": "gemini-3-flash-preview",
    "pro-3": "gemini-3-pro-preview",
    "fast": "gemini-2.0-flash-lite",
    "latest": "gemini-3-flash-preview",
}


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider using the Google AI Studio API.

    Supports all Gemini models including:
    - Gemini 3.0 Flash/Pro (Preview)
    - Gemini 2.5 Flash/Flash-Lite
    - Gemini 2.0 Flash/Flash-Lite

    Includes automatic cost tracking with budget enforcement.

    Guardrails:
    - Rate limiting (requests per minute)
    - Circuit breaker (consecutive failure protection)
    - Budget enforcement (daily/monthly limits)
    """

    provider_name = "google"

    # Shared rate limiter and circuit breaker across all instances
    _rate_limiter: Optional[RateLimiter] = None
    _circuit_breaker: Optional[CircuitBreaker] = None

    @classmethod
    def _get_rate_limiter(cls) -> RateLimiter:
        """Get or create the shared rate limiter."""
        if cls._rate_limiter is None:
            cls._rate_limiter = RateLimiter(
                requests_per_minute=settings.gemini_rate_limit
            )
        return cls._rate_limiter

    @classmethod
    def _get_circuit_breaker(cls) -> CircuitBreaker:
        """Get or create the shared circuit breaker."""
        if cls._circuit_breaker is None:
            cls._circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                half_open_max_calls=3,
            )
        return cls._circuit_breaker

    async def _track_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Track API usage for cost monitoring."""
        tracker = get_cost_tracker()
        await tracker.record_usage(
            provider=Provider.GOOGLE_GEMINI,
            operation="generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    def _check_budget(self):
        """Check if we can proceed with API call."""
        tracker = get_cost_tracker()
        can_proceed, msg = tracker.check_can_proceed(Provider.GOOGLE_GEMINI)
        if not can_proceed:
            raise RuntimeError(f"Budget exceeded: {msg}")

    def _check_circuit_breaker(self):
        """Check if circuit breaker allows the call."""
        breaker = self._get_circuit_breaker()
        can_proceed, msg = breaker.can_proceed()
        if not can_proceed:
            raise RuntimeError(f"Circuit breaker: {msg}")

    async def _apply_rate_limit(self):
        """Wait for rate limit token."""
        limiter = self._get_rate_limiter()
        await limiter.async_wait()

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI API key. If not provided, uses GOOGLE_API_KEY env var.
            default_model: Default model to use for requests.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key."
            )

        genai.configure(api_key=self.api_key)
        self.default_model = self._resolve_model(default_model)

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model ID."""
        return MODEL_ALIASES.get(model, model)

    def _get_client(self, model: str) -> genai.GenerativeModel:
        """Get a GenerativeModel client for the specified model."""
        resolved = self._resolve_model(model)
        return genai.GenerativeModel(resolved)

    @observe(name="gemini.generate")
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
        """Generate a response from Gemini."""
        # === GUARDRAILS: Check all safety gates before API call ===
        # 1. Check budget
        self._check_budget()
        # 2. Check circuit breaker
        self._check_circuit_breaker()
        # 3. Apply rate limit (waits if needed)
        await self._apply_rate_limit()

        model_id = self._resolve_model(model or self.default_model)
        client = self._get_client(model_id)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Build content with optional system prompt
        content = prompt
        if system_prompt:
            content = f"{system_prompt}\n\n{prompt}"

        breaker = self._get_circuit_breaker()
        try:
            response = await client.generate_content_async(
                content,
                generation_config=config,
            )
            # Record success for circuit breaker
            breaker.record_success()

        except Exception:
            # Record failure for circuit breaker
            breaker.record_failure()
            raise

        # Extract usage info
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        # Track usage for cost monitoring
        await self._track_usage(
            model_id,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        return LLMResponse(
            content=response.text,
            model=model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown",
            raw_response=response,
        )

    def get_guardrails_status(self) -> dict:
        """Get status of all guardrails."""
        tracker = get_cost_tracker()
        return {
            "rate_limiter": {
                "requests_per_minute": settings.gemini_rate_limit,
                "tokens_available": self._get_rate_limiter().tokens,
            },
            "circuit_breaker": self._get_circuit_breaker().get_status(),
            "budget": {
                "daily_spent": tracker.get_daily_spend(Provider.GOOGLE_GEMINI),
                "monthly_spent": tracker.get_monthly_spend(Provider.GOOGLE_GEMINI),
                "daily_limit": tracker.budgets[Provider.GOOGLE_GEMINI].daily_limit_usd,
                "monthly_limit": tracker.budgets[Provider.GOOGLE_GEMINI].monthly_limit_usd,
            },
        }

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
        """Stream response tokens from Gemini."""
        model_id = self._resolve_model(model or self.default_model)
        client = self._get_client(model_id)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        content = prompt
        if system_prompt:
            content = f"{system_prompt}\n\n{prompt}"

        response = await client.generate_content_async(
            content,
            generation_config=config,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    @observe(name="gemini.chat")
    async def chat(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Multi-turn chat with Gemini."""
        # === GUARDRAILS: Check all safety gates before API call ===
        self._check_budget()
        self._check_circuit_breaker()
        await self._apply_rate_limit()

        model_id = self._resolve_model(model or self.default_model)
        client = self._get_client(model_id)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Convert messages to Gemini format
        history = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                history.append({"role": "model", "parts": [msg.content]})

        # Start chat with history (excluding last user message)
        chat = client.start_chat(history=history[:-1] if len(history) > 1 else [])

        # Send last message
        last_message = history[-1]["parts"][0] if history else ""
        if system_instruction:
            last_message = f"{system_instruction}\n\n{last_message}"

        breaker = self._get_circuit_breaker()
        try:
            response = await chat.send_message_async(
                last_message,
                generation_config=config,
            )
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        # Track usage
        await self._track_usage(
            model_id,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        return LLMResponse(
            content=response.text,
            model=model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown",
            raw_response=response,
        )

    def list_models(self) -> list[ModelInfo]:
        """List available Gemini models."""
        return list(GEMINI_MODELS.values())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info about a Gemini model."""
        resolved = self._resolve_model(model_id)
        return GEMINI_MODELS.get(resolved)


class VertexGeminiProvider(LLMProvider):
    """
    Google Gemini via Vertex AI for enterprise/production use.

    Requires Google Cloud project setup and authentication.
    """

    provider_name = "vertex"

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        default_model: str = "gemini-2.5-flash",
    ):
        """
        Initialize Vertex AI Gemini provider.

        Args:
            project_id: GCP project ID. Uses GOOGLE_CLOUD_PROJECT if not provided.
            location: GCP region for Vertex AI.
            default_model: Default model to use.
        """
        import vertexai
        from vertexai.generative_models import GenerativeModel

        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.default_model = self._resolve_model(default_model)

        vertexai.init(project=self.project_id, location=self.location)
        self._GenerativeModel = GenerativeModel

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model ID."""
        return MODEL_ALIASES.get(model, model)

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
        """Generate using Vertex AI."""
        from vertexai.generative_models import GenerationConfig

        model_id = self._resolve_model(model or self.default_model)
        client = self._GenerativeModel(model_id)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        content = prompt
        if system_prompt:
            content = f"{system_prompt}\n\n{prompt}"

        response = await client.generate_content_async(
            content,
            generation_config=config,
        )

        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        return LLMResponse(
            content=response.text,
            model=model_id,
            provider=self.provider_name,
            usage=usage,
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
        """Stream from Vertex AI."""
        from vertexai.generative_models import GenerationConfig

        model_id = self._resolve_model(model or self.default_model)
        client = self._GenerativeModel(model_id)

        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        content = prompt
        if system_prompt:
            content = f"{system_prompt}\n\n{prompt}"

        response = await client.generate_content_async(
            content,
            generation_config=config,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def chat(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Chat via Vertex AI."""
        # Similar implementation to GeminiProvider.chat
        model_id = self._resolve_model(model or self.default_model)

        # Convert last message to prompt
        last_content = messages[-1].content if messages else ""
        system_prompt = None
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                break

        return await self.generate(
            last_content,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def list_models(self) -> list[ModelInfo]:
        """List available models."""
        return list(GEMINI_MODELS.values())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info."""
        resolved = self._resolve_model(model_id)
        return GEMINI_MODELS.get(resolved)
