"""Multi-model router for intelligent model selection."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import LLMProvider, LLMResponse, Message, ModelCapability, ModelInfo


class TaskType(Enum):
    """Types of tasks for model routing."""
    FAST_RESPONSE = "fast"          # Quick, simple tasks
    CODE_GENERATION = "code"         # Writing code
    REASONING = "reasoning"          # Complex reasoning
    ANALYSIS = "analysis"            # Analyzing data/text
    CREATIVE = "creative"            # Creative writing
    SUMMARIZATION = "summarization"  # Summarizing content
    EXTRACTION = "extraction"        # Extracting info
    GENERAL = "general"              # General purpose


@dataclass
class RoutingConfig:
    """Configuration for model routing."""
    # Default models by task type
    task_defaults: dict[TaskType, str]
    # Fallback model if routing fails
    fallback_model: str = "gemini-2.5-flash"
    # Whether to prefer preview models
    allow_preview: bool = True
    # Max cost per request (for cost-aware routing)
    max_cost_per_request: Optional[float] = None


DEFAULT_ROUTING = RoutingConfig(
    task_defaults={
        TaskType.FAST_RESPONSE: "gemini-2.5-flash-lite",
        TaskType.CODE_GENERATION: "gemini-3-flash-preview",
        TaskType.REASONING: "gemini-3-pro-preview",
        TaskType.ANALYSIS: "gemini-2.5-flash",
        TaskType.CREATIVE: "gemini-2.5-flash",
        TaskType.SUMMARIZATION: "gemini-2.5-flash-lite",
        TaskType.EXTRACTION: "gemini-2.0-flash-lite",
        TaskType.GENERAL: "gemini-2.5-flash",
    },
    fallback_model="gemini-2.5-flash",
    allow_preview=True,
)


class ModelRouter:
    """
    Intelligent router for multi-model orchestration.

    Routes requests to optimal models based on:
    - Task type
    - Required capabilities
    - Cost constraints
    - Latency requirements
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        config: Optional[RoutingConfig] = None,
    ):
        """
        Initialize the model router.

        Args:
            providers: Dict of provider_name -> LLMProvider instances
            config: Routing configuration
        """
        self.providers = providers
        self.config = config or DEFAULT_ROUTING

        # Build model -> provider mapping
        self._model_to_provider: dict[str, LLMProvider] = {}
        for provider in providers.values():
            for model in provider.list_models():
                self._model_to_provider[model.id] = provider

    def _get_provider_for_model(self, model_id: str) -> Optional[LLMProvider]:
        """Get the provider that serves a model."""
        return self._model_to_provider.get(model_id)

    def select_model(
        self,
        task_type: TaskType = TaskType.GENERAL,
        required_capabilities: Optional[list[ModelCapability]] = None,
        prefer_fast: bool = False,
        prefer_cheap: bool = False,
    ) -> str:
        """
        Select the best model for a task.

        Args:
            task_type: Type of task to perform
            required_capabilities: Capabilities the model must have
            prefer_fast: Prioritize speed over quality
            prefer_cheap: Prioritize cost over quality

        Returns:
            Model ID to use
        """
        # Start with task default
        model_id = self.config.task_defaults.get(task_type, self.config.fallback_model)

        # Quick preferences
        if prefer_fast:
            model_id = "gemini-2.5-flash-lite"
        elif prefer_cheap:
            model_id = "gemini-2.0-flash-lite"

        # Check capabilities
        if required_capabilities:
            provider = self._get_provider_for_model(model_id)
            if provider:
                model_info = provider.get_model_info(model_id)
                if model_info:
                    has_all = all(
                        cap in model_info.capabilities
                        for cap in required_capabilities
                    )
                    if not has_all:
                        # Find a model with required capabilities
                        model_id = self._find_capable_model(required_capabilities)

        # Check preview preference
        if not self.config.allow_preview and "preview" in model_id:
            # Fall back to stable
            if "3-flash" in model_id:
                model_id = "gemini-2.5-flash"
            elif "3-pro" in model_id:
                model_id = "gemini-2.5-flash"

        return model_id

    def _find_capable_model(
        self,
        required: list[ModelCapability],
    ) -> str:
        """Find a model with all required capabilities."""
        for provider in self.providers.values():
            for model in provider.list_models():
                if all(cap in model.capabilities for cap in required):
                    if self.config.allow_preview or not model.is_preview:
                        return model.id
        return self.config.fallback_model

    async def generate(
        self,
        prompt: str,
        *,
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        prefer_fast: bool = False,
        prefer_cheap: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate with automatic model routing.

        Args:
            prompt: User prompt
            task_type: Type of task for routing
            model: Explicit model override
            temperature: Sampling temperature
            max_tokens: Max output tokens
            system_prompt: System instructions
            prefer_fast: Prioritize speed
            prefer_cheap: Prioritize cost

        Returns:
            LLMResponse
        """
        # Select model
        model_id = model or self.select_model(
            task_type=task_type,
            prefer_fast=prefer_fast,
            prefer_cheap=prefer_cheap,
        )

        # Get provider
        provider = self._get_provider_for_model(model_id)
        if not provider:
            # Fallback to first available provider
            provider = next(iter(self.providers.values()))
            model_id = self.config.fallback_model

        return await provider.generate(
            prompt,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        task_type: TaskType = TaskType.GENERAL,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """
        Multi-turn chat with automatic routing.
        """
        model_id = model or self.select_model(task_type=task_type)
        provider = self._get_provider_for_model(model_id)

        if not provider:
            provider = next(iter(self.providers.values()))
            model_id = self.config.fallback_model

        return await provider.chat(
            messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def list_all_models(self) -> list[ModelInfo]:
        """List all available models across all providers."""
        models = []
        seen = set()
        for provider in self.providers.values():
            for model in provider.list_models():
                if model.id not in seen:
                    models.append(model)
                    seen.add(model.id)
        return models

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info about any model."""
        provider = self._get_provider_for_model(model_id)
        if provider:
            return provider.get_model_info(model_id)
        return None
