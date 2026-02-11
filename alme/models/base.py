"""Base model adapter for ALME evaluation."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelResponse:
    """Response from a model adapter."""

    raw_output: str
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error: Optional[str] = None


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model."""

    @abstractmethod
    def supports_mode(self, mode: str) -> bool:
        """Check if the model supports a given mode.

        Args:
            mode: One of "audio_only", "text_only",
                  "audio_text_aligned", "audio_text_conflict"
        """

    @abstractmethod
    def run(
        self,
        audio_path: str,
        text: Optional[str],
        question: str,
        choices: List[str],
        system_prompt: str,
        condition: str = "",
    ) -> ModelResponse:
        """Run inference on a single stimulus.

        Args:
            audio_path: Path to audio file (may be None for text_only)
            text: Text input (transcript or conflict text, may be None for audio_only)
            question: Question to answer
            choices: Answer choices
            system_prompt: System prompt for the model
            condition: Evaluation condition — one of "audio_only", "text_only",
                       "audio_text_aligned", "audio_text_conflict". Used to select
                       prompt template. Empty string falls back to inference from
                       audio_path/text presence.

        Returns:
            ModelResponse with raw output and metadata
        """

    def run_batch(
        self,
        items: List[tuple],
        system_prompt: str,
        condition: str = "",
    ) -> List[ModelResponse]:
        """Run inference on a batch of stimuli.

        Default implementation falls back to sequential run() calls.
        Override in subclasses to enable true batched inference.

        Args:
            items: List of (audio_path, text, question, choices) tuples
            system_prompt: System prompt for the model
            condition: Evaluation condition (passed through to run())

        Returns:
            List of ModelResponse, one per item
        """
        results = []
        for audio_path, text, question, choices in items:
            results.append(self.run(audio_path, text, question, choices, system_prompt, condition))
        return results

    def load(self) -> None:
        """Load model weights. Optional for API-based models."""

    def unload(self) -> None:
        """Unload model weights. Optional for API-based models."""
