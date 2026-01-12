"""
Training Data Export Module.

This module is responsible for buffering evaluation trials and exporting them
to disk in standard formats (JSONL, JSON) suitable for fine-tuning LLMs or
analyzing performance offline.
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .trial import Trial

logger = logging.getLogger(__name__)


class TrainingDataExporter:
    """
    Export evaluation trials as labeled training data.

    Attributes:
        output_dir (Path): Directory where export files will be saved.
        _buffer (List[Trial]): Internal storage for trials waiting to be exported.
    """

    def __init__(self, output_dir: str = "D:/Research/training-data"):
        """
        Initialize the exporter.

        Args:
            output_dir (str): The target directory for exported files.
                              Defaults to D:/Research/training-data.
        """
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create export directory {output_dir}: {e}")
            # Fallback to local directory if specified path fails
            self.output_dir = Path("./training-data")
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: List[Trial] = []

    def add_trial(self, trial: Trial) -> None:
        """
        Add a trial to the export buffer.

        Args:
            trial (Trial): The completed trial to add.
        """
        self._buffer.append(trial)
        logger.debug(f"Added trial {trial.id} to export buffer. Size: {len(self._buffer)}")

    def buffer_size(self) -> int:
        """Return the number of trials in the buffer."""
        return len(self._buffer)

    async def export(
        self,
        format: str = "jsonl",
        filename: Optional[str] = None
    ) -> Path:
        """
        Export buffered trials to file asynchronously.

        Args:
            format (str): Output format, either "jsonl" or "json".
            filename (Optional[str]): Specific filename. If None, generates timestamped name.

        Returns:
            Path: The absolute path to the written file.

        Raises:
            ValueError: If buffer is empty or format is unknown.
        """
        if not self._buffer:
            raise ValueError("No trials to export")

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"evals_{timestamp}.{format}"

        filepath = self.output_dir / filename

        # Snapshot buffer to prevent modification during async write
        trials_to_export = list(self._buffer)

        try:
            # Offload blocking I/O to a thread
            if format == "jsonl":
                await asyncio.to_thread(self._write_jsonl, filepath, trials_to_export)
            elif format == "json":
                await asyncio.to_thread(self._write_json, filepath, trials_to_export)
            else:
                raise ValueError(f"Unknown format: {format}")

            # Only clear buffer if write was successful
            self._buffer.clear()
            logger.info(f"Successfully exported {len(trials_to_export)} trials to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    def _write_jsonl(self, filepath: Path, trials: List[Trial]) -> None:
        """Synchronous write handler for JSONL."""
        with open(filepath, "w", encoding="utf-8") as f:
            for trial in trials:
                example = self._trial_to_training_example(trial)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _write_json(self, filepath: Path, trials: List[Trial]) -> None:
        """Synchronous write handler for JSON."""
        examples = [self._trial_to_training_example(t) for t in trials]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    def _trial_to_training_example(self, trial: Trial) -> Dict[str, Any]:
        """
        Convert a Trial object into a serializable dictionary.

        Args:
            trial (Trial): The trial to convert.

        Returns:
            Dict[str, Any]: A dictionary representing the training example.
        """
        return {
            "id": str(trial.id),
            "prompt": trial.input_prompt,
            "response": str(trial.output) if trial.output is not None else "",
            "label": "good" if trial.pass_fail else "bad",
            "scores": {r.name: r.score for r in trial.grader_results},
            "grader_details": [r.to_dict() for r in trial.grader_results],
            "model": trial.model,
            "cost_usd": trial.cost_usd,
            "latency_ms": trial.latency_ms,
            "timestamp": trial.created_at.isoformat() if hasattr(trial.created_at, 'isoformat') else str(trial.created_at),
            "metadata": trial.metadata,
        }


# Global exporter instance
_exporter: Optional[TrainingDataExporter] = None


def get_exporter() -> TrainingDataExporter:
    """
    Get the singleton instance of the TrainingDataExporter.

    Returns:
        TrainingDataExporter: The global exporter instance.
    """
    global _exporter
    if _exporter is None:
        _exporter = TrainingDataExporter()
    return _exporter
