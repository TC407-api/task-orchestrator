"""
Training pipeline for failure prediction model.

This module handles loading training data from JSONL exports
and orchestrating the model training process.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .classifier import FailurePredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Pipeline to load data, process it, and train the FailurePredictor.
    """

    def __init__(
        self,
        data_dir: str = "D:/Research/training-data",
        model_dir: str = "models/failure_prediction",
    ):
        """
        Initialize the trainer.

        Args:
            data_dir: Directory containing JSONL training data
            model_dir: Directory to save model artifacts
        """
        self.data_dir = data_dir
        self.model_dir = model_dir

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Load JSONL files and extract features/labels.

        Expected JSONL format:
        {
            "prompt": "...",
            "tool": "...",
            "success": bool,  // or "error": "...",
            "context": {...}
        }

        Returns:
            Tuple of (X, y) where X is list of dicts, y is list of labels
        """
        X: List[Dict[str, Any]] = []
        y: List[int] = []

        pattern = os.path.join(self.data_dir, "**/*.jsonl")
        files = glob.glob(pattern, recursive=True)

        logger.info(f"Found {len(files)} training data files in {self.data_dir}")

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            sample = self._parse_entry(entry)
                            if sample:
                                X.append(sample[0])
                                y.append(sample[1])
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON error in {filepath}:{line_num}: {e}")
                        except Exception as e:
                            logger.warning(f"Error parsing {filepath}:{line_num}: {e}")
            except Exception as e:
                logger.warning(f"Error reading file {filepath}: {e}")

        logger.info(f"Loaded {len(X)} samples. Failures: {sum(y)}, Successes: {len(y) - sum(y)}")
        return X, y

    def _parse_entry(self, entry: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Parse a single JSONL entry into a training sample.

        Args:
            entry: The JSONL entry

        Returns:
            Tuple of (input_dict, label) or None if invalid
        """
        # Determine label: 1 for Failure, 0 for Success
        is_failure = 0

        # Check various ways failure might be indicated
        if entry.get('error'):
            is_failure = 1
        elif entry.get('success') is False:
            is_failure = 1
        elif entry.get('pass_fail') is False:
            is_failure = 1
        elif entry.get('label') == 'bad':
            is_failure = 1

        # Extract input
        prompt = entry.get('prompt') or entry.get('input_prompt') or entry.get('input', '')
        if not prompt:
            return None

        tool = entry.get('tool_name') or entry.get('tool') or entry.get('operation', 'unknown')

        return (
            {
                "prompt": prompt,
                "tool": tool,
                "context": entry.get('context', {}),
            },
            is_failure,
        )

    def run_pipeline(
        self,
        test_size: float = 0.2,
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Args:
            test_size: Fraction of data to use for testing
            min_samples: Minimum samples required to train

        Returns:
            Training results and metrics
        """
        from .classifier import FailurePredictor

        # Load data
        X, y = self.load_data()

        if len(X) < min_samples:
            error_msg = f"Insufficient data: {len(X)} samples (min: {min_samples})"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "samples": len(X)}

        # Split data
        try:
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # Stratification might fail with very imbalanced data
            logger.warning("Stratified split failed, using random split")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Train
        predictor = FailurePredictor(model_dir=self.model_dir)
        train_result = predictor.train(X_train, y_train)

        if not train_result.get("success"):
            return train_result

        # Evaluate on test set
        metrics = self._evaluate(predictor, X_test, y_test)

        return {
            "success": True,
            "training": train_result,
            "evaluation": metrics,
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def _evaluate(
        self,
        predictor: "FailurePredictor",
        X_test: List[Dict[str, Any]],
        y_test: List[int],
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            predictor: The trained predictor
            X_test: Test inputs
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
            )

            # Get predictions
            y_pred = []
            y_prob = []
            for sample in X_test:
                result = predictor.predict(
                    sample['prompt'],
                    sample['tool'],
                    sample.get('context'),
                )
                y_pred.append(1 if result.is_high_risk else 0)
                y_prob.append(result.risk_score)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }

            # AUC if we have both classes
            if len(set(y_test)) > 1:
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}


def main():
    """Command-line entry point for training."""
    logging.basicConfig(level=logging.INFO)

    trainer = ModelTrainer()
    result = trainer.run_pipeline()

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)

    if result.get("success"):
        print(f"Total samples: {result['total_samples']}")
        print(f"Train samples: {result['train_samples']}")
        print(f"Test samples: {result['test_samples']}")
        print("\nEvaluation Metrics:")
        for metric, value in result.get('evaluation', {}).items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print(f"Training failed: {result.get('error')}")


if __name__ == "__main__":
    main()


__all__ = ["ModelTrainer", "main"]
