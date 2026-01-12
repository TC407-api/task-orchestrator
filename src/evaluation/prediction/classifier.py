"""
ML Classifier for failure prediction.

This module provides the FailurePredictor class that wraps
a scikit-learn classifier for predicting operation failures.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Result from failure prediction.

    Attributes:
        risk_score: Probability of failure (0.0-1.0)
        is_high_risk: Whether risk exceeds threshold
        confidence: Model confidence
        details: Additional prediction details
    """
    risk_score: float
    is_high_risk: bool
    confidence: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_score": self.risk_score,
            "is_high_risk": self.is_high_risk,
            "confidence": self.confidence,
            "details": self.details,
        }


class FailurePredictor:
    """
    ML-based predictor for operation failure.

    Wraps a scikit-learn RandomForest classifier with the
    FeatureExtractor for end-to-end prediction.
    """

    def __init__(
        self,
        model_dir: str = "models/failure_prediction",
        threshold: float = 0.7,
    ):
        """
        Initialize the predictor.

        Args:
            model_dir: Directory for model artifacts
            threshold: Probability threshold for high risk
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "classifier.joblib")
        self.extractor_path = os.path.join(model_dir, "extractor.joblib")

        self.model = None
        self.extractor = None
        self.threshold = threshold
        self._model_version = "rf_v1"

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Attempt to load model artifacts if they exist."""
        if os.path.exists(self.model_path) and os.path.exists(self.extractor_path):
            try:
                import joblib
                from .features import FeatureExtractor

                self.model = joblib.load(self.model_path)
                self.extractor = FeatureExtractor.load(self.extractor_path)

                logger.info("Failure prediction model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load prediction model: {e}")
                self.model = None
                self.extractor = None
        else:
            logger.debug("No saved prediction model found.")

    @property
    def is_active(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None and self.extractor is not None

    def train(
        self,
        X_raw: list,
        y: list,
        n_estimators: int = 100,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """
        Train the prediction pipeline.

        Args:
            X_raw: List of dicts with 'prompt', 'tool', 'context'
            y: List of integers (0=Success, 1=Failure)
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth

        Returns:
            Training metrics
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from .features import FeatureExtractor

            logger.info(f"Training failure predictor on {len(X_raw)} samples...")

            # Initialize and fit extractor
            self.extractor = FeatureExtractor()
            self.extractor.fit(X_raw)
            X_vec = self.extractor.transform(X_raw)

            # Train classifier
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X_vec, y)

            # Save artifacts
            self._save_model()

            # Calculate training metrics
            y_pred = self.model.predict(X_vec)
            accuracy = (y_pred == y).mean()

            logger.info(f"Model training complete. Accuracy: {accuracy:.2%}")

            return {
                "success": True,
                "samples": len(X_raw),
                "training_accuracy": accuracy,
                "model_version": self._model_version,
            }

        except ImportError as e:
            logger.error(f"scikit-learn not installed: {e}")
            return {"success": False, "error": "scikit-learn not installed"}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def _save_model(self) -> None:
        """Save model artifacts to disk."""
        import joblib

        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.extractor.save(self.extractor_path)

        logger.info(f"Model saved to {self.model_dir}")

    def predict(
        self,
        prompt: str,
        tool: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """
        Predict failure probability for a given operation.

        Args:
            prompt: The input prompt
            tool: Tool name being used
            context: Additional context

        Returns:
            PredictionResult with risk assessment
        """
        if not self.is_active:
            logger.debug("Model not loaded. Returning neutral prediction.")
            return PredictionResult(
                risk_score=0.5,
                is_high_risk=False,
                confidence=0.0,
                details={"reason": "model_not_loaded"},
            )

        try:
            input_data = [{
                "prompt": prompt,
                "tool": tool,
                "context": context or {},
            }]

            X_vec = self.extractor.transform(input_data)

            # Get probability of class 1 (Failure)
            probs = self.model.predict_proba(X_vec)
            failure_prob = probs[0][1]

            # Calculate confidence (how certain the model is)
            confidence = abs(failure_prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

            return PredictionResult(
                risk_score=float(failure_prob),
                is_high_risk=failure_prob > self.threshold,
                confidence=float(confidence),
                details={
                    "model_version": self._model_version,
                    "threshold": self.threshold,
                    "class_probabilities": {
                        "success": float(probs[0][0]),
                        "failure": float(probs[0][1]),
                    },
                },
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return PredictionResult(
                risk_score=0.5,
                is_high_risk=False,
                confidence=0.0,
                details={"error": str(e)},
            )

    def predict_risk(
        self,
        prompt: str,
        tool: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Legacy method for compatibility with ImmuneSystem.

        Args:
            prompt: The input prompt
            tool: Tool name
            context: Additional context

        Returns:
            Tuple of (risk_score, details_dict)
        """
        result = self.predict(prompt, tool, context)
        return result.risk_score, result.details

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get the most important features for prediction.

        Args:
            top_n: Number of top features to return

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_active:
            return {}

        try:
            importances = self.model.feature_importances_
            feature_names = self.extractor.get_feature_names()

            # Sort by importance
            indices = importances.argsort()[::-1][:top_n]

            return {
                feature_names[i]: float(importances[i])
                for i in indices
            }

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}


__all__ = ["FailurePredictor", "PredictionResult"]
