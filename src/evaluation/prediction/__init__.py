"""
Phase 8.4: Predictive Failure Detection Module.

This package provides ML-based failure prediction using
scikit-learn classifiers trained on historical evaluation data.
"""

from .features import FeatureExtractor
from .classifier import FailurePredictor, PredictionResult
from .training import ModelTrainer

__all__ = [
    "FeatureExtractor",
    "FailurePredictor",
    "PredictionResult",
    "ModelTrainer",
]
