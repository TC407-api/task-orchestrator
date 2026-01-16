"""Test-Time Training (TTT) Memory Layer for O(1) pattern lookup.

This module implements a memory layer that compresses failure patterns into a
weight matrix for constant-time risk prediction, with online learning support.
"""
import pickle
import numpy as np
from typing import List, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer

# Import FailurePattern from existing context


class TTTMemoryLayer:
    """
    Test-Time Training (TTT) Memory Layer.

    Compresses historical failure patterns into a weight matrix for O(1)
    risk prediction and supports online learning (gradient updates)
    during inference.

    Attributes:
        embedding_dim: Dimension size for internal vector representation.
        learning_rate: Step size for online weight updates.
    """

    def __init__(self, embedding_dim: int = 256, learning_rate: float = 0.01):
        """
        Initialize the TTT Memory Layer.

        Args:
            embedding_dim: Dimension size for internal vector representation.
            learning_rate: Step size for online weight updates.
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Internal state
        self._weights: Optional[np.ndarray] = None
        self._vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=embedding_dim,
            stop_words='english'
        )
        self._is_initialized: bool = False

    def compress_patterns(self, patterns: List[Any]) -> None:
        """
        Batch compression of failure patterns into the weight matrix.

        This effectively 'trains' the memory on historical data.
        Replaces the O(N) similarity search with a linear model fit.

        Args:
            patterns: List of historical FailurePattern objects.
        """
        if not patterns:
            return

        # Extract context text from patterns
        contexts = []
        for p in patterns:
            # Handle both real FailurePattern objects and mocks
            if hasattr(p, 'context'):
                contexts.append(p.context)
            elif hasattr(p, 'input_summary'):
                contexts.append(p.input_summary)
            else:
                contexts.append(str(p))

        # Fit vectorizer to the known failure contexts
        tfidf_matrix = self._vectorizer.fit_transform(contexts)

        # Initialize weights as the mean direction of failure patterns (centroid)
        # Converting sparse matrix mean to dense array
        self._weights = np.array(tfidf_matrix.mean(axis=0)).flatten()
        self._is_initialized = True

    def predict_risk_fast(self, prompt: str) -> float:
        """
        O(1) Risk Prediction.

        Performs a dot product between the encoded prompt and the
        memory weight matrix to estimate failure risk.

        Args:
            prompt: The user prompt or context string.

        Returns:
            float: Normalized risk score (0.0 to 1.0).
        """
        if not self._is_initialized or self._weights is None:
            return 0.0

        # Transform input using fitted vectorizer
        vec = self._vectorizer.transform([prompt])

        # Calculate risk score via dot product (similarity to failure centroid)
        # vec is sparse (1, features), weights is dense (features,)
        raw_score = vec.dot(self._weights)[0]

        # Normalize to 0.0-1.0 range
        return float(np.clip(raw_score, 0.0, 1.0))

    def update_weights(self, new_failure: Any) -> None:
        """
        Online Learning (Test-Time Training).

        Updates the internal weight matrix based on a single new failure
        observation without retraining on the whole dataset.

        Args:
            new_failure: The new failure pattern to learn from.
        """
        if not self._is_initialized:
            self.compress_patterns([new_failure])
            return

        # Get context from failure
        if hasattr(new_failure, 'context'):
            context = new_failure.context
        elif hasattr(new_failure, 'input_summary'):
            context = new_failure.input_summary
        else:
            context = str(new_failure)

        # Transform new failure context
        vec = self._vectorizer.transform([context]).toarray().flatten()

        # Gradient update: move weights towards the new failure vector
        self._weights = self._weights + self.learning_rate * vec

    def save_state(self, filepath: str = "ttt_memory_state.pkl") -> None:
        """
        Persist the current weights and vectorizer to disk.

        Args:
            filepath: Destination path for the pickle file.
        """
        state = {
            'weights': self._weights,
            'vectorizer': self._vectorizer,
            'initialized': self._is_initialized,
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str) -> None:
        """
        Load weights and vectorizer from disk.

        Args:
            filepath: Source path of the pickle file.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self._weights = state['weights']
            self._vectorizer = state['vectorizer']
            self._is_initialized = state['initialized']
