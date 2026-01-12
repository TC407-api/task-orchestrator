"""
Feature extraction for failure prediction.

This module handles the transformation of raw prompts and context
into numerical vectors suitable for ML classification.
"""

import re
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Risky keywords that often appear in problematic prompts
RISKY_KEYWORDS = {
    'delete', 'remove', 'rm', 'drop', 'truncate', 'shutdown',
    'restart', 'exec', 'system', 'format', 'chmod', 'chown',
    'update', 'alter', 'grant', 'revoke', 'sudo', 'admin',
    'password', 'credential', 'secret', 'token', 'api_key',
}

# Keywords indicating complex operations
COMPLEXITY_KEYWORDS = {
    'recursively', 'all', 'every', 'entire', 'complete',
    'complex', 'advanced', 'sophisticated', 'multiple',
    'async', 'concurrent', 'parallel', 'batch',
}


class FeatureExtractor:
    """
    Extracts features from prompt and context data for ML prediction.

    Combines:
    - Text features (TF-IDF)
    - Structural features (length, complexity)
    - Semantic features (risky keywords, sentiment)
    - Context features (tool type, time)
    """

    def __init__(self, max_text_features: int = 500):
        """
        Initialize the feature extractor.

        Args:
            max_text_features: Maximum number of TF-IDF features
        """
        self.max_text_features = max_text_features
        self._tfidf = None
        self._is_fitted = False

    def fit(self, X: List[Dict[str, Any]], y: Optional[List[int]] = None) -> "FeatureExtractor":
        """
        Fit the TF-IDF vectorizer on the prompts.

        Args:
            X: List of dicts with 'prompt', 'tool', 'context' keys
            y: Labels (not used, but kept for sklearn compatibility)

        Returns:
            Self for chaining
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            prompts = [item.get('prompt', '') for item in X]

            self._tfidf = TfidfVectorizer(
                max_features=self.max_text_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )
            self._tfidf.fit(prompts)
            self._is_fitted = True

            logger.info(f"FeatureExtractor fitted on {len(prompts)} samples")
            return self

        except ImportError:
            logger.error("scikit-learn not installed. Cannot fit TF-IDF.")
            raise

    def transform(self, X: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform raw data into feature vectors.

        Args:
            X: List of dicts with 'prompt', 'tool', 'context' keys

        Returns:
            Feature matrix as numpy array
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform.")

        # Text features (TF-IDF)
        prompts = [item.get('prompt', '') for item in X]
        text_features = self._tfidf.transform(prompts).toarray()

        # Metadata/structural features
        meta_features = []
        for item in X:
            prompt = item.get('prompt', '')
            context = item.get('context', {})
            tool = item.get('tool', 'unknown')

            features = self._extract_meta_features(prompt, tool, context)
            meta_features.append(features)

        # Combine text and meta features
        return np.hstack((text_features, np.array(meta_features)))

    def _extract_meta_features(
        self,
        prompt: str,
        tool: str,
        context: Dict[str, Any],
    ) -> List[float]:
        """
        Extract metadata features from a single sample.

        Args:
            prompt: The input prompt
            tool: Tool name
            context: Additional context

        Returns:
            List of feature values
        """
        features = []

        # 1. Text length (normalized)
        length = len(prompt)
        features.append(min(length / 1000, 10))  # Cap at 10

        # 2. Complexity score (special char ratio)
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', prompt))
        complexity = special_chars / length if length > 0 else 0
        features.append(complexity)

        # 3. Risky keyword count
        prompt_lower = prompt.lower()
        risk_count = sum(1 for kw in RISKY_KEYWORDS if kw in prompt_lower)
        features.append(risk_count)

        # 4. Complexity keyword count
        complexity_count = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in prompt_lower)
        features.append(complexity_count)

        # 5. Sentiment (using simple heuristic - negative words)
        negative_words = ['error', 'fail', 'problem', 'issue', 'bug', 'broken', 'wrong']
        negative_count = sum(1 for w in negative_words if w in prompt_lower)
        features.append(negative_count)

        # 6. Question marks (uncertainty indicator)
        question_count = prompt.count('?')
        features.append(min(question_count, 5))

        # 7. Code indicators
        code_indicators = ['```', 'def ', 'function', 'class ', 'import ', 'from ']
        has_code = any(ind in prompt for ind in code_indicators)
        features.append(1.0 if has_code else 0.0)

        # 8. Tool type features (one-hot style)
        is_fs_tool = 1.0 if any(kw in tool.lower() for kw in ['fs', 'file', 'read', 'write']) else 0.0
        is_db_tool = 1.0 if any(kw in tool.lower() for kw in ['db', 'sql', 'query', 'database']) else 0.0
        is_spawn_tool = 1.0 if 'spawn' in tool.lower() else 0.0
        features.extend([is_fs_tool, is_db_tool, is_spawn_tool])

        # 9. Word count
        word_count = len(prompt.split())
        features.append(min(word_count / 100, 10))

        # 10. Uppercase ratio (possible shouting/emphasis)
        upper_count = sum(1 for c in prompt if c.isupper())
        upper_ratio = upper_count / length if length > 0 else 0
        features.append(upper_ratio)

        return features

    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretation."""
        if not self._is_fitted:
            return []

        # TF-IDF feature names
        tfidf_names = list(self._tfidf.get_feature_names_out())

        # Meta feature names
        meta_names = [
            'text_length',
            'special_char_ratio',
            'risky_keyword_count',
            'complexity_keyword_count',
            'negative_word_count',
            'question_count',
            'has_code',
            'is_fs_tool',
            'is_db_tool',
            'is_spawn_tool',
            'word_count',
            'uppercase_ratio',
        ]

        return tfidf_names + meta_names

    def save(self, path: str) -> None:
        """Save the fitted extractor to disk."""
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureExtractor":
        """Load a fitted extractor from disk."""
        import joblib
        return joblib.load(path)


__all__ = ["FeatureExtractor", "RISKY_KEYWORDS", "COMPLEXITY_KEYWORDS"]
