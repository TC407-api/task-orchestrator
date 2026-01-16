"""Session Risk Scoring for continuous session integrity validation.

Provides continuous risk assessment based on:
- Geographic velocity (location change speed)
- Device consistency (fingerprint matching)
- Behavior anomaly (action pattern deviation)
- Capability abuse (unexpected capability usage)
- Rate anomaly (unusual request frequency)
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import structlog

from src.governance.session_context import SessionContext

logger = structlog.get_logger(__name__)


class RiskFactor(str, Enum):
    """Risk factor classifications."""
    GEO_VELOCITY = "geo_velocity"
    DEVICE_CONSISTENCY = "device_consistency"
    BEHAVIOR_ANOMALY = "behavior_anomaly"
    CAPABILITY_ABUSE = "capability_abuse"
    RATE_ANOMALY = "rate_anomaly"


# Actions classified by risk level
HIGH_RISK_ACTIONS = {
    "delete_production_data",
    "delete_file",
    "drop_table",
    "deploy_production",
    "modify_credentials",
    "escalate_privileges",
    "disable_security",
}

MEDIUM_RISK_ACTIONS = {
    "code_write",
    "config_modify",
    "user_create",
    "permission_grant",
}

LOW_RISK_ACTIONS = {
    "code_read",
    "file_read",
    "search",
    "query",
    "list",
}


@dataclass
class SessionRiskScore:
    """
    Risk score for a session at a point in time.

    All scores are 0.0 to 1.0, where higher = more risky.

    Attributes:
        overall: Weighted combination of all factors
        geo_velocity: Location change speed risk
        device_consistency: Fingerprint mismatch risk
        behavior_anomaly: Action pattern deviation risk
        capability_abuse: Unexpected capability usage risk
        rate_anomaly: Request frequency anomaly risk
        timestamp: When this score was calculated
    """
    overall: float
    geo_velocity: float
    device_consistency: float
    behavior_anomaly: float
    capability_abuse: float
    rate_anomaly: float
    timestamp: datetime

    def is_high_risk(self, threshold: float = 0.7) -> bool:
        """Check if overall risk exceeds threshold."""
        return self.overall >= threshold

    def is_critical_risk(self, threshold: float = 0.9) -> bool:
        """Check if overall risk is critical."""
        return self.overall >= threshold

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "overall": self.overall,
            "geo_velocity": self.geo_velocity,
            "device_consistency": self.device_consistency,
            "behavior_anomaly": self.behavior_anomaly,
            "capability_abuse": self.capability_abuse,
            "rate_anomaly": self.rate_anomaly,
            "timestamp": self.timestamp.isoformat(),
        }


class SessionRiskScorer:
    """
    Calculates continuous risk scores for sessions.

    Evaluates multiple risk factors and produces weighted scores
    to determine if sessions require re-authentication or termination.
    """

    # Factor weights for overall score calculation
    WEIGHTS = {
        RiskFactor.GEO_VELOCITY: 0.15,
        RiskFactor.DEVICE_CONSISTENCY: 0.25,
        RiskFactor.BEHAVIOR_ANOMALY: 0.20,
        RiskFactor.CAPABILITY_ABUSE: 0.25,
        RiskFactor.RATE_ANOMALY: 0.15,
    }

    # Thresholds
    REAUTH_THRESHOLD = 0.7
    TERMINATE_THRESHOLD = 0.9

    # Rate limits (requests per minute)
    NORMAL_RATE = 60
    SUSPICIOUS_RATE = 200
    CRITICAL_RATE = 500

    def __init__(self, threshold: float = 0.7):
        """
        Initialize the risk scorer.

        Args:
            threshold: Risk threshold for triggering re-auth (default: 0.7)
        """
        self.threshold = threshold
        self._session_scores: Dict[str, SessionRiskScore] = {}
        self._session_history: Dict[str, List[str]] = {}

    def calculate_risk(
        self,
        session: SessionContext,
        action: str,
        env_fingerprint: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
        allowed_capabilities: Optional[List[str]] = None,
        requests_per_minute: Optional[int] = None,
    ) -> SessionRiskScore:
        """
        Calculate risk score for a session action.

        Args:
            session: The session context
            action: Action being performed
            env_fingerprint: Current environment fingerprint
            expected_fingerprint: Expected fingerprint from session
            allowed_capabilities: List of allowed capabilities
            requests_per_minute: Current request rate

        Returns:
            SessionRiskScore with all factors evaluated
        """
        # Calculate individual factor scores
        geo_score = self._calculate_geo_velocity(session)
        device_score = self._calculate_device_consistency(
            env_fingerprint, expected_fingerprint
        )
        behavior_score = self._calculate_behavior_anomaly(session, action)
        capability_score = self._calculate_capability_abuse(action, allowed_capabilities)
        rate_score = self._calculate_rate_anomaly(requests_per_minute)

        # Calculate weighted overall score
        overall = (
            geo_score * self.WEIGHTS[RiskFactor.GEO_VELOCITY]
            + device_score * self.WEIGHTS[RiskFactor.DEVICE_CONSISTENCY]
            + behavior_score * self.WEIGHTS[RiskFactor.BEHAVIOR_ANOMALY]
            + capability_score * self.WEIGHTS[RiskFactor.CAPABILITY_ABUSE]
            + rate_score * self.WEIGHTS[RiskFactor.RATE_ANOMALY]
        )

        # Clamp to 0-1 range
        overall = max(0.0, min(1.0, overall))

        score = SessionRiskScore(
            overall=overall,
            geo_velocity=geo_score,
            device_consistency=device_score,
            behavior_anomaly=behavior_score,
            capability_abuse=capability_score,
            rate_anomaly=rate_score,
            timestamp=datetime.now(),
        )

        # Cache the score
        self._session_scores[session.session_id] = score

        # Track action history
        if session.session_id not in self._session_history:
            self._session_history[session.session_id] = []
        self._session_history[session.session_id].append(action)

        if score.is_high_risk(self.threshold):
            logger.warning(
                "high_risk_session_detected",
                session_id=session.session_id,
                overall_score=overall,
                action=action,
            )

        return score

    def _calculate_geo_velocity(self, session: SessionContext) -> float:
        """
        Calculate geo velocity risk.

        In a full implementation, this would compare IP geolocation
        between requests to detect impossible travel.
        """
        # Placeholder - would need IP tracking and geolocation
        # For now, return low risk
        return 0.0

    def _calculate_device_consistency(
        self,
        current_fingerprint: Optional[str],
        expected_fingerprint: Optional[str],
    ) -> float:
        """Calculate device consistency risk based on fingerprint match."""
        if current_fingerprint is None or expected_fingerprint is None:
            return 0.1  # Slight risk if fingerprints not tracked

        if current_fingerprint != expected_fingerprint:
            return 0.9  # High risk for fingerprint mismatch

        return 0.0  # No risk for matching fingerprint

    def _calculate_behavior_anomaly(
        self,
        session: SessionContext,
        action: str,
    ) -> float:
        """Calculate behavior anomaly based on action patterns."""
        # Check if action is high-risk
        if action in HIGH_RISK_ACTIONS:
            return 0.7

        if action in MEDIUM_RISK_ACTIONS:
            return 0.3

        return 0.1  # Low risk for normal actions

    def _calculate_capability_abuse(
        self,
        action: str,
        allowed_capabilities: Optional[List[str]],
    ) -> float:
        """Calculate capability abuse risk."""
        if allowed_capabilities is None:
            return 0.0  # No restrictions

        # Check if action is within allowed capabilities
        action_base = action.split("_")[0] if "_" in action else action

        # Map actions to capabilities
        action_to_capability = {
            "code": "code_write",
            "delete": "delete",
            "deploy": "deploy",
            "config": "config_modify",
            "read": "code_read",
        }

        required_cap = action_to_capability.get(action_base)

        if required_cap and required_cap not in allowed_capabilities:
            # Check if related capability exists
            if not any(cap.startswith(action_base) for cap in allowed_capabilities):
                return 0.9  # High risk for capability abuse

        # High-risk actions always have some risk
        if action in HIGH_RISK_ACTIONS:
            return 0.5

        return 0.0

    def _calculate_rate_anomaly(
        self,
        requests_per_minute: Optional[int],
    ) -> float:
        """Calculate rate anomaly risk."""
        if requests_per_minute is None:
            return 0.0

        if requests_per_minute >= self.CRITICAL_RATE:
            return 1.0
        elif requests_per_minute >= self.SUSPICIOUS_RATE:
            return 0.7
        elif requests_per_minute >= self.NORMAL_RATE:
            return 0.3

        return 0.0

    def should_require_reauth(self, score: SessionRiskScore) -> bool:
        """
        Determine if session requires re-authentication.

        Args:
            score: The session's risk score

        Returns:
            True if re-auth should be required
        """
        return score.overall >= self.threshold

    def should_terminate(self, score: SessionRiskScore) -> bool:
        """
        Determine if session should be terminated.

        Args:
            score: The session's risk score

        Returns:
            True if session should be terminated
        """
        return score.overall >= self.TERMINATE_THRESHOLD

    def get_risk_factors(self, session_id: str) -> List[str]:
        """
        Get list of contributing risk factors for a session.

        Args:
            session_id: The session identifier

        Returns:
            List of human-readable risk factor descriptions
        """
        score = self._session_scores.get(session_id)
        if not score:
            return []

        factors = []

        if score.geo_velocity > 0.5:
            factors.append("Suspicious geographic velocity detected")

        if score.device_consistency > 0.5:
            factors.append("Device fingerprint mismatch detected")

        if score.behavior_anomaly > 0.5:
            factors.append("Unusual behavior pattern detected")

        if score.capability_abuse > 0.5:
            factors.append("Capability abuse detected - action outside allowed scope")

        if score.rate_anomaly > 0.5:
            factors.append("Abnormal request rate detected")

        return factors

    def get_session_score(self, session_id: str) -> Optional[SessionRiskScore]:
        """Get the most recent risk score for a session."""
        return self._session_scores.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear cached data for a session."""
        self._session_scores.pop(session_id, None)
        self._session_history.pop(session_id, None)
