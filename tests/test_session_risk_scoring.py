"""Tests for Session Risk Scoring system.

TDD RED Phase: These tests define the expected behavior for SessionRiskScorer.
They should fail initially until the implementation is complete.
"""
import pytest
from datetime import datetime, timedelta

from src.governance.session_risk_scoring import (
    SessionRiskScore,
    SessionRiskScorer,
    RiskFactor,
)
from src.governance.session_context import SessionContext


class TestSessionRiskScore:
    """Tests for SessionRiskScore dataclass."""

    def test_risk_score_has_all_factors(self):
        """SessionRiskScore should have all risk factor scores."""
        score = SessionRiskScore(
            overall=0.3,
            geo_velocity=0.1,
            device_consistency=0.0,
            behavior_anomaly=0.2,
            capability_abuse=0.5,
            rate_anomaly=0.1,
            timestamp=datetime.now(),
        )

        assert score.overall == 0.3
        assert score.geo_velocity == 0.1
        assert score.device_consistency == 0.0
        assert score.behavior_anomaly == 0.2
        assert score.capability_abuse == 0.5
        assert score.rate_anomaly == 0.1
        assert score.timestamp is not None

    def test_risk_score_values_bounded(self):
        """Risk score values should be between 0.0 and 1.0."""
        score = SessionRiskScore(
            overall=0.5,
            geo_velocity=0.0,
            device_consistency=1.0,
            behavior_anomaly=0.5,
            capability_abuse=0.25,
            rate_anomaly=0.75,
            timestamp=datetime.now(),
        )

        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.geo_velocity <= 1.0
        assert 0.0 <= score.device_consistency <= 1.0
        assert 0.0 <= score.behavior_anomaly <= 1.0
        assert 0.0 <= score.capability_abuse <= 1.0
        assert 0.0 <= score.rate_anomaly <= 1.0

    def test_risk_score_is_high_risk(self):
        """is_high_risk() should return True when overall >= 0.7."""
        high_risk = SessionRiskScore(
            overall=0.8,
            geo_velocity=0.5,
            device_consistency=0.5,
            behavior_anomaly=0.5,
            capability_abuse=0.5,
            rate_anomaly=0.5,
            timestamp=datetime.now(),
        )
        assert high_risk.is_high_risk() is True

        low_risk = SessionRiskScore(
            overall=0.3,
            geo_velocity=0.1,
            device_consistency=0.1,
            behavior_anomaly=0.1,
            capability_abuse=0.1,
            rate_anomaly=0.1,
            timestamp=datetime.now(),
        )
        assert low_risk.is_high_risk() is False


class TestSessionRiskScorerCalculation:
    """Tests for risk score calculation."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    @pytest.fixture
    def mock_session(self) -> SessionContext:
        """Create a mock session context."""
        return SessionContext(
            session_id="test-session-123",
            created_at=datetime.now() - timedelta(hours=1),
            last_active=datetime.now(),
        )

    def test_risk_score_calculation(
        self, scorer: SessionRiskScorer, mock_session: SessionContext
    ):
        """calculate_risk() should return a valid SessionRiskScore."""
        score = scorer.calculate_risk(
            session=mock_session,
            action="code_write",
        )

        assert isinstance(score, SessionRiskScore)
        assert 0.0 <= score.overall <= 1.0
        assert score.timestamp is not None

    def test_risk_score_includes_all_factors(
        self, scorer: SessionRiskScorer, mock_session: SessionContext
    ):
        """calculate_risk() should evaluate all risk factors."""
        score = scorer.calculate_risk(
            session=mock_session,
            action="delete_file",
        )

        # All factors should be set (not None)
        assert score.geo_velocity is not None
        assert score.device_consistency is not None
        assert score.behavior_anomaly is not None
        assert score.capability_abuse is not None
        assert score.rate_anomaly is not None

    def test_high_risk_action_increases_score(
        self, scorer: SessionRiskScorer, mock_session: SessionContext
    ):
        """High-risk actions should result in higher risk scores."""
        low_risk_score = scorer.calculate_risk(
            session=mock_session,
            action="code_read",
        )

        high_risk_score = scorer.calculate_risk(
            session=mock_session,
            action="delete_production_data",
        )

        assert high_risk_score.overall > low_risk_score.overall


class TestSessionRiskScorerReauth:
    """Tests for re-authentication requirements."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    def test_high_risk_triggers_reauth(self, scorer: SessionRiskScorer):
        """should_require_reauth() should return True for high-risk scores."""
        high_risk_score = SessionRiskScore(
            overall=0.8,
            geo_velocity=0.7,
            device_consistency=0.3,
            behavior_anomaly=0.6,
            capability_abuse=0.5,
            rate_anomaly=0.4,
            timestamp=datetime.now(),
        )

        assert scorer.should_require_reauth(high_risk_score) is True

    def test_low_risk_does_not_trigger_reauth(self, scorer: SessionRiskScorer):
        """should_require_reauth() should return False for low-risk scores."""
        low_risk_score = SessionRiskScore(
            overall=0.3,
            geo_velocity=0.1,
            device_consistency=0.1,
            behavior_anomaly=0.2,
            capability_abuse=0.1,
            rate_anomaly=0.1,
            timestamp=datetime.now(),
        )

        assert scorer.should_require_reauth(low_risk_score) is False

    def test_custom_threshold_affects_reauth(self):
        """Custom threshold should affect reauth decision."""
        strict_scorer = SessionRiskScorer(threshold=0.5)
        lenient_scorer = SessionRiskScorer(threshold=0.9)

        medium_risk_score = SessionRiskScore(
            overall=0.6,
            geo_velocity=0.3,
            device_consistency=0.3,
            behavior_anomaly=0.4,
            capability_abuse=0.3,
            rate_anomaly=0.3,
            timestamp=datetime.now(),
        )

        assert strict_scorer.should_require_reauth(medium_risk_score) is True
        assert lenient_scorer.should_require_reauth(medium_risk_score) is False


class TestSessionRiskScorerTermination:
    """Tests for session termination."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    def test_critical_risk_terminates_session(self, scorer: SessionRiskScorer):
        """should_terminate() should return True for critical risk (>0.9)."""
        critical_score = SessionRiskScore(
            overall=0.95,
            geo_velocity=0.9,
            device_consistency=0.8,
            behavior_anomaly=0.9,
            capability_abuse=0.9,
            rate_anomaly=0.8,
            timestamp=datetime.now(),
        )

        assert scorer.should_terminate(critical_score) is True

    def test_high_risk_does_not_terminate(self, scorer: SessionRiskScorer):
        """should_terminate() should return False for high but not critical risk."""
        high_risk_score = SessionRiskScore(
            overall=0.8,
            geo_velocity=0.5,
            device_consistency=0.5,
            behavior_anomaly=0.6,
            capability_abuse=0.5,
            rate_anomaly=0.5,
            timestamp=datetime.now(),
        )

        assert scorer.should_terminate(high_risk_score) is False


class TestEnvFingerprintMismatch:
    """Tests for environment fingerprint detection."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    def test_env_fingerprint_mismatch_increases_risk(self, scorer: SessionRiskScorer):
        """Fingerprint mismatch should increase device_consistency risk."""
        # Session with original fingerprint
        session = SessionContext(
            session_id="test-session",
            created_at=datetime.now() - timedelta(hours=1),
            last_active=datetime.now(),
        )

        # Calculate with matching fingerprint
        score_matching = scorer.calculate_risk(
            session=session,
            action="code_read",
            env_fingerprint="fingerprint-A",
            expected_fingerprint="fingerprint-A",
        )

        # Calculate with mismatched fingerprint
        score_mismatch = scorer.calculate_risk(
            session=session,
            action="code_read",
            env_fingerprint="fingerprint-B",
            expected_fingerprint="fingerprint-A",
        )

        assert score_mismatch.device_consistency > score_matching.device_consistency


class TestCapabilityAbuseDetection:
    """Tests for capability abuse detection."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    def test_capability_abuse_detected(self, scorer: SessionRiskScorer):
        """Using unexpected capabilities should increase capability_abuse risk."""
        session = SessionContext(
            session_id="test-session",
            created_at=datetime.now() - timedelta(hours=1),
            last_active=datetime.now(),
        )

        # Action within allowed capabilities
        score_normal = scorer.calculate_risk(
            session=session,
            action="code_read",
            allowed_capabilities=["code_read", "code_write"],
        )

        # Action outside allowed capabilities
        score_abuse = scorer.calculate_risk(
            session=session,
            action="delete_production_data",
            allowed_capabilities=["code_read", "code_write"],
        )

        assert score_abuse.capability_abuse > score_normal.capability_abuse


class TestRateAnomalyDetection:
    """Tests for rate anomaly detection."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    def test_rate_anomaly_detection(self, scorer: SessionRiskScorer):
        """Unusual request frequency should increase rate_anomaly risk."""
        session = SessionContext(
            session_id="test-session",
            created_at=datetime.now() - timedelta(hours=1),
            last_active=datetime.now(),
        )

        # Normal request rate (10 per minute)
        score_normal = scorer.calculate_risk(
            session=session,
            action="code_read",
            requests_per_minute=10,
        )

        # Abnormal request rate (1000 per minute)
        score_anomaly = scorer.calculate_risk(
            session=session,
            action="code_read",
            requests_per_minute=1000,
        )

        assert score_anomaly.rate_anomaly > score_normal.rate_anomaly


class TestRiskFactorRetrieval:
    """Tests for getting risk factors."""

    @pytest.fixture
    def scorer(self) -> SessionRiskScorer:
        """Create a SessionRiskScorer with default threshold."""
        return SessionRiskScorer(threshold=0.7)

    def test_get_risk_factors(self, scorer: SessionRiskScorer):
        """get_risk_factors() should return list of contributing factors."""
        # First calculate a risk score for a session
        session = SessionContext(
            session_id="test-session-456",
            created_at=datetime.now() - timedelta(hours=1),
            last_active=datetime.now(),
        )

        scorer.calculate_risk(
            session=session,
            action="delete_file",
            env_fingerprint="new-fingerprint",
            expected_fingerprint="original-fingerprint",
        )

        factors = scorer.get_risk_factors("test-session-456")

        assert isinstance(factors, list)
        # Should include device_consistency factor due to fingerprint mismatch
        assert any("fingerprint" in f.lower() or "device" in f.lower() for f in factors)


class TestRiskFactor:
    """Tests for RiskFactor enum."""

    def test_all_risk_factors_defined(self):
        """All expected risk factors should be defined."""
        assert RiskFactor.GEO_VELOCITY is not None
        assert RiskFactor.DEVICE_CONSISTENCY is not None
        assert RiskFactor.BEHAVIOR_ANOMALY is not None
        assert RiskFactor.CAPABILITY_ABUSE is not None
        assert RiskFactor.RATE_ANOMALY is not None
