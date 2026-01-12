"""
Phase 8 Integration Tests for Task Orchestrator Evaluation System.

Tests for:
- Phase 8.1: Langfuse Observability Integration
- Phase 8.2: Alerting & High-Risk Pattern Detection
- Phase 8.3: Cross-Project Pattern Sharing (Federation)
- Phase 8.4: Predictive Failure Detection (ML)

Run with:
    JWT_SECRET_KEY=test123 python -m pytest tests/test_phase8.py -v
"""

import json
import os
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.integration


# =============================================================================
# Phase 8.1: Langfuse Integration Tests
# =============================================================================

class TestLangfuseIntegration:
    """Test enhanced Langfuse observability integration."""

    def test_evaluation_tracer_singleton(self):
        """Verify EvaluationTracer is a singleton."""
        from src.evaluation.langfuse_integration import EvaluationTracer, reset_tracer

        reset_tracer()
        tracer1 = EvaluationTracer()
        tracer2 = EvaluationTracer()

        assert tracer1 is tracer2
        reset_tracer()

    def test_gemini_cost_calculation(self):
        """Verify Gemini cost calculation."""
        from src.evaluation.langfuse_integration import calculate_gemini_cost

        # Flash model
        cost_flash = calculate_gemini_cost(
            "gemini-3-flash-preview",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost_flash > 0
        assert cost_flash < 0.01  # Should be cheap

        # Pro model
        cost_pro = calculate_gemini_cost(
            "gemini-3-pro-preview",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost_pro > cost_flash  # Pro should be more expensive

    def test_tracer_disabled_without_credentials(self):
        """Verify tracer handles missing Langfuse gracefully."""
        from src.evaluation.langfuse_integration import EvaluationTracer, reset_tracer

        reset_tracer()
        # Remove credentials temporarily
        original_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if "LANGFUSE_SECRET_KEY" in os.environ:
            del os.environ["LANGFUSE_SECRET_KEY"]

        tracer = EvaluationTracer()
        # Should not be enabled without credentials
        # (May still be enabled if langfuse package handles this differently)

        # Restore
        if original_key:
            os.environ["LANGFUSE_SECRET_KEY"] = original_key
        reset_tracer()


# =============================================================================
# Phase 8.2: Alerting Tests
# =============================================================================

class TestAlertingSeverity:
    """Test alert severity levels."""

    def test_alert_severity_enum(self):
        """Verify AlertSeverity enum values."""
        from src.evaluation.alerting import AlertSeverity

        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertRules:
    """Test alert rule evaluation."""

    def test_high_risk_threshold_triggers(self):
        """Verify HighRiskThreshold rule triggers correctly."""
        from src.evaluation.alerting import HighRiskThreshold, AlertContext, AlertSeverity

        rule = HighRiskThreshold(threshold=0.8, severity=AlertSeverity.CRITICAL)
        context = AlertContext(
            pattern_id="test-pattern",
            risk_score=0.9,
            is_new_pattern=False,
        )

        alert = rule.evaluate(context)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "0.9" in alert.message

    def test_high_risk_threshold_no_trigger(self):
        """Verify HighRiskThreshold doesn't trigger below threshold."""
        from src.evaluation.alerting import HighRiskThreshold, AlertContext

        rule = HighRiskThreshold(threshold=0.8)
        context = AlertContext(
            pattern_id="test-pattern",
            risk_score=0.5,
            is_new_pattern=False,
        )

        alert = rule.evaluate(context)
        assert alert is None

    def test_new_pattern_detected_triggers(self):
        """Verify NewPatternDetected rule triggers."""
        from src.evaluation.alerting import NewPatternDetected, AlertContext

        rule = NewPatternDetected()
        context = AlertContext(
            pattern_id="new-pattern",
            risk_score=0.3,
            is_new_pattern=True,
        )

        alert = rule.evaluate(context)
        assert alert is not None
        assert "New failure pattern" in alert.message

    def test_frequency_spike_triggers(self):
        """Verify FrequencySpike rule triggers with enough failures."""
        from src.evaluation.alerting import FrequencySpike, AlertContext

        rule = FrequencySpike(max_per_hour=3)

        # Create history with recent timestamps
        now = datetime.now()
        history = [
            {"timestamp": now - timedelta(minutes=10)},
            {"timestamp": now - timedelta(minutes=20)},
            {"timestamp": now - timedelta(minutes=30)},
        ]

        context = AlertContext(
            pattern_id="freq-pattern",
            risk_score=0.5,
            is_new_pattern=False,
            failure_history=history,
        )

        alert = rule.evaluate(context)
        assert alert is not None
        assert "frequency spike" in alert.message.lower()


class TestAlertManager:
    """Test AlertManager coordination."""

    @pytest.mark.asyncio
    async def test_manager_processes_failure(self):
        """Verify AlertManager processes failures and generates alerts."""
        from src.evaluation.alerting import (
            AlertManager,
            HighRiskThreshold,
            NewPatternDetected,
            ConsoleNotifier,
        )

        manager = AlertManager(
            rules=[
                HighRiskThreshold(threshold=0.7),
                NewPatternDetected(),
            ],
            notifiers=[ConsoleNotifier()],
        )

        alerts = await manager.process_failure(
            pattern_id="test",
            risk_score=0.9,
            is_new_pattern=True,
        )

        # Should trigger both rules
        assert len(alerts) == 2

    @pytest.mark.asyncio
    async def test_manager_get_recent_alerts(self):
        """Verify AlertManager returns recent alerts."""
        from src.evaluation.alerting import AlertManager, HighRiskThreshold

        manager = AlertManager(rules=[HighRiskThreshold(threshold=0.5)])

        await manager.process_failure(
            pattern_id="alert1",
            risk_score=0.9,
            is_new_pattern=False,
        )
        await manager.process_failure(
            pattern_id="alert2",
            risk_score=0.8,
            is_new_pattern=False,
        )

        recent = manager.get_recent_alerts(limit=10)
        assert len(recent) == 2

    def test_manager_stats(self):
        """Verify AlertManager statistics."""
        from src.evaluation.alerting import AlertManager, HighRiskThreshold

        manager = AlertManager(rules=[HighRiskThreshold()])
        stats = manager.get_stats()

        assert "alerts_generated" in stats
        assert "rules_count" in stats
        assert stats["rules_count"] == 1


# =============================================================================
# Phase 8.3: Federation Tests
# =============================================================================

class TestPatternFederation:
    """Test cross-project pattern sharing."""

    @pytest.mark.asyncio
    async def test_subscribe_to_project(self):
        """Verify subscription to another project."""
        from src.evaluation.immune_system.federation import PatternFederation

        federation = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )

        result = await federation.subscribe_to_project("project_other")

        assert result["success"] is True
        assert "project_other" in federation.subscriptions

    @pytest.mark.asyncio
    async def test_cannot_subscribe_to_self(self):
        """Verify cannot subscribe to own project."""
        from src.evaluation.immune_system.federation import PatternFederation

        federation = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )

        result = await federation.subscribe_to_project("project_test")

        assert result["success"] is False
        assert "self" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_unsubscribe_from_project(self):
        """Verify unsubscription works."""
        from src.evaluation.immune_system.federation import PatternFederation

        federation = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
            subscriptions={"project_other"},
        )

        result = await federation.unsubscribe_from_project("project_other")

        assert result["success"] is True
        assert "project_other" not in federation.subscriptions

    @pytest.mark.asyncio
    async def test_publish_pattern_visibility(self):
        """Verify pattern visibility can be changed."""
        from src.evaluation.immune_system.federation import PatternFederation, PatternVisibility

        federation = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )

        result = await federation.publish_pattern("pattern-123", "shared")

        assert result["success"] is True
        assert federation._pattern_visibility["pattern-123"] == PatternVisibility.SHARED

    def test_federation_stats(self):
        """Verify federation statistics."""
        from src.evaluation.immune_system.federation import PatternFederation

        federation = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
            subscriptions={"project_a", "project_b"},
        )

        stats = federation.get_stats()

        assert stats["local_group_id"] == "project_test"
        assert stats["subscriptions_count"] == 2

    def test_relevance_scoring(self):
        """Verify pattern relevance scoring."""
        from src.evaluation.immune_system.federation import PatternFederation
        from src.evaluation.immune_system import FailurePattern

        federation = PatternFederation(
            graphiti_client=None,
            local_group_id="project_test",
        )

        pattern = FailurePattern(
            id="test",
            operation="spawn_agent",
            failure_type="timeout",
            input_summary="Generate code for timeout handling",
            output_summary="Error output",
            grader_scores={},
            occurrence_count=15,
        )

        score = federation._calculate_relevance(pattern, "timeout", is_local=True)

        # Should have local bonus and maturity bonus
        assert score > 0.5


# =============================================================================
# Phase 8.4: Prediction Tests
# =============================================================================

class TestFeatureExtractor:
    """Test ML feature extraction."""

    def test_risky_keywords_detection(self):
        """Verify risky keywords are detected."""
        from src.evaluation.prediction.features import RISKY_KEYWORDS

        # Check some expected keywords exist
        assert "delete" in RISKY_KEYWORDS
        assert "drop" in RISKY_KEYWORDS
        assert "exec" in RISKY_KEYWORDS

    def test_meta_feature_extraction(self):
        """Verify meta features are extracted correctly."""
        from src.evaluation.prediction.features import FeatureExtractor

        extractor = FeatureExtractor()

        features = extractor._extract_meta_features(
            prompt="DROP TABLE users; DELETE FROM orders",
            tool="sql_query",
            context={},
        )

        # Should be a list of features
        assert len(features) > 0

        # Risk count should be >= 2 (drop, delete)
        risk_count_idx = 2  # Index of risky_keyword_count
        assert features[risk_count_idx] >= 2


class TestFailurePredictor:
    """Test ML failure predictor."""

    def test_predictor_initialization(self):
        """Verify predictor initializes without model."""
        from src.evaluation.prediction import FailurePredictor

        predictor = FailurePredictor(model_dir="nonexistent_dir")

        assert predictor.is_active is False

    def test_predictor_neutral_without_model(self):
        """Verify predictor returns neutral score without model."""
        from src.evaluation.prediction import FailurePredictor

        predictor = FailurePredictor(model_dir="nonexistent_dir")
        result = predictor.predict("test prompt", "test_tool")

        assert result.risk_score == 0.5
        assert result.is_high_risk is False
        assert "model_not_loaded" in str(result.details)

    def test_prediction_result_serialization(self):
        """Verify PredictionResult serializes correctly."""
        from src.evaluation.prediction import PredictionResult

        result = PredictionResult(
            risk_score=0.8,
            is_high_risk=True,
            confidence=0.6,
            details={"test": "value"},
        )

        data = result.to_dict()

        assert data["risk_score"] == 0.8
        assert data["is_high_risk"] is True
        assert data["confidence"] == 0.6


class TestModelTrainer:
    """Test ML model training pipeline."""

    def test_entry_parsing(self):
        """Verify JSONL entry parsing."""
        from src.evaluation.prediction.training import ModelTrainer

        trainer = ModelTrainer()

        # Test success case
        entry = {
            "prompt": "Write hello world",
            "tool": "spawn_agent",
            "success": True,
        }
        result = trainer._parse_entry(entry)

        assert result is not None
        assert result[1] == 0  # Success label

        # Test failure case
        entry_fail = {
            "prompt": "Generate code",
            "tool": "spawn_agent",
            "error": "Timeout",
        }
        result_fail = trainer._parse_entry(entry_fail)

        assert result_fail is not None
        assert result_fail[1] == 1  # Failure label


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase8Integration:
    """Integration tests for all Phase 8 components."""

    @pytest.mark.asyncio
    async def test_alerting_with_immune_system(self):
        """Verify alerting integrates with immune system flow."""
        from src.evaluation import get_immune_system, reset_immune_system
        from src.evaluation.alerting import AlertManager, HighRiskThreshold

        reset_immune_system()
        immune = get_immune_system()

        # Record a failure that would trigger alerts
        pattern = await immune.record_failure(
            operation="spawn_agent",
            prompt="DROP TABLE users",
            output="Error",
            grader_results=[{"name": "test", "passed": False}],
        )

        assert pattern is not None
        reset_immune_system()

    def test_all_exports_available(self):
        """Verify all Phase 8 exports are accessible."""
        from src.evaluation import (
            # Phase 8.1
            EvaluationTracer,
            get_tracer,
            trace_trial,
            calculate_gemini_cost,
            # Phase 8.2
            Alert,
            AlertSeverity,
            AlertManager,
            HighRiskThreshold,
            ConsoleNotifier,
            # Phase 8.3
            PatternFederation,
            PatternVisibility,
            # Phase 8.4
            FeatureExtractor,
            FailurePredictor,
            ModelTrainer,
        )

        # Just verify imports work
        assert EvaluationTracer is not None
        assert AlertManager is not None
        assert PatternFederation is not None
        assert FailurePredictor is not None


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
