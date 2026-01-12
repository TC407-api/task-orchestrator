"""
Evaluation Module for Task Orchestrator.

This package provides the core primitives for the evaluation system,
allowing for the creation of Trials and the application of Graders to assess
agent performance.

Components:
- Trial: Complete record of an agent execution with evaluation
- GraderResult: Result from a single grader evaluation
- Graders: Various code-based validators (JSON, regex, length, etc.)
- Integration: Langfuse Scores API wrapper
- Export: Training data exporter
- Immune System: Failure pattern storage and guardrails
- Alerting (Phase 8.2): High-risk pattern detection and notifications
- Prediction (Phase 8.4): ML-based failure prediction
"""

from .trial import Trial, GraderResult
from .graders import (
    Grader,
    GraderPipeline,
    NonEmptyGrader,
    JSONValidGrader,
    JSONSchemaGrader,
    RegexGrader,
    LengthGrader,
    ContainsGrader,
    NotContainsGrader,
    # Model-based graders
    ModelGrader,
    RelevanceGrader,
    CompletenessGrader,
    AccuracyGrader,
    FormatGrader,
    # Specialized model graders (Phase 7)
    CodeQualityGrader,
    SafetyGrader,
    PerformanceGrader,
)
from .integration import score_trial, score_grader_result, create_eval_span
from .export import TrainingDataExporter, get_exporter
from .immune_system import (
    ImmuneSystem,
    ImmuneResponse,
    get_immune_system,
    reset_immune_system,
    FailurePattern,
    FailurePatternStore,
    PatternMatcher,
    MatchedPattern,
    PromptGuardrails,
    GuardrailResult,
    # Dashboard (Phase 7)
    ImmuneDashboard,
    create_dashboard,
    # Federation (Phase 8.3)
    PatternFederation,
    PatternVisibility,
)
# Phase 8.1: Enhanced Langfuse Integration
from .langfuse_integration import (
    EvaluationTracer,
    get_tracer,
    trace_trial,
    trace_grader,
    calculate_gemini_cost,
)
# Phase 8.2: Alerting
from .alerting import (
    Alert,
    AlertSeverity,
    AlertManager,
    AlertRule,
    HighRiskThreshold,
    FrequencySpike,
    NewPatternDetected,
    ConsecutiveFailures,
    ConsoleNotifier,
    WebhookNotifier,
    SlackNotifier,
)
# Phase 8.4: Prediction
from .prediction import (
    FeatureExtractor,
    FailurePredictor,
    PredictionResult,
    ModelTrainer,
)

__all__ = [
    # Core
    "Trial",
    "GraderResult",
    # Code-based Graders
    "Grader",
    "GraderPipeline",
    "NonEmptyGrader",
    "JSONValidGrader",
    "JSONSchemaGrader",
    "RegexGrader",
    "LengthGrader",
    "ContainsGrader",
    "NotContainsGrader",
    # Model-based Graders (LLM-as-judge)
    "ModelGrader",
    "RelevanceGrader",
    "CompletenessGrader",
    "AccuracyGrader",
    "FormatGrader",
    # Specialized Model Graders (Phase 7)
    "CodeQualityGrader",
    "SafetyGrader",
    "PerformanceGrader",
    # Integration
    "score_trial",
    "score_grader_result",
    "create_eval_span",
    # Export
    "TrainingDataExporter",
    "get_exporter",
    # Immune System
    "ImmuneSystem",
    "ImmuneResponse",
    "get_immune_system",
    "reset_immune_system",
    "FailurePattern",
    "FailurePatternStore",
    "PatternMatcher",
    "MatchedPattern",
    "PromptGuardrails",
    "GuardrailResult",
    # Dashboard (Phase 7)
    "ImmuneDashboard",
    "create_dashboard",
    # Federation (Phase 8.3)
    "PatternFederation",
    "PatternVisibility",
    # Langfuse Integration (Phase 8.1)
    "EvaluationTracer",
    "get_tracer",
    "trace_trial",
    "trace_grader",
    "calculate_gemini_cost",
    # Alerting (Phase 8.2)
    "Alert",
    "AlertSeverity",
    "AlertManager",
    "AlertRule",
    "HighRiskThreshold",
    "FrequencySpike",
    "NewPatternDetected",
    "ConsecutiveFailures",
    "ConsoleNotifier",
    "WebhookNotifier",
    "SlackNotifier",
    # Prediction (Phase 8.4)
    "FeatureExtractor",
    "FailurePredictor",
    "PredictionResult",
    "ModelTrainer",
]
