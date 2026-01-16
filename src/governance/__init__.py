"""Enterprise AI Governance Platform.

This module provides governance, audit, and cost control capabilities for enterprise AI development.

Modules:
- audit_log: Immutable audit logging with cryptographic integrity
- compliance: SOC2/HIPAA compliance report generation
- ttt_memory: Test-Time Training memory layer for O(1) pattern lookup
- context_compressor: Context window compression
- cli_orchestrator: Agent chain orchestration
- session_context: Persistent session state
- cost_dashboard: Enterprise cost dashboard
- budget_allocator: Department budget management
- agent_identity: Non-Human Identity (NHI) lifecycle management
- session_risk_scoring: Continuous session integrity validation
"""

__version__ = "0.1.0"

# Core audit and compliance
from src.governance.audit_log import AuditEntry, ImmutableAuditLog
from src.governance.compliance import (
    ComplianceFramework,
    ComplianceReport,
    ReportGenerator,
)

# Memory and compression
from src.governance.ttt_memory import TTTMemoryLayer
from src.governance.context_compressor import CompressedContext, ContextCompressor

# Orchestration and session
from src.governance.cli_orchestrator import (
    AgentChain,
    AgentChainStep,
    ChainResult,
    TEST_FIX_REVIEW,
    BUILD_VALIDATE_DEPLOY,
)
from src.governance.session_context import SessionContext, SessionManager

# Cost management
from src.governance.cost_dashboard import (
    CostDashboard,
    DashboardSummary,
    CostAlert,
    TimeRange,
    TrendPoint,
    ProjectCosts,
)
from src.governance.budget_allocator import (
    BudgetAllocator,
    BudgetCheckResult,
    Department,
    ChargebackReport,
)

# Agent Identity (NHI)
from src.governance.agent_identity import (
    AgentCredential,
    AgentIdentityManager,
    ArchetypeType,
)

# Session Risk Scoring
from src.governance.session_risk_scoring import (
    SessionRiskScore,
    SessionRiskScorer,
    RiskFactor,
)

__all__ = [
    # Audit
    "AuditEntry",
    "ImmutableAuditLog",
    # Compliance
    "ComplianceFramework",
    "ComplianceReport",
    "ReportGenerator",
    # TTT Memory
    "TTTMemoryLayer",
    # Context Compression
    "CompressedContext",
    "ContextCompressor",
    # CLI Orchestration
    "AgentChain",
    "AgentChainStep",
    "ChainResult",
    "TEST_FIX_REVIEW",
    "BUILD_VALIDATE_DEPLOY",
    # Session
    "SessionContext",
    "SessionManager",
    # Cost Dashboard
    "CostDashboard",
    "DashboardSummary",
    "CostAlert",
    "TimeRange",
    "TrendPoint",
    "ProjectCosts",
    # Budget
    "BudgetAllocator",
    "BudgetCheckResult",
    "Department",
    "ChargebackReport",
    # Agent Identity (NHI)
    "AgentCredential",
    "AgentIdentityManager",
    "ArchetypeType",
    # Session Risk Scoring
    "SessionRiskScore",
    "SessionRiskScorer",
    "RiskFactor",
]
