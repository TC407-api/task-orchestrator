"""Task orchestrator agents."""
from .email_agent import EmailAgent
from .calendar_agent import CalendarAgent
from .coordinator import CoordinatorAgent
from .audit_workflow import AuditWorkflow
from .archetype_registry import (
    Archetype,
    ArchetypeRegistry,
    ArchetypeConfig,
    get_archetype_registry,
    reset_registry,
)
from .background_tasks import (
    BackgroundTaskScheduler,
    ScheduledTask,
    TaskResult,
    TaskScheduleType,
    TaskStatus,
    # Batch 2: BackgroundTaskManager
    BackgroundTaskManager,
    TaskManagerStatus,
    TaskManagerDefinition,
    TaskManagerExecution,
)
from .terminal_loop import (
    TerminalListener,
    ErrorCapture,
    StackTraceParser,
    FixProposer,
    DetectedError,
    FixProposal,
    StackTraceLocation,
    ErrorLanguage,
    ErrorSeverity,
    # Batch 2: TerminalLoop
    TerminalLoop,
    LoopState,
    LoopIteration,
)
from .shadow_validator import (
    # Batch 2: Shadow comparison features
    ASTNodeType,
    ASTNode,
    ShadowValidationResult,
    ShadowComparator,
)
from .workflows import (
    # Batch 2: @workflow/@step decorators
    step,
    workflow,
    StepStatus,
    WorkflowStepDef,
    DecoratorWorkflowState,
)

__all__ = [
    "EmailAgent",
    "CalendarAgent",
    "CoordinatorAgent",
    "AuditWorkflow",
    "Archetype",
    "ArchetypeRegistry",
    "ArchetypeConfig",
    "get_archetype_registry",
    "reset_registry",
    # Background tasks (original)
    "BackgroundTaskScheduler",
    "ScheduledTask",
    "TaskResult",
    "TaskScheduleType",
    "TaskStatus",
    # Background tasks (Batch 2)
    "BackgroundTaskManager",
    "TaskManagerStatus",
    "TaskManagerDefinition",
    "TaskManagerExecution",
    # Terminal loop (original)
    "TerminalListener",
    "ErrorCapture",
    "StackTraceParser",
    "FixProposer",
    "DetectedError",
    "FixProposal",
    "StackTraceLocation",
    "ErrorLanguage",
    "ErrorSeverity",
    # Terminal loop (Batch 2)
    "TerminalLoop",
    "LoopState",
    "LoopIteration",
    # Shadow validator (Batch 2)
    "ASTNodeType",
    "ASTNode",
    "ShadowValidationResult",
    "ShadowComparator",
    # Workflows (Batch 2)
    "step",
    "workflow",
    "StepStatus",
    "WorkflowStepDef",
    "DecoratorWorkflowState",
]
