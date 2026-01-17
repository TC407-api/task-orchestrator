"""Tests for OperationClassifier - TDD RED phase."""
from src.agents.operation_classifier import (
    OperationClassifier,
    OperationCategory,
)


def test_safe_operations_auto_execute():
    """Test that safe operations are classified as SAFE."""
    classifier = OperationClassifier()

    # Read-only operations should be SAFE
    result = classifier.classify("tasks_list")
    assert result.category == OperationCategory.SAFE

    result = classifier.classify("inbox_status")
    assert result.category == OperationCategory.SAFE

    result = classifier.classify("cost_summary")
    assert result.category == OperationCategory.SAFE

    result = classifier.classify("file_read", {"path": "/tmp/test.txt"})
    assert result.category == OperationCategory.SAFE


def test_approval_required_for_file_delete():
    """Test that destructive operations require approval."""
    classifier = OperationClassifier()

    # File delete should require approval
    result = classifier.classify("file_delete", {"path": "/important/file.txt"})
    assert result.category == OperationCategory.REQUIRES_APPROVAL

    # Database write should require approval
    result = classifier.classify("database_write", {"table": "users", "data": {}})
    assert result.category == OperationCategory.REQUIRES_APPROVAL

    # Email send should require approval
    result = classifier.classify("email_send", {"to": "user@example.com"})
    assert result.category == OperationCategory.REQUIRES_APPROVAL


def test_blocked_patterns_raise_error():
    """Test that blocked patterns are classified as BLOCKED."""
    classifier = OperationClassifier()

    # rm -rf / should be blocked
    result = classifier.classify("shell_command", {"command": "rm -rf /"})
    assert result.category == OperationCategory.BLOCKED

    # DROP DATABASE should be blocked
    result = classifier.classify("database_query", {"sql": "DROP DATABASE production"})
    assert result.category == OperationCategory.BLOCKED

    # Force push to main should be blocked
    result = classifier.classify("git_command", {"command": "git push --force origin main"})
    assert result.category == OperationCategory.BLOCKED


def test_timeout_handling_auto_rejects():
    """Test that approval timeouts are set correctly."""
    classifier = OperationClassifier()

    # File delete should have default timeout
    result = classifier.classify("file_delete")
    assert result.timeout_seconds == 300

    # Deployment should have longer timeout
    result = classifier.classify("deployment")
    assert result.timeout_seconds == 600

    # Database migration should have longer timeout
    result = classifier.classify("database_migrate")
    assert result.timeout_seconds == 600


def test_classification_includes_timeout_seconds():
    """Test that all classifications include timeout_seconds."""
    classifier = OperationClassifier()

    # Safe operations should have timeout
    result = classifier.classify("tasks_list")
    assert hasattr(result, "timeout_seconds")
    assert isinstance(result.timeout_seconds, int)

    # Approval-required operations should have timeout
    result = classifier.classify("file_delete")
    assert hasattr(result, "timeout_seconds")
    assert isinstance(result.timeout_seconds, int)
    assert result.timeout_seconds > 0


def test_is_blocked_checks_patterns():
    """Test the is_blocked method with various patterns."""
    classifier = OperationClassifier()

    assert classifier.is_blocked("cmd", {"command": "rm -rf /"}) is True
    assert classifier.is_blocked("cmd", {"command": "DROP DATABASE test"}) is True
    assert classifier.is_blocked("cmd", {"command": "ls -la"}) is False


def test_unknown_operations_default_to_approval():
    """Test that unknown operations require approval by default."""
    classifier = OperationClassifier()

    # Unknown operation should require approval (safe default)
    result = classifier.classify("unknown_dangerous_operation")
    assert result.category == OperationCategory.REQUIRES_APPROVAL
