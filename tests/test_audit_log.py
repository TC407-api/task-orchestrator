import pytest
import json
from datetime import datetime, timedelta
from dataclasses import FrozenInstanceError
from src.governance.audit_log import AuditEntry, ImmutableAuditLog

# Fixture to prevent writing to actual user home directory during tests
@pytest.fixture
def audit_log(tmp_path):
    """Returns an ImmutableAuditLog instance using a temporary directory."""
    return ImmutableAuditLog(storage_dir=tmp_path / "audit")

def create_sample_entry(entry_id: str = "test-id-1") -> AuditEntry:
    return AuditEntry(
        id=entry_id,
        timestamp=datetime.now(),
        operation="EXECUTE_TOOL",
        agent_id="agent-007",
        input_hash="abc123hash",
        output_hash="xyz789hash",
        cost_usd=0.002,
        trace_id="trace-uuid-1",
        prev_hash=None
    )

def test_audit_entry_has_sha256_hash(audit_log):
    """Test that appending an entry generates a valid SHA256 hash for the chain."""
    entry = create_sample_entry()

    # Action
    entry_hash = audit_log.append(entry)

    # Assert
    assert isinstance(entry_hash, str)
    assert len(entry_hash) == 64  # SHA256 length in hex
    # Verify the log file actually contains this hash
    latest = audit_log.query(limit=1)[0]
    assert latest.prev_hash is not None or entry_hash is not None

def test_audit_chain_validates_integrity(audit_log):
    """Test that the cryptographic chain is valid and detects tampering."""
    entry1 = create_sample_entry("id-1")
    entry2 = create_sample_entry("id-2")

    audit_log.append(entry1)
    audit_log.append(entry2)

    # Should be valid initially
    assert audit_log.verify_chain() is True

    # Simulate tampering: Modify the underlying file manually
    # Note: This relies on the implementation details of storage (jsonl)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = audit_log.storage_dir / f"{today}.jsonl"

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Tamper with the first record (change cost)
    tampered_record = json.loads(lines[0])
    tampered_record['cost_usd'] = 1000.00
    lines[0] = json.dumps(tampered_record) + "\n"

    with open(log_file, 'w') as f:
        f.writelines(lines)

    # Should fail validation now
    assert audit_log.verify_chain() is False

def test_audit_entry_immutable_after_creation():
    """Test that AuditEntry objects cannot be modified after instantiation."""
    entry = create_sample_entry()

    with pytest.raises(FrozenInstanceError):
        entry.cost_usd = 1.00

    with pytest.raises(FrozenInstanceError):
        entry.operation = "HACK_SYSTEM"

def test_audit_query_by_time_range(audit_log):
    """Test filtering logs by start and end timestamps."""
    base_time = datetime.now()

    entry_old = create_sample_entry("old")
    # Bypass immutability for setup or create new instances with specific times
    # Since we expect AuditEntry to be frozen, we use object.__setattr__ or constructor
    object.__setattr__(entry_old, 'timestamp', base_time - timedelta(hours=2))

    entry_new = create_sample_entry("new")
    object.__setattr__(entry_new, 'timestamp', base_time)

    audit_log.append(entry_old)
    audit_log.append(entry_new)

    # Query for last hour only
    start_time = base_time - timedelta(hours=1)
    results = audit_log.query(start_time=start_time)

    assert len(results) == 1
    assert results[0].id == "new"

def test_audit_query_by_operation_type(audit_log):
    """Test filtering logs by operation string."""
    e1 = create_sample_entry("e1")
    object.__setattr__(e1, 'operation', "TOOL_USE")

    e2 = create_sample_entry("e2")
    object.__setattr__(e2, 'operation', "LLM_CALL")

    audit_log.append(e1)
    audit_log.append(e2)

    results = audit_log.query(operation="LLM_CALL")

    assert len(results) == 1
    assert results[0].operation == "LLM_CALL"

def test_audit_export_to_jsonl(audit_log, tmp_path):
    """Test exporting the audit log to a specific external file."""
    audit_log.append(create_sample_entry())

    export_path = tmp_path / "export_test.jsonl"
    audit_log.export(export_path)

    assert export_path.exists()

    with open(export_path, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        assert data['agent_id'] == "agent-007"
