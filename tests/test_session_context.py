import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from src.governance.session_context import SessionManager, SessionContext


# Fixture to prevent writing to real user home directory during tests
@pytest.fixture
def temp_storage_path(tmp_path):
    sessions_dir = tmp_path / ".claude" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def test_session_persists_to_disk(temp_storage_path):
    """Test that creating a session writes a JSON file to disk."""
    manager = SessionManager(storage_path=temp_storage_path)

    # Act
    session = manager.create_session()

    # Assert
    expected_file = temp_storage_path / f"{session.session_id}.json"
    assert expected_file.exists(), "Session file was not created on disk"

    with open(expected_file, 'r') as f:
        data = json.load(f)
        assert data['session_id'] == session.session_id


def test_session_restores_on_startup(temp_storage_path):
    """Test that a session can be loaded from disk by ID."""
    # Setup: Create a session with one manager
    manager_1 = SessionManager(storage_path=temp_storage_path)
    original_session = manager_1.create_session()
    session_id = original_session.session_id

    # Act: Try to restore with a fresh manager instance
    manager_2 = SessionManager(storage_path=temp_storage_path)
    restored_session = manager_2.restore_session(session_id)

    # Assert
    assert restored_session is not None
    assert restored_session.session_id == session_id
    assert restored_session.created_at == original_session.created_at


def test_session_tracks_file_context(temp_storage_path):
    """Test that accessed files are recorded in the session."""
    manager = SessionManager(storage_path=temp_storage_path)
    session = manager.create_session()

    files_to_track = ["src/main.py", "tests/test_core.py"]

    # Act
    manager.update_context(
        session_id=session.session_id,
        files=files_to_track,
        messages=[]
    )

    # Reload to verify persistence
    reloaded = manager.restore_session(session.session_id)

    # Assert
    assert reloaded is not None
    assert set(files_to_track).issubset(set(reloaded.files_accessed))


def test_session_tracks_conversation_history(temp_storage_path):
    """Test that conversation summary/history is updated."""
    manager = SessionManager(storage_path=temp_storage_path)
    session = manager.create_session()

    messages = [
        {"role": "user", "content": "Analyze this code"},
        {"role": "assistant", "content": "The code looks good."}
    ]

    # Act
    manager.update_context(
        session_id=session.session_id,
        files=[],
        messages=messages
    )

    # Reload
    reloaded = manager.restore_session(session.session_id)

    # Assert
    assert reloaded is not None
    # Assuming the summary appends or stores the last interaction
    assert len(reloaded.conversation_summary) > 0
    assert "Analyze this code" in str(reloaded.conversation_summary)


def test_session_expires_after_timeout(temp_storage_path):
    """Test that expired sessions are cleaned up."""
    # Setup: Manager with 0 hours timeout (immediate expiration logic)
    # Note: In a real scenario we might mock datetime, but here we rely on
    # passing a small timeout or manipulating the file timestamp.
    manager = SessionManager(storage_path=temp_storage_path, timeout_hours=0.0001)
    session = manager.create_session()

    # Ensure file exists
    expected_file = temp_storage_path / f"{session.session_id}.json"
    assert expected_file.exists()

    # Sleep briefly to ensure timeout threshold is passed
    time.sleep(0.5)

    # Act
    deleted_count = manager.cleanup_expired()

    # Assert
    assert deleted_count == 1
    assert not expected_file.exists(), "Expired session file should be deleted"
    assert manager.restore_session(session.session_id) is None
