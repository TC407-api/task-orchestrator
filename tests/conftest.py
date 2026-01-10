"""Pytest fixtures for Task Orchestrator tests."""
import pytest
from datetime import timedelta
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.api.server import app
from src.api.auth.jwt import create_access_token


# --- Authentication Fixtures ---

@pytest.fixture
def valid_token() -> str:
    """Generate a valid JWT token for testing."""
    return create_access_token(
        subject="test-user",
        scopes=["read", "write"],
        expires_delta=timedelta(hours=1),
    )


@pytest.fixture
def expired_token() -> str:
    """Generate an expired JWT token for testing."""
    return create_access_token(
        subject="test-user",
        scopes=["read", "write"],
        expires_delta=timedelta(seconds=-1),  # Already expired
    )


@pytest.fixture
def token_with_scopes() -> callable:
    """Factory fixture to create tokens with specific scopes."""
    def _create_token(scopes: list[str]) -> str:
        return create_access_token(
            subject="test-user",
            scopes=scopes,
            expires_delta=timedelta(hours=1),
        )
    return _create_token


@pytest.fixture
def auth_headers(valid_token: str) -> dict[str, str]:
    """Generate Authorization headers with a valid token."""
    return {"Authorization": f"Bearer {valid_token}"}


@pytest.fixture
def expired_auth_headers(expired_token: str) -> dict[str, str]:
    """Generate Authorization headers with an expired token."""
    return {"Authorization": f"Bearer {expired_token}"}


# --- Mock Fixtures ---

@pytest.fixture
def mock_coordinator() -> MagicMock:
    """Create a mock CoordinatorAgent."""
    coordinator = MagicMock()
    coordinator.tasks = {}
    coordinator.add_task = AsyncMock()
    coordinator.complete_task = AsyncMock()
    coordinator.schedule_task = AsyncMock()
    coordinator.sync_from_email = AsyncMock(return_value=[])
    coordinator.get_daily_summary = AsyncMock(return_value={})
    coordinator.prioritize_tasks = AsyncMock(return_value=[])
    coordinator.get_overdue_tasks = MagicMock(return_value=[])
    coordinator.auto_schedule_pending = AsyncMock(return_value=[])
    coordinator.calendar_agent = None
    return coordinator


@pytest.fixture
def mock_gmail_client() -> MagicMock:
    """Create a mock Gmail client."""
    client = MagicMock()
    client.get_unread_emails = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_calendar_client() -> MagicMock:
    """Create a mock Calendar client."""
    client = MagicMock()
    client.get_events = AsyncMock(return_value=[])
    client.create_event = AsyncMock()
    return client


# --- Client Fixtures ---

@pytest.fixture
def test_client(mock_coordinator: MagicMock) -> Generator[TestClient, None, None]:
    """Create a test client with mocked coordinator."""
    with patch("src.api.server.coordinator", mock_coordinator):
        with TestClient(app) as client:
            yield client


@pytest.fixture
async def async_client(mock_coordinator: MagicMock) -> AsyncClient:
    """Create an async test client with mocked coordinator."""
    with patch("src.api.server.coordinator", mock_coordinator):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# --- Rate Limiting Fixtures ---

@pytest.fixture
def disable_rate_limiting():
    """Disable rate limiting for tests."""
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    # Create a disabled limiter
    disabled_limiter = Limiter(key_func=get_remote_address, enabled=False)

    with patch.object(app.state, "limiter", disabled_limiter):
        yield


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_task_data() -> dict:
    """Sample task creation data."""
    return {
        "title": "Test Task",
        "description": "A test task description",
        "priority": "high",
        "tags": ["test", "sample"],
        "estimated_minutes": 60,
        "auto_schedule": False,
    }


@pytest.fixture
def sample_task_update() -> dict:
    """Sample task update data."""
    return {
        "title": "Updated Task Title",
        "priority": "low",
        "notes": "Updated notes",
    }
