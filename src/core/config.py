"""Configuration settings for Task Orchestrator."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # GCP Configuration
    gcp_project_id: str = "claude-code-ultimate"
    gcp_region: str = "us-central1"

    # Credentials paths
    oauth_credentials_path: Path = Path.home() / ".claude" / "oauth-credentials.json"
    oauth_token_path: Path = Path.home() / ".claude" / "oauth-token.pickle"
    service_account_path: Path = Path.home() / ".claude" / "vertex-ai-key.json"

    # API Scopes
    gmail_scopes: list[str] = [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.compose",
    ]
    calendar_scopes: list[str] = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    ]
    drive_scopes: list[str] = [
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    # Rate limits (requests per minute)
    gmail_rate_limit: int = 250
    calendar_rate_limit: int = 500
    drive_rate_limit: int = 1000
    vertex_ai_rate_limit: int = 60
    gemini_rate_limit: int = 1000

    # Model configuration - Gemini models (latest as of Dec 2024)
    default_fast_model: str = "gemini-2.5-flash-lite"
    default_balanced_model: str = "gemini-2.5-flash"
    default_reasoning_model: str = "gemini-3-pro-preview"
    default_code_model: str = "gemini-3-flash-preview"

    # API Server
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    class Config:
        env_prefix = "TASK_ORCHESTRATOR_"
        env_file = ".env"


settings = Settings()
