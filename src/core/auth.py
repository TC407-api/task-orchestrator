"""Authentication utilities for Google APIs."""
import json
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .config import settings


def get_oauth_credentials(
    scopes: list[str],
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> Credentials:
    """
    Get or refresh OAuth credentials for user-consent APIs.

    Args:
        scopes: List of OAuth scopes required
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to store/load token pickle

    Returns:
        Valid Credentials object
    """
    credentials_path = credentials_path or settings.oauth_credentials_path
    token_path = token_path or settings.oauth_token_path

    creds = None

    # Load existing token if available (JSON format for security)
    if token_path.exists():
        with open(token_path, "r") as token:
            token_data = json.load(token)
            creds = Credentials.from_authorized_user_info(token_data, scopes)

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"OAuth credentials not found at {credentials_path}. "
                    "Download from GCP Console > APIs & Services > Credentials"
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), scopes
            )
            creds = flow.run_local_server(port=0)

        # Save token for next run (JSON format for security)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w") as token:
            token_data = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": list(creds.scopes) if creds.scopes else scopes,
            }
            json.dump(token_data, token)

    return creds


def get_service_credentials(
    scopes: Optional[list[str]] = None,
    service_account_path: Optional[Path] = None,
) -> service_account.Credentials:
    """
    Get service account credentials for server-to-server APIs.

    Args:
        scopes: Optional list of scopes to restrict
        service_account_path: Path to service account JSON key

    Returns:
        Service account Credentials object
    """
    service_account_path = service_account_path or settings.service_account_path

    if not service_account_path.exists():
        raise FileNotFoundError(
            f"Service account key not found at {service_account_path}. "
            "Create in GCP Console > IAM & Admin > Service Accounts"
        )

    credentials = service_account.Credentials.from_service_account_file(
        str(service_account_path),
        scopes=scopes or ["https://www.googleapis.com/auth/cloud-platform"],
    )

    return credentials


def get_all_scopes() -> list[str]:
    """Get combined list of all OAuth scopes."""
    return (
        settings.gmail_scopes
        + settings.calendar_scopes
        + settings.drive_scopes
    )
