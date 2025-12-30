"""OAuth setup script - run this to authorize Gmail/Calendar access."""
import pickle
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes needed for Task Orchestrator
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Paths
CREDENTIALS_PATH = Path.home() / ".claude" / "oauth-credentials.json"
TOKEN_PATH = Path.home() / ".claude" / "oauth-token.pickle"


def main():
    """Run OAuth flow and save token."""
    print("=" * 60)
    print("Task Orchestrator OAuth Setup")
    print("=" * 60)
    print()

    if not CREDENTIALS_PATH.exists():
        print(f"ERROR: OAuth credentials not found at {CREDENTIALS_PATH}")
        print("Download from GCP Console > APIs & Services > Credentials")
        return

    print(f"Using credentials: {CREDENTIALS_PATH}")
    print(f"Token will be saved to: {TOKEN_PATH}")
    print()
    print("A browser window will open for authorization.")
    print("Sign in with your Google account and approve access.")
    print()

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH),
            SCOPES
        )

        # Run the OAuth flow - this opens a browser
        creds = flow.run_local_server(port=0)

        # Save the token
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

        print()
        print("=" * 60)
        print("SUCCESS! OAuth token saved.")
        print("=" * 60)
        print()
        print("You can now restart the Task Orchestrator API:")
        print("  cd ~/Projects/task-orchestrator")
        print("  uvicorn src.api.server:app --reload")

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("If you see 'access_denied', make sure:")
        print("1. Your email is added as a test user in GCP Console")
        print("2. Go to: APIs & Services > OAuth consent screen > Test users")
        print("3. Add your email address")


if __name__ == "__main__":
    main()
