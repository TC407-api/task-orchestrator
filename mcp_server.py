#!/usr/bin/env python3
"""Entry point for running task-orchestrator as MCP server."""
import asyncio
import sys
import os
from pathlib import Path

# Get the directory where this script lives
SCRIPT_DIR = Path(__file__).parent.absolute()

# Add src to path
sys.path.insert(0, str(SCRIPT_DIR))

# Load .env from the project directory (not cwd)
from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / ".env")

from src.mcp.server import run_mcp_server

if __name__ == "__main__":
    asyncio.run(run_mcp_server())
