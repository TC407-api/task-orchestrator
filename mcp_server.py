#!/usr/bin/env python3
"""Entry point for running task-orchestrator as MCP server."""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.mcp.server import run_mcp_server

if __name__ == "__main__":
    asyncio.run(run_mcp_server())
