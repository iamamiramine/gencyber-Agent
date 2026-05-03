"""HTTP client for the workbench ``terminal-session`` service (``/api/execute``)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _terminal_base_url() -> str:
    return os.getenv("TERMINAL_SESSION_URL", "http://gencyber-workbench:3000").strip().rstrip("/")


def execute_in_terminal_session(
    command: str,
    session_id: str = "default",
    timeout: int = 30,
    *,
    terminal_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``command`` in a persistent PTY via ``POST {TERMINAL_SESSION_URL}/api/execute``."""
    base = (terminal_url or _terminal_base_url()).rstrip("/")
    try:
        requests.post(
            f"{base}/api/sessions",
            json={"sessionId": session_id},
            timeout=5,
        )
        response = requests.post(
            f"{base}/api/execute",
            json={
                "sessionId": session_id,
                "command": command,
                "timeout": timeout,
            },
            timeout=timeout + 10,
        )
        if response.status_code == 200:
            return response.json()
        return {
            "stdout": "",
            "stderr": f"HTTP error: {response.status_code}",
            "exit_code": -1,
            "success": False,
            "command": command,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "success": False,
            "command": command,
        }


def list_terminal_sessions(terminal_url: Optional[str] = None) -> Dict[str, Any]:
    base = (terminal_url or _terminal_base_url()).rstrip("/")
    try:
        response = requests.get(f"{base}/api/sessions", timeout=5)
        if response.status_code == 200:
            return {"success": True, "sessions": response.json()}
        return {
            "success": False,
            "error": f"HTTP {response.status_code}",
            "sessions": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sessions": [],
        }
