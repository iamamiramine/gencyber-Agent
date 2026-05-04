"""
Write multi-line scripts on the workbench volume via the terminal session (non-interactive).

Mirrors the NYUCTF pattern of materializing a file then running ``python path`` / ``bash path``
with ``execute_script_tool``. Scripts are watermarked (header comment + filename token) so
the agent can recognize its own artifacts under ``GENCYBER_AGENT_SCRIPT_DIR``.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shlex
import uuid
from typing import Any, Dict

from core.helpers.terminal_session_client import execute_in_terminal_session
from core.tools.script_execution_tool import _compose_script_output_for_agent

logger = logging.getLogger(__name__)

SCRIPT_DIR = "/workspace/.gencyber-agent-scripts"
MARKER_PREFIX = "# gencyber-agent-managed"
MARKER_TOKEN = "gencyber"
_MAX_BYTES = 262144


def _safe_session_slug(session_id: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(session_id or "default")).strip("._-")
    return (t[:48] or "default") if t else "default"


def _normalize_language(raw: str | None) -> str:
    if not raw or not str(raw).strip():
        return "py"
    ext = str(raw).strip().lower().lstrip(".")
    if not re.fullmatch(r"[a-z0-9]{1,12}", ext):
        return "py"
    return ext


def _watermark(script_id: str, remote_path: str, body: str, ext: str) -> str:
    header = (
        f"{MARKER_PREFIX}\n"
        f"# script-id: {script_id}\n"
        f"# saved-path: {remote_path}\n"
        f"# hint-next: run e.g. ``python3 {remote_path}`` or ``bash {remote_path}``\n"
    )
    if ext in ("sh", "bash") and body.lstrip().startswith("#!"):
        first_nl = body.find("\n")
        if first_nl != -1:
            shebang, rest = body[: first_nl + 1], body[first_nl + 1 :]
            return shebang + header + rest
    return header + body


def _remote_write_command(remote_path: str, raw_bytes: bytes) -> str:
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    parent = os.path.dirname(remote_path)
    inner = (
        f"import pathlib,base64; p=pathlib.Path({repr(remote_path)}); "
        f"p.write_bytes(base64.b64decode({repr(b64)}))"
    )
    return f"mkdir -p {shlex.quote(parent)} && python3 -c {shlex.quote(inner)}"


class WriteScriptTool:
    """
    Same role as ``ExecuteScriptTool``: used as a LangGraph node callable; writes via
    ``execute_in_terminal_session`` so files land on the workbench volume.
    """

    name: str = "write_script"
    session_id: str = "default"

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        body = state.get("write_script")
        if body is None or not str(body).strip():
            return {
                "script_output": "[write_script] No ``write_script`` body in state.",
                "write_script": None,
                "write_script_language": None,
            }
        ext = _normalize_language(state.get("write_script_language"))
        raw_body = str(body)
        if len(raw_body.encode("utf-8")) > _MAX_BYTES:
            return {
                "script_output": f"[write_script] Script exceeds max size ({_MAX_BYTES} bytes).",
                "write_script": None,
                "write_script_language": None,
            }

        script_id = uuid.uuid4().hex
        slug = _safe_session_slug(self.session_id)
        filename = f"{slug}_{MARKER_TOKEN}_{script_id}.{ext}"
        remote_path = f"{SCRIPT_DIR.rstrip('/')}/{filename}"
        stamped = _watermark(script_id, remote_path, raw_body, ext)
        payload = stamped.encode("utf-8")

        cmd = _remote_write_command(remote_path, payload)
        try:
            result = execute_in_terminal_session(
                command=cmd,
                session_id=self.session_id,
                timeout=min(120, int(os.getenv("GENCYBER_WRITE_SCRIPT_TIMEOUT", "120"))),
            )
        except Exception as e:
            logger.exception("write_script remote write failed")
            return {
                "script_output": f"[write_script] {e}",
                "write_script": None,
                "write_script_language": None,
            }

        out = _compose_script_output_for_agent(result)
        chmod = ""
        if ext in ("sh", "bash"):
            chmod_res = execute_in_terminal_session(
                command=f"chmod +x {shlex.quote(remote_path)}",
                session_id=self.session_id,
                timeout=15,
            )
            chmod = "\n" + _compose_script_output_for_agent(chmod_res)

        summary = (
            f"[write_script] Wrote watermarked file.\n"
            f"- path: ``{remote_path}``\n"
            f"- script-id: ``{script_id}``\n"
            f"- Next turn: set ``command`` to run it, e.g. "
            f"``python3 {remote_path}`` or ``bash {remote_path}`` (not a one-liner ``python3 -c`` for long logic).\n"
            f"\n--- remote write stdout/stderr ---\n{out}{chmod}"
        )
        print(
            f"WRITE_SCRIPT_TOOL path={remote_path!r} script_id={script_id} ext={ext}",
            flush=True,
        )
        return {
            "script_output": summary,
            "write_script": None,
            "write_script_language": None,
            "command": None,
        }

