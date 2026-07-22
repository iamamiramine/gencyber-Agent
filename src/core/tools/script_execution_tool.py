"""Run shell commands in the workbench shared PTY (HTTP).

Faithful to the gencyber-0 baseline: the tool dispatches the agent's command and
returns the raw terminal output as ``script_output``. It does NOT interpret,
summarize, or annotate output — the agent reads the raw text and infers what
happened. The workbench PTY merges stderr into the same stream as stdout, so
``stdout`` already carries the full terminal output; we only fall back to the
``stderr`` field for infrastructure-level errors (HTTP / transport failures) where
stdout is empty, so such a failure is never a silent blank turn.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.helpers.terminal_session_client import execute_in_terminal_session


def _compose_script_output_for_agent(result: Dict[str, Any]) -> str:
    """Return the raw terminal output, faithful to gencyber-0 (stdout only).

    No interpretation, no exit/cwd annotation. Because the workbench PTY folds
    stderr into the stdout stream, ``stdout`` is the complete terminal output. The
    only fallback is an infrastructure error reported on ``stderr`` while stdout is
    empty (e.g. the HTTP call to the workbench failed).
    """
    stdout = result.get("stdout")
    if stdout:
        return stdout
    stderr = result.get("stderr")
    if stderr:
        return stderr
    # Infrastructure-level failures (a busy PTY session, transport errors) report
    # on an ``error`` field with empty stdout/stderr. Surface it so a rejected or
    # failed call is never a silent blank turn the agent misreads as "no output".
    error = result.get("error")
    if error:
        return f"[workbench] {error}"
    return ""


# Public alias for tests / write_script_tool.
compose_script_output_for_agent = _compose_script_output_for_agent


class ExecuteScriptTool:
    """LangGraph node callable: run ``state['command']`` in the workbench PTY."""

    name: str = "execute_script"
    session_id: str = "default"
    default_timeout: int = 60

    def __init__(self, session_id: str = "default", timeout: int = 60) -> None:
        self.session_id = session_id
        self.default_timeout = timeout

    def _run(self, command: str, timeout: int | None = None) -> Dict[str, Any]:
        """Execute a shell command and return graph state updates."""
        timeout = self.default_timeout if timeout is None else timeout
        try:
            result = execute_in_terminal_session(
                command=command,
                session_id=self.session_id,
                timeout=timeout,
            )
            text = _compose_script_output_for_agent(result)
            # SSH-exit hint, faithful to gencyber-0: when the agent leaves an SSH
            # session, remind it that it is back at the local terminal.
            if command and command.strip().lower() == "exit":
                text = (
                    text
                    + "\n\nYou have exited the SSH session and are now at the local "
                    "terminal. Output your next command to continue the task."
                )
            print(
                f"EXECUTE_SCRIPT_TOOL command={command!r} stdout_preview_len={len(text[:500])}",
                flush=True,
            )
            return {"script_output": text, "command": None}
        except Exception as e:
            print(f"EXECUTE_SCRIPT_TOOL error={e!r}", flush=True)
            return {
                "script_output": f"An unexpected error occurred: {e}",
                "command": None,
            }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        command = (state.get("command") or "").strip()
        if not command:
            return {
                "script_output": "[execute_script] No ``command`` in state.",
                "command": None,
            }
        explicit = state.get("command_timeout")
        timeout = int(explicit) if explicit is not None else None
        return self._run(command, timeout=timeout)
