from langchain.tools import BaseTool
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field, model_validator

from core.helpers.terminal_session_client import execute_in_terminal_session


def _compose_script_output_for_agent(result: Dict[str, Any]) -> str:
    """
    Merge stdout, stderr, and exit metadata so downstream agents see failures
    (curl and many tools write errors to stderr only).
    """
    chunks: List[str] = []
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if stdout:
        chunks.append(stdout)
    if stderr:
        chunks.append("[stderr]\n" + stderr)
    exit_code = result.get("exit_code")
    success = result.get("success")
    meta: List[str] = []
    if exit_code is not None:
        meta.append(f"exit_code={exit_code}")
    if success is False:
        meta.append("success=false")
    if meta:
        chunks.append("[" + " ".join(meta) + "]")
    if not chunks:
        return "(no output)"
    return "\n\n".join(chunks)

class ScriptExecutionInput(BaseModel):
    """Input for ExecuteScriptTool."""
    command: Optional[str] = Field(default=None, description="The command to execute")
    timeout: Optional[int] = Field(default=60, description="Timeout in seconds for command execution")
    
    @model_validator(mode="after")
    def validate_and_set_command(self) -> "ScriptExecutionInput":
        """Validate that command is provided and set command if missing."""
        if self.command is None:
            raise ValueError("'command' field must be provided")
        return self

class ExecuteScriptTool(BaseTool):
    """Run shell commands in the workbench terminal-session (PTY over HTTP)."""

    name: str = "execute_script"
    description: str = (
        "Execute a shell command in the shared workbench terminal and return stdout/stderr. "
        "Uses ``TERMINAL_SESSION_URL`` and the graph session id."
    )
    args_schema: Type[BaseModel] = ScriptExecutionInput
    session_id: str = "default"

    def __init__(self, session_id: str = "default"):
        """Initialize with the terminal session id (align with UI / thread)."""
        super().__init__()
        self.session_id = session_id
    
    def _run(self, command: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
        """Execute a shell command and return its output.
        
        Args:
            command (str, optional): The command to execute.
            timeout (int, optional): Timeout in seconds. Defaults to 60.
            
        Returns:
            Dict[str, Any]: Dictionary with execution results to update the graph state
        """            
        try:
            # Sync HTTP — safe inside LangGraph ``astream_events`` / async FastAPI (no asyncio.run).
            result = execute_in_terminal_session(
                command=command or "",
                session_id=self.session_id,
                timeout=timeout,
            )

            return_dict = {
                "script_output": _compose_script_output_for_agent(result),
            }

            out_preview = (return_dict.get("script_output") or "")[:500]
            print(
                f"EXECUTE_SCRIPT_TOOL command={command!r} stdout_preview_len={len(out_preview)}",
                flush=True,
            )
            return return_dict
            
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(f"EXECUTE_SCRIPT_TOOL error={e!r}", flush=True)
            return {
                "script_output": error_msg,
            }