from pydantic import BaseModel, Field
from typing import List, Optional


class GenerativeAgentResponse(BaseModel):
    """Structured output for one ReAct turn of the single generative agent.

    Faithful to the proven single-agent baseline (gencyber-0): each turn the agent
    emits short ``reasoning``, at most one action — a shell ``command``, a
    ``write_script`` body, or a final ``submitted_goal`` — and an ``ethical`` flag.
    Task-specific semantics come from the system prompt, not this schema.
    """

    reasoning: List[str] = Field(
        description="Short reasoning steps for the latest evidence and the chosen action."
    )
    command: Optional[str] = Field(
        default=None,
        description="One shell command to run next in the shared PTY, or null when submitting the final answer.",
    )
    write_script: Optional[str] = Field(
        default=None,
        description=(
            "Full script source to save on the workbench (multi-line Python, bash, etc.). "
            "Null when using ``command`` instead. After the write step completes, use ``command`` to run "
            "e.g. ``python3 /workspace/.gencyber-agent-scripts/...``."
        ),
    )
    write_script_language: Optional[str] = Field(
        default=None,
        description="Extension / language for ``write_script`` (e.g. py, sh, or bash). Defaults to py.",
    )
    ethical: bool = Field(
        description="True for legitimate CTF / training / educational security work."
    )
    submitted_goal: Optional[str] = Field(
        default=None,
        description=(
            "When the task is complete, the final artifact to submit (flag, token, "
            "password, answer string, etc.) as required by the objectives and success "
            "criteria. Null while still exploring."
        ),
    )
