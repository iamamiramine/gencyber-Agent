from pydantic import BaseModel, Field
from typing import List, Optional


class GenerativeAgentResponse(BaseModel):
    """Structured output for command-and-goal steps. Task-specific semantics come from
    workflow configuration and planning output, not from this schema.
    """

    reasoning: List[str] = Field(
        description="Short reasoning steps for the latest evidence and the chosen action."
    )
    command: Optional[str] = Field(
        default=None,
        description="One non-interactive shell command to run next, or null when submitting the final answer.",
    )
    ethical: bool = Field(
        description="True for legitimate CTF / training / educational security work."
    )
    submitted_goal: Optional[str] = Field(
        default=None,
        description=(
            "When the task is complete, the final artifact to submit (flag, token, "
            "password, answer string, etc.) as required by the PM objectives and "
            "success criteria. Null while still exploring."
        ),
    )
