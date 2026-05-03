from typing import List, Optional

from pydantic import BaseModel, Field


class ReasoningAgentResponse(BaseModel):
    """Structured output for hierarchical task-tree maintenance and next-step selection."""

    reasoning: List[str] = Field(
        description=(
            "Short reasoning bullets describing why the tree was initialized or updated "
            "and why the recommended task was selected."
        )
    )
    task_tree: str = Field(
        description=(
            "The full hierarchical task tree in numbered format "
            "(e.g., 1, 1.1, 1.1.1) with statuses: to-do, completed, not applicable, "
            "failed, blocked."
        )
    )
    candidate_tasks: List[str] = Field(
        default_factory=list,
        description=(
            "Actionable candidate leaf lines still open: to-do, failed, or blocked "
            "(full lines with ids, matching task_tree)."
        ),
    )
    recommended_task: Optional[str] = Field(
        default=None,
        description="The single highest-priority next task selected from candidate leaf tasks.",
    )
    evidence_summary: Optional[str] = Field(
        default=None,
        description="Compact summary of the latest execution evidence used for this update.",
    )
    update_applied: bool = Field(
        default=True,
        description="Whether the latest evidence changed the task tree.",
    )
