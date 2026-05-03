from typing import List

from pydantic import BaseModel, Field


class PMAgentResponse(BaseModel):
    """Structured session plan derived from the user briefing."""

    high_level_context: str = Field(
        description="Brief summary of task type, scenario, and overall goal."
    )
    objectives: List[str] = Field(
        default_factory=list,
        description="Concrete outcomes to achieve for this session."
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Rules, limits, allowed or forbidden behavior, environment bounds."
    )
    flag_or_goal_format: str = Field(
        description="Expected format of the final answer or artifact (e.g. flag string, password style, token)."
    )
    challenge_summary: str = Field(
        description="Short factual recap of key details from the briefing for later session steps."
    )

    def to_injectable_context(self) -> str:
        """Format as markdown for injection into combined session context."""
        lines = [
            "## High-level context",
            self.high_level_context,
            "",
            "## Objectives",
            *("- " + o for o in self.objectives),
            "",
            "## Constraints",
            *("- " + c for c in self.constraints),
            "",
            "## Flag / goal format",
            self.flag_or_goal_format,
            "",
            "## Challenge summary",
            self.challenge_summary,
        ]
        return "\n".join(lines)
