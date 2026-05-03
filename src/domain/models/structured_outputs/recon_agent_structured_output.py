from typing import List, Optional

from pydantic import BaseModel, Field


class ReconAgentResponse(BaseModel):
    """Structured output for recon: environmental analysis and next-focus priorities for the session."""

    environmental_context: str = Field(
        description="Summary of the current environment: what is known (hosts, ports, services, versions, paths, credentials), what is unknown, and what to probe next. When execution output was provided, include concise analysis and port/service/version findings."
    )
    recon_priorities: List[str] = Field(
        default_factory=list,
        description="Prioritized list of next checks or focus areas aligned with session objectives and constraints."
    )
    recon_findings_summary: str = Field(
        default="",
        description="When enough evidence exists: short structured summary (e.g., port → service/version, key paths, OS). Empty or minimal when no output or analysis still in progress."
    )
    assumptions_and_notes: str = Field(
        default="",
        description="Assumptions about the target or notes for downstream session steps."
    )

    def to_injectable_context(self) -> str:
        """Format as a single block for downstream prompt injection."""
        lines = [
            "## Environmental context",
            self.environmental_context,
            "",
            "## Recon priorities",
            *("- " + p for p in self.recon_priorities),
        ]
        if self.recon_findings_summary:
            lines.extend(["", "## Recon findings summary", self.recon_findings_summary])
        if self.assumptions_and_notes:
            lines.extend(["", "## Assumptions and notes", self.assumptions_and_notes])
        return "\n".join(lines)
