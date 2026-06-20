"""Structured output for the Reasoning agent (PenTestGPT-v2-inspired EGATS + TDA).

The Reasoning agent owns the plan. Each turn it (1) updates an Evidence-Guided
Attack Tree (EGATS) from the latest evidence, (2) assesses task difficulty for the
node it intends to pursue, (3) selects ONE node and emits ONE concrete directive
that grounds the execution (generative) agent, and (4) decides whether the run
continues or ends. The execution agent may submit a flag, but it never decides to
end the run — that decision belongs to the Reasoning agent.

The attack tree is carried as annotated natural-language text (not a recursive
Pydantic schema): this mirrors the working text "Pentesting Task Tree" of v1 and is
far more robust for LLM structured output than a deep nested tree, while still
capturing the v2 EGATS node annotations (type / status / promise / TDI).
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TaskDifficultyAssessment(BaseModel):
    """Task Difficulty Assessment (TDA) for the node the Reasoning agent will pursue.

    Four execution-measurable dimensions are combined into a single Task Difficulty
    Index (TDI). TDI drives the search mode: a confident, low-difficulty lead is
    exploited depth-first; a high-difficulty / low-evidence frontier triggers
    breadth-first reconnaissance to gather more evidence first.
    """

    horizon: float = Field(
        description=(
            "Ĥ — estimated remaining steps to the next checkpoint for this node, "
            "normalized to [0,1] (0 = essentially one step away, 1 = many uncertain "
            "steps remain)."
        )
    )
    evidence_confidence: float = Field(
        description=(
            "E ∈ [0,1] — how strongly the REAL evidence so far supports the hypothesis "
            "behind this node (0 = pure guess, 1 = directly confirmed by tool output)."
        )
    )
    context_load: float = Field(
        description=(
            "C ∈ [0,1] — how much state / context this task requires to execute "
            "correctly (0 = self-contained single command, 1 = many interdependent "
            "facts must be tracked)."
        )
    )
    success_rate: float = Field(
        description=(
            "S ∈ [0,1] — historical success of similar actions already attempted this "
            "run (1 = this class of action has been working, 0 = it keeps failing)."
        )
    )
    tdi: float = Field(
        description=(
            "Composite Task Difficulty Index ∈ [0,1], computed as "
            "0.3*horizon + 0.3*(1-evidence_confidence) + 0.2*context_load + "
            "0.2*(1-success_rate). Higher = harder / less certain."
        )
    )
    mode: str = Field(
        description=(
            "Search mode implied by the TDI: 'exploit' (depth-first, commit to the "
            "lead) when TDI <= 0.3; 'recon' (breadth-first, gather more evidence) when "
            "TDI >= 0.6; 'balanced' in between."
        )
    )


class SkillFileRequest(BaseModel):
    """A Tier-3 request to read one supporting file of a previously catalogued skill."""

    skill_name: str = Field(
        description="Exact skill name from the catalog that owns the file."
    )
    filename: str = Field(
        description=(
            "Relative path of the supporting file to read, taken verbatim from the "
            "skill's <supporting_files> list (e.g. 'sql-injection.md')."
        )
    )


class ReasoningAgentResponse(BaseModel):
    """One planning turn of the Reasoning agent."""

    reasoning: List[str] = Field(
        description=(
            "Short reasoning steps: what the latest GENERATIVE FEEDBACK shows, how it "
            "updates the attack tree, and why the selected node / directive is the most "
            "promising next move."
        )
    )
    attack_tree: str = Field(
        description=(
            "The FULL, updated Evidence-Guided Attack Tree as annotated text. Use a "
            "hierarchical outline (1, 1.1, 1.1.1 ...). Tag every node with its type "
            "[observation|hypothesis|action], a status [todo|in-progress|done|pruned|"
            "n/a], a promise score phi in [0,1], and a tdi when assessed. Only add nodes "
            "that real evidence justifies; never invent tasks for services/ports not yet "
            "discovered. Re-emit the whole tree each turn so it evolves statefully."
        )
    )
    task_difficulty: TaskDifficultyAssessment = Field(
        description="Task Difficulty Assessment for the node selected this turn."
    )
    selected_task: Optional[str] = Field(
        default=None,
        description=(
            "Id and short title of the single attack-tree node chosen to pursue this "
            "turn (e.g. '1.2.1 brute-force SSH login'). Null only when ending."
        ),
    )
    directive: Optional[str] = Field(
        default=None,
        description=(
            "ONE concrete, bounded instruction for the execution agent that realizes the "
            "selected node — precise, simple language, achievable with a single shell "
            "command or one script. The execution agent is grounded to ONLY this "
            "directive. Null when decision == 'end', when loading a skill, or when "
            "submitting (set 'submitted_goal' instead)."
        ),
    )
    submitted_goal: Optional[str] = Field(
        default=None,
        description=(
            "The recovered flag/answer to submit for validation. YOU (the reasoning "
            "agent) own submission: when the execution feedback shows the objective "
            "value, set this to that exact value — it is routed to the submit tool and "
            "the ACCEPTED/REJECTED result returns to you. Do NOT set 'directive' on the "
            "same turn, and do NOT set decision='end' claiming a flag 'is being "
            "submitted' — emit it here and wait for the ACCEPTED result, then end. Only "
            "submit a value actually recovered from real execution output, never one "
            "copied from documentation. Null when not submitting this turn."
        ),
    )
    decision: str = Field(
        description=(
            "'continue' to hand the directive to the execution agent, or 'end' to "
            "terminate the run. End when the objective is satisfied (a submission was "
            "ACCEPTED), the tree is exhausted, or no promising branch remains. The "
            "execution agent never ends the run — only this field does."
        )
    )
    end_reason: Optional[str] = Field(
        default=None,
        description="Why the run is ending (goal achieved / exhausted / blocked). Null while continuing.",
    )
    load_skill: Optional[str] = Field(
        default=None,
        description=(
            "Progressive disclosure (Tier 2): the EXACT name of a skill from the "
            "<SkillsCatalog> whose full instructions you want before deciding the next "
            "directive. When set, this planning turn loads that skill's knowledge and "
            "returns to you — do NOT also set 'directive' on the same turn. Use only "
            "when a catalogued skill matches the current task; null otherwise."
        ),
    )
    read_skill_file: Optional[SkillFileRequest] = Field(
        default=None,
        description=(
            "Progressive disclosure (Tier 3): request ONE supporting file of an "
            "already-loaded skill, by skill name + filename from its <supporting_files> "
            "list. When set, do NOT also set 'directive' on the same turn. Null unless "
            "you need a specific reference document."
        ),
    )
