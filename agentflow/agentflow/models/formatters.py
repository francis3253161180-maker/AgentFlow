from typing import Literal

from pydantic import BaseModel, Field

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str
    # Empty remains accepted for historical persisted action parsing.  New
    # hierarchical Planner prompts require one exact current-step gap ID and
    # Solver validates it before execution.
    target_gap: str = ""


class HierarchicalNextStep(NextStep):
    """Guided-JSON action used only for new hierarchical rollouts."""

    target_gap: str


class PlanStep(BaseModel):
    """One evidence-oriented dependency in a generic high-level plan."""

    step_id: str
    objective: str
    success_criteria: str
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    verified_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class HighLevelPlan(BaseModel):
    # This is vLLM guided JSON.  Reject an otherwise valid but unusable empty
    # plan before it can reach the execution loop.
    steps: list[PlanStep] = Field(min_length=1)


class StepVerification(BaseModel):
    """Evidence-only verifier result for the current plan step."""

    completed: bool
    missing_evidence: list[str] = Field(default_factory=list)
    verified_evidence: list[str] = Field(default_factory=list)
    contradiction: bool = False
    invalidated_step_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str
