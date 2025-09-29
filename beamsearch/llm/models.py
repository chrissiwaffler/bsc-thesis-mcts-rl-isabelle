from pydantic import BaseModel, Field


class ThoughtStep(BaseModel):
    """a single step in the reasoning process"""

    question: str = Field(..., description="What am I trying to figure out?")
    answer: str = Field(..., description="My reasoning about this quesion")
    next_question: str | None = Field(None, description="Follow-up question if needed")


class ActionObservation(BaseModel):
    """a single action-observation pair in ReAct style"""

    thought: str = Field(..., description="What I'm thinking")
    action: str = Field(..., description="What action to take")
    observation: str | None = Field(None, description="What I observed from the action")


class NodeReasoning(BaseModel):
    """reasoning state for a proof node"""

    react_history: list[ActionObservation] = Field(
        default_factory=list, description="ReAct history"
    )
    final_decision: str = Field(default="", description="Final decision for this node")
