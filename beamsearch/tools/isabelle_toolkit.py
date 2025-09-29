from pydantic import BaseModel, ConfigDict, Field

from ..core import QIsabelleSession


class IsabelleSuccessResult(BaseModel):
    """Successful Isabelle command execution result"""

    success: bool = True
    is_done: bool
    result: str
    state_name: str


class IsabelleErrorResult(BaseModel):
    """Failed Isabelle command execution result"""

    success: bool = False
    error: str


IsabelleResult = IsabelleSuccessResult | IsabelleErrorResult


class IsabelleExecuteCommand(BaseModel):
    """execute an isabelle command in the current proof state"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_state: str = Field(..., description="Current state name in the proof graph")
    command: str = Field(..., description="The Isabelle command to execute")
    new_state: str = Field(..., description="Name for the new state after execution")

    session: QIsabelleSession | None = Field(None, exclude=True)

    def call(self) -> IsabelleResult:
        """executes the tactic and returns result"""
        if not self.session:
            return IsabelleErrorResult(error="No Isabelle session available")

        try:
            is_done, result = self.session.execute(
                self.current_state,
                self.command,
                self.new_state,
            )
            return IsabelleSuccessResult(
                is_done=is_done,
                result=result,
                state_name=self.new_state,
            )
        except Exception as e:
            error_msg = str(e).split("Traceback:")[0].strip()
            return IsabelleErrorResult(error=error_msg)
