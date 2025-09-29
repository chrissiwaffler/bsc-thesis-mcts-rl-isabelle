from typing import Any, TypedDict

from mcts.isabelle_interface import (
    IsabelleErrorResult,
    IsabelleInterface,
    IsabelleResult,
    IsabelleSuccessResult,
)


class RewardConfig(TypedDict):
    success_reward: float
    error_penalty: float
    step_reward: float
    timeout_reward: float


DEFAULT_REWARD_CONFIG: RewardConfig = {
    "success_reward": 1.0,
    "error_penalty": -0.1,
    "step_reward": 0.0,
    "timeout_reward": -0.05,
}


class StepResult(TypedDict):
    reward: float
    result: IsabelleResult


class TheoremEnv:
    """multi-turn theorem proving environment for skyrl with mcts state routing support

    gym style wrapper around the isabelle interface.
    manages episode lifecycle and reward computation.

    - episode management: reset/step cycle
    - reward computation
    - state tracking
    - context management for the isabelle session
    """

    def __init__(
        self,
        theorem_name: str,
        theorem_content: str,
        max_steps: int | None = None,
        reward_config: RewardConfig | None = None,
        use_sledgehammer: bool = True,
    ) -> None:
        """initialize a new environment
        - theorem_name: str
        - theorem_content: str
        - max_steps: number
        """
        super().__init__()

        # set default max_steps if not provided
        if not max_steps:
            max_steps = 200
        self.max_steps = max_steps

        # set default reward config if not provided
        if not reward_config:
            reward_config = DEFAULT_REWARD_CONFIG
        self.reward_config = reward_config

        self.theorem_name = theorem_name
        self.theorem_content = theorem_content
        self.use_sledgehammer = use_sledgehammer

        self.theorem_statement = self._extract_theorem_statement(self.theorem_content)
        print(f"initialize new theorem for {self.theorem_statement}")

        # environment state, will be fully initialized in init()
        self.isabelle: IsabelleInterface | None = None
        self.current_state = "init"
        self.setup_count = 0
        self.is_initialized = False

    def _extract_theorem_statement(self, theorem_content: str) -> str:
        """extract clean theorem statement from content"""
        # simple extraction: find theorem line
        lines = theorem_content.split("\n")
        for line in lines:
            if "theorem" in line.lower() and ":" in line:
                # extract theorem name and statement
                if "theorem" in line and '"' in line:
                    # extract quoted part
                    start = line.find('"')
                    end = line.rfind('"')
                    if start != -1 and end != -1 and end > start:
                        return line[start + 1 : end]
                return line.strip()
        # fallback
        return "theorem to prove"

    def init(self) -> None:
        """initialize new theorem proving episode"""
        # reset state
        self.step_count = 0
        self.current_state = "init"

        # cleanup previous session
        if self.isabelle:
            self.isabelle.cleanup()

        self.isabelle = IsabelleInterface()

        # start proof
        result = self.isabelle.start_proof(
            theorem_name=self.theorem_name,
            theorem_content=self.theorem_content,
        )

        if isinstance(result, IsabelleErrorResult):
            print("error in initializing theorem")
            raise RuntimeError(
                f"failed to initialize theorem: {self.theorem_name} with content: {self.theorem_content}, error from isabelleinterface: {result.error}"
            )

        self.current_state = result.state_name
        self.is_initialized = True

    def step(self, action: str) -> StepResult:
        """execute isabelle command with state routing support"""
        if not self.isabelle:
            return self._error_response("environment not initialized")

        self.step_count += 1

        # parse state routing: "state_id:command" or just "command"
        state_id, command = self._parse_action(action)

        # validate state exists
        if not self._is_valid_state(state_id):
            return self._error_response(f"invalid state: {state_id}")

        # execute command on specified state
        result = self.isabelle.next_step(state_id, command)

        # update current state if using sequential mode
        if state_id == self.current_state or state_id == "current":
            self.current_state = result.state_name

        # handle different result types
        if isinstance(result, IsabelleErrorResult):
            return self._handle_result_error(result)
        else:
            return self._handle_result_success(result, command)

    def _handle_result_error(
        self,
        result: IsabelleErrorResult,
    ) -> StepResult:
        """handle isabelle command application errors"""
        return {
            "reward": -0.1,
            "result": result,
        }

    def _handle_result_success(
        self, result: IsabelleSuccessResult, command: str
    ) -> StepResult:
        """handle successful isabelle execution"""

        # check for trivial commands that don't represent meaningful proof completion
        command_stripped = command.strip().lower()
        is_trivial_command = any(
            trivial in command_stripped for trivial in ["oops", "sorry"]
        )

        if result.is_done:
            if is_trivial_command:
                # trivial commands like "oops" or "sorry" are terminal but don't complete the proof
                # give negative reward to discourage these
                return {"reward": -0.5, "result": result}
            else:
                # proof completed with meaningful command
                return {
                    "reward": self.reward_config["success_reward"],
                    "result": result,
                }
        elif self.step_count >= self.max_steps:
            # step limit reached
            return {
                # small timeout penalty
                "reward": self.reward_config["timeout_reward"],
                "result": result,
            }
        else:
            # valid intermediate step
            return {
                # neutral; let value head handle progress estimation
                "reward": self.reward_config["step_reward"],
                "result": result,
            }

    def close(self):
        """cleanup environment resources"""
        if self.isabelle:
            self.isabelle.cleanup()
            self.isabelle = None
        self.is_initialized = False

    def _error_response(self, message: str) -> StepResult:
        """generate error response"""
        return {
            "reward": self.reward_config["error_penalty"],
            "result": IsabelleErrorResult(
                state_name="error",
                error=message,
                is_done=True,
            ),
        }

    def _parse_action(self, action: str) -> tuple[str, str]:
        """parse action format: 'state_id#command' or 'command'"""
        if "#" in action:
            parts = action.split("#", 1)
            if len(parts) == 2:
                state_id, command = parts
                # resolve "current" alias
                if state_id.strip() == "current":
                    state_id = self.current_state
                return state_id.strip(), command.strip()

        # default: use current state
        return self.current_state, action.strip()

    def _is_valid_state(self, state_id: str) -> bool:
        """check if state exists in isabelle session"""
        if not self.isabelle or not self.isabelle._session:
            return False
        # type ignore to handle dynamic attribute access
        return state_id in self.isabelle.active_states or state_id == "init"  # type: ignore[attr-defined]

    def _count_subgoals(self, proof_state: str) -> int:
        """heuristic to count remaining subgoals"""
        # simple heuristic: count numbered goals
        lines = proof_state.split("\n")
        goal_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:5]:
                goal_count += 1
        return goal_count

    def get_state_stats(self) -> dict[str, Any]:
        """get environment statistics for debugging"""
        if not self.isabelle:
            return {"error": "not_initialized"}

        return {
            "active_states": len(self.isabelle.active_states),  # type: ignore[attr-defined]
            "current_state": self.current_state,
            "step_count": self.step_count,
            "is_initialized": self.is_initialized,
            "worker_info": self.isabelle.get_worker_info(),
        }

    async def start_sledgehammer_async(self, state_id: str) -> None:
        """start sledgehammer asynchronously for the given state"""
        if not self.isabelle or not self.isabelle._sledgehammer_manager:
            return

        try:
            await self.isabelle._sledgehammer_manager.start_sledgehammer_async(state_id)
        except Exception as e:
            print(f"failed to start sledgehammer for state {state_id}: {e}")

    def check_sledgehammer_result(self, state_id: str) -> list[str] | None:
        """check if sledgehammer has results for the given state"""
        if not self.isabelle or not self.isabelle._sledgehammer_manager:
            return None

        try:
            return self.isabelle._sledgehammer_manager.check_sledgehammer_result(state_id)
        except Exception as e:
            print(f"failed to check sledgehammer result for state {state_id}: {e}")
            return None
