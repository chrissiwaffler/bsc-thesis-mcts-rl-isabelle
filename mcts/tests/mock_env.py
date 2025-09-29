"""Mock theorem environment for testing MCTS without Isabelle"""

from typing import Any

from pydantic import BaseModel


# Mock classes to avoid circular imports
class IsabelleSuccessResult(BaseModel):
    """Successful Isabelle command execution result"""

    success: bool = True
    is_done: bool
    result: str
    state_name: str


class IsabelleErrorResult(BaseModel):
    """Failed Isabelle command execution result"""

    success: bool = False
    is_done: bool
    state_name: str
    error: str


IsabelleResult = IsabelleSuccessResult | IsabelleErrorResult


class MockTheoremEnv:
    """Mock environment that simulates Isabelle theorem proving"""

    def __init__(
        self, theorem_name: str = "", theorem_content: str = "", max_steps: int = 200
    ):
        """Initialize mock environment"""
        self.theorem_name = theorem_name
        self.theorem_content = theorem_content
        self.theorem_statement = theorem_content  # Add this for MCTS compatibility
        self.max_steps = max_steps
        self.current_state = "initial_state"
        self.step_count = 0
        self.is_initialized = False

        # Track all states and their content for MCTS
        self.all_states = {}

        # Mock state transitions
        self.state_transitions = {
            "initial_state": {
                "apply (induction n)": "induction_state",
                "simp": "simp_state",
                "auto": "auto_state",
                "rule conjI": "conj_state",
                "assumption": "assumption_state",
            },
            "induction_state": {"simp": "base_case_state", "auto": "proof_complete"},
            "simp_state": {"auto": "proof_complete"},
            "auto_state": {
                "auto": "proof_complete"
            },  # auto should lead to proof_complete
            "conj_state": {"auto": "proof_complete"},
            "assumption_state": {"simp": "proof_complete"},
            "base_case_state": {"auto": "proof_complete"},
        }

        # Mock rewards for state transitions
        self.rewards = {
            "initial_state": 0.0,
            "induction_state": 0.3,
            "simp_state": 0.2,
            "auto_state": 0.4,
            "conj_state": 0.1,
            "assumption_state": 0.0,
            "base_case_state": 0.6,
            "proof_complete": 1.0,
        }

    def init(self) -> None:
        """Initialize environment"""
        print(f"[mock-env] initialized theorem: {self.theorem_name}")
        self.current_state = "initial_state"
        self.step_count = 0
        self.is_initialized = True

    def reset(self, theorem_name: str, theorem_content: str) -> None:
        """Reset environment with new theorem"""
        print(f"[mock-env] reset theorem: {theorem_name}")
        self.theorem_name = theorem_name
        self.theorem_content = theorem_content
        self.theorem_statement = theorem_content  # Add this for MCTS compatibility
        self.current_state = "initial_state"
        self.step_count = 0
        self.is_initialized = True

        # Initialize all states tracking
        self.all_states = {
            "initial_state": "Initial proof state: A = A",
            "induction_state": "Induction state: n case",
            "simp_state": "Simplified state: A = A",
            "auto_state": "Auto-solved state: A = A",
            "conj_state": "Conjunction state: A ∧ B",
            "assumption_state": "Assumption state: A = A",
            "base_case_state": "Base case state: n = 0",
            "proof_complete": "Proof complete: QED",
            "error_state": "Error state: invalid command",
        }

    def step(self, action: str) -> dict[str, Any]:
        """Execute a command and return StepResult dictionary"""
        self.step_count += 1
        print(f"[mock-env] step {self.step_count}: action={action}")

        # Parse action format: "state_id#command" or just "command"
        if "#" in action:
            parts = action.split("#", 1)
            if len(parts) == 2:
                state_id, command = parts
                state_id = state_id.strip()
                command = command.strip()
            else:
                state_id = self.current_state
                command = action.strip()
        else:
            state_id = self.current_state
            command = action.strip()

        # Get next state
        current_transitions = self.state_transitions.get(state_id, {})
        next_state = current_transitions.get(command, "error_state")

        # Calculate reward
        reward = self.rewards.get(next_state, -0.1)

        # Check if proof is complete
        done = (next_state == "proof_complete") or (self.step_count >= self.max_steps)

        # Create mock result
        if next_state == "error_state":
            result = IsabelleErrorResult(
                state_name=next_state,
                error=f"Invalid command '{command}' for state '{state_id}'",
                is_done=True,
            )
        else:
            result = IsabelleSuccessResult(
                state_name=next_state,
                is_done=done,
                result=f"Mock proof state for {next_state}",
            )

        print(
            f"[mock-env] {state_id} --{command}--> {next_state} (reward: {reward}, done: {done})"
        )

        # Update current state for MCTS compatibility
        self.current_state = next_state

        return {"reward": reward, "result": result}

    def get_current_state(self) -> str:
        """Get current proof state"""
        return self.current_state

    def is_proof_complete(self) -> bool:
        """Check if proof is complete"""
        return self.current_state == "proof_complete"

    def get_available_commands(self) -> list[str]:
        """Get available commands for current state (mock implementation)"""
        return list(self.state_transitions.get(self.current_state, {}).keys())

    def cleanup(self):
        """Clean up resources (no-op for mock)"""
        print("[mock-env] cleanup called")
        self.is_initialized = False

    def close(self):
        """Close environment"""
        self.cleanup()

    def get_state_stats(self) -> dict[str, Any]:
        """Get environment statistics for debugging"""
        return {
            "current_state": self.current_state,
            "step_count": self.step_count,
            "is_initialized": self.is_initialized,
            "theorem_name": self.theorem_name,
            "max_steps": self.max_steps,
        }

    def get_theorem_statement(self) -> str:
        """Get theorem statement for MCTS"""
        return self.theorem_content or "A = A"
