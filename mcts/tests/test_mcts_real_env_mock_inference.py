"""Test MCTS with real environment and mocked inference manager"""

import os
import sys

import pytest

# Add parent directory to path to import mcts_accelerate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.base_env import TheoremEnv
from mcts.inference_mcts import MCTSInference
from mcts.search import MCTSConfig, MCTSNode, MCTSSearch


class MockInferenceManager:
    """Mock inference manager that provides realistic fake LLM responses"""

    def __init__(self):
        self.call_count = 0
        self.command_responses = []
        self.value_responses = []

    def generate_commands(self, prompt, stop=None, n=1, **kwargs):
        """Generate mock commands that simulate realistic theorem proving tactics"""
        self.call_count += 1

        # Extract current state from prompt to provide context-aware responses
        current_state = self._extract_state_from_prompt(prompt)

        # Define realistic commands based on proof state
        if "initial" in current_state.lower() or "prove" in prompt:
            commands = ["apply (induction n)", "simp", "auto", "rule conjI"]
        elif "induction" in current_state.lower():
            commands = ["simp", "auto", "case (Suc n)"]
        elif "simp" in current_state.lower():
            commands = ["auto", "simp add: assms", "assumption"]
        else:
            commands = ["auto", "simp", "assumption", "rule refl"]

        # Limit to requested number of commands
        selected_commands = commands[: min(n, len(commands))]

        # Wrap commands in XML format
        xml_responses = []
        for cmd in selected_commands:
            xml_response = f"<response><think>Mock thinking for {cmd}</think>\n<command>{cmd}</command></response>"
            xml_responses.append(xml_response)

        # Return list of XML strings as expected by MCTSInference
        print(f"[mock-inference] Generated commands: {xml_responses}")
        return xml_responses

    def estimate_value(self, prompt, stop=None, **kwargs):
        """Generate mock value estimates that simulate realistic LLM scoring"""
        self.call_count += 1

        # Extract command and result from prompt to provide context-aware values
        command = self._extract_command_from_prompt(prompt)
        result = self._extract_result_from_prompt(prompt)

        # Simulate realistic value estimation based on command and result
        if "auto" in command and "complete" in result.lower():
            value = 0.9  # High value for auto that completes proof
        elif "induction" in command:
            value = 0.7  # Good value for induction in appropriate contexts
        elif "simp" in command:
            value = 0.5  # Moderate value for simplification
        elif "error" in result.lower():
            value = -0.3  # Negative value for errors
        else:
            value = 0.2  # Default moderate value

        # Add some noise to make it more realistic
        import random

        value += random.uniform(-0.1, 0.1)
        value = max(-1.0, min(1.0, value))  # Clamp to [-1, 1]

        # Return XML formatted string as expected by MCTSInference
        xml_response = f"<response><think>Mock thinking for value estimation</think>\n<score>{value:.3f}</score></response>"
        print(f"[mock-inference] Estimated value: {value:.3f} for command: {command}")
        return xml_response

    def _extract_state_from_prompt(self, prompt):
        """Extract current state information from prompt"""
        lines = prompt.split("\n")
        for line in lines:
            if "CURRENT STATE:" in line:
                return line.replace("CURRENT STATE:", "").strip()
        return "unknown"

    def _extract_command_from_prompt(self, prompt):
        """Extract command information from prompt"""
        lines = prompt.split("\n")
        for line in lines:
            if "EXECUTED COMMAND:" in line:
                return line.replace("EXECUTED COMMAND:", "").strip()
        return "unknown"

    def _extract_result_from_prompt(self, prompt):
        """Extract result information from prompt"""
        lines = prompt.split("\n")
        for line in lines:
            if "RESULTING SUBGOALS:" in line:
                return line.replace("RESULTING SUBGOALS:", "").strip()
        return "unknown"


class TestMCTSRealEnvMockInference:
    """Test MCTS with real environment and mocked inference"""

    @pytest.fixture
    def mock_inference_manager(self):
        """Fixture providing mock inference manager"""
        return MockInferenceManager()

    @pytest.fixture
    def simple_theorem_env(self):
        """Fixture providing real theorem environment with simple theorem"""
        # Simple theorem that should be provable with basic tactics
        theorem_content = """
theorem test_simple:
  "A = A"
proof -
  assume "A = A"
  show "A = A" by assumption
qed
"""
        env = TheoremEnv(
            theorem_name="test_simple", theorem_content=theorem_content, max_steps=50
        )
        return env

    @pytest.fixture
    def induction_theorem_env(self):
        """Fixture providing real theorem environment with induction theorem"""
        # Theorem that requires induction
        theorem_content = """
theorem test_induction:
  fixes n :: nat
  shows "n + 0 = n"
proof (induction n)
  case 0
    then show "0 + 0 = 0" by simp
next
  case (Suc n)
    then show "Suc n + 0 = Suc n" by simp
qed
"""
        env = TheoremEnv(
            theorem_name="test_induction",
            theorem_content=theorem_content,
            max_steps=100,
        )
        return env

    def test_mcts_with_real_env_simple_theorem(
        self, mock_inference_manager, simple_theorem_env
    ):
        """Test MCTS with real environment and simple theorem"""
        print("\n=== Testing MCTS with Real Environment (Simple Theorem) ===")

        # Configure MCTS for quick testing
        config = MCTSConfig(
            n_simulations=20,  # Small number for quick test
            max_actions_per_expand=3,
            c_puct=1.0,
            max_depth=10,
        )

        # Create MCTS inference with mock manager
        mcts_inference = MCTSInference(model_manager=mock_inference_manager)  # type: ignore

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mcts_inference, config=config)

        # Initialize environment
        simple_theorem_env.init()

        # Run MCTS search
        try:
            root_node = mcts.search(
                env=simple_theorem_env,
                initial_state_id=simple_theorem_env.current_state,
            )

            # Verify search completed
            assert root_node is not None
            assert root_node.visit_count > 0
            assert root_node.visit_count <= config.n_simulations

            print(f"✅ MCTS search completed with {root_node.visit_count} visits")
            print(f"✅ Root node has {len(root_node.children)} children")
            print(f"✅ Root average value: {root_node.average_value:.3f}")

            # Verify tree structure
            if len(root_node.children) > 0:
                child_visits = sum(
                    child.visit_count for child in root_node.children.values()
                )
                # Child visits should be at least root visits - 1 (accounting for root visit)
                assert child_visits >= root_node.visit_count - 1
                print(
                    f"✅ Child visits: {child_visits}, Root visits: {root_node.visit_count}"
                )

        except Exception as e:
            pytest.fail(f"MCTS search failed with error: {e}")
        finally:
            # Clean up
            simple_theorem_env.close()

    def test_mcts_with_real_env_induction_theorem(
        self, mock_inference_manager, induction_theorem_env
    ):
        """Test MCTS with real environment and induction theorem"""
        print("\n=== Testing MCTS with Real Environment (Induction Theorem) ===")

        # Configure MCTS for induction theorem (more complex)
        config = MCTSConfig(
            n_simulations=30,  # More simulations for complex theorem
            max_actions_per_expand=4,
            c_puct=1.5,  # Higher exploration for complex problems
            max_depth=15,
        )

        # Create MCTS inference with mock manager
        mcts_inference = MCTSInference(model_manager=mock_inference_manager)  # type: ignore

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mcts_inference, config=config)

        # Initialize environment
        induction_theorem_env.init()

        # Run MCTS search
        try:
            root_node = mcts.search(
                env=induction_theorem_env,
                initial_state_id=induction_theorem_env.current_state,
            )

            # Verify search completed
            assert root_node is not None
            assert root_node.visit_count > 0
            assert root_node.visit_count <= config.n_simulations

            print(f"✅ MCTS search completed with {root_node.visit_count} visits")
            print(f"✅ Root node has {len(root_node.children)} children")
            print(f"✅ Root average value: {root_node.average_value:.3f}")

            # For induction theorem, we expect more exploration
            assert len(root_node.children) >= 1  # Should have at least one child

        except Exception as e:
            pytest.fail(f"MCTS search failed with error: {e}")
        finally:
            # Clean up
            induction_theorem_env.close()

    def test_mcts_node_expansion_with_real_env(
        self, mock_inference_manager, simple_theorem_env
    ):
        """Test MCTS node expansion with real environment"""
        print("\n=== Testing MCTS Node Expansion with Real Environment ===")

        config = MCTSConfig(n_simulations=10, max_actions_per_expand=2, c_puct=1.0)

        mcts_inference = MCTSInference(model_manager=mock_inference_manager)  # type: ignore
        _ = MCTSSearch(mcts_inference=mcts_inference, config=config)

        # Initialize environment
        simple_theorem_env.init()

        try:
            # Create root node manually
            root_node = MCTSNode(
                state_id=simple_theorem_env.current_state,
                proof_mode="prove",
                available_lemmas=[],
            )

            # Test node expansion using the expand method
            children = root_node.expand(mcts_inference, simple_theorem_env, config)

            # Verify expansion worked
            assert len(children) > 0
            assert all(isinstance(child, MCTSNode) for child in children)

            print(f"✅ Node expansion created {len(children)} children")

            # Verify children have proper attributes
            for child in children:
                assert child.state_id != ""
                assert child.parent == root_node
                assert child.action is not None
                # Terminal nodes have visit_count = 1, non-terminal have 0
                if child.is_terminal:
                    assert child.visit_count == 1
                else:
                    assert child.visit_count == 0
                # value_sum can be non-zero if the environment gave immediate rewards
                assert isinstance(child.value_sum, float)

        except Exception as e:
            pytest.fail(f"Node expansion failed with error: {e}")
        finally:
            simple_theorem_env.close()

    def test_mcts_simulation_with_real_env(
        self, mock_inference_manager, simple_theorem_env
    ):
        """Test MCTS simulation (evaluation) with real environment"""
        print("\n=== Testing MCTS Simulation with Real Environment ===")

        config = MCTSConfig(n_simulations=5, max_actions_per_expand=2, c_puct=1.0)

        mcts_inference = MCTSInference(model_manager=mock_inference_manager)  # type: ignore
        mcts = MCTSSearch(mcts_inference=mcts_inference, config=config)

        # Initialize environment
        simple_theorem_env.init()

        try:
            # Create root node manually
            root_node = MCTSNode(
                state_id=simple_theorem_env.current_state,
                proof_mode="prove",
                available_lemmas=[],
            )

            # Test node evaluation (simulation)
            value = mcts._evaluate_node(root_node, simple_theorem_env)

            # Verify evaluation returned a valid value
            assert isinstance(value, float)
            assert -1.0 <= value <= 1.0

            print(f"✅ Simulation returned value: {value:.3f}")

        except Exception as e:
            pytest.fail(f"Simulation failed with error: {e}")
        finally:
            simple_theorem_env.close()

    def test_mock_inference_manager_responses(self, mock_inference_manager):
        """Test that mock inference manager provides reasonable responses"""
        print("\n=== Testing Mock Inference Manager Responses ===")

        # Test command generation
        prompt = """THEOREM:
A = A

PROOF HISTORY:

CURRENT STATE:
initial state

MODE: prove

Generate the next proof step.

<think>
</think>
<command>
"""

        commands = mock_inference_manager.generate_commands(prompt, n=3)
        assert len(commands) == 3
        assert all(isinstance(cmd, str) for cmd in commands)
        assert all(len(cmd) > 0 for cmd in commands)

        print(f"✅ Generated {len(commands)} commands")

        # Test value estimation
        value_prompt = """THEOREM:
A = A

PROOF HISTORY:

EXECUTED COMMAND:
auto

RESULTING SUBGOALS:
proof complete

MODE: prove

Evaluate this executed command for mathematical correctness and logical progression.

<think>
</think>
<score>
"""

        value_response = mock_inference_manager.estimate_value(value_prompt)
        # Mock returns XML format, parse it to extract the score
        import re

        score_match = re.search(r"<score>([-+]?\d*\.?\d+)</score>", value_response)
        assert score_match is not None, "XML response should contain a score"
        value_float = float(score_match.group(1))
        assert isinstance(value_float, float)
        assert -1.0 <= value_float <= 1.0

        print(f"✅ Generated value estimate: {value_float:.3f}")


if __name__ == "__main__":
    # Run tests manually if called directly
    test_instance = TestMCTSRealEnvMockInference()

    # Test mock inference manager
    mock_mgr = MockInferenceManager()
    test_instance.test_mock_inference_manager_responses(mock_mgr)

    # Test with simple theorem
    simple_env = test_instance.simple_theorem_env()
    test_instance.test_mcts_with_real_env_simple_theorem(mock_mgr, simple_env)

    # Test with induction theorem
    induction_env = test_instance.induction_theorem_env()
    test_instance.test_mcts_with_real_env_induction_theorem(mock_mgr, induction_env)

    # Test node expansion
    simple_env2 = test_instance.simple_theorem_env()
    test_instance.test_mcts_node_expansion_with_real_env(mock_mgr, simple_env2)

    # Test simulation
    simple_env3 = test_instance.simple_theorem_env()
    test_instance.test_mcts_simulation_with_real_env(mock_mgr, simple_env3)

    print("\n🎉 All tests passed!")
