"""Test reward fix for oops/sorry commands"""

import os
import sys

# Add parent directory to path to import mcts modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.base_env import DEFAULT_REWARD_CONFIG, TheoremEnv
from mcts.isabelle_interface import IsabelleSuccessResult


class TestRewardFix:
    """Test that oops/sorry commands get negative rewards"""

    def test_oops_sorry_negative_rewards(self):
        """Test that oops and sorry commands get negative rewards when is_done=True"""
        print("\n=== Testing Oops/Sorry Negative Rewards ===")

        # Create environment with default config
        env = TheoremEnv(
            theorem_name="test",
            theorem_content='theorem test: "True" sorry',
            reward_config=DEFAULT_REWARD_CONFIG,
        )
        # Initialize step_count for testing
        env.step_count = 0

        # Test oops command with is_done=True (simulating successful execution)
        oops_result = IsabelleSuccessResult(
            is_done=True, result="Proof undone", state_name="undone_state"
        )

        step_result = env._handle_result_success(oops_result, "test_state", "oops")
        print(f"Oops command reward: {step_result['reward']}")
        assert (
            step_result["reward"] == -0.5
        ), f"Expected -0.5, got {step_result['reward']}"

        # Test sorry command with is_done=True
        sorry_result = IsabelleSuccessResult(
            is_done=True, result="Goal admitted", state_name="admitted_state"
        )

        step_result = env._handle_result_success(sorry_result, "test_state", "sorry")
        print(f"Sorry command reward: {step_result['reward']}")
        assert (
            step_result["reward"] == -0.5
        ), f"Expected -0.5, got {step_result['reward']}"

        # Test substring matching for sorry variations
        sorry_variations = ["apply sorry", "by sorry", "using sorry", "sorry."]
        for cmd in sorry_variations:
            step_result = env._handle_result_success(sorry_result, "test_state", cmd)
            assert (
                step_result["reward"] == -0.5
            ), f"Expected -0.5 for '{cmd}', got {step_result['reward']}"

        # Test substring matching for oops variations
        oops_variations = ["apply oops", "by oops", "oops."]
        for cmd in oops_variations:
            step_result = env._handle_result_success(oops_result, "test_state", cmd)
            assert (
                step_result["reward"] == -0.5
            ), f"Expected -0.5 for '{cmd}', got {step_result['reward']}"

        # Test normal successful command (should get positive reward)
        success_result = IsabelleSuccessResult(
            is_done=True,
            result="Proof completed successfully",
            state_name="completed_state",
        )

        step_result = env._handle_result_success(success_result, "test_state", "auto")
        print(f"Auto command reward: {step_result['reward']}")
        assert (
            step_result["reward"] == 1.0
        ), f"Expected 1.0, got {step_result['reward']}"

        # Test that legitimate commands don't get flagged as trivial
        legitimate_commands = ["apply auto", "simp", "induct n", "blast", "force"]
        for cmd in legitimate_commands:
            step_result = env._handle_result_success(success_result, "test_state", cmd)
            assert (
                step_result["reward"] == 1.0
            ), f"Expected 1.0 for '{cmd}', got {step_result['reward']}"

        # Test intermediate step (should get neutral reward)
        intermediate_result = IsabelleSuccessResult(
            is_done=False, result="Intermediate step", state_name="intermediate_state"
        )

        step_result = env._handle_result_success(
            intermediate_result, "test_state", "simp"
        )
        print(f"Simp command reward: {step_result['reward']}")
        assert (
            step_result["reward"] == 0.0
        ), f"Expected 0.0, got {step_result['reward']}"

        print("✅ All reward tests passed!")

    def test_case_insensitive_commands(self):
        """Test that command matching is case-insensitive"""
        print("\n=== Testing Case-Insensitive Command Matching ===")

        env = TheoremEnv(
            theorem_name="test",
            theorem_content='theorem test: "True" sorry',
            reward_config=DEFAULT_REWARD_CONFIG,
        )

        # Test with different cases
        test_cases = ["OOPS", "Oops", "oOpS", "SORRY", "Sorry", "SoRrY"]

        for command in test_cases:
            result = IsabelleSuccessResult(
                is_done=True,
                result=f"Command {command} executed",
                state_name=f"{command.lower()}_state",
            )

            step_result = env._handle_result_success(result, "test_state", command)
            print(f"{command} command reward: {step_result['reward']}")
            assert (
                step_result["reward"] == -0.5
            ), f"Expected -0.5 for {command}, got {step_result['reward']}"

        print("✅ Case-insensitive matching works!")


if __name__ == "__main__":
    test = TestRewardFix()
    test.test_oops_sorry_negative_rewards()
    test.test_case_insensitive_commands()
    print("\n🎉 All reward fix tests passed!")
