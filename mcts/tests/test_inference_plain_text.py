"""Test the updated MCTS inference system with XML prompts"""

import os
import sys
from unittest.mock import Mock

import pytest

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_loader import ModelInferenceManager
from inference_mcts import MCTSInference


class TestMCTSInferenceXML:
    """Test MCTS inference with XML prompts"""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock inference manager"""
        manager = Mock(spec=ModelInferenceManager)
        manager.policy_client = Mock()
        manager.value_client = Mock()
        return manager

    @pytest.fixture
    def inference(self, mock_manager):
        """Create MCTS inference instance"""
        return MCTSInference(mock_manager)

    def test_prompt_template_policy_xml(self, inference):
        """Test that policy prompt template uses XML format"""
        theorem = 'theorem test: "A = A"'
        proof_history = []
        current_state = "A = A"
        available_lemmas = None

        prompt = inference._prompt_template_policy(
            theorem_statement=theorem,
            proof_mode="prove",
            proof_history=proof_history,
            current_state=current_state,
            available_lemmas=available_lemmas,
        )

        # Check that it uses XML format
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "<command>" in prompt
        assert "THEOREM:" in prompt
        assert "CURRENT STATE:" in prompt
        # Should not contain old plain text format
        assert "Thinking:" not in prompt
        assert "Command:" not in prompt

    def test_prompt_template_value_xml(self, inference):
        """Test that value prompt template uses XML format"""
        theorem = 'theorem test: "A = A"'
        command = "refl"
        execution_result = "No subgoals!"
        proof_history = []

        prompt = inference._prompt_template_value(
            theorem_statement=theorem,
            command=command,
            execution_result=execution_result,
            proof_history=proof_history,
            proof_mode="prove",
        )

        # Check that it uses XML format
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "<score>" in prompt
        assert "THEOREM:" in prompt
        assert "EXECUTED COMMAND:" in prompt
        assert "RESULTING SUBGOALS:" in prompt
        # Should not contain old plain text format
        assert "Thinking:" not in prompt
        assert "Score:" not in prompt

    def test_generate_commands_with_mock_client(self, inference, mock_manager):
        """Test generate_commands with mocked inference client"""
        # Mock the client response
        mock_manager.generate_commands.return_value = [
            "<response><think>Use reflexivity.</think>\n<command>refl</command></response>",
            "<response><think>Try assumption.</think>\n<command>assumption</command></response>",
        ]

        theorem = 'theorem test: "A = A"'
        current_state = "A = A"

        commands = inference.generate_commands(
            theorem_statement=theorem,
            proof_mode="prove",
            proof_history=[],
            current_state=current_state,
            available_lemmas=None,
            n=2,
        )

        assert commands is not None
        assert len(commands) == 2
        assert commands[0]["command"] == "refl"
        assert commands[1]["command"] == "assumption"
        # Check that thinking is included
        assert "thinking" in commands[0]
        assert "thinking" in commands[1]

    def test_estimate_value_with_mock_client(self, inference, mock_manager):
        """Test estimate_value with mocked inference client"""
        # Mock the client response
        mock_manager.estimate_value.return_value = (
            "<response><think>Command succeeded.</think>\n<score>0.8</score></response>"
        )

        theorem = 'theorem test: "A = A"'
        command = "refl"
        execution_result = "No subgoals!"

        value_result = inference.estimate_value(
            theorem_statement=theorem,
            command=command,
            execution_result=execution_result,
            proof_history=[],
            proof_mode="prove",
        )

        assert value_result is not None
        assert value_result["value"] == 0.8

    def test_generate_commands_empty_response(self, inference, mock_manager):
        """Test generate_commands with empty response"""
        # Mock empty response
        mock_manager.generate_commands.return_value = []

        commands = inference.generate_commands(
            theorem_statement='theorem test: "A = A"',
            proof_mode="prove",
            proof_history=[],
            current_state="A = A",
            available_lemmas=None,
            n=1,
        )

        assert commands is None

    def test_estimate_value_empty_response(self, inference, mock_manager):
        """Test estimate_value with empty response"""
        # Mock empty response
        mock_manager.estimate_value.return_value = ""

        value_result = inference.estimate_value(
            theorem_statement='theorem test: "A = A"',
            command="refl",
            execution_result="No subgoals!",
            proof_history=[],
            proof_mode="prove",
        )

        # Empty response should return None
        assert value_result is None

    def test_generate_commands_xml_parsing(self, inference, mock_manager):
        """Test generate_commands with XML response"""
        # Mock XML response
        mock_manager.generate_commands.return_value = [
            "<response><think>Use reflexivity</think>\n<command>refl</command></response>"
        ]

        commands = inference.generate_commands(
            theorem_statement='theorem test: "A = A"',
            proof_mode="prove",
            proof_history=[],
            current_state="A = A",
            available_lemmas=None,
            n=1,
        )

        assert commands is not None
        assert len(commands) == 1
        assert commands[0]["command"] == "refl"
        assert commands[0]["thinking"] == "Use reflexivity"

    def test_estimate_value_xml_parsing(self, inference, mock_manager):
        """Test estimate_value with XML response"""
        # Mock XML response
        mock_manager.estimate_value.return_value = (
            "<response><think>Good command</think>\n<score>0.9</score></response>"
        )

        value_result = inference.estimate_value(
            theorem_statement='theorem test: "A = A"',
            command="refl",
            execution_result="No subgoals!",
            proof_history=[],
            proof_mode="prove",
        )

        assert value_result is not None
        assert value_result["value"] == 0.9

    def test_generate_commands_malformed_response(self, inference, mock_manager):
        """Test generate_commands with malformed response"""
        # Mock malformed response
        mock_manager.generate_commands.return_value = [
            "Just some text without proper format"
        ]

        commands = inference.generate_commands(
            theorem_statement='theorem test: "A = A"',
            proof_mode="prove",
            proof_history=[],
            current_state="A = A",
            available_lemmas=None,
            n=1,
        )

        # Should return empty list when parsing fails
        assert commands == []

    def test_estimate_value_malformed_response(self, inference, mock_manager):
        """Test estimate_value with malformed response"""
        # Mock malformed response
        mock_manager.estimate_value.return_value = (
            "Just some text without proper format"
        )

        value_result = inference.estimate_value(
            theorem_statement='theorem test: "A = A"',
            command="refl",
            execution_result="No subgoals!",
            proof_history=[],
            proof_mode="prove",
        )

        # Should return neutral value when parsing fails
        assert value_result is not None
        assert value_result["value"] == 0.0

    def test_estimate_value_clipping(self, inference, mock_manager):
        """Test that values are clipped to [-1, 1] range"""
        # Test high value
        mock_manager.estimate_value.return_value = (
            "<response><think>Very good</think>\n<score>2.5</score></response>"
        )
        value_result = inference.estimate_value(
            theorem_statement='theorem test: "A = A"',
            command="refl",
            execution_result="No subgoals!",
            proof_history=[],
            proof_mode="prove",
        )
        assert value_result["value"] == 1.0

        # Test low value
        mock_manager.estimate_value.return_value = (
            "<response><think>Very bad</think>\n<score>-2.5</score></response>"
        )
        value_result = inference.estimate_value(
            theorem_statement='theorem test: "A = A"',
            command="refl",
            execution_result="No subgoals!",
            proof_history=[],
            proof_mode="prove",
        )
        assert value_result["value"] == -1.0
