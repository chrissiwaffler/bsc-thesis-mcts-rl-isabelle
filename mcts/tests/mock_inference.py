"""Mock inference manager for testing MCTS without external inference engines"""

import os
import sys

# Add parent directory to path to import mcts_accelerate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockInferenceManager:
    """Mock inference manager that mimics MCTSInference interface"""

    """Mock inference manager that returns predefined responses for testing"""

    def __init__(self):
        self.command_counter = 0
        self.value_counter = 0

        # Predefined commands for testing
        self.mock_commands = [
            "apply (induction n)",
            "simp",
            "auto",
            "rule conjI",
            "assumption",
        ]

        # Predefined values for testing
        self.mock_values = [0.7, 0.3, 0.9, -0.2, 0.5]

    def generate_commands(
        self,
        prompt: str,
        stop: list[str] | None = None,
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> list[str]:
        """Generate mock commands for testing"""
        print(f"[mock-inference] generate_commands called with n={n}")

        responses = []
        for _i in range(n):
            # Cycle through predefined commands
            cmd = self.mock_commands[self.command_counter % len(self.mock_commands)]

            # Return XML formatted response as string
            xml_response = f"<response><think>Mock thinking for {cmd}</think>\n<command>{cmd}</command></response>"
            responses.append(xml_response)
            self.command_counter += 1

        print(f"[mock-inference] Generated commands: {responses}")
        return responses

    def estimate_value(
        self,
        prompt: str,
        stop: list[str],
        n: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """Generate mock value estimation for testing"""
        print("[mock-inference] estimate_value called")

        # Return a mock response with XML structure
        value = self.mock_values[self.value_counter % len(self.mock_values)]
        self.value_counter += 1

        mock_response = f"""<response><think>Mock thinking for value estimation</think>
<score>{value}</score></response>"""

        print(f"[mock-inference] Estimated value: {value}")
        return mock_response

    def cleanup(self):
        """Clean up resources (no-op for mock)"""
        print("[mock] cleanup called")
