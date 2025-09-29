"""Test MCTS algorithm with mock inference"""

import os
import sys

# Add parent directory to path to import mcts_accelerate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.inference_mcts import MCTSInference
from mcts.search import MCTSConfig, MCTSSearch

from .mock_env import MockTheoremEnv
from .mock_inference import MockInferenceManager


class TestMCTSMock:
    """Test MCTS algorithm with mock components"""

    def test_basic_mcts_functionality(self):
        """Test basic MCTS search functionality"""
        print("\n=== Testing Basic MCTS Functionality ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(n_simulations=10)  # Small number for testing

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        print("Starting MCTS search...")
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Verify search completed
        print(f"MCTS search completed. Root node: {root_node}")
        print(f"Root visit count: {root_node.visit_count if root_node else 'None'}")
        print(
            f"Root children count: {len(root_node.children) if root_node else 'None'}"
        )

        assert root_node is not None
        assert root_node.visit_count > 0
        assert len(root_node.children) > 0

        print(f"✅ MCTS search completed with {root_node.visit_count} visits")
        print(f"✅ Root node has {len(root_node.children)} children")

    def test_mcts_with_successful_path(self):
        """Test MCTS can find a successful proof path"""
        print("\n=== Testing MCTS with Successful Path ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(n_simulations=20)  # More simulations for better search

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Check if we found a successful path
        assert root_node is not None
        assert root_node.visit_count > 0

        # Look for successful paths in the tree
        successful_nodes = []
        self._collect_successful_nodes(root_node, successful_nodes)

        print(f"✅ Found {len(successful_nodes)} successful nodes")

        # Should find at least one successful path
        assert len(successful_nodes) > 0

    def test_mcts_value_estimates(self):
        """Test MCTS value estimation and backpropagation"""
        print("\n=== Testing MCTS Value Estimates ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(n_simulations=15)

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Verify value estimates are reasonable
        assert root_node is not None
        assert root_node.visit_count > 0

        # Root value should be between -1 and 1
        assert -1.0 <= root_node.average_value <= 1.0

        print(f"✅ Root node value: {root_node.average_value}")
        print(f"✅ Root node visit count: {root_node.visit_count}")

    def test_mcts_expansion(self):
        """Test MCTS node expansion"""
        print("\n=== Testing MCTS Expansion ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(n_simulations=5, max_actions_per_expand=3)

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Verify expansion happened
        assert root_node is not None
        assert len(root_node.children) > 0

        # Check that children have proper structure
        for child in root_node.children.values():
            assert child.parent == root_node
            assert child.action is not None
            assert child.state_id is not None

        print(f"✅ Root expanded to {len(root_node.children)} children")

    def _collect_successful_nodes(self, node, successful_nodes):
        """Recursively collect nodes that lead to successful proofs"""
        if (
            node.is_terminal and node.average_value > 0.5
        ):  # Consider high-value terminal nodes as successful
            successful_nodes.append(node)

        for child in node.children.values():
            self._collect_successful_nodes(child, successful_nodes)
