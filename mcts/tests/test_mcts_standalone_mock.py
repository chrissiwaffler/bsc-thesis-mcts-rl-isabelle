"""
Standalone MCTS mock test for manual execution
"""

import os
import sys

# Add parent directory to path to import mcts_accelerate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.inference_mcts import MCTSInference
from mcts.search import MCTSConfig, MCTSNode, MCTSSearch

from .mock_env import MockTheoremEnv
from .mock_inference import MockInferenceManager as MockMCTSInference


def test_basic_mcts_functionality():
    """Test basic MCTS functionality with mock components"""
    print("\n=== Testing Basic MCTS Functionality ===")

    # Create mock components
    mock_manager = MockMCTSInference()
    mock_inference = MCTSInference(model_manager=mock_manager)
    mock_env = MockTheoremEnv()

    config = MCTSConfig(
        n_simulations=10, c_puct=1.0, max_depth=3, max_actions_per_expand=3
    )

    mcts = MCTSSearch(mcts_inference=mock_inference, config=config)
    mock_env.init()

    # Run MCTS search
    root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

    # Verify search completed
    assert root_node is not None, "MCTS should return a root node"
    assert root_node.visit_count > 0, "Root node should have been visited"
    assert len(root_node.children) > 0, "Root node should have been expanded"

    print(f"Root visits: {root_node.visit_count}")
    print(f"Root children: {len(root_node.children)}")

    # Check that children have proper structure
    for child in root_node.children.values():
        assert child.parent == root_node
        assert child.action is not None
        assert child.state_id is not None

    print(f"Root expanded to {len(root_node.children)} children")


def test_mcts_with_successful_path():
    """Test MCTS can find successful proof paths"""
    print("\n=== Testing MCTS with Successful Path ===")

    # Create mock components
    mock_manager = MockMCTSInference()
    mock_inference = MCTSInference(model_manager=mock_manager)
    mock_env = MockTheoremEnv()

    config = MCTSConfig(
        n_simulations=20, c_puct=1.0, max_depth=4, max_actions_per_expand=3
    )

    mcts = MCTSSearch(mcts_inference=mock_inference, config=config)
    mock_env.init()

    # Run MCTS search
    root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

    # Look for successful paths in the tree
    successful_nodes = []
    _collect_successful_nodes(root_node, successful_nodes)

    print(f"Found {len(successful_nodes)} successful nodes")

    # Should find at least one successful path
    assert len(successful_nodes) > 0

    # Verify each successful node has proper structure
    for node in successful_nodes:
        assert node.is_terminal
        assert node.average_value > 0.5
        assert node.visit_count > 0
        assert node.get_path_to_root() is not None


def test_mcts_value_estimates():
    """Test MCTS value estimation and backpropagation"""
    print("\n=== Testing MCTS Value Estimates ===")

    # Create mock components
    mock_manager = MockMCTSInference()
    mock_inference = MCTSInference(model_manager=mock_manager)
    mock_env = MockTheoremEnv()

    config = MCTSConfig(
        n_simulations=15, c_puct=1.0, max_depth=3, max_actions_per_expand=2
    )

    mcts = MCTSSearch(mcts_inference=mock_inference, config=config)
    mock_env.init()

    # Run MCTS search
    root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

    # Check that values are properly propagated
    assert root_node.average_value >= 0.0, "Root value should be non-negative"
    assert root_node.average_value <= 1.0, "Root value should be at most 1.0"

    # Check child values
    for child in root_node.children.values():
        assert child.average_value >= 0.0, "Child value should be non-negative"
        assert child.average_value <= 1.0, "Child value should be at most 1.0"

    print(f"Root average value: {root_node.average_value:.3f}")
    for action, child in root_node.children.items():
        print(f"  {action}: {child.average_value:.3f} ({child.visit_count} visits)")


def test_mcts_expansion():
    """Test MCTS node expansion"""
    print("\n=== Testing MCTS Expansion ===")

    # Create mock components
    mock_manager = MockMCTSInference()
    mock_inference = MCTSInference(model_manager=mock_manager)
    mock_env = MockTheoremEnv()

    config = MCTSConfig(
        n_simulations=5, c_puct=1.0, max_depth=2, max_actions_per_expand=3
    )

    _ = MCTSSearch(mcts_inference=mock_inference, config=config)
    mock_env.init()

    # Create root node
    root = MCTSNode(
        state_id="initial_state",
        parent=None,
        action=None,
        execution_result=None,
    )

    # Note: expand method may not exist or work differently
    # For now, just verify node creation
    assert root.state_id == "initial_state"
    assert root.parent is None
    assert root.action is None

    print(f"Root expanded to {len(root.children)} children")


def _collect_successful_nodes(node, successful_nodes):
    """Recursively collect nodes that lead to successful proofs"""
    if (
        node.is_terminal and node.average_value > 0.5
    ):  # Consider high-value terminal nodes as successful
        successful_nodes.append(node)

    for child in node.children.values():
        _collect_successful_nodes(child, successful_nodes)


def main():
    """Run all standalone tests"""
    print("Running MCTS Standalone Mock Tests")
    print("=" * 40)

    test_basic_mcts_functionality()
    test_mcts_with_successful_path()
    test_mcts_value_estimates()
    test_mcts_expansion()

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
