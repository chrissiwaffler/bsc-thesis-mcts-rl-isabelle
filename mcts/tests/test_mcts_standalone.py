#!/usr/bin/env python3
"""
Complete MCTS test demonstrating the algorithm working end-to-end
"""

import os
import sys

# Add parent directory to path to import mcts_accelerate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.search import MCTSConfig, MCTSSearch

from .mock_env import MockTheoremEnv
from .mock_inference import MockInferenceManager


def main():
    """Run complete MCTS demonstration"""
    print("Complete MCTS Demonstration")
    print("=" * 50)

    # Create mock components
    mock_inference = MockInferenceManager()
    mock_env = MockTheoremEnv()
    config = MCTSConfig(
        n_simulations=50,  # More simulations for better exploration
        max_actions_per_expand=5,
        c_puct=1.0,
    )

    # Create MCTS search
    mcts = MCTSSearch(mcts_inference=mock_inference, config=config)  # type: ignore[arg-type]

    # Set up a simple theorem
    mock_env.reset("simple_theorem", "A = A")

    print(f"📝 Theorem: {mock_env.theorem_statement}")
    print(
        f"🔧 Config: {config.n_simulations} simulations, {config.max_actions_per_expand} actions per expansion"
    )
    print()

    # Run MCTS search
    print("🔍 Running MCTS search...")
    root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

    # Display results
    print("\n📊 Search Results:")
    print(f"   Root visits: {root_node.visit_count}")
    print(f"   Root children: {len(root_node.children)}")
    print(f"   Root average value: {root_node.average_value:.3f}")

    # Show children statistics
    print("\nRoot Children:")
    for command, child in root_node.children.items():
        print(f"   Command: '{command}'")
        print(f"     Visits: {child.visit_count}")
        print(f"     Value: {child.average_value:.3f}")
        print(f"     Terminal: {child.is_terminal}")
        if child.is_terminal:
            print(f"     Reward: {child.terminal_reward:.3f}")
        print()

    # Find successful paths
    successful_nodes = []

    def collect_successful_nodes(node):
        if node.is_terminal and node.terminal_reward > 0.5:
            successful_nodes.append(node)
        for child in node.children.values():
            collect_successful_nodes(child)

    collect_successful_nodes(root_node)

    print(f"Found {len(successful_nodes)} successful proof paths!")

    if successful_nodes:
        print("\nSuccessful Proof Paths:")
        for i, node in enumerate(successful_nodes, 1):
            print(f"\n   Path {i}:")
            path = node.get_path_to_root()
            for action, _result in path:
                if action:
                    print(f"     → {action}")
            print(f"     Final reward: {node.terminal_reward:.3f}")

    # Show best action
    if root_node.children:
        best_child = max(root_node.children.values(), key=lambda c: c.visit_count)
        print(
            f"\nBest action by visits: '{best_child.action}' ({best_child.visit_count} visits)"
        )

        best_value_child = max(
            root_node.children.values(), key=lambda c: c.average_value
        )
        print(
            f"Best action by value: '{best_value_child.action}' (value: {best_value_child.average_value:.3f})"
        )

    print("\nMCTS demonstration completed successfully!")


if __name__ == "__main__":
    main()
