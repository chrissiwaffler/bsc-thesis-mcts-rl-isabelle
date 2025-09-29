"""Complete MCTS integration tests"""

import os
import sys

# Add parent directory to path to import mcts_accelerate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.inference_mcts import MCTSInference
from mcts.search import MCTSConfig, MCTSSearch

from .mock_env import MockTheoremEnv
from .mock_inference import MockInferenceManager


class TestMCTSComplete:
    """Complete MCTS integration tests"""

    def test_mcts_complete_search(self):
        """Test complete MCTS search with analysis"""
        print("\n=== Testing Complete MCTS Search ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(
            n_simulations=30,  # Moderate number for testing
            max_actions_per_expand=4,
            c_puct=1.0,
        )

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Verify search completed successfully
        assert root_node is not None
        assert root_node.visit_count == config.n_simulations
        assert len(root_node.children) > 0
        assert root_node.average_value >= -1.0 and root_node.average_value <= 1.0

        print(f"✅ MCTS search completed with {root_node.visit_count} visits")
        print(f"✅ Root node has {len(root_node.children)} children")
        print(f"✅ Root average value: {root_node.average_value:.3f}")

    def test_mcts_successful_path_discovery(self):
        """Test MCTS can discover successful proof paths"""
        print("\n=== Testing Successful Path Discovery ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(
            n_simulations=40,  # Enough simulations to find successful paths
            max_actions_per_expand=5,
            c_puct=1.0,
        )

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Find successful paths
        successful_nodes = []

        def collect_successful_nodes(node):
            if node.is_terminal and node.terminal_reward > 0.5:
                successful_nodes.append(node)
            for child in node.children.values():
                collect_successful_nodes(child)

        collect_successful_nodes(root_node)

        # Should find at least one successful path
        assert (
            len(successful_nodes) > 0
        ), "MCTS should find at least one successful proof path"

        print(f"✅ Found {len(successful_nodes)} successful proof paths")

        # Verify each successful node has proper structure
        for node in successful_nodes:
            assert node.is_terminal
            assert node.terminal_reward > 0.5
            assert node.visit_count > 0
            assert node.get_path_to_root() is not None

    def test_mcts_action_selection_statistics(self):
        """Test MCTS action selection and visit statistics"""
        print("\n=== Testing Action Selection Statistics ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(n_simulations=25, max_actions_per_expand=3, c_puct=1.0)

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Verify children have reasonable statistics
        assert len(root_node.children) > 0

        total_visits = sum(child.visit_count for child in root_node.children.values())
        # Total visits should be close to n_simulations (might be less due to terminal states)
        assert total_visits <= config.n_simulations
        assert (
            total_visits >= config.n_simulations - 5
        )  # Allow some tolerance for early termination

        # Check that visits are distributed (not all on one child)
        visit_counts = [child.visit_count for child in root_node.children.values()]
        assert (
            max(visit_counts) < config.n_simulations
        ), "Visits should be distributed among children"

        # Find best actions by different metrics
        best_by_visits = max(root_node.children.values(), key=lambda c: c.visit_count)
        best_by_value = max(root_node.children.values(), key=lambda c: c.average_value)

        print(
            f"✅ Best action by visits: '{best_by_visits.action}' ({best_by_visits.visit_count} visits)"
        )
        print(
            f"✅ Best action by value: '{best_by_value.action}' (value: {best_by_value.average_value:.3f})"
        )

        # Verify best actions have reasonable properties
        assert best_by_visits.visit_count > 0
        assert -1.0 <= best_by_value.average_value <= 1.0

    def test_mcts_tree_structure(self):
        """Test MCTS tree structure and node relationships"""
        print("\n=== Testing Tree Structure ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()
        config = MCTSConfig(n_simulations=20, max_actions_per_expand=3, c_puct=1.0)

        # Create MCTS search
        mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

        # Set up a simple theorem
        mock_env.reset("test_theorem", "A = A")

        # Run MCTS search
        root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

        # Verify root structure
        assert root_node.parent is None
        assert root_node.state_id == "initial_state"
        assert root_node.visit_count > 0

        # Verify children structure
        for command, child in root_node.children.items():
            assert child.parent == root_node
            assert child.action == command
            assert child.state_id is not None
            assert child.state_id != root_node.state_id

            # Verify path to root works
            path = child.get_path_to_root()
            assert len(path) == 1
            assert path[0][0] == command  # action
            assert path[0][1] is not None  # execution result

        print(f"✅ Tree structure verified for {len(root_node.children)} children")

    def test_mcts_convergence_with_more_simulations(self):
        """Test MCTS convergence behavior with more simulations"""
        print("\n=== Testing Convergence with More Simulations ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()

        # Test with increasing number of simulations
        simulation_counts = [10, 25, 50]
        root_values = []

        for n_sims in simulation_counts:
            config = MCTSConfig(
                n_simulations=n_sims, max_actions_per_expand=4, c_puct=1.0
            )

            # Reset environment for each run
            mock_env.reset("test_theorem", "A = A")

            # Create MCTS search
            mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

            # Run MCTS search
            root_node = mcts.search(env=mock_env, initial_state_id="initial_state")
            root_values.append(root_node.average_value)

            print(
                f"   {n_sims} simulations: root value = {root_node.average_value:.3f}"
            )

        # Verify that values are reasonable and show some convergence
        for value in root_values:
            assert -1.0 <= value <= 1.0

        # With more simulations, value should generally improve (be closer to successful outcomes)
        # This is a weak test since it's stochastic, but checks basic sanity
        assert (
            root_values[-1] >= -0.5
        ), "Final value should be reasonable with more simulations"

        print(f"✅ Convergence test completed: values {root_values}")

    def test_mcts_different_configurations(self):
        """Test MCTS with different configuration parameters"""
        print("\n=== Testing Different Configurations ===")

        # Create mock components
        mock_manager = MockInferenceManager()
        mock_inference = MCTSInference(model_manager=mock_manager)
        mock_env = MockTheoremEnv()

        configs = [
            MCTSConfig(n_simulations=15, max_actions_per_expand=2, c_puct=0.5),
            MCTSConfig(n_simulations=15, max_actions_per_expand=4, c_puct=1.5),
            MCTSConfig(n_simulations=15, max_actions_per_expand=6, c_puct=2.0),
        ]

        for i, config in enumerate(configs):
            print(
                f"   Testing config {i + 1}: {config.n_simulations} sims, {config.max_actions_per_expand} actions, c_puct={config.c_puct}"
            )

            # Reset environment for each run
            mock_env.reset("test_theorem", "A = A")

            # Create MCTS search
            mcts = MCTSSearch(mcts_inference=mock_inference, config=config)

            # Run MCTS search
            root_node = mcts.search(env=mock_env, initial_state_id="initial_state")

            # Verify basic properties
            assert root_node.visit_count == config.n_simulations
            assert len(root_node.children) <= config.max_actions_per_expand
            assert -1.0 <= root_node.average_value <= 1.0

            print(
                f"     → {len(root_node.children)} children, value: {root_node.average_value:.3f}"
            )

        print(f"✅ All {len(configs)} configurations tested successfully")
