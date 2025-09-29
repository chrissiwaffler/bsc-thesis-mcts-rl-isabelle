import copy
import os
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from ..core import ProofGraph, ProofState


class ProofTreeVisualizer:
    """Live visualization for proof trees during algorithm execution

    Usage with ProofGraph:
        viz = ProofTreeVisualizer(save_frames=True)

        # during your algorithm:
        graph = ProofGraph()
        state = graph.add_state("theorem: P and Q")
        graph.update_result(state.state_name, False, "goal: ...")
        viz.update(graph, "Step 1")

        # at the end:
        viz.show()
    """

    def __init__(
        self,
        save_frames=False,
        output_dir="proof_frames",
        show_probabilities=True,
    ):
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.frames = []  # store graph snapshots
        self.current_frame = 0
        self.save_frames = save_frames
        self.output_dir = output_dir
        self.is_paused = False
        self.show_probabilities = show_probabilities

        if save_frames and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # setup keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # interactive mode
        plt.ion()

        print("=== PROOF TREE VISUALIZER (BEAM SEARCH) ===")
        print("← →: Navigate frames")
        print("P: Toggle probability display")
        print("Space: Pause/resume")
        print("S: Save current frame")
        print("Q: Quit")
        print("==========================================\n")

    def _tree_layout(self, G, root):
        """Create tree layout for graph"""
        pos = {}
        try:
            levels = nx.single_source_shortest_path_length(G, root)
        except nx.NetworkXError:
            # fallback if root not in graph
            return nx.spring_layout(G)

        # Group by level and sort by probability within each level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        # Sort each level by probability
        for _level, nodes in level_nodes.items():
            nodes.sort(key=lambda n: G.nodes[n]["data"].probability, reverse=True)

        # Position nodes
        for level, nodes in level_nodes.items():
            width = len(nodes)
            for i, node in enumerate(nodes):
                # Spread nodes more for better visibility
                x = (i - (width - 1) / 2) * 1.5
                y = -level * 2.0
                pos[node] = (x, y)

        return pos

    def _get_node_color(self, state: "ProofState") -> str:
        """get node color based on state"""
        if state.is_done:
            return "#90EE90"  # Light green for completed
        elif state.probability < 0.01:
            return "#FFE4E1"  # Light red for very low probability
        elif state.beam_rank and state.beam_rank <= 5:
            return "#87CEEB"  # Sky blue for top beam nodes
        else:
            return "#FFB6C1"  # Light pink for active nodes

    def _draw_graph(self, proof_graph: "ProofGraph", title="", highlight_beam=None):
        """Draw the enhanced graph"""
        self.ax.clear()

        G = proof_graph.G

        if len(G) == 0:
            self.ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=20)
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            return

        # Find root
        roots = [n for n in G.nodes() if len(list(G.predecessors(n))) == 0]
        root = roots[0] if roots else next(iter(G.nodes()))

        pos = self._tree_layout(G, root)

        # Draw edges with probability-based styling
        for edge in G.edges():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]

            child_state = proof_graph.get_state(edge[1])
            edge_width = max(0.5, min(5.0, child_state.probability * 5))
            edge_alpha = max(0.3, min(1.0, child_state.probability))

            self.ax.plot(
                [x1, x2], [y1 - 0.1, y2 + 0.1], "gray", alpha=edge_alpha, lw=edge_width
            )

        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            state = proof_graph.get_state(node)

            # Node size based on probability
            node_size = max(0.3, min(0.6, 0.3 + state.probability * 0.3))

            # Node shape and color
            if state.beam_rank and state.beam_rank <= 3:
                # Star shape for top beam nodes
                rect = mpatches.FancyBboxPatch(
                    (x - node_size / 2, y - 0.1),
                    node_size,
                    0.2,
                    boxstyle="round,pad=0.02",
                    facecolor=self._get_node_color(state),
                    edgecolor="gold" if state.beam_rank == 1 else "black",
                    linewidth=3 if state.beam_rank == 1 else 2,
                    linestyle="-" if state.beam_rank == 1 else "--",
                )
            else:
                rect = mpatches.FancyBboxPatch(
                    (x - node_size / 2, y - 0.1),
                    node_size,
                    0.2,
                    boxstyle="round,pad=0.02",
                    facecolor=self._get_node_color(state),
                    edgecolor="black",
                    linewidth=2,
                )
            self.ax.add_patch(rect)

            # Node label
            self.ax.text(
                x,
                y,
                state.state_name,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

            # Probability display
            if self.show_probabilities and state.probability < 1.0:
                self.ax.text(
                    x,
                    y - 0.25,
                    f"p={state.probability:.3f}",
                    ha="center",
                    fontsize=8,
                    color="darkblue",
                    weight="bold",
                )

                # Temperature and completion index
                if state.temperature > 0:
                    self.ax.text(
                        x,
                        y - 0.35,
                        f"τ={state.temperature}",
                        ha="center",
                        fontsize=7,
                        color="gray",
                        style="italic",
                    )

            # Tactic
            tactic_text = (
                state.command[:40] + "..." if len(state.command) > 40 else state.command
            )
            self.ax.text(
                x,
                y + 0.25,
                tactic_text,
                ha="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

            # Beam rank badge
            if state.beam_rank:
                self.ax.text(
                    x + node_size / 2 + 0.1,
                    y + 0.1,
                    f"#{state.beam_rank}",
                    ha="center",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "gold" if state.beam_rank == 1 else "silver",
                        "alpha": 0.9,
                    },
                )

        # Set limits
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            x_margin = 1.0
            y_margin = 1.0
            self.ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            self.ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

        # Title and statistics
        stats = proof_graph.get_statistics()
        frame_info = (
            f" - Frame {self.current_frame + 1}/{len(self.frames)}"
            if self.frames
            else ""
        )
        status = " [PAUSED]" if self.is_paused else ""

        title_text = f"{title}{frame_info}{status}\n"
        title_text += f"Nodes: {stats.total_nodes}, "
        title_text += f"Completed: {stats.completed_proofs}, "
        title_text += f"Max Depth: {stats.max_depth}"

        self.ax.set_title(title_text, fontsize=14)
        self.ax.axis("off")

        # Legend
        legend_elements = [
            mpatches.Patch(color="#90EE90", label="Completed"),
            mpatches.Patch(color="#87CEEB", label="Top Beam"),
            mpatches.Patch(color="#FFB6C1", label="Active"),
            mpatches.Patch(color="#FFE4E1", label="Low Prob"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        self.ax.grid(True, alpha=0.05, linestyle=":")

    def update(
        self, proof_graph: "ProofGraph", title="Beam Search", highlight_beam=None
    ):
        """Update visualization"""
        if not self.is_paused:
            # Store snapshot
            snapshot = {
                "graph": proof_graph.G.copy(),
                "states": {
                    node: copy.deepcopy(proof_graph.get_state(node))
                    for node in proof_graph.G.nodes()
                },
                "title": title,
                "highlight_beam": highlight_beam,
            }
            self.frames.append(snapshot)
            self.current_frame = len(self.frames) - 1

            # Draw
            self._draw_graph(proof_graph, title, highlight_beam)

            # Save frame if requested
            if self.save_frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = os.path.join(
                    self.output_dir,
                    f"beam_frame_{len(self.frames):04d}_{timestamp}.png",
                )
                self.fig.savefig(filename, dpi=150, bbox_inches="tight")

            plt.pause(0.001)

    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == "left" and self.frames:
            self.current_frame = max(0, self.current_frame - 1)
            self._redraw_frame()
        elif event.key == "right" and self.frames:
            self.current_frame = min(len(self.frames) - 1, self.current_frame + 1)
            self._redraw_frame()
        elif event.key == " ":
            self.is_paused = not self.is_paused
            print(f"Visualization {'PAUSED' if self.is_paused else 'RESUMED'}")
        elif event.key == "p":
            self.show_probabilities = not self.show_probabilities
            self._redraw_frame()
            print(f"Probabilities {'ON' if self.show_probabilities else 'OFF'}")
        elif event.key == "s":
            if self.frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"beam_proof_tree_{self.current_frame}_{timestamp}.png"
                self.fig.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved: {filename}")
        elif event.key == "q":
            plt.close(self.fig)

    def _redraw_frame(self):
        """Redraw current frame"""
        if self.frames and 0 <= self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]

            # Create a mock ProofGraph-like object to avoid circular imports
            class TempGraph:
                def __init__(self):
                    self.G = frame["graph"]

                def get_state(self, node_name):
                    return self.G.nodes[node_name]["data"]

                def get_statistics(self):
                    # Simplified statistics for redraw
                    total_nodes = self.G.number_of_nodes()
                    completed_proofs = sum(
                        1 for n in self.G.nodes() if self.get_state(n).is_done
                    )
                    max_depth = max(
                        (self.get_state(n).depth for n in self.G.nodes()), default=0
                    )

                    from ..core import ProofGraph

                    return ProofGraph.SearchStats(
                        total_nodes=total_nodes,
                        completed_proofs=completed_proofs,
                        max_depth=max_depth,
                        beam_history=[],
                    )

            temp_graph = TempGraph()
            for node, state in frame["states"].items():
                temp_graph.G.nodes[node]["data"] = state
            self._draw_graph(temp_graph, frame["title"], frame.get("highlight_beam"))
            plt.draw()

    def close(self):
        """Close visualization"""
        plt.close(self.fig)

    def show(self):
        """Keep window open"""
        print(f"\nVisualization complete. Total frames: {len(self.frames)}")
        print("Use arrow keys to navigate, P to toggle probabilities, Q to quit.")
        plt.ioff()
        plt.show()
