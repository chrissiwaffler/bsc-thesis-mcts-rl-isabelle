from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field


@dataclass
class ProofState:
    state_name: str
    command: str
    is_done: bool = False
    result: str = ""
    # parent's state_name
    parent: str | None = None
    # state_name of the children
    # use a factory, so every instance has an own new list
    children: list[str] = field(default_factory=list)
    failed_tactics: list[tuple[str, str]] = field(default_factory=list)

    # Enhanced failure tracking
    failure_error_messages: list[str] = field(default_factory=list)

    # track proof mode
    mode: str = "prove"  # prove or state mode
    available_lemmas: list[str] = field(
        default_factory=list
    )  # already proven "have" statements

    # beam search properties
    probability: float = 1.0  # same as context_score in our simplified approach
    # context_score: float = 1.0  # context-aware score from external judge

    # generation metadata
    # what temperature was used to generate this tactic
    temperature: float = 0.0

    # reasoning and history
    # reasoning: NodeReasoning | None = None  # self-ask reasoning
    reasoning: Any | None = None

    # exploration_history: list[ActionObservation] = field(default_factory=list)
    exploration_history: list[Any] = field(default_factory=list)

    # search metadata
    fully_explored: bool = False
    depth: int = 0  # depth in proof tree
    beam_rank: int | None = None  # rank in current beam (1=best)

    def has_failure_pattern(self, min_occurrences: int = 2) -> bool:
        """Check if there's a repeated failure pattern"""
        if len(self.failure_error_messages) < min_occurrences:
            return False

        error_counts = {}
        for error_msg in self.failure_error_messages:
            # Normalize error message for comparison
            normalized = error_msg.lower().strip()
            error_counts[normalized] = error_counts.get(normalized, 0) + 1

        # Check if any error message occurs too often
        return any(count >= min_occurrences for count in error_counts.values())


class ProofGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.counter = 0
        self.beam_history: list[list[str]] = []  # history of beams at each level
        self.probability_threshold: float = 0.001  # min probability to keep node

    def add_state(
        self,
        command: str,
        parent: str | None = None,
        temperature: float = 0.0,
        **kw,
    ) -> ProofState:
        state_name = f"state{self.counter}"
        self.counter += 1

        depth = self.get_state(parent).depth + 1 if parent else 0

        node = ProofState(
            state_name=state_name,
            command=command,
            parent=parent,
            temperature=temperature,
            depth=depth,
            **kw,
        )
        self.G.add_node(state_name, data=node)

        if parent:
            self.G.add_edge(parent, state_name)
            self.get_state(parent).children.append(state_name)

        return node

    def update_result(
        self,
        state_name: str,
        is_done: bool,
        result: str,
    ):
        node = self.get_state(state_name)
        node.is_done = is_done
        node.result = result

    def update_probability(self, state_name: str, probability: float):
        self.get_state(state_name).probability = probability

    def get_state(self, state_name: str) -> ProofState:
        return self.G.nodes[state_name]["data"]

    def add_failed_tactic(self, state_name: str, tactic: str, error_msg: str):
        state = self.get_state(state_name)
        state.failed_tactics.append((tactic, error_msg))

    def get_beam_at_depth(self, depth: int, beam_width: int) -> list[ProofState]:
        """get top k nodes at specific depth by probability"""
        nodes = [
            self.get_state(node)
            for node in self.G.nodes
            if self.get_state(node).depth == depth
        ]

        # highest prob score first
        nodes.sort(key=lambda x: x.probability, reverse=True)

        for i, node in enumerate(nodes):
            node.beam_rank = i + 1

        return nodes[:beam_width]

    # TODO: needed?
    # def prune_low_probability_nodes(self, threshold: float | None = None):
    #     """removes nodes with probability below threshold"""
    #     if threshold is None:
    #         threshold = self.probability_threshold
    #
    #     nodes_to_remove = []
    #     for node_name in self.G.nodes():
    #         node = self.get_state(node_name)
    #         if node.probability < threshold and not node.is_done:
    #             nodes_to_remove.append(node_name)
    #
    #     for node_name in nodes_to_remove:
    #         # update parent's children list
    #         node = self.get_state(node_name)
    #         if node.parent:
    #             parent = self.get_state(node.parent)
    #             if node_name in parent.children:
    #                 parent.children.remove(node_name)
    #
    #         self.G.remove_node(node_name)

    class SearchStats(BaseModel):
        total_nodes: int = Field(ge=0)
        completed_proofs: int = Field(ge=0)
        max_depth: int = Field(ge=0)
        beam_history: list[list[str]] = Field(default_factory=list)

        class Config:
            frozen = True

    def get_statistics(self) -> ProofGraph.SearchStats:
        total_nodes = self.G.number_of_nodes()
        completed_proofs = sum(1 for n in self.G.nodes() if self.get_state(n).is_done)
        max_depth = max((self.get_state(n).depth for n in self.G.nodes()), default=0)

        return ProofGraph.SearchStats(
            total_nodes=total_nodes,
            completed_proofs=completed_proofs,
            max_depth=max_depth,
            beam_history=self.beam_history,
        )
