import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional, cast

from mcts.base_env import TheoremEnv
from mcts.inference_mcts import MCTSInference
from mcts.isabelle_interface import IsabelleErrorResult
from mcts.shared_types import CommandGenerationResponse, ValueEstimateResponse
from mcts.utils import parse_isabelle_response

UNKNOWN_STATE = "unknown_state"

# configure simple logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    n_simulations: int = 1500

    # ucb exploration constant
    c_puct: float = 1.0

    max_depth: int = 100

    # number of actions to try per expansion
    # (= k parameter for response generation with llm)
    max_actions_per_expand: int = 5


class MCTSNode:
    def __init__(
        self,
        state_id: str,
        parent: Optional["MCTSNode"] = None,
        action: str | None = None,
        execution_result: str | None = None,
        command_generation_response: CommandGenerationResponse | None = None,
        value_estimate_response: ValueEstimateResponse | None = None,
        proof_mode: str = "prove",
        available_lemmas: list[str] | None = None,
        prior: float = 0.0,
    ) -> None:
        self.state_id = state_id
        self.parent = parent

        # clean command that lead here
        self.action = action
        # subgoal result by Isabelle after applying this action
        self.execution_result = execution_result
        # the mode (prove, state) isabelle is in after the current command (action)

        self.command_generation_response = command_generation_response
        self.value_estimate_response = value_estimate_response

        self.proof_mode = proof_mode
        self.available_lemmas = available_lemmas

        # neural network prior
        self.prior = prior

        # mcts statistics
        self.visit_count = 0
        self.value_sum = 0.0
        # action -> child node
        self.children: dict[str, MCTSNode] = {}

        # terminal state info
        self.is_terminal = False
        self.terminal_reward: float = 0.0

        # sledgehammer state tracking
        self.sledgehammer_started = False
        self.sledgehammer_task: asyncio.Task | None = None

    @property
    def average_value(self) -> float:
        """average value from mcts backups"""
        return self.value_sum / max(1, self.visit_count)

    def ucb_score(self, c_puct: float) -> float:
        """ucb1 score for node selection"""
        if self.visit_count == 0:
            return float("inf")

        exploitation = self.average_value

        if self.parent is None:
            # root node doesn't need exploration bonus
            exploration = 0.0
        else:
            exploration = (
                c_puct
                * self.prior
                * math.sqrt(self.parent.visit_count)
                / (1 + self.visit_count)
            )

        return exploitation + exploration

    def select_child(self, c_puct: float) -> "MCTSNode":
        """select best child with the highest UCB score"""
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))

    def expand(
        self,
        mcts_inference: MCTSInference,
        env: TheoremEnv,
        config: MCTSConfig,
    ) -> list["MCTSNode"]:
        """expand node by executing actions in environment"""
        if self.is_terminal or self.children:
            logger.debug(f"node {self.state_id} already expanded or terminal")
            return list(self.children.values())

        # get state content for this node
        state_content = self._get_state_content(env)
        if state_content == UNKNOWN_STATE:
            return []

        proof_context = self.get_proof_context(env)

        # generate new commands using xml structure
        responses = mcts_inference.generate_commands(
            theorem_statement=proof_context["theorem_statement"],
            proof_mode=self.proof_mode,
            proof_history=proof_context["proof_history"],
            current_state=state_content,
            available_lemmas=self.available_lemmas,
            n=config.max_actions_per_expand,
        )

        if not responses:
            logger.info(f"no responses generated for node {self.state_id}")
            return []

        # responses is guaranteed to be a list after the None check
        responses = cast(list[CommandGenerationResponse], responses)

        logger.info(f"generated {len(responses)} commands for node {self.state_id}")
        for i, response in enumerate(responses):
            logger.debug(f"command {i + 1}: {response['command']}")

        # start sledgehammer synchronously if enabled and not already started
        if (
            hasattr(env, "use_sledgehammer")
            and env.use_sledgehammer
            and not self.sledgehammer_started
        ):  # type: ignore
            try:
                # start sledgehammer synchronously for this state
                # Note: This assumes start_sledgehammer_async can be called synchronously
                # If it needs to be async, we'll need to handle this differently
                if hasattr(env, "start_sledgehammer_sync"):
                    env.start_sledgehammer_sync(self.state_id)  # type: ignore
                else:
                    # Fallback: try to call the async method synchronously (may not work)
                    logger.warning(f"Sledgehammer start not available synchronously for node {self.state_id}")
                self.sledgehammer_started = True
                logger.debug(f"started sledgehammer for node {self.state_id}")
            except Exception as e:
                logger.warning(
                    f"failed to start sledgehammer for node {self.state_id}: {e}"
                )

        # check for sledgehammer results and add them to responses
        sledgehammer_tactics = []
        if hasattr(env, "check_sledgehammer_result"):  # type: ignore
            sledgehammer_result = env.check_sledgehammer_result(self.state_id)  # type: ignore
            if sledgehammer_result:
                for tactic in sledgehammer_result:
                    sledgehammer_tactics.append(
                        {
                            "command": tactic,
                            "thinking": "",
                            "full_prompt": "",
                            "full_response": "",
                            "logprobs": 0.0,
                        }
                    )
                logger.info(
                    f"added {len(sledgehammer_tactics)} sledgehammer tactics for node {self.state_id}"
                )

        # combine LLM responses with sledgehammer tactics
        all_responses = cast(
            list[CommandGenerationResponse], responses + sledgehammer_tactics
        )

        # execute each command to create child nodes with real state IDs

        # uniform priors for now
        if all_responses:
            prior_prob = 1.0 / len(all_responses)
        else:
            prior_prob = 1.0

        logger.debug(
            f"expanding node {self.state_id} with {len(all_responses)} commands"
        )
        for response in all_responses:
            command = response["command"]

            if command in self.children:
                logger.debug(f"command already exists: {command}")
                continue

            # execute action in environment for this current state
            routed_action = f"{self.state_id}#{command}"
            logger.debug(f"executing command: {command}")
            result = env.step(routed_action)

            # create child node
            child = MCTSNode(
                state_id=result["result"].state_name,
                parent=self,
                action=command,
                command_generation_response=response,
                prior=prior_prob,
            )

            if result["result"].is_done or isinstance(
                result["result"], IsabelleErrorResult
            ):
                # handle terminal states
                child.is_terminal = True
                child.terminal_reward = result["reward"]
                # mark as visited to be included in training data
                child.visit_count = 1
                child.value_sum = result["reward"]
                # proof can be successfully or unsuccesfully be terminated
                child.execution_result = str(result["result"])

                # log command validation result
                if isinstance(result["result"], IsabelleErrorResult):
                    logger.warning(
                        f"invalid command: {command} - error: {result['result']}"
                    )
                else:
                    logger.info(
                        f"valid command completed proof: {command} - reward: {result['reward']}"
                    )
            else:
                # get isabelle result from environment
                child.execution_result = str(result["result"])

                # update isabelle's current mode ("state" or "prove"")
                # also store available lemmas
                if hasattr(result["result"], "result"):
                    child.proof_mode, child.available_lemmas = parse_isabelle_response(
                        result["result"].result  # type: ignore[attr-defined]
                    )
                else:
                    # IsabelleErrorResult doesn't have .result attribute
                    child.proof_mode = "unknown"
                    child.available_lemmas = []

                logger.info(
                    f"valid command: {command} - created state: {child.state_id}"
                )

            self.children[command] = child
            logger.debug(f"created child node {child.state_id} for command: {command}")

        return list(self.children.values())

    def get_path_to_root(self) -> list[tuple[str | None, str | None]]:
        """reconstruct proof history from root to this node; returns [(action, execution_result)]"""
        path = []
        node = self

        # traverse up to root, collecting (action, result) pairs
        while node.parent is not None:
            path.append((node.action, node.execution_result))
            node = node.parent

        # reverse to get root-to-current order
        return path[::-1]

    def get_proof_context(self, env: TheoremEnv) -> dict:
        """get proof context specific to this node's path"""
        return {
            "theorem_statement": env.theorem_statement,
            "proof_history": self.get_path_to_root(),
        }

    def _get_state_content(self, env: TheoremEnv) -> str:
        """get proof state content (isabelle result after command application) from environment"""

        # original logic for real isabelle - check this FIRST
        if (
            hasattr(env, "isabelle")
            and env.isabelle
            and self.state_id in env.isabelle.active_states
        ):
            _, content = env.isabelle.active_states[self.state_id]
            return str(content)

        # for mock environment, check all_states
        if hasattr(env, "all_states"):
            all_states = getattr(env, "all_states", {})
            if self.state_id in all_states:
                return str(all_states[self.state_id])

        # for mock environment, return a simple state content
        # only if it's NOT a real isabelle environment; checked above
        if hasattr(env, "current_state") and (
            self.state_id == env.current_state or self.state_id == "initial_state"
        ):
            return f"Mock proof state for {self.state_id}"

        return UNKNOWN_STATE

    def backup(self, value: float):
        """backpropagate value up the tree"""
        self.visit_count += 1
        self.value_sum += value

        if self.parent:
            self.parent.backup(value)

    def to_dict(self) -> dict:
        """Convert MCTSNode to a JSON-serializable dictionary"""
        # convert execution_result to dict if it's an IsabelleResult object
        execution_result_dict = None
        if self.execution_result is not None:
            if hasattr(self.execution_result, "model_dump"):
                # Pydantic v2
                execution_result_dict = self.execution_result.model_dump()  # type: ignore[attr-defined]
            elif hasattr(self.execution_result, "dict"):
                # Pydantic v1
                execution_result_dict = self.execution_result.dict()  # type: ignore[attr-defined]
            else:
                execution_result_dict = str(self.execution_result)

        # convert response objects to dicts if they exist
        command_response_dict = None
        if self.command_generation_response is not None:
            command_response_dict = (
                dict(self.command_generation_response)
                if hasattr(self.command_generation_response, "keys")
                else str(self.command_generation_response)
            )

        value_response_dict = None
        if self.value_estimate_response is not None:
            value_response_dict = (
                dict(self.value_estimate_response)
                if hasattr(self.value_estimate_response, "keys")
                else str(self.value_estimate_response)
            )

        return {
            "state_id": self.state_id,
            "action": self.action,
            "execution_result": execution_result_dict,
            "proof_mode": self.proof_mode,
            "available_lemmas": self.available_lemmas,
            "prior": self.prior,
            "visit_count": self.visit_count,
            "value_sum": self.value_sum,
            "average_value": self.average_value,
            "is_terminal": self.is_terminal,
            "terminal_reward": self.terminal_reward,
            "command_generation_response": command_response_dict,
            "value_estimate_response": value_response_dict,
            # don't include parent/children to avoid circular references
        }


class MCTSSearch:
    """Monte Carlo Tree Search for theorem proving"""

    def __init__(
        self,
        mcts_inference: MCTSInference,
        config: MCTSConfig,
    ) -> None:
        self.mcts_inference = mcts_inference
        self.config = config
        self.root: MCTSNode | None = None

    def search(
        self, env: TheoremEnv, initial_state_id: str, timeout_seconds: int | None = None
    ) -> MCTSNode:
        """run mcts search from initial state with optional timeout"""
        logger.info(
            f"starting mcts search with {self.config.n_simulations} simulations"
        )
        # init root
        self.root = MCTSNode(state_id=initial_state_id)

        # generate initial responses for the root node to include it in training data
        proof_context = {
            "theorem_statement": env.theorem_statement,
            "proof_history": [],  # empty history for root
        }

        # get state content for root node
        state_content = self.root._get_state_content(env)
        if state_content != UNKNOWN_STATE:
            # generate command generation response for root (initial state)
            command_responses = self.mcts_inference.generate_commands(
                theorem_statement=str(proof_context["theorem_statement"]),
                # root starts in prove mode
                proof_mode="prove",
                proof_history=proof_context["proof_history"],  # type: ignore[arg-type]
                current_state=state_content,
                # no lemmas at root
                available_lemmas=None,
                # just one response for root
                n=1,
            )
            if command_responses:
                self.root.command_generation_response = command_responses[0]

            # generate value estimate response for root
            value_response = self.mcts_inference.estimate_value(
                theorem_statement=str(proof_context["theorem_statement"]),
                # special marker for root
                command="initial_state",
                execution_result=state_content,
                proof_history=proof_context["proof_history"],  # type: ignore[arg-type]
                proof_mode="prove",
            )
            if value_response:
                self.root.value_estimate_response = value_response

        # run the search loop with timeout if specified
        import time
        start_time = time.time()

        for i in range(self.config.n_simulations):
            # Check timeout
            if timeout_seconds is not None and (time.time() - start_time) > timeout_seconds:
                logger.warning(
                    f"mcts search timed out after {timeout_seconds} seconds, root visits: {self.root.visit_count}"
                )
                break

            logger.debug(f"simulation {i + 1}/{self.config.n_simulations}")
            # 1. selection: traverse tree to leaf using UCB
            assert self.root is not None
            node = self._select_leaf(self.root)

            # special case: if we selected the root node and it's unvisited,
            # we need to evaluate it directly since it can't be expanded
            if node == self.root and node.visit_count == 0:
                # evaluate the root node directly
                value = self._evaluate_node(node, env)
                node.backup(value)
                continue

            # 2. expansion: add children if not terminal
            if not node.is_terminal and node.visit_count > 0:
                # note: aggressive expansion approach - expand only after first visit
                # alternative (standard alphazero): expand on first visit, remove visit_count check
                # if not node.is_terminal and not node.children:

                children = node.expand(
                    self.mcts_inference,
                    env,
                    self.config,
                )

                if children:
                    # select best child using UCB
                    non_terminal_children = [
                        c for c in children if not c.is_terminal
                    ]
                    if non_terminal_children:
                        node = max(
                            non_terminal_children,
                            key=lambda c: c.ucb_score(self.config.c_puct),
                        )
                        # note: aggressive approach - jump to child after expansion
                        # alternative (standard alphazero): keep 'node' as the expanded leaf
                        # remove the node reassignment, evaluate expanded leaf instead

                    else:
                        # all children are terminal, pick the one with the highes terminal reward
                        node = max(children, key=lambda c: c.terminal_reward)

            # 3. simulation: get value estimate by nn
            value = self._evaluate_node(node, env)

            # 4. backpropagation: update statistics in the tree upwards
            node.backup(value)

        if timeout_seconds is None or (time.time() - start_time) <= timeout_seconds:
            logger.info(f"mcts search completed, root visits: {self.root.visit_count}")
        else:
            logger.warning(f"mcts search timed out, completed {self.root.visit_count} simulations")

        assert self.root is not None
        return self.root

    def _select_leaf(self, root: Optional["MCTSNode"]) -> "MCTSNode":
        """traverse tree using UCB until leaf"""
        if root is None:
            raise ValueError("Root node cannot be None")
        node = root
        path = []

        while node.children and not node.is_terminal:
            child = node.select_child(self.config.c_puct)
            path.append(child.action or "root")
            node = child

        if path:
            logger.debug(f"selected path: {' -> '.join(path)}")
        return node

    def _evaluate_node(self, node: MCTSNode, env: TheoremEnv) -> float:
        """evaluate node value using neural network and environment feedback"""
        # if terminal, use environment reward
        if node.is_terminal:
            return node.terminal_reward

        # get state content for neural network eval
        state_content = node._get_state_content(env)
        if state_content == UNKNOWN_STATE:
            return 0.0

        proof_context = node.get_proof_context(env)

        # root node or nodes without actions should get a neutral value
        # this is normal for the root node which represents the initial state
        if not node.action or not node.execution_result:
            # for root node, return a neutral value and don't call estimate_value
            if node.parent is None:  # this is the root node
                # neutral value for initial state
                return 0.0
            else:
                # non-root nodes should have actions and execution results
                if not node.action:
                    logger.warning(
                        f"node {node} should have had an action (command), but doesn't"
                    )
                if not node.execution_result:
                    logger.warning(
                        f"node {node} should have had an execution result (updated subgoals), but doesn't"
                    )
                return 0.0

        # get nn value estimate; cached to avoid redundant calls
        if node.value_estimate_response is not None:
            # use cached value estimate
            return node.value_estimate_response["value"]

        response = self.mcts_inference.estimate_value(
            theorem_statement=proof_context["theorem_statement"],
            command=node.action,
            execution_result=node.execution_result or "",
            proof_history=proof_context["proof_history"],
            proof_mode=node.proof_mode,
        )

        if not response:
            logger.debug(f"no value estimate generated for node {node.state_id}")
            return 0.0

        # cache response for future use
        node.value_estimate_response = response
        return response["value"]

    def extract_all_trajectories(
        self, env: TheoremEnv
    ) -> tuple[list[MCTSNode], list[MCTSNode]]:
        """extract selective training trajectories"""
        if not self.root:
            logger.warning("no root found!")
            return [], []

        trajectories_policy: list[MCTSNode] = []
        trajectories_value: list[MCTSNode] = []
        self._collect_trajectories_recursive(
            self.root, env, trajectories_policy, trajectories_value
        )

        return trajectories_policy, trajectories_value

    def get_best_trajectory(self, env: TheoremEnv) -> list[MCTSNode]:
        """Get the best trajectory from the root node based on visit counts"""
        if not self.root or self.root.visit_count == 0:
            return []

        # follow the path of highest visit counts
        trajectory = []
        current = self.root

        while current and not current.is_terminal:
            if not current.children:
                break

            # select child with highest visit count
            best_child = max(
                current.children.values(), key=lambda child: child.visit_count
            )
            trajectory.append(best_child)
            current = best_child

        return trajectory

    def _collect_trajectories_recursive(
        self,
        node: MCTSNode,
        env: TheoremEnv,
        trajectories_policy: list[MCTSNode],
        trajectories_value: list[MCTSNode],
    ):
        """collect training data selectively
        - include all visited nodes (from search tree traversal)
        - include all terminal children (positive and negative examples)
        - include root node (initial state) even if unvisited
        - exclude non-terminal unvisited children (no eval info)

        returns: list[MCTSNode] for the policy model and value model
        """
        # collect current node if visited during search OR if it's the root node
        if node.visit_count <= 0 and not node.is_terminal and node.parent is not None:
            return

        if node.command_generation_response:
            trajectories_policy.append(node)
        if node.value_estimate_response:
            trajectories_value.append(node)

        for child in node.children.values():
            # include if: visited during search OR terminal (negative examples)
            if child.visit_count > 0 or child.is_terminal:
                self._collect_trajectories_recursive(
                    child,
                    env,
                    trajectories_policy,
                    trajectories_value,
                )


def get_max_depth(node: MCTSNode, current_depth: int = 0) -> int:
    """Calculate the maximum depth of an MCTS tree from the given node."""
    max_depth = current_depth
    for child in node.children.values():
        max_depth = max(max_depth, get_max_depth(child, current_depth + 1))
    return max_depth


if __name__ == "__main__":
    root = MCTSNode("init")
    child_a = MCTSNode("state_a", root, "auto")
    child_a.execution_result = "1 subgoal remaining"

    child_b = MCTSNode("state_b", root, "induction n")
    child_b.execution_result = "2 subgoals: base case, inductive step"

    grandchild_a1 = MCTSNode("state_a1", child_a, "simp")
    grandchild_b1 = MCTSNode("state_b1", child_b, "auto")

    # path reconstruction
    assert child_a.get_path_to_root() == [("auto", "1 subgoal remaining")]
    assert grandchild_a1.get_path_to_root() == [
        ("auto", "1 subgoal remaining"),
        ("simp", None),
    ]
    assert grandchild_b1.get_path_to_root() == [
        ("induction n", "2 subgoals: base case, inductive step"),
        ("auto", None),
    ]
