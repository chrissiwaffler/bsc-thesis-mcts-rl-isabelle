import re

from mirascope import llm, prompt_template

from ..core import ProofGraph, QIsabelleSession
from ..data import few_shot_examples
from ..tools import (
    IsabelleExecuteCommand,
    IsabelleSuccessResult,
)
from ..utils import (
    ProofTreeVisualizer,
    get_proof_path_to_current,
    parse_isabelle_response,
)
from .constants import MODEL, PROVIDER
from .models import ActionObservation, NodeReasoning
from .prompts import suggest_next_tactic


@llm.call(provider=PROVIDER, model=MODEL, response_model=ActionObservation)
@prompt_template(
    """You are using ReAct (Reasoning + Acting) for Isabelle theorem proving.

    Current situation:
    Theorem: {theorem}
    Current state: {current_state}
    Previous actions: {action_history_str}

    Based on the reasoning: {reasoning_conclusion}

    Decide on the next action to take. Think step by step about:
    1. What the current situation tells us
    2. What action would make progress
    3. What we expect to observe
    """
)
def _react_step(
    theorem: str,
    current_state: str,
    action_history: list[ActionObservation],
    reasoning_conclusion: str,
    action_history_str: str,
): ...


def explore_with_reasoning(
    session: QIsabelleSession,
    graph: ProofGraph,
    viz: ProofTreeVisualizer,
    theorem: str,
    current_state_name: str,
    max_attempts: int = 16,
) -> list[str]:
    """explore from node using self-ask reasoning and ReAct"""

    state = graph.get_state(current_state_name)
    if state.is_done:
        print("Proof complete!")
        return []

    # parse current state
    mode, lemmas = parse_isabelle_response(state.result)
    state.mode = mode
    state.available_lemmas = lemmas

    print(f"\nExploring from: {current_state_name}")
    print(f"Current goals: {state.result[:150]}...")

    # step 1: self-ask reasoning
    print("\nSelf-ask reasoning phase...")

    # Get proof path for this state
    proof_path = get_proof_path_to_current(graph, current_state_name)

    # Build the query string
    query_parts = [
        f"Theorem: {theorem}",
    ]

    # Add proof steps if any
    if proof_path:
        query_parts.append("Proof steps so far:")
        for i, cmd in enumerate(proof_path, 1):
            query_parts.append(f"  {i}. {cmd}")

    query_parts.extend(
        [
            f"Current goals: {state.result}",
            f"Current mode: {mode}",
        ]
    )

    if mode == "state":
        query_parts[0] = f"Current mode: {mode}"
        if "proof (state)" in state.result:
            query_parts.append("proof (state)")

    if lemmas:
        query_parts.append(f"Available lemmas: {', '.join(lemmas)}")

    if state.failed_tactics:
        failed_str = str(
            [
                (t, e[:50] + "..." if len(e) > 50 else e)
                for t, e in state.failed_tactics[-5:]
            ]
        )
        query_parts.append(f"Failed tactics: {failed_str}")

    query = "\n".join(query_parts)

    # Format examples
    examples_str = "\n\n".join(
        [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in few_shot_examples]
    )

    # Build failure context from error messages
    failure_context = (
        "\n".join(state.failure_error_messages[-5:])
        if state.failure_error_messages
        else ""
    )

    print("\nLLM API Call: self_ask_reasoning (temperature=N/A - using default)")
    print("\nSELF-ASK QUERY:")
    print("=" * 80)
    print(query)
    print("=" * 80 + "\n")

    # Build mode description and instructions
    mode_description = (
        "structured proof mode" if mode == "state" else "tactical proof mode"
    )

    # Get mode instructions from agent's method (simplified version)
    if mode == "state":
        mode_instructions = """Valid commands in STATE MODE:
- have "statement" - State a fact to prove later
- have "statement" by method - State and prove immediately
- show ?thesis by method - Prove the main goal
- show "statement" by method - Prove a specific statement
- proof - - Enter nested prove mode

NEVER use 'apply' commands in state mode."""
    else:
        mode_instructions = """Valid commands in PROVE MODE:
- apply simp, apply auto, apply blast - Basic automation
- apply (simp add: lemma) - Simplify with specific lemmas
- using assms apply method - Use assumptions
- apply (induct x) - Induction on variable x
- proof - - Switch to structured proof mode

NEVER use 'have' or 'show' commands in prove mode."""

    # Format failed attempts
    if not state.failed_tactics:
        failed_attempts = "No failed attempts yet."
    else:
        lines = []
        for tactic, error in state.failed_tactics[-5:]:  # Last 5 failures
            error_short = error[:100] + "..." if len(error) > 100 else error
            lines.append(f"- {tactic}\n  Error: {error_short}")
        failed_attempts = "\n".join(lines)

    # Format available lemmas
    available_lemmas = ", ".join(lemmas) if lemmas else "none"

    # Format proof history
    if not proof_path:
        proof_history = "No steps yet - this is the first tactic."
    else:
        lines = []
        for i, cmd in enumerate(proof_path, 1):
            lines.append(f"{i}. {cmd}")
        proof_history = "\n".join(lines)

    reasoning_response = suggest_next_tactic(
        theorem=theorem,
        proof_state=state.result,
        mode=mode,
        mode_description=mode_description,
        proof_history=proof_history,
        failed_attempts=failed_attempts,
        available_lemmas=available_lemmas,
        mode_specific_instructions=mode_instructions,
    )

    # Extract suggested tactics from the response
    content = reasoning_response.content

    # Print full self-ask reasoning
    print(f"\nSelf-ask reasoning:\n{content}\n")

    suggested_tactics = []

    # Look for the standard self-ask format
    if "So the final answer is:" in content:
        final_tactic = content.split("So the final answer is:")[-1].strip()
        suggested_tactics.append(final_tactic)

    # Also extract any other tactics mentioned in the reasoning
    lines = content.strip().split("\n")
    for line in lines:
        if (
            "tactic" in line.lower()
            or "apply" in line
            or "proof" in line
            or "by" in line
        ):
            # Extract potential tactics from the line
            tactic_patterns = [
                r"apply\s*\([^)]+\)",
                r"by\s+\w+",
                r"proof\s*\([^)]+\)",
                r"proof\s*-",
                r"using\s+.*?\s+by\s+\w+",
                r'thus\s+"[^"]*"\s+by\s+\w+',
                r'have\s+"[^"]*"\s+by\s+\w+',
                r"show\s+\?thesis",
            ]
            for pattern in tactic_patterns:
                matches = re.findall(pattern, line)
                suggested_tactics.extend(matches)

    # Remove duplicates and clean up
    suggested_tactics = list(dict.fromkeys(suggested_tactics))[:5]  # Keep top 5 unique

    print(f"self-ask suggested tactics: {suggested_tactics}")

    state.reasoning = NodeReasoning(react_history=[], final_decision="")

    successful_children = []

    # Track commands already tried from this state
    tried_commands = set()

    # Add already failed tactics to tried commands
    for failed_tactic, _ in state.failed_tactics:
        tried_commands.add(failed_tactic)

    # Add already successful children to tried commands
    for child_name in state.children:
        child_state = graph.get_state(child_name)
        tried_commands.add(child_state.command)

    # step 2: react loop for trying tactics
    for i, suggested_tactic in enumerate(suggested_tactics[:max_attempts]):
        if suggested_tactic in tried_commands:
            print(f"Skipping already tried: {suggested_tactic}")
            continue

        # Mark as tried
        tried_commands.add(suggested_tactic)

        print(f"\nAttempt {i + 1}: {suggested_tactic}")

        # react step
        # Format action history
        action_history_str = "\n".join(
            [
                f"- Thought: {a.thought}\n  Action: {a.action}"
                for a in state.reasoning.react_history
            ]
        )

        print("\nLLM API Call: react_step (temperature=N/A - using default)")
        action_obs = _react_step(
            theorem=theorem,
            current_state=state.result,
            action_history=state.reasoning.react_history,
            reasoning_conclusion=content,
            action_history_str=action_history_str,
        )

        print(f"Thought: {action_obs.thought}")
        print(f"Action: {action_obs.action}")

        # execute the tactic
        child = graph.add_state(suggested_tactic, parent=current_state_name)

        viz.update(graph, f"trying: {suggested_tactic}")

        tool = IsabelleExecuteCommand(
            current_state=current_state_name,
            command=suggested_tactic,
            new_state=child.state_name,
            session=session,
        )
        result = tool.call()

        if isinstance(result, IsabelleSuccessResult):
            # success case
            is_done = result.is_done
            observation = f"Tactic executed successfully. Done: {is_done}. Result: {result.result}"

            action_obs.observation = observation
            state.reasoning.react_history.append(action_obs)

            graph.update_result(child.state_name, is_done, result.result)
            successful_children.append(child.state_name)
            print("Success!")

            if is_done:
                print("PROOF COMPLETE!")
                break
        else:
            # failure case - result is IsabelleErrorResult
            error_msg = result.error
            observation = f"Tactic failed: {error_msg}"

            action_obs.observation = observation
            state.reasoning.react_history.append(action_obs)

            graph.add_failed_tactic(current_state_name, suggested_tactic, error_msg)
            state.children.remove(child.state_name)
            graph.G.remove_node(child.state_name)
            print(f"Failed: {error_msg[:80]}...")

    # final decision
    state.reasoning.final_decision = (
        f"explored {len(successful_children)} successful tactics"
    )

    print(
        f"\nSummary: {len(successful_children)} valid children, {len(state.failed_tactics)} failed"
    )
    viz.update(graph, f"explored: {len(successful_children)} paths")

    return successful_children
