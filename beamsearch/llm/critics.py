import lilypad
from mirascope import llm, prompt_template

from .constants import CRITIC_MODEL, PROVIDER


@lilypad.trace(name="Evaluate Score")
def evaluate_command_in_context(
    command: str,
    proof_path: list[str],
    current_goals: str,
    theorem: str,
    result_after_command: str | None = None,
) -> float:
    """Evaluate how good a command is in the current proof context.

    This function calls an external LLM judge that considers:
    - The command being evaluated
    - The proof history (commands from root to current state)
    - The current proof goals before applying the command
    - The resulting subgoals after applying the command (if available)

    Args:
        command: The Isabelle command to evaluate
        proof_path: List of commands from root to current state
        current_goals: Current proof goals/state before command
        result_after_command: The resulting state after applying the command (optional)

    Returns:
        Score between 0.0 and 1.0 indicating command quality in context
    """
    # Add metadata for evaluation using a span
    with lilypad.span("Evaluation Metadata") as span:
        span.metadata(
            {
                "command": command,
                "proof_depth": len(proof_path),
                "has_result_after": result_after_command is not None,
            }
        )

    # reconstruct proof context from path
    proof_history = "\n".join(proof_path) if proof_path else "No previous commands"

    # determine current mode based on goals
    mode = "prove"
    if "proof (prove):" in current_goals:
        mode = "prove"
    elif "proof (state):" in current_goals:
        mode = "state"

    # call the LLM critic to evaluate the tactic
    response = _evaluate_tactic_critic(
        theorem=theorem,
        proof_state=current_goals,
        mode=mode,
        tactic=command,
        proof_history=proof_history,
        result_after_command=result_after_command,
    )

    # extract score from response
    score = response

    # Check if this tactic reduced the number of subgoals
    if result_after_command:
        import re

        # Extract subgoal counts
        before_match = re.search(r"goal \((\d+) subgoal", current_goals)
        after_match = re.search(r"goal \((\d+) subgoal", result_after_command)

        if before_match and after_match:
            before_count = int(before_match.group(1))
            after_count = int(after_match.group(1))
            if after_count < before_count:
                # Boost score for tactics that reduce subgoals
                reduction_bonus = 0.2 * (before_count - after_count) / before_count
                score = min(1.0, score + reduction_bonus)
                print(
                    f"CRITIC: {command} -> {score:.3f} (boosted for reducing {before_count} -> {after_count} subgoals)"
                )
            else:
                print(f"CRITIC: {command} -> {score:.3f}")
        else:
            print(f"CRITIC: {command} -> {score:.3f}")
    else:
        print(f"CRITIC: {command} -> {score:.3f}")

    # ensure score is in valid range
    return max(0.0, min(1.0, score))


@llm.call(provider=PROVIDER, model=CRITIC_MODEL, response_model=float)
@prompt_template(
    """You are a proof step critic evaluating the quality of an Isabelle tactic.

    Theorem: {theorem}
    Proof history: {proof_history}
    Current proof mode: {mode}
    Current proof state: {proof_state}
    Proposed tactic: {tactic}
    Result after applying tactic: {result_after_command}

    Evaluate how promising this tactic is for making progress toward completing the proof.
    Consider:
    - Does the tactic match the proof state structure and the remaining subgoals?
    - Is it appropriate for the current mode (prove/state)?
    - Does it avoid patterns that previously failed?
    - How likely is it to make meaningful progress?
    - If result is provided, did it actually make progress (reduce goals, simplify, etc.)?

    Return a single float between 0.0 and 1.0 representing the tactic quality.
    - 0.0-0.2: Very poor tactic, likely to fail or make no progress
    - 0.2-0.4: Weak tactic, might work but not optimal
    - 0.4-0.6: Average tactic, reasonable attempt
    - 0.6-0.8: Good tactic, likely to make progress
    - 0.8-1.0: Excellent tactic, highly likely to succeed or has succeeded
    """
)
def _evaluate_tactic_critic(
    theorem: str,
    proof_state: str,
    mode: str,
    tactic: str,
    proof_history: str,
    result_after_command: str | None,
): ...
