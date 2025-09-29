from mirascope import llm, prompt_template

from .constants import MODEL, PROVIDER


@llm.call(provider=PROVIDER, model=MODEL)
@prompt_template(
    """You are reflecting on multiple proof failures to identify patterns and adapt strategy.

    Current theorem: {theorem}
    Current proof state: {proof_state}

    Failure history:
    {failure_summary}

    Detected pattern: {pattern_type}

    Based on these failures, provide strategic guidance:
    1. Is our current approach fundamentally flawed?
    2. What alternative proof strategies should we consider?
    3. What prerequisites or lemmas should we establish first?
    4. Should we switch proof modes or tactics?

    Suggest a new approach that addresses the root causes of these failures.
    """
)
def reflect_on_failures(
    theorem: str,
    proof_state: str,
    pattern_type: str,
    failure_summary: str,
): ...


# def self_ask_reasoning(
#     theorem: str,
#     proof_state: str,
#     mode: str,
#     available_lemmas: list[str],
#     failed_tactics: list[tuple[str, str]],
#     examples: str,
#     query: str,
#     failure_context: str,
#     proof_path: list[str],
# ) -> str:
#     """Dispatcher that calls the appropriate mode-specific function"""
#     if mode == "state":
#         response = _self_ask_reasoning_state_mode(
#             theorem=theorem,
#             proof_state=proof_state,
#             mode=mode,
#             available_lemmas=available_lemmas,
#             failed_tactics=failed_tactics,
#             examples=examples,
#             query=query,
#             failure_context=failure_context,
#             proof_path=proof_path,
#         )
#     else:  # prove mode
#         response = _self_ask_reasoning_prove_mode(
#             theorem=theorem,
#             proof_state=proof_state,
#             mode=mode,
#             available_lemmas=available_lemmas,
#             failed_tactics=failed_tactics,
#             examples=examples,
#             query=query,
#             failure_context=failure_context,
#             proof_path=proof_path,
#         )
#
#     return response
#
#
# @llm.call(provider=PROVIDER, model=MODEL)
# @prompt_template(
#     """You are an Isabelle theorem proving assistant in PROVE MODE using self-ask reasoning.
#     You are working with tactical proof methods (apply, by, using...by).
#
#     Examples:
#     {examples}
#
#     Query: {query}
#
#     {failure_context}
#
#     Think step by step, ask yourself questions, then provide the final answer.
#     Consider the failure patterns and adapt your approach accordingly.
#
#     IMPORTANT: Your final answer after "So the final answer is:" must be a SINGLE Isabelle command.
#     Do NOT use code blocks, markdown formatting, or multiple lines.
#
#     Valid commands in PROVE MODE:
#     - apply (rule ...) - Apply a specific rule
#     - apply (simp add: ...) - Simplify with additional rules
#     - apply (auto simp: ...) - Automatic proof with simplification
#     - apply blast, apply force - Powerful automated provers
#     - using ... apply ... - Use facts then prove
#     - using assms apply ... - Use the assumptions of the theorem
#     - proof - - Switch to structured proof mode (state mode)
#
#     NEVER use: have, show, fix, assume, thus, hence (these are for state mode)
#     """
# )
# def _self_ask_reasoning_prove_mode(
#     theorem: str,
#     proof_state: str,
#     mode: str,
#     available_lemmas: list[str],
#     failed_tactics: list[tuple[str, str]],
#     examples: str,
#     query: str,
#     failure_context: str,
#     proof_path: list[str],
# ): ...
#
#
# @llm.call(provider=PROVIDER, model=MODEL)
# @prompt_template(
#     """You are an Isabelle theorem proving assistant in STATE MODE using self-ask reasoning.
#     You are in structured proof mode where you build proofs step by step.
#
#     Examples:
#     {examples}
#
#     Query: {query}
#
#     {failure_context}
#
#     Think step by step, ask yourself questions, then provide the final answer.
#     Consider the failure patterns and adapt your approach accordingly.
#
#     IMPORTANT: Your final answer after "So the final answer is:" must be a SINGLE Isabelle command.
#     Do NOT use code blocks, markdown formatting, or multiple lines.
#
#     Valid commands in STATE MODE:
#     - have "statement" - State a fact to prove later (switches to prove mode)
#     - have "statement" by method - State and prove a fact immediately (stays in state mode)
#     - have label: "statement" - Labeled statement for later reference
#     - show ?thesis by method - Prove the main goal
#     - show "statement" by method - Prove a specific statement
#     - fix x y z - Introduce fixed variables
#     - assume label: "assumption" - Introduce an assumption
#     - thus "statement" by method - Therefore... (uses previous fact)
#     - hence "statement" by method - Hence... (uses previous fact)
#     - moreover, ultimately, finally - Chaining facts
#     - proof - Enter a nested prove mode
#     - qed, done - Complete the proof
#
#     NEVER use: apply (these are for prove mode)
#
#     Decision guide:
#     - Simple facts that follow directly: use "by simp" or "by auto"
#     - Complex statements needing work: omit the proof method to enter prove mode
#     - Need multiple steps: use labeled have statements
#     """
# )
# def _self_ask_reasoning_state_mode(
#     theorem: str,
#     proof_state: str,
#     mode: str,
#     available_lemmas: list[str],
#     failed_tactics: list[tuple[str, str]],
#     examples: str,
#     query: str,
#     failure_context: str,
#     proof_path: list[str],
# ): ...


@llm.call(provider=PROVIDER, model=MODEL)
@prompt_template(
    """You are an expert Isabelle proof assistant helping to find the next tactic.

    Theorem: {theorem}
    Current proof state: {proof_state}
    Mode: {mode} ({mode_description})
    
    Progress so far:
    {proof_history}
    
    Failed attempts:
    {failed_attempts}
    
    Available lemmas: {available_lemmas}
    
    {mode_specific_instructions}
    
    Based on the current state and failures, reason step by step about what tactic would make the most progress.
    Consider why previous attempts failed and adapt your approach.
    
    Your response MUST end with exactly this format:
    TACTIC: <your single isabelle command here>
    
    The command after TACTIC: must be executable Isabelle syntax with no explanation.
    """
)
def suggest_next_tactic(
    theorem: str,
    proof_state: str,
    mode: str,
    mode_description: str,
    proof_history: str,
    failed_attempts: str,
    available_lemmas: str,
    mode_specific_instructions: str,
): ...
