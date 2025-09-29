import re

from ..core import ProofGraph, ProofState, QIsabelleServerError, QIsabelleSession


def get_proof_path_to_current(graph: ProofGraph, current_state: str) -> list[str]:
    """Get all commands from root to current state"""
    path = []
    current = current_state

    while current:
        state = graph.get_state(current)
        if state.parent:  # Don't include the root node's command
            path.append(state.command)
        current = state.parent

    # Reverse to get root to current order
    path.reverse()
    return path


def reconstruct_proof(
    graph: ProofGraph,
    final_state: str,
) -> tuple[list[str], str]:
    """reconstruct proof path from final winning state back to root"""

    # Use the existing function to get the proof path
    path = get_proof_path_to_current(graph, final_state)

    # format as proof
    if not path:
        return [], ""

    # simple formatting - just join with newlines and indent
    proof_lines = []
    for tactic in path:
        if tactic.startswith("proof"):
            proof_lines.append(tactic)
        else:
            proof_lines.append(f"  {tactic}")

    proof_text = "\n".join(proof_lines)

    return path, proof_text


def verify_complete_proof(
    session: QIsabelleSession,
    theorem: str,
    proof_text: str,
) -> tuple[bool, str]:
    """verify the complete proof is valid"""

    # sorry not allowed
    if "sorry" in proof_text.lower():
        return False, "Proof contains 'sorry'"

    # create fresh start for verification
    try:
        # execute theorem + proof together
        complete_code = f"{theorem}\n{proof_text}"

        is_done, result = session.execute(
            session.initial_state_name, complete_code, "verification_state"
        )

        if is_done:
            return True, "Proof verified successfully!"
        else:
            return False, f"Proof incomplete: {result}"

    except QIsabelleServerError as e:
        error_msg = str(e).split("Traceback:")[0].strip()
        return False, f"Proof verification failed: {error_msg}"


def finalize_proof(
    session: QIsabelleSession,
    graph: ProofGraph,
    theorem: str,
    final_state: str,
) -> tuple[bool, str, str]:
    """reconstruct and verify the complete proof"""

    print(f">>reconstructing proof from {final_state}...")

    # reconstruct
    tactics, proof_text = reconstruct_proof(graph, final_state)
    print(f">>reconstructed proof with {len(tactics)} steps")

    for i, tactic in enumerate(tactics, 1):
        print(f"  {i}. {tactic}")

    print(f">>complete proof:\n{proof_text}")

    # verify
    print(">>verifying complete proof...")
    is_valid, message = verify_complete_proof(session, theorem, proof_text)

    if is_valid:
        print(f"VALID: {message}")
    else:
        print(f"INVALID: {message}")

    return is_valid, proof_text, message


def would_be_duplicate_statement(
    statement: str, parent_state: ProofState, graph: ProofGraph
) -> bool:
    """Check if a statement would be a duplicate of an existing SUCCESSFUL child statement.

    This checks:
    1. Direct children of the parent state that succeeded
    2. Normalizes statements to catch variations
    3. Considers chaining keywords (moreover, ultimately) as potential duplicates

    Important: We only check against successful children because failed statements
    might have had syntax/type errors in the proof part that could work when
    we try just the bare statement.
    """

    def normalize_statement(command: str) -> str:
        """Normalize a statement for comparison purposes.

        This handles variations like:
        - Whitespace differences
        - Optional labels
        - Different keywords that mean the same thing in context

        Returns a normalized form for duplicate detection.
        """
        # Remove extra whitespace
        normalized = " ".join(command.split())

        # Extract the core statement (remove labels if present)
        # Pattern: optional_keyword [label:] "statement"
        label_pattern = r"^(\w+)\s+(\w+):\s*(.*)$"
        match = re.match(label_pattern, normalized)
        if match:
            keyword, label, rest = match.groups()
            normalized = f"{keyword} {rest}"

        return normalized

    normalized_new = normalize_statement(statement)

    # Extract just the quoted content for comparison
    quote_pattern = r'"([^"]+)"'
    new_quote_match = re.search(quote_pattern, normalized_new)
    if not new_quote_match:
        return False
    new_content = new_quote_match.group(1)

    # Check all successful children of the parent
    for child_name in parent_state.children:
        child_state = graph.get_state(child_name)
        child_command = child_state.command

        # Skip failed children - we want to allow retrying failed statements
        # without their proofs in case the statement itself is valid
        if child_command in [t[0] for t in parent_state.failed_tactics]:
            continue

        normalized_child = normalize_statement(child_command)

        # Check if the quoted content matches
        child_quote_match = re.search(quote_pattern, normalized_child)
        if child_quote_match and child_quote_match.group(1) == new_content:
            # Same statement content found
            return True

        # Also check if child is the same statement but with proof
        child_without_proof = extract_statement_without_proof(child_command)
        if child_without_proof:
            normalized_child_without = normalize_statement(child_without_proof)
            if normalized_child_without == normalized_new:
                return True

    return False


def extract_statement_without_proof(command: str) -> str | None:
    """Extract a statement command without its proof method.

    Examples:
    - 'have "P" by simp' -> 'have "P"'
    - 'have label: "P" by auto' -> 'have label: "P"'
    - 'show ?thesis by blast' -> 'show ?thesis'
    - 'thus "P" by (simp add: assms)' -> 'thus "P"'
    - 'hence "P" using foo by auto' -> 'hence "P" using foo'

    Returns None if the command doesn't have a proof method to remove.
    """
    # Keywords that can have statements with proofs
    statement_keywords = [
        "have",
        "show",
        "thus",
        "hence",
        "moreover",
        "ultimately",
        "finally",
    ]

    # Check if command starts with one of these keywords
    command_lower = command.strip().lower()
    if not any(command_lower.startswith(kw) for kw in statement_keywords):
        return None

    # Look for ' by ' pattern (with spaces to avoid matching strings)
    by_pattern = r"\s+by\s+"
    match = re.search(by_pattern, command, re.IGNORECASE)

    if match:
        # Extract everything before ' by '
        statement_part = command[: match.start()].strip()
        return statement_part

    return None
