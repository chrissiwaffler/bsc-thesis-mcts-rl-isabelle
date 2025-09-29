import re


def extract_theorem_statement(isabelle_content: str) -> str:
    """extract theorem from isabelle theory file"""
    # try to find named theorem first: theorem NAME: "statement"
    match = re.search(
        r"(theorem\s+\w+:.*?)(?=\s+using|\s+by|\s+proof|\s+sorry)",
        isabelle_content,
        re.DOTALL,
    )

    if not match:
        # try to find unnamed theorem: theorem "statement"
        match = re.search(
            r"(theorem\s+\".*?\")(?=\s+using|\s+by|\s+proof|\s+sorry)",
            isabelle_content,
            re.DOTALL,
        )

    theorem = match.group(1).strip() if match else None
    assert theorem is not None, f"No theorem found in content: {isabelle_content}"
    return theorem


def extract_imports(isabelle_content: str) -> list[str]:
    """extract imports from Isabelle theory file"""
    # look for pattern: theory NAME imports IMPORTS begin
    # This handles multi-line imports
    match = re.search(
        r"theory\s+\w+\s+imports\s+(.*?)(?:\s+begin)", isabelle_content, re.DOTALL
    )

    if match:
        imports_str = match.group(1).strip()
        # split by whitespace and newlines, handle quotes if present
        imports = re.split(r"[\s\n]+", imports_str)
        # remove quotes and filter empty strings
        imports = [imp.replace('"', "") for imp in imports if imp.strip()]
        return imports

    # default to Main if no imports found
    return ["Main"]


def parse_isabelle_response(result: str) -> tuple[str, list[str]]:
    """extract mode and available facts from isabelle output"""
    mode = "prove"
    lemmas = []

    if "proof (state)" in result:
        mode = "state"
    elif "proof (prove)" in result or "goal (" in result or "subgoal" in result:
        mode = "prove"

    # extract lemmas in state mode
    if mode == "state":
        if "this:" in result:
            lemmas.append("this")

        fact_pattern = r'(\w+):\s*(?:"[^"]*"|.*?)(?=\n|$)'
        matches = re.findall(fact_pattern, result)
        for match in matches:
            if match not in ["this", "goal", "proof"]:
                lemmas.append(match)

        if "assms" in result:
            lemmas.append("assms")

    return mode, lemmas


def is_valid_isabelle_command(command: str) -> bool:
    """Check if a string looks like a valid Isabelle command"""
    if not command or not command.strip():
        return False

    # Remove any markdown or code block artifacts
    command = command.strip()

    # Reject if it contains markdown code blocks
    if "```" in command:
        return False

    # NEVER allow sorry - it's not a valid proof
    if "sorry" in command.lower():
        return False

    # Reject if it looks like prose or explanation
    prose_indicators = [
        "the problem",
        "we need",
        "this uses",
        "involves",
        "solving",
        "correctly",
        "properties",
        "expression",
        "rational number",
    ]
    command_lower = command.lower()
    if any(indicator in command_lower for indicator in prose_indicators):
        return False

    # Valid commands usually start with these
    valid_starts = [
        "apply",
        "by",
        "proof",
        "using",
        "have",
        "show",
        "thus",
        "hence",
        "moreover",
        "ultimately",
        "finally",
        "next",
        "assume",
        "obtain",
        "fix",
        "let",
        "done",
        "qed",
        "oops",
        ".",
        "..",
    ]

    # Check if it starts with a valid command
    for start in valid_starts:
        if command.startswith(start):
            return True

    # Also accept if it's a single identifier (like a lemma name)
    return bool(command.isidentifier())
