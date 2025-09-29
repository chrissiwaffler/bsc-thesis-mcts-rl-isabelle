import gc
import os
import re
import time
from contextlib import suppress

import ray
import torch
from accelerate import Accelerator

from mcts.logging_utils import MCTSLogger


def init_ray():
    if not ray.is_initialized():
        # set multiprocessing start method to 'spawn' for CUDA compatibility
        import multiprocessing as mp

        if mp.get_start_method() != "spawn":
            mp.set_start_method("spawn", force=True)

        ray.init(
            runtime_env={
                "excludes": [
                    # git directories
                    "**/.git",
                    "**/.git/**",
                    # large model files and caches
                    "**/.cache",
                    "**/__pycache__",
                    "**/*.pyc",
                    "**/*.pyo",
                    "**/*.pyd",
                    # dataset files
                    "**/miniF2F-facebook/**",
                    "**/datasets/**",
                    # model weights and checkpoints
                    "**/checkpoints/**",
                    "**/models/**",
                    "**/*.pth",
                    "**/*.pt",
                    "**/*.bin",
                    "**/*.safetensors",
                    # documentation and examples
                    "**/docs/**",
                    "**/examples/**",
                    "**/README*",
                    "**/*.md",
                    # build artifacts
                    "**/build/**",
                    "**/dist/**",
                    "**/node_modules/**",
                    "**/.venv/**",
                    "**/venv/**",
                    "**/env/**",
                    # specific large directories
                    "**/atropos/**",
                    "**/prime-rl/**",
                    "**/xformers/**",
                    "**/.pytest_cache/**",
                    "**/.mypy_cache/**",
                    "**/.ruff_cache/**",
                    # log files
                    "**/*.log",
                    "**/logs/**",
                    "**/outputs/**",
                    # temporary files
                    "**/tmp/**",
                    "**/temp/**",
                    "**/.tmp/**",
                    "**/.temp/**",
                    "**/afp_*/**",
                    "**/dockerheaps/**"
                    # ide files
                    "**/.vscode/**",
                    "**/.idea/**",
                    "**/*.swp",
                    "**/*.swo",
                    # os files
                    "**/.DS_Store",
                    "**/Thumbs.db",
                    # docker files
                    "**/Dockerfile",
                    "**/docker-compose*.yml",
                    "**/.dockerignore",
                    # virtual environment files
                    "**/Pipfile*",
                    "**/poetry.lock",
                    "**/conda-lock.yml",
                    "**/environment.yml",
                    # coverage files
                    "**/.coverage",
                    "**/htmlcov/**",
                    "**/coverage.xml",
                    # jupyter notebooks
                    "**/*.ipynb",
                    "**/.ipynb_checkpoints/**",
                    # large data files
                    "**/*.zip",
                    "**/*.tar",
                    "**/*.gz",
                    "**/*.7z",
                    # additional large directories that might exist
                    "**/data/**",
                    "**/raw_data/**",
                    "**/processed_data/**",
                    "**/training_data/**",
                    "**/test_data/**",
                    "**/validation_data/**",
                    "**/saved_models/**",
                    "**/model_outputs/**",
                    "**/experiment_results/**",
                    "**/wandb/**",
                    "**/tensorboard/**",
                    "**/mlruns/**",
                    "**/ray_results/**",
                    "**/ray_session/**",
                    "**/ray_logs/**",
                ],
            },
        )


def extract_theorem_statement(isabelle_content: str) -> str:
    """extract theorem from isabelle theory file"""
    # try to find named theorem first: theorem NAME: "statement"
    # look for theorem ending with sorry, proof, by, or using (can be on next line)
    match = re.search(
        r"(theorem\s+\w+\s*:.*?)(\n\s*(?:sorry|proof|by|using)|\s+(?:sorry|proof|by|using))",
        isabelle_content,
        re.DOTALL,
    )

    if not match:
        # try to find unnamed theorem: theorem "statement"
        match = re.search(
            r'(theorem\s+".*?")(\n\s*(?:sorry|proof|by|using)|\s+(?:sorry|proof|by|using))',
            isabelle_content,
            re.DOTALL,
        )

    if match:
        theorem = match.group(1).strip()
        return theorem

    # fallback: try to extract just the theorem line without the ending keyword
    lines = isabelle_content.split("\n")
    for line in lines:
        if line.strip().startswith("theorem"):
            return line.strip()

    raise AssertionError(f"No theorem found in content: {isabelle_content}")


def extract_imports(isabelle_content: str) -> list[str]:
    """extract imports from Isabelle theory file"""
    # look for pattern: theory NAME imports IMPORTS begin
    # this handles multi-line imports
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


def cleanup_memory(
    accelerator: Accelerator | None = None, message: str = "Memory cleanup completed"
) -> None:
    """
    Comprehensive memory cleanup for both GPU and general Python memory.

    Args:
        accelerator: Optional accelerator object for logging
        message: Message to log after cleanup
    """
    # force garbage collection
    gc.collect()

    if torch.cuda.is_available():
        # get current memory state before cleanup
        if accelerator:
            with suppress(Exception):
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                accelerator.print(
                    f"[main] Before cleanup - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
                )

        # empty cuda cache
        torch.cuda.empty_cache()

        # synchronize to ensure all operations are complete
        torch.cuda.synchronize()

        # additional cleanup for fragmented memory
        if hasattr(torch.cuda, "memory_snapshot"):
            with suppress(Exception):
                torch.cuda.memory_snapshot()

        # small delay to ensure cleanup completes
        time.sleep(0.5)

        # final garbage collection
        gc.collect()

        # log memory state after cleanup
        if accelerator:
            with suppress(Exception):
                memory_allocated_after = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved_after = torch.cuda.memory_reserved() / 1024**3  # GB
                accelerator.print(
                    f"[main] After cleanup - Allocated: {memory_allocated_after:.2f}GB, Reserved: {memory_reserved_after:.2f}GB"
                )

        if accelerator:
            accelerator.print(f"[main] {message}")
    else:
        if accelerator:
            accelerator.print(f"[main] {message} (no CUDA available)")


def save_proof_to_thy_file(proof_data: dict, output_dir: str) -> None:
    """Save a successful proof to a .thy file
    - proof_data:
        - theorem_name
        - theorem_content
        - commands: list[str]
    """
    theorem_name = proof_data["theorem_name"]
    theorem_content = proof_data["theorem_content"]
    commands = proof_data["commands"]

    # create filename
    filename = f"{theorem_name}_proof.thy"
    filepath = os.path.join(output_dir, filename)

    logger = MCTSLogger.get_logger("proof_saving")

    try:
        # extract components using existing utility functions
        theorem_statement = extract_theorem_statement(theorem_content)
        imports = extract_imports(theorem_content)

        # extract theory name from the theory line
        theory_match = re.search(r"theory\s+(\w+)", theorem_content)
        theory_name_from_content = (
            theory_match.group(1) if theory_match else theorem_name
        )

        with open(filepath, "w", encoding="utf-8") as f:
            # write header comment
            f.write(f"(* Generated proof for {theorem_name} *)\n\n")

            # write theory declaration with imports
            f.write(
                f"theory {theory_name_from_content}_proof imports {' '.join(imports)}\n"
            )
            f.write("begin\n\n")

            # write theorem statement
            f.write(f"{theorem_statement}\n")

            # write proof
            for cmd in commands:
                if cmd.strip():
                    f.write(f"  {cmd.strip()}\n")
            # end theory
            f.write("end\n")

        logger.info(f"Saved proof for {theorem_name} to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save proof for {theorem_name}: {e}")
