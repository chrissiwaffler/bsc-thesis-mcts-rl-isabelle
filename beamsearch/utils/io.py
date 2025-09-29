import json
import random
from datetime import datetime
from pathlib import Path


def load_minif2f_files(
    base_path="miniF2F/isabelle",
    # test or split
    split="test",
    limit=100,
    shuffle=True,
):
    """Load miniF2F problem files"""
    split_dir = Path(base_path) / split

    if not split_dir.exists():
        print(f"Error: {split_dir} not found")
        return []

    # get all .thy files
    thy_files = list(split_dir.glob("*.thy"))

    if shuffle:
        random.seed(13)
        random.shuffle(thy_files)

    # limit number of files
    thy_files = thy_files[:limit]

    problems = []
    for thy_file in thy_files:
        with open(thy_file) as f:
            content = f.read()

        problems.append(
            {
                "name": thy_file.stem,
                "path": str(thy_file),
                "content": content,
            }
        )

    print(f"Loaded {len(problems)} problems from {split_dir}")
    return problems


def create_results_dir(method_name: str):
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/minif2f_run_{method_name}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # create subdirs
    (results_dir / "successful").mkdir(exist_ok=True)
    (results_dir / "failed").mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)

    return results_dir


def save_result(results_dir, problem_name, success, proof_text, stats, elapsed_time):
    """Save individual result"""
    result = {
        "problem": problem_name,
        "success": success,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
    }

    # save proof file
    if success:
        proof_file = results_dir / "successful" / f"{problem_name}.thy"
    else:
        proof_file = results_dir / "failed" / f"{problem_name}.thy"

    with open(proof_file, "w") as f:
        f.write(f"(* Generated proof for {problem_name}*)\n")
        f.write(f"(* Success: {success} *)\n")
        f.write(f"(* Time: {elapsed_time:.2f}s *)\n\n")
        f.write(proof_text)

    # save metadata
    meta_file = results_dir / "logs" / f"{problem_name}_meta.json"
    with open(meta_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
