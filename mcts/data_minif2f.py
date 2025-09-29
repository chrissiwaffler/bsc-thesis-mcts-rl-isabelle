from pathlib import Path
from typing import TypedDict


class TheoremData(TypedDict):
    name: str
    content: str


def load_dataset(path_to_minif2f: str, split: str) -> list[TheoremData]:
    """
    load minif2f dataset from the specified path and split.

    args:
        path_to_minif2f: path to the minif2f directory containing isabelle subfolder
        split: either 'test' or 'valid' specifying which split to load

    returns:
        list of theoremdata objects containing theorem names and content

    raises:
        value error: if split is not test or valid
        file not found error: if the specified directory doesn't exist
    """
    # check split validity
    if split not in ["test", "valid"]:
        raise ValueError("split must be either 'test' or 'valid'")

    # construct the path to the isabelle split directory
    isabelle_path = Path(path_to_minif2f) / "isabelle" / split

    if not isabelle_path.exists():
        raise FileNotFoundError(f"directory not found: {isabelle_path}")

    theorems = []

    # load all .thy files in the directory
    for thy_file in isabelle_path.glob("*.thy"):
        # filename without .thy extension
        theorem_name = thy_file.stem

        try:
            with open(thy_file, encoding="utf-8") as f:
                content = f.read()

            theorems.append(TheoremData(name=theorem_name, content=content))
        except Exception as e:
            print(f"warning: failed to read {thy_file}: {e}")
            continue

    return theorems
