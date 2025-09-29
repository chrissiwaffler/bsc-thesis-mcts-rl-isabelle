from dataclasses import dataclass
from typing import Any, TypedDict


class ValueEstimateResponse(TypedDict):
    value: float
    full_prompt: str
    full_response: str
    # per-token logprobs
    logprobs: list[float]


class CommandGenerationResponse(TypedDict):
    command: str
    thinking: str
    full_prompt: str
    full_response: str
    # per-token logprobs
    logprobs: list[float]


@dataclass
class TrainingConfig:
    policy_model_path: str
    value_model_path: str
    path_to_minif2f: str
    # how many iterations of mcts rollouts for the training data & finetuning the model
    training_iterations: int
    # maximum total number of steps to generate in the mcts search
    max_mcts_steps: int
    # maximum depth of mcts tree
    max_mcts_depth: int
    # number of mcts rollouts for one theorem
    num_episodes_per_theorem: int
    # number of mcts simulations per episode
    num_mcts_simulations: int
    # number of new commands per node expand
    num_expand_per_node: int
    # number of theorem envs to be run in parallel
    max_workers: int

    # timeout per theorem rollout in seconds (prevents getting stuck on single theorems)
    mcts_rollout_timeout_seconds: int = 300  # 5 minutes default

    # set the environment variable CUDA_VISIBLE_DEVICES before launching value and policy model
    policy_vllm_cuda_visible_devices: str = ""
    value_vllm_cuda_visible_devices: str = ""

    # arguments to pass down to the vllm server when starting the policy model
    policy_vllm_args: dict[str, Any] | None = None
    # arguments to pass down to the vllm server when starting the value model
    value_vllm_args: dict[str, Any] | None = None

    temperature_policy: float = 0.8
    temperature_value: float = 0.3

    # training settings
    # number of training epochs
    num_epochs: int = 2

    # learning rate for training
    # For 4xH100: increased to 1e-4 for faster learning
    learning_rate: float = 1e-4

    # batch size for training (effective = 16 * 8 = 128 with grad accum)
    # For 4xH100: set to 16 for effective batch size 256
    batch_size: int = 16

    # huggingface upload settings
    # whether to upload models to hf hub
    upload_to_hf: bool = True
    hf_username: str = "chrissi"
    hf_repo_name: str = "isabelle-mcts-rl"
    # hf token (can also use env var)
    hf_token: str | None = None

    # evaluation settings
    eval_frequency: int = 1  # evaluate every iteration for monitoring
    eval_max_theorems: int = 50  # more comprehensive evaluation
    eval_mcts_simulations: int = 50  # balanced evaluation quality
    # save proofs in final evaluation
    final_eval_save_proofs: bool = True
    # output directory for proofs
    final_eval_output_dir: str = "./final_evaluation_proofs"

    # sledgehammer settings
    # whether to use sledgehammer for automation during the final evaluation; for training it's always off for now
    use_sledgehammer: bool = True
