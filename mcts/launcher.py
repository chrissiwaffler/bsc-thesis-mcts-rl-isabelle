import logging
import os

from mcts.logging_utils import MCTSLogger
from mcts.main import training_loop
from mcts.shared_types import TrainingConfig
from mcts.utils import init_ray
from mcts.wandb_manager import finish_wandb, init_wandb

"""
adjust the configuration down below according to your hardware
and then launch the training with
uv run mcts/launcher.py
"""

if __name__ == "__main__":
    # Set PyTorch CUDA memory configuration to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger = MCTSLogger.get_logger("launcher")

    init_ray()

    MCTSLogger.setup_logging(
        log_level="info",
        console_output=True,
        file_output=True,
    )

    logger.info("Starting PPO training pipeline...")
    logging.getLogger("mcts_accelerate.mcts").setLevel(logging.INFO)

    # configure training
    config = TrainingConfig(
        policy_model_path="EleutherAI/llemma_7b",
        value_model_path="EleutherAI/llemma_7b",
        path_to_minif2f="./miniF2F-facebook/",
        training_iterations=3,  # Gradually increase from 1
        num_episodes_per_theorem=1,
        # this is rather just an upper bound for the isabelle connection
        max_mcts_steps=1024,
        # those next 2 are more important to set
        max_mcts_depth=16,
        num_mcts_simulations=64,
        num_expand_per_node=16,
        max_workers=32,
        # timeout per theorem rollout to prevent getting stuck (5 minutes)
        mcts_rollout_timeout_seconds=300,
        # INFO: adjust this to your available gpus
        # Current: 2xA100 setup (policy=GPU0, value=GPU1)
        # For 4xA100: consider "0,1" for policy and "2,3" for value with tensor_parallel_size=2
        # For 8xA100: "0,1,2,3" for policy and "4,5,6,7" for value with tensor_parallel_size=4
        policy_vllm_cuda_visible_devices="0,1",
        value_vllm_cuda_visible_devices="2,3",
        # Memory-optimized training settings for 4-GPU setup
        num_epochs=1,
        batch_size=16,  # Reduced from 32 to prevent memory issues
        # Hardware-specific adjustments for 4xH100 (80GB each)
        policy_vllm_args={
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.7,
            "trust_remote_code": True,
            "max_model_len": 4096,
        },
        value_vllm_args={
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.7,
            "trust_remote_code": True,
            "max_model_len": 4096,
        },
    )

    # initialize wandb early for better control, using config values
    wandb_initialized = init_wandb(
        project="mcts-training",
        name="main_training_run",
        config=config,
    )

    if wandb_initialized:
        logger.info("Wandb initialized successfully")
    else:
        logger.info("Wandb not available or disabled")
    result = training_loop(config)

    # handle return values
    if isinstance(result, tuple) and len(result) >= 2:
        policy_path, value_path = result[:2]
    else:
        policy_path, value_path = None, None

    logger.info("training completed! final models saved at:")
    logger.info(f"policy: {policy_path}")
    logger.info(f"value: {value_path}")

    # finish wandb run
    finish_wandb()
