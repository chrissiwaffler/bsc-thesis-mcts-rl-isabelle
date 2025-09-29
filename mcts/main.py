import json
import os
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import upload_folder
from transformers import PreTrainedModel

from mcts.base_env import TheoremEnv
from mcts.data_mcts import ModelType
from mcts.data_minif2f import TheoremData, load_dataset
from mcts.inference_loader import ModelInferenceManager
from mcts.inference_mcts import MCTSInference
from mcts.logging_utils import MCTSLogger
from mcts.search import MCTSConfig, MCTSNode, MCTSSearch, get_max_depth
from mcts.shared_types import TrainingConfig
from mcts.trainer_ppo import PPOConfig, train_ppo
from mcts.utils import cleanup_memory, save_proof_to_thy_file
from mcts.wandb_manager import (
    log_evaluation_metrics,
    log_final_evaluation_metrics,
    log_mcts_metrics,
)

logger = MCTSLogger.get_logger("main")


# load environment variables from .env file
def load_env_file():
    """load environment variables from .env file if it exists"""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    if key and value:
                        os.environ[key.strip()] = value.strip()


# load .env file at import time
load_env_file()


def collect_mcts_metrics(
    traj_policy: list[MCTSNode], traj_value: list[MCTSNode]
) -> dict[str, Any]:
    """collect mcts-specific metrics for wandb logging"""

    metrics = {}

    if not traj_policy and not traj_value:
        return metrics

    # tree statistics; wandb_manager will add mcts/ prefix
    visit_counts = [node.visit_count for node in traj_policy if node.visit_count > 0]
    if visit_counts:
        metrics["avg_visit_count"] = float(np.mean(visit_counts))
        metrics["max_visit_count"] = float(np.max(visit_counts))
        metrics["min_visit_count"] = float(np.min(visit_counts))

    # value statistics
    values = [node.average_value for node in traj_policy if node.visit_count > 0]
    if values:
        metrics["avg_value"] = float(np.mean(values))
        metrics["std_value"] = float(np.std(values))
        metrics["max_value"] = float(np.max(values))
        metrics["min_value"] = float(np.min(values))

    # terminal states
    terminal_nodes = [node for node in traj_policy if node.is_terminal]
    metrics["num_terminal_nodes"] = len(terminal_nodes)
    metrics["terminal_rate"] = (
        len(terminal_nodes) / len(traj_policy) if traj_policy else 0.0
    )

    # terminal rewards
    if terminal_nodes:
        terminal_rewards = [node.terminal_reward for node in terminal_nodes]
        metrics["avg_terminal_reward"] = float(np.mean(terminal_rewards))
        metrics["successful_proofs"] = sum(1 for r in terminal_rewards if r > 0.5)
        metrics["success_rate"] = metrics["successful_proofs"] / len(terminal_rewards)

    # tree depth
    depths = []
    for node in traj_policy:
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        if depth > 0:
            depths.append(depth)

    if depths:
        metrics["avg_tree_depth"] = float(np.mean(depths))
        metrics["max_tree_depth"] = float(np.max(depths))

    # exploration metrics
    total_children = sum(len(node.children) for node in traj_policy)
    metrics["avg_children_per_node"] = (
        total_children / len(traj_policy) if traj_policy else 0.0
    )

    # value estimation metrics
    if traj_value:
        metrics["num_value_estimates"] = len(traj_value)

        # value estimates from the model
        value_estimates = [
            node.value_estimate_response["value"]
            for node in traj_value
            if node.value_estimate_response
        ]
        if value_estimates:
            metrics["avg_value_estimate"] = float(np.mean(value_estimates))
            metrics["std_value_estimate"] = float(np.std(value_estimates))
            metrics["max_value_estimate"] = float(np.max(value_estimates))
            metrics["min_value_estimate"] = float(np.min(value_estimates))

        # log probabilities for value estimation
        value_logprobs = [
            node.value_estimate_response["logprobs"]
            for node in traj_value
            if node.value_estimate_response
        ]
        if value_logprobs:
            # Flatten all logprobs since they have variable lengths
            all_logprobs = []
            for logprob_list in value_logprobs:
                if isinstance(logprob_list, list):
                    all_logprobs.extend(logprob_list)
                else:
                    all_logprobs.append(logprob_list)

            if all_logprobs:
                metrics["avg_value_logprob"] = float(np.mean(all_logprobs))
                metrics["std_value_logprob"] = float(np.std(all_logprobs))

    return metrics


def single_mcts_rollout(
    theorem_data: dict,
    episode_id: int,
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
) -> tuple[list[MCTSNode], list[MCTSNode]]:
    """single mcts run - one theorem, one episode"""
    logger = MCTSLogger.get_logger("single_mcts")
    mcts_inference = MCTSInference(
        inference_manager,
        config.temperature_policy,
        config.temperature_value,
    )

    theorem_name = theorem_data["name"]
    theorem_content = theorem_data["content"]

    env = None
    try:
        env = TheoremEnv(
            theorem_name=theorem_name,
            theorem_content=theorem_content,
            max_steps=config.max_mcts_steps,
            # disable sledgehammer during training
            use_sledgehammer=False,
        )
        env.init()
        initial_state = env.current_state

        # validate environment initialization
        if not initial_state or initial_state in ["error", "init", ""]:
            logger.error(
                f"environment initialization failed for theorem {theorem_name}: {initial_state}"
            )
            return [], []

        mcts_config = MCTSConfig(
            n_simulations=config.num_mcts_simulations,
            c_puct=1.0,
            max_depth=config.max_mcts_depth,
            max_actions_per_expand=config.num_expand_per_node,
        )
        mcts = MCTSSearch(mcts_inference, mcts_config)
        root_node = mcts.search(env, initial_state)

        # validate mcts search results
        if not root_node or root_node.visit_count == 0:
            logger.error(f"MCTS search failed for theorem {theorem_name}")
            return [], []

        traj_policy, traj_value = mcts.extract_all_trajectories(env)

        # validate trajectory extraction
        if len(traj_policy) == 0 or len(traj_value) == 0:
            logger.error(f"No trajectories extracted for theorem {theorem_name}")
            return [], []

        logger.info(
            f"Episode {episode_id} for {theorem_name}: {len(traj_policy)} policy, {len(traj_value)} value nodes"
        )
        return traj_policy, traj_value

    except Exception as e:
        logger.error(
            f"MCTS rollout failed for theorem {theorem_name}, episode {episode_id}: {e}"
        )
        logger.debug(traceback.format_exc())
        return [], []
    finally:
        if env is not None:
            env.close()


def run_mcts_training_episodes_parallel(
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
) -> tuple[list[MCTSNode], list[MCTSNode]]:
    """Run MCTS training episodes in parallel using ThreadPoolExecutor"""
    logger = MCTSLogger.get_logger("parallel_mcts")

    theorem_data = load_dataset(config.path_to_minif2f, "valid")
    theorem_data = theorem_data[:-2]

    # create all theorem x episode combinations
    work_items = []
    for theorem in theorem_data:
        for episode in range(config.num_episodes_per_theorem):
            work_items.append((theorem, episode))

    logger.info(f"Starting {len(work_items)} parallel MCTS rollouts")

    all_traj_policy = []
    all_traj_value = []

    # use ThreadPoolExecutor for parallel execution
    max_workers = config.max_workers
    logger.info(f"Using {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        future_to_item = {
            executor.submit(
                single_mcts_rollout,
                theorem,
                episode,
                inference_manager,
                config,
            ): (theorem, episode)
            for theorem, episode in work_items
        }

        # collect results as they complete
        for future in as_completed(future_to_item):
            theorem, episode = future_to_item[future]
            try:
                traj_p, traj_v = future.result()
                all_traj_policy.extend(traj_p)
                all_traj_value.extend(traj_v)

                # log mcts metrics for this episode
                mcts_metrics = collect_mcts_metrics(traj_p, traj_v)
                mcts_metrics["theorem_name"] = theorem["name"]
                mcts_metrics["episode"] = episode
                log_mcts_metrics(mcts_metrics)

            except Exception as e:
                logger.error(
                    f"Exception in MCTS rollout for {theorem['name']}, episode {episode}: {e}"
                )
                logger.debug(traceback.format_exc())

    logger.info(
        f"Completed parallel rollouts: {len(all_traj_policy)} policy nodes, {len(all_traj_value)} value nodes"
    )
    return all_traj_policy, all_traj_value


def run_mcts_training_episodes(
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
) -> tuple[list[MCTSNode], list[MCTSNode]]:
    logger = MCTSLogger.get_logger("main")
    mcts_inference = MCTSInference(
        inference_manager,
        config.temperature_policy,
        config.temperature_value,
    )
    all_traj_policy: list[MCTSNode] = []
    all_traj_value: list[MCTSNode] = []

    theorem_data = load_dataset(config.path_to_minif2f, "valid")

    logger.info(f"starting new mcts training for {len(theorem_data)} theorems")

    for theorem in theorem_data:
        theorem_name = theorem["name"]
        theorem_content = theorem["content"]

        logger.info(f"running MCTS for theorem: {theorem_name}")

        for episode in range(config.num_episodes_per_theorem):
            if episode % 5 == 0:
                logger.info(f"episode {episode}/{config.num_episodes_per_theorem}")

            env = None
            try:
                env = TheoremEnv(
                    theorem_name=theorem_name,
                    theorem_content=theorem_content,
                    max_steps=config.max_mcts_steps,
                    # disable sledgehammer during training
                    use_sledgehammer=False,
                )
                env.init()
                initial_state = env.current_state

                # validate environment initialization
                if not initial_state or initial_state in ["error", "init", ""]:
                    logger.error(
                        f"Environment initialization failed for theorem {theorem_name}: {initial_state}"
                    )
                    continue

                mcts_config = MCTSConfig(
                    n_simulations=config.num_mcts_simulations,
                    c_puct=1.0,
                    max_depth=config.max_mcts_depth,
                    max_actions_per_expand=config.num_expand_per_node,
                )
                mcts = MCTSSearch(mcts_inference, mcts_config)
                root_node = mcts.search(env, initial_state, timeout_seconds=config.mcts_rollout_timeout_seconds)

                # validate mcts search results
                if not root_node or root_node.visit_count == 0:
                    logger.error(f"MCTS search failed for theorem {theorem_name}")
                    continue

                traj_policy, traj_value = mcts.extract_all_trajectories(env)

                # validate trajectory extraction
                if len(traj_policy) == 0 or len(traj_value) == 0:
                    logger.error(
                        f"No trajectories extracted for theorem {theorem_name}"
                    )
                    continue

                logger.info(
                    f"adding trajectories: {len(traj_policy)} for policy, {len(traj_value)} for value"
                )
                all_traj_policy.extend(traj_policy)
                all_traj_value.extend(traj_value)

                # log mcts metrics for this episode
                mcts_metrics = collect_mcts_metrics(traj_policy, traj_value)
                mcts_metrics["theorem_name"] = theorem_name
                mcts_metrics["episode"] = episode
                log_mcts_metrics(mcts_metrics)

            except Exception as e:
                logger.error(f"{type(e).__name__} in episode {episode}: {e}")
                logger.debug(traceback.format_exc())
            finally:
                if env:
                    env.close()

    return all_traj_policy, all_traj_value


def upload_to_huggingface(
    model_path: str,
    repo_name: str,
    token: str | None = None,
    model_type: str = "model",
    path_in_repo: str | None = None,
    logger=None,
) -> None:
    """Upload trained model to HuggingFace Hub"""
    try:
        # use provided token, or fall back to hf_token environment variable
        final_token = token or os.getenv("HF_TOKEN")

        if not final_token:
            raise ValueError(
                "No HuggingFace token provided. Set hf_token in TrainingConfig "
                "or HF_TOKEN environment variable"
            )

        upload_path = f"{repo_name}/{path_in_repo}" if path_in_repo else repo_name

        if logger:
            logger.info(
                f"Uploading {model_type} model from {model_path} to {upload_path}"
            )

        upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            path_in_repo=path_in_repo,
            token=final_token,
            commit_message=f"Upload {model_type} model trained with MCTS+RL",
        )

        if logger:
            logger.info(
                f"Successfully uploaded {model_type} model to https://huggingface.co/{upload_path}"
            )

    except Exception as e:
        if logger:
            logger.error(f"Failed to upload {model_type} model to HuggingFace: {e}")
        raise


def save_model_with_tokenizer(
    model: PreTrainedModel,
    tokenizer,
    save_path: str,
    model_name: str,
    logger,
    base_model: str | None = None,
) -> None:
    """Save model and tokenizer in standard HuggingFace"""

    logger.info(f"saving {model_name} model to {save_path}")

    # create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # save model in standard huggingface format
    model.save_pretrained(save_path)

    # ensure tokenizer has pad token set
    if (
        hasattr(tokenizer, "pad_token")
        and tokenizer.pad_token is None
        and hasattr(tokenizer, "eos_token")
    ):
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"set pad token to eos token for {model_name}")

    # save tokenizer
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_path)

    # fix README.md base_model metadata if it exists
    readme_path = os.path.join(save_path, "README.md")
    if os.path.exists(readme_path) and base_model:
        try:
            with open(readme_path, encoding="utf-8") as f:
                content = f.read()

            # replace any base_model that looks like a temp path, checkpoint path, or other invalid path
            invalid_base_model_patterns = [
                r'(base_model:\s*)(/tmp/[^"\s]+)',  # /tmp/ paths
                r'(base_model:\s*)(\./checkpoints/[^"\s]+)',  # checkpoint paths
                r"(base_model:\s*)(/tmp/[^'\s]+)",  # quoted /tmp/ paths
                r"(base_model:\s*)(\./checkpoints/[^'\s]+)",  # quoted checkpoint paths
                r'(base_model:\s*)"([^"]*tmp/[^"]*)"',  # double quoted temp paths
                r"(base_model:\s*)'([^']*tmp/[^']*)'",  # single quoted temp paths
            ]

            fixed = False
            for pattern in invalid_base_model_patterns:
                if re.search(pattern, content):
                    # Always use the base LLaMA model for Isabella
                    base_model = "EleutherAI/llemma_7b"
                    content = re.sub(pattern, rf'\1"{base_model}"', content)
                    fixed = True
                    break

            if fixed:
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"fixed base_model in README.md to {base_model}")
            else:
                logger.info("no invalid base_model path found in README.md")
        except Exception as e:
            logger.warning(f"failed to fix README.md base_model: {e}")

    # verify saved files
    saved_files = sorted(os.listdir(save_path))
    logger.info(f"{model_name} saved files: {saved_files}")

    # check for essential files
    essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    # check for model files
    # could be safetensors or pytorch_model.bin
    model_files = [
        f
        for f in saved_files
        if f.startswith("model.") or f.startswith("pytorch_model.")
    ]
    if not model_files:
        logger.warning(f"{model_name} missing model weight files")
    else:
        logger.info(f"{model_name} model files: {model_files}")

    missing_files = [f for f in essential_files if f not in saved_files]
    if missing_files:
        logger.warning(f"{model_name} missing essential files: {missing_files}")
    else:
        logger.info(f"{model_name} all essential files present")


def evaluate_on_test_split(
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
    save_proofs: bool = False,
    output_dir: str = "./evaluation_proofs",
    max_workers: int = 32,
) -> dict[str, Any]:
    """Evaluate current models on miniF2F test split"""
    logger = MCTSLogger.get_logger("evaluation")

    # load test dataset
    test_theorems = load_dataset(config.path_to_minif2f, "test")

    # limit for faster evaluation during training
    max_eval_theorems = min(config.eval_max_theorems, len(test_theorems))
    test_theorems = test_theorems[:max_eval_theorems]

    logger.info(f"Evaluating on {len(test_theorems)} test theorems")

    evaluation_results = {
        "total_theorems": len(test_theorems),
        "successful_proofs": 0,
        "failed_proofs": 0,
        "avg_proof_length": 0.0,
        "total_steps": 0,
        "success_rate": 0.0,
        # store successful proof data
        "successful_proofs_data": [],
    }

    proof_lengths = []

    # create output directory if saving proofs
    if save_proofs:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving successful proofs to {output_dir}")

    logger.info(f"Using {max_workers} parallel workers for training evaluation")

    # use threadpoolexecutor for parallel evaluation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all evaluation tasks
        future_to_theorem = {
            executor.submit(
                evaluate_single_theorem_training,
                theorem,
                inference_manager,
                config,
                save_proofs,
                output_dir,
            ): theorem
            for theorem in test_theorems
        }

        # collect results as they complete
        for future in as_completed(future_to_theorem):
            theorem = future_to_theorem[future]
            theorem_name = theorem["name"]

            try:
                proof_data, _ = future.result()

                if proof_data is not None:
                    # success case
                    evaluation_results["successful_proofs"] += 1
                    proof_lengths.append(proof_data["proof_length"])
                    evaluation_results["successful_proofs_data"].append(proof_data)
                else:
                    # failed case
                    evaluation_results["failed_proofs"] += 1

            except Exception as e:
                logger.error(
                    f"Exception in training evaluation for {theorem_name}: {e}"
                )
                evaluation_results["failed_proofs"] += 1

    # calculate metrics
    evaluation_results["success_rate"] = (
        evaluation_results["successful_proofs"] / evaluation_results["total_theorems"]
        if evaluation_results["total_theorems"] > 0
        else 0.0
    )

    evaluation_results["avg_proof_length"] = (
        sum(proof_lengths) / len(proof_lengths) if proof_lengths else 0.0
    )

    # log to wandb
    eval_metrics = {
        "total_theorems": evaluation_results["total_theorems"],
        "successful_proofs": evaluation_results["successful_proofs"],
        "failed_proofs": evaluation_results["failed_proofs"],
        "success_rate": evaluation_results["success_rate"],
        "avg_proof_length": evaluation_results["avg_proof_length"],
    }
    log_evaluation_metrics(eval_metrics)

    logger.info(
        f"Evaluation results: {evaluation_results['success_rate']:.2%} success rate"
    )
    return evaluation_results


def evaluate_single_theorem(
    theorem: TheoremData,
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
    output_dir: str = "./final_evaluation_proofs",
) -> dict:
    """Evaluate a single theorem and return comprehensive metadata"""
    logger = MCTSLogger.get_logger("single_evaluation")

    theorem_name = theorem["name"]
    theorem_content = theorem["content"]

    start_time = time.time()

    env = None
    try:
        env = TheoremEnv(
            theorem_name=theorem_name,
            theorem_content=theorem_content,
            max_steps=config.max_mcts_steps,
            use_sledgehammer=config.use_sledgehammer,
        )
        env.init()
        initial_state = env.current_state

        if not initial_state or initial_state in ["error", "init", ""]:
            logger.warning(f"Environment init failed for {theorem_name}")
            elapsed_time = time.time() - start_time
            return {
                "theorem_name": theorem_name,
                "theorem_content": theorem_content,
                "success": False,
                "error": "Environment initialization failed",
                "time_seconds": elapsed_time,
                "total_steps": 0,
                "tree_depth": 0,
                "proof_length": 0,
                "commands": [],
                "proof_trajectory": [],
            }

        # run mcts search with full simulations for final eval
        mcts_inference = MCTSInference(
            inference_manager,
            config.temperature_policy,
            config.temperature_value,
        )
        mcts_config = MCTSConfig(
            n_simulations=config.num_mcts_simulations,
            c_puct=1.0,
            max_depth=config.max_mcts_depth,
            max_actions_per_expand=config.num_expand_per_node,
        )
        mcts = MCTSSearch(mcts_inference, mcts_config)
        root_node = mcts.search(env, initial_state, timeout_seconds=config.mcts_rollout_timeout_seconds)

        elapsed_time = time.time() - start_time

        # calculate tree depth and total steps
        total_steps = (
            len(
                [
                    node
                    for node in [root_node, *list(root_node.children.values())]
                    if node.visit_count > 0
                ]
            )
            if root_node
            else 0
        )

        # get tree depth
        tree_depth = 0
        if root_node:
            tree_depth = get_max_depth(root_node)

        # check if proof was successful
        if not root_node or root_node.visit_count <= 0:
            return {
                "theorem_name": theorem_name,
                "theorem_content": theorem_content,
                "success": False,
                "error": "MCTS search failed",
                "time_seconds": elapsed_time,
                "total_steps": total_steps,
                "tree_depth": tree_depth,
                "proof_length": 0,
                "commands": [],
                "proof_trajectory": [],
            }

        best_trajectory = mcts.get_best_trajectory(env)
        if not best_trajectory or len(best_trajectory) == 0:
            return {
                "theorem_name": theorem_name,
                "theorem_content": theorem_content,
                "success": False,
                "error": "No valid trajectory found",
                "time_seconds": elapsed_time,
                "total_steps": total_steps,
                "tree_depth": tree_depth,
                "proof_length": 0,
                "commands": [],
                "proof_trajectory": [],
            }

        # Convert trajectory to serializable format
        serializable_trajectory = [node.to_dict() for node in best_trajectory]

        final_node = best_trajectory[-1]
        if not (final_node.is_terminal and final_node.terminal_reward > 0.5):
            return {
                "theorem_name": theorem_name,
                "theorem_content": theorem_content,
                "success": False,
                "error": "Proof not completed successfully",
                "time_seconds": elapsed_time,
                "total_steps": total_steps,
                "tree_depth": tree_depth,
                "proof_length": len(best_trajectory),
                "commands": [node.action for node in best_trajectory if node.action],
                "proof_trajectory": serializable_trajectory,
            }

        # success case - create comprehensive proof data
        proof_data = {
            "theorem_name": theorem_name,
            "theorem_content": theorem_content,
            "success": True,
            "time_seconds": elapsed_time,
            "total_steps": total_steps,
            "tree_depth": tree_depth,
            "proof_length": len(best_trajectory),
            "commands": [node.action for node in best_trajectory if node.action],
            "proof_trajectory": serializable_trajectory,
        }

        # save proof to .thy file
        save_proof_to_thy_file(proof_data, output_dir)

        return proof_data

    except Exception as e:
        logger.error(f"Final evaluation failed for {theorem_name}: {e}")
        elapsed_time = time.time() - start_time
        return {
            "theorem_name": theorem_name,
            "theorem_content": theorem_content,
            "success": False,
            "error": str(e),
            "time_seconds": elapsed_time,
            "total_steps": 0,
            "tree_depth": 0,
            "proof_length": 0,
            "commands": [],
            "proof_trajectory": [],
        }
    finally:
        if env:
            env.close()


def evaluate_single_theorem_training(
    theorem: TheoremData,
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
    save_proofs: bool = False,
    output_dir: str = "./evaluation_proofs",
) -> tuple[dict | None, str | None]:
    """Evaluate a single theorem for training evaluation and return proof data if successful"""
    logger = MCTSLogger.get_logger("single_training_evaluation")

    theorem_name = theorem["name"]
    theorem_content = theorem["content"]

    env = None
    try:
        env = TheoremEnv(
            theorem_name=theorem_name,
            theorem_content=theorem_content,
            max_steps=config.max_mcts_steps,
            # disable sledgehammer during training evaluation
            use_sledgehammer=False,
        )
        env.init()
        initial_state = env.current_state

        if not initial_state or initial_state in ["error", "init", ""]:
            logger.warning(f"Environment init failed for {theorem_name}")
            return None, theorem_name

        # run mcts search for evaluation usesng eval_mcts_simulations
        mcts_inference = MCTSInference(
            inference_manager,
            config.temperature_policy,
            config.temperature_value,
        )
        mcts_config = MCTSConfig(
            # use eval-specific config
            n_simulations=config.eval_mcts_simulations,
            c_puct=1.0,
            max_depth=config.max_mcts_depth,
            max_actions_per_expand=config.num_expand_per_node,
        )
        mcts = MCTSSearch(mcts_inference, mcts_config)
        root_node = mcts.search(env, initial_state, timeout_seconds=config.mcts_rollout_timeout_seconds)

        # check if proof was successful
        if not root_node or root_node.visit_count <= 0:
            return None, theorem_name

        best_trajectory = mcts.get_best_trajectory(env)
        if not best_trajectory or len(best_trajectory) == 0:
            return None, theorem_name

        final_node = best_trajectory[-1]
        if not (final_node.is_terminal and final_node.terminal_reward > 0.5):
            return None, theorem_name

        # Convert trajectory to serializable format
        serializable_trajectory = [node.to_dict() for node in best_trajectory]

        # success case, create proof data
        proof_data = {
            "theorem_name": theorem_name,
            "theorem_content": theorem_content,
            "proof_trajectory": serializable_trajectory,
            "proof_length": len(best_trajectory),
            "commands": [node.action for node in best_trajectory if node.action],
        }

        # save proof to .thy file if requested
        if save_proofs:
            save_proof_to_thy_file(proof_data, output_dir)

        return proof_data, None

    except Exception as e:
        logger.error(f"Training evaluation failed for {theorem_name}: {e}")
        return None, theorem_name
    finally:
        if env:
            env.close()


def run_final_evaluation(
    inference_manager: ModelInferenceManager,
    config: TrainingConfig,
    output_dir: str = "./final_evaluation_proofs",
    max_workers: int = 32,
) -> dict[str, Any]:
    """Run comprehensive final evaluation and save all successful proofs"""
    logger = MCTSLogger.get_logger("final_evaluation")

    logger.info("Starting final evaluation on full test split...")

    # load full test dataset for final evaluation
    test_theorems = load_dataset(config.path_to_minif2f, "test")

    logger.info(f"Evaluating on {len(test_theorems)} test theorems")

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # create comprehensive json file for all evaluation data
    json_path = os.path.join(output_dir, "evaluation_results.json")

    evaluation_results = {
        "summary": {
            "total_theorems": len(test_theorems),
            "successful_proofs": 0,
            "failed_proofs": 0,
            "success_rate": 0.0,
            "avg_proof_length": 0.0,
            "avg_time_seconds": 0.0,
            "avg_total_steps": 0.0,
            "avg_tree_depth": 0.0,
        },
        "per_theorem": [],
    }

    successful_times = []
    successful_steps = []
    successful_depths = []
    proof_lengths = []

    # use threadpoolexecutor for parallel evaluation
    logger.info(f"Using {max_workers} parallel workers for evaluation")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all evaluation tasks
        future_to_theorem = {
            executor.submit(
                evaluate_single_theorem,
                theorem,
                inference_manager,
                config,
                output_dir,
            ): theorem
            for theorem in test_theorems
        }

        # collect results as they complete
        for future in as_completed(future_to_theorem):
            theorem = future_to_theorem[future]
            theorem_name = theorem["name"]

            try:
                theorem_result = future.result()

                # store all theorem results
                evaluation_results["per_theorem"].append(theorem_result)

                if theorem_result["success"]:
                    # success case
                    evaluation_results["summary"]["successful_proofs"] += 1
                    successful_times.append(theorem_result["time_seconds"])
                    successful_steps.append(theorem_result["total_steps"])
                    successful_depths.append(theorem_result["tree_depth"])
                    proof_lengths.append(theorem_result["proof_length"])
                else:
                    # failed case
                    evaluation_results["summary"]["failed_proofs"] += 1

            except Exception as e:
                logger.error(f"Exception in evaluation for {theorem_name}: {e}")
                evaluation_results["summary"]["failed_proofs"] += 1
                # add failed theorem entry
                evaluation_results["per_theorem"].append(
                    {
                        "theorem_name": theorem_name,
                        "success": False,
                        "error": str(e),
                        "time_seconds": 0.0,
                        "total_steps": 0,
                        "tree_depth": 0,
                        "proof_length": 0,
                        "commands": [],
                        "proof_trajectory": [],
                    }
                )

    # calculate final metrics
    evaluation_results["summary"]["success_rate"] = (
        evaluation_results["summary"]["successful_proofs"]
        / evaluation_results["summary"]["total_theorems"]
        if evaluation_results["summary"]["total_theorems"] > 0
        else 0.0
    )

    evaluation_results["summary"]["avg_proof_length"] = (
        sum(proof_lengths) / len(proof_lengths) if proof_lengths else 0.0
    )

    evaluation_results["summary"]["avg_time_seconds"] = (
        sum(successful_times) / len(successful_times) if successful_times else 0.0
    )

    evaluation_results["summary"]["avg_total_steps"] = (
        sum(successful_steps) / len(successful_steps) if successful_steps else 0.0
    )

    evaluation_results["summary"]["avg_tree_depth"] = (
        sum(successful_depths) / len(successful_depths) if successful_depths else 0.0
    )

    # write comprehensive json file
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Comprehensive evaluation results saved to {json_path}")

    except Exception as e:
        logger.error(f"Failed to write JSON file: {e}")
        logger.warning("Continuing with evaluation despite JSON serialization failure")

    # log final evaluation metrics
    try:
        final_metrics = {
            "total_theorems": evaluation_results["summary"]["total_theorems"],
            "successful_proofs": evaluation_results["summary"]["successful_proofs"],
            "failed_proofs": evaluation_results["summary"]["failed_proofs"],
            "success_rate": evaluation_results["summary"]["success_rate"],
            "avg_proof_length": evaluation_results["summary"]["avg_proof_length"],
            "avg_time_seconds": evaluation_results["summary"]["avg_time_seconds"],
            "avg_total_steps": evaluation_results["summary"]["avg_total_steps"],
            "avg_tree_depth": evaluation_results["summary"]["avg_tree_depth"],
        }
        log_final_evaluation_metrics(final_metrics)

        logger.info(
            f"Final evaluation complete: {evaluation_results['summary']['success_rate']:.2%} success rate"
        )
    except KeyError as e:
        logger.error(f"Missing key in evaluation_results: {e}")
        logger.error("This indicates a structural issue with the evaluation results")

    logger.info(f"Successful proofs saved to {output_dir}")

    return evaluation_results


def training_loop(config: TrainingConfig):
    logger = MCTSLogger.get_logger("main")
    inference_manager = ModelInferenceManager(
        config.policy_model_path,
        config.value_model_path,
        policy_cuda_visible_devices=config.policy_vllm_cuda_visible_devices,
        value_cuda_visible_devices=config.value_vllm_cuda_visible_devices,
        policy_vllm_kwargs=config.policy_vllm_args,
        value_vllm_kwargs=config.value_vllm_args,
    )

    policy_save_path = None
    value_save_path = None

    # track separate step counters for mcts and evaluation
    mcts_global_step = 0
    evaluation_global_step = 0

    try:
        for iteration in range(config.training_iterations):
            logger.info(
                f"\n=== ITERATION {iteration + 1} / {config.training_iterations} ==="
            )

            # === PHASE 1: LOAD vLLM ENGINES  ===
            logger.info("Loading vllm engines for inference phase...")

            # check gpu memory before loading vllm engines
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"GPU memory before vllm load - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
                )

            try:
                inference_manager.start_vllm_server()
                # validate that engines are actually loaded
                if not inference_manager.are_servers_running():
                    raise RuntimeError("inference engines failed to load properly")
                logger.info("inference engines loaded")

                # check gpu memory after vllm load
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(
                        f"GPU memory after inference load - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
                    )

            except Exception as e:
                logger.error(f"Failed to vllm engines: {e}")
                return None, None

            # === PHASE 2: RUN MCTS ROLLOUTS (inference phase) ===
            logger.info("Running MCTS rollouts with vllm inference...")

            # Verify servers are still running before rollouts
            if not inference_manager.are_servers_running():
                logger.error("vLLM servers not running before MCTS rollouts!")
                return None, None

            policy_traj, value_traj = run_mcts_training_episodes_parallel(
                inference_manager,
                config,
            )

            # update mcts step after episodes
            mcts_global_step += config.num_episodes_per_theorem

            logger.info(
                f"collected {len(policy_traj)} nodes for policy and {len(value_traj)} nodes for value"
            )

            # Limit training data size and prioritize high-reward nodes
            max_train_nodes = 500

            def compute_node_reward(node: MCTSNode) -> float:
                """Compute reward score for node prioritization (higher = better)"""
                if node.is_terminal:
                    # Terminal nodes: use actual outcome reward
                    if node.terminal_reward > 0.5:
                        return 1.0  # Successful proofs get highest priority
                    elif node.terminal_reward < -0.1:
                        return -1.0  # Errors get lowest priority
                    else:
                        return 0.0  # Timeouts/other failures
                else:
                    # Non-terminal nodes: use MCTS value as quality indicator
                    if node.visit_count > 0:
                        return float(np.clip(node.average_value, -0.5, 0.5))
                    else:
                        return -0.1  # Unvisited nodes get low priority

            # Sort and limit policy trajectory by reward (highest first)
            if len(policy_traj) > max_train_nodes:
                policy_traj.sort(key=compute_node_reward, reverse=True)
                limited_count = min(max_train_nodes, len(policy_traj))
                policy_traj = policy_traj[:limited_count]
                logger.info(f"Limited policy training data to top {limited_count} nodes by reward")

            # Sort and limit value trajectory by reward (highest first)
            if len(value_traj) > max_train_nodes:
                value_traj.sort(key=compute_node_reward, reverse=True)
                limited_count = min(max_train_nodes, len(value_traj))
                value_traj = value_traj[:limited_count]
                logger.info(f"Limited value training data to top {limited_count} nodes by reward")

            logger.info(f"Final training data: {len(policy_traj)} policy nodes, {len(value_traj)} value nodes")

            # check how many nodes have response data
            policy_with_responses = sum(
                1 for node in policy_traj if node.command_generation_response
            )
            value_with_responses = sum(
                1 for node in value_traj if node.value_estimate_response
            )
            logger.info(
                f"nodes with responses: policy={policy_with_responses}/{len(policy_traj)}, value={value_with_responses}/{len(value_traj)}"
            )

            if len(policy_traj) == 0 or len(value_traj) == 0:
                logger.warning("no training data collected, skipping iteration")
                # still need to shutdown vllm engines even if we skip
                inference_manager.shutdown_servers()
                continue

            # check if any nodes have the required response data
            if policy_with_responses == 0 or value_with_responses == 0:
                logger.error(
                    f"no nodes have response data: policy={policy_with_responses}, value={value_with_responses}"
                )
                logger.error("inference is failing, cannot proceed with training")
                # still need to shutdown vllm engines
                inference_manager.shutdown_servers()
                return None, None

            # check if any nodes have the required response data
            policy_with_responses = sum(
                1 for node in policy_traj if node.command_generation_response
            )
            value_with_responses = sum(
                1 for node in value_traj if node.value_estimate_response
            )

            if policy_with_responses == 0 or value_with_responses == 0:
                logger.error(
                    f"no nodes have response data: policy={policy_with_responses}, value={value_with_responses}"
                )
                logger.error("inference is failing, cannot proceed with training")
                # still need to shutdown vllm engines
                inference_manager.shutdown_servers()
                return None, None

            # === PHASE 3: SHUTDOWN VLLM ENGINES FOR TRAINING ===
            logger.info("Shutting down vllm engines for training...")
            inference_manager.shutdown_servers()

            # check gpu memory after shutdown
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"GPU memory after vllm shutdown - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
                )

            # enhanced cleanup after vllm shutdown to prevent oom
            cleanup_memory(None, "After vllm shutdown before training")

            # Additional memory cleanup before creating datasets
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # check gpu memory after cleanup
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"GPU memory after cleanup - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
                )

            # === PHASE 4: TRAIN MODELS ===
            try:
                # create PPO config for training with optimized settings
                ppo_config = PPOConfig(
                    num_train_epochs=config.num_epochs,
                    per_device_train_batch_size=config.batch_size,
                    per_device_eval_batch_size=config.batch_size,
                    gradient_accumulation_steps=4,  # Reduced from 8 for memory efficiency
                    learning_rate=config.learning_rate,
                    warmup_steps=100,
                    logging_steps=5,
                    save_steps=1000,
                    eval_steps=500,
                    mixed_precision="bf16",
                    gradient_checkpointing=True,
                    dataloader_num_workers=4,
                )

                logger.info("Training policy model...")
                policy_trainer = train_ppo(
                    model_path=inference_manager.policy_path,
                    train_nodes=policy_traj,
                    eval_nodes=policy_traj[: min(10, len(policy_traj))],
                    model_type=ModelType.POLICY,
                    output_dir="./checkpoints/policy_temp",
                    config=ppo_config,
                )

                # Memory cleanup between policy and value model training
                cleanup_memory(None, "After policy model training, before value model")
                del policy_trainer  # Explicitly delete trainer to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logger.info("Training value model...")
                value_trainer = train_ppo(
                    model_path=inference_manager.value_path,
                    train_nodes=value_traj,
                    eval_nodes=value_traj[: min(10, len(value_traj))],
                    model_type=ModelType.VALUE,
                    output_dir="./checkpoints/value_temp",
                    config=ppo_config,
                )

                # Memory cleanup after training
                cleanup_memory(None, "After value model training")
                del value_trainer  # Explicitly delete trainer to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                training_successful = True

            except Exception as e:
                logger.error(f"Training failed in iteration {iteration + 1}: {e}")
                training_successful = False
                # continue to next iteration or final evaluation
                continue

            # the trainer saves models automatically to temp directories
            # copy them to iteration-specific paths for the next iteration
            policy_temp_path = "./checkpoints/policy_temp"
            value_temp_path = "./checkpoints/value_temp"
            policy_save_path = f"./checkpoints/policy_iter_{iteration + 1}"
            value_save_path = f"./checkpoints/value_iter_{iteration + 1}"

            # copy the trained models to iteration paths
            policy_saved = False
            value_saved = False

            if os.path.exists(policy_temp_path):
                try:
                    if os.path.exists(policy_save_path):
                        shutil.rmtree(policy_save_path)
                    shutil.copytree(policy_temp_path, policy_save_path)
                    logger.info(
                        f"Successfully copied policy model from {policy_temp_path} to {policy_save_path}"
                    )
                    policy_saved = True
                except Exception as e:
                    logger.error(
                        f"Failed to copy policy model from {policy_temp_path} to {policy_save_path}: {e}"
                    )

            if os.path.exists(value_temp_path):
                try:
                    if os.path.exists(value_save_path):
                        shutil.rmtree(value_save_path)
                    shutil.copytree(value_temp_path, value_save_path)
                    logger.info(
                        f"Successfully copied value model from {value_temp_path} to {value_save_path}"
                    )
                    value_saved = True
                except Exception as e:
                    logger.error(
                        f"Failed to copy value model from {value_temp_path} to {value_save_path}: {e}"
                    )

            if policy_saved or value_saved:
                saved_paths = []
                if policy_saved:
                    saved_paths.append(policy_save_path)
                if value_saved:
                    saved_paths.append(value_save_path)
                logger.info(f"Successfully saved models to: {', '.join(saved_paths)}")
            else:
                logger.warning(
                    "No models were successfully saved to checkpoint directories"
                )

            # === PHASE 5: EVALUATION ===
            if training_successful and iteration % config.eval_frequency == 0:
                logger.info("Running evaluation on test split...")

                # ensure proper cleanup before evaluation
                cleanup_memory(
                    message=f"Memory cleanup before evaluation in iteration {iteration + 1}"
                )

                try:
                    # reload vllm engines for evaluation with trained models
                    inference_manager.reload_models(policy_save_path, value_save_path)

                    # regular evaluation (during training)
                    eval_results = evaluate_on_test_split(
                        inference_manager,
                        config,
                        # don't save proofs during training
                        save_proofs=False,
                        max_workers=config.max_workers,
                    )

                    evaluation_global_step += 1

                    logger.info(
                        f"Iteration {iteration + 1} evaluation: {eval_results['success_rate']:.2%} success rate"
                    )

                except Exception as e:
                    logger.error(f"Evaluation failed in iteration {iteration + 1}: {e}")
                finally:
                    # always shutdown engines after evaluation
                    inference_manager.shutdown_servers()

                    # additional cleanup after evaluation
                    cleanup_memory(
                        message=f"Memory cleanup after evaluation in iteration {iteration + 1}"
                    )

            # === PHASE 6: CLEANUP AND PREPARE FOR NEXT ITERATION ===
            if iteration < config.training_iterations - 1:
                logger.info("Preparing for next iteration...")
                # perform memory cleanup before reloading models to prevent fragmentation
                cleanup_memory(
                    message=f"Memory cleanup before iteration {iteration + 2}"
                )

    except Exception as e:
        logger.error(f"training loop error: {e}")
        training_failed = True
    else:
        training_failed = False

    # === FINAL EVALUATION ===
    if not training_failed:
        logger.info("Running final comprehensive evaluation...")

        # reload engines with final trained models for evaluation
        try:
            final_policy_path = (
                f"./checkpoints/policy_iter_{config.training_iterations}"
            )
            final_value_path = f"./checkpoints/value_iter_{config.training_iterations}"

            # check if final checkpoints exist, otherwise use initial models
            if not os.path.exists(final_policy_path) or not os.path.exists(
                final_value_path
            ):
                logger.warning(
                    f"Final checkpoints not found at {final_policy_path} and {final_value_path}"
                )
                logger.warning("Using initial model paths for final evaluation")
                # Use hardcoded fallback to avoid config corruption issues
                final_policy_path = "EleutherAI/llemma_7b"
                final_value_path = "EleutherAI/llemma_7b"

            inference_manager.reload_models(final_policy_path, final_value_path)

            # run final evaluation with proof saving
            final_results = run_final_evaluation(
                inference_manager,
                config,
                output_dir=config.final_eval_output_dir,
                max_workers=config.max_workers,
            )

            logger.info(
                f"Final results: {final_results['summary']['success_rate']:.2%} success rate"
            )

            # upload models to huggingface if enabled
            if config.upload_to_hf:
                try:
                    logger.info("Uploading trained models to HuggingFace Hub...")

                    # upload both models to the same repository with subdirectories
                    base_repo = f"{config.hf_username}/{config.hf_repo_name}"

                    # upload policy model to policy/ subdirectory
                    upload_to_huggingface(
                        final_policy_path,
                        base_repo,
                        config.hf_token,
                        "policy",
                        path_in_repo="policy",
                        logger=logger,
                    )

                    # upload value model to value/ subdirectory
                    upload_to_huggingface(
                        final_value_path,
                        base_repo,
                        config.hf_token,
                        "value",
                        path_in_repo="value",
                        logger=logger,
                    )

                    logger.info("Model upload to HuggingFace completed successfully!")

                except Exception as e:
                    logger.error(f"Model upload to HuggingFace failed: {e}")
                    # don't fail the entire training for upload issues

        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
        finally:
            # final cleanup
            inference_manager.shutdown_servers()
    else:
        logger.warning("Skipping final evaluation due to training failure")
        # still perform final cleanup
        inference_manager.shutdown_servers()

    return policy_save_path, value_save_path
