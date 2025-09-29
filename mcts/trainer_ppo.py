import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from mcts.data_mcts import MCTSNodeDataset, ModelType
from mcts.logging_utils import MCTSLogger
from mcts.search import MCTSNode
from mcts.wandb_manager import log_training_metrics

logger = MCTSLogger.get_logger("trainer_ppo")


def value_loss_fn(
    predicted_values: torch.Tensor, target_values: torch.Tensor
) -> torch.Tensor:
    """MSE loss for value model training"""
    return torch.nn.functional.mse_loss(predicted_values, target_values)


@dataclass
class PPOConfig:
    """Configuration for PPO training"""

    num_train_epochs: int = 3
    # for 4xA100: set to 16
    per_device_train_batch_size: int = 8
    # for 4xA100: set to 16
    per_device_eval_batch_size: int = 8
    # effective batch size = 64
    # for 4xA100: set to 16 for effective batch size 256
    gradient_accumulation_steps: int = 8
    # for 4xA100: can increase to 1e-4
    learning_rate: float = 5e-5
    # for 4xA100: can reduce to 50 if stable
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    # for 4xA100: can increase to 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    clip_epsilon: float = 0.2
    # kl penalty coefficient; set to 0 to disable
    beta: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    # for 4xA100: set to 8
    dataloader_num_workers: int = 4


def get_per_token_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-token log probabilities for the generated sequence.

    Returns:
        torch.Tensor: [batch_size, seq_len-1] tensor of per-token log probabilities
                      for the generated tokens (excluding the input prompt).
    """
    # shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # get log probabilities for each token
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # apply attention mask to zero out padding tokens
    per_token_logps = per_token_logps * shift_attention_mask

    return per_token_logps


def get_sequence_logprobs(
    logits: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    stored_logprobs: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute total log probability for the generated sequence.

    Args:
        response_mask: [batch_size, seq_len] mask where 1 indicates response tokens, 0 indicates prompt/padding

    Returns:
        torch.Tensor: [batch_size] tensor of summed log probabilities per sequence.
    """
    if stored_logprobs is not None:
        # use pre-computed per-token logprobs
        per_token_logps = stored_logprobs
        # stored_logprobs has shape [batch_size, seq_len-1], need to create valid_token_mask
        valid_token_mask = (
            (attention_mask[..., 1:] > 0).float()
            if attention_mask is not None
            else torch.ones_like(stored_logprobs)
        )
    elif logits is not None and input_ids is not None and attention_mask is not None:
        # compute from logits
        per_token_logps = get_per_token_logprobs(logits, input_ids, attention_mask)
        valid_token_mask = (attention_mask[..., 1:] > 0).float()
    else:
        raise ValueError(
            "Either stored_logprobs or (logits, input_ids, attention_mask) must be provided"
        )

    # apply response mask if provided -> only logprobs for response tokens
    if response_mask is not None:
        # response_mask has shape [batch_size, seq_len], we need [batch_size, seq_len-1] for logprobs
        # shift to align with logprobs
        response_token_mask = response_mask[..., 1:].float()
        # combining with attention mask to ensure to only consider valid (non-padded) response tokens
        valid_token_mask = valid_token_mask * response_token_mask

    # apply mask and sum
    masked_logprobs = per_token_logps * valid_token_mask
    sequence_logprobs = masked_logprobs.sum(dim=-1)
    sequence_lengths = valid_token_mask.sum(dim=-1)

    # avoid division by zero for empty sequences
    sequence_lengths = torch.clamp(sequence_lengths, min=1.0)

    # normalizing by sequence length to make log probs comparable across different lengths
    normalized_logprobs = sequence_logprobs / sequence_lengths

    # handle any remaining NaN values
    normalized_logprobs = torch.nan_to_num(normalized_logprobs, nan=0.0)

    return normalized_logprobs


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: [batch_size] rewards
        values: [batch_size] value estimates
        dones: [batch_size] done flags (1 if terminal, 0 otherwise)
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: [batch_size] computed advantages
    """
    if dones is None:
        # if no done flags provided, assume all are non-terminal
        dones = torch.zeros_like(rewards)

    advantages = torch.zeros_like(rewards)
    gae = 0

    # compute GAE backwards through the trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # terminal state has value 0
            next_value = torch.tensor(0.0)
        else:
            next_value = values[t + 1].detach()

        # TD error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t].detach()

        # GAE accumulation
        gae = delta + gamma * lam * (1 - dones[t]) * gae

        advantages[t] = gae

    return advantages


def compute_policy_loss(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Compute PPO policy loss (clipped surrogate objective).

    Args:
        policy_logprobs: [batch_size] log probs from current policy
        ref_logprobs: [batch_size] log probs from reference policy
        rewards: [batch_size] reward values
        values: [batch_size] value estimates for advantage computation
        dones: [batch_size] done flags
        clip_epsilon: clipping parameter for PPO

    Returns:
        tuple[torch.Tensor, dict]: (policy_loss, loss_components)
    """
    # handle NaN values
    policy_logprobs = torch.nan_to_num(policy_logprobs, nan=0.0)
    ref_logprobs = torch.nan_to_num(ref_logprobs, nan=0.0)
    rewards = torch.nan_to_num(rewards, nan=0.0)
    values = torch.nan_to_num(values, nan=0.0)

    # compute advantages using GAE
    advantages = compute_advantages(rewards, values, dones)
    advantages = torch.nan_to_num(advantages, nan=0.0)

    # normalize advantages for stability
    adv_std = advantages.std()
    if adv_std > 0 and not torch.isnan(adv_std):
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

    # policy ratio and clipped surrogate loss
    log_ratio = policy_logprobs - ref_logprobs
    ratio = torch.exp(torch.clamp(log_ratio, -10, 10))
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    loss_components = {
        "policy_loss": policy_loss.item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }

    return policy_loss, loss_components


def compute_value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    """Compute value function loss (MSE).

    Args:
        predicted_values: [batch_size] predicted values
        target_values: [batch_size] target values

    Returns:
        torch.Tensor: MSE loss
    """
    return torch.nn.functional.mse_loss(predicted_values, target_values)


class PPOTrainer:
    """PPO trainer using Accelerate"""

    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None,
        accelerator: Accelerator,
        config: PPOConfig,
        output_dir: str = "./checkpoints",
        model_type: ModelType = ModelType.POLICY,
        original_model_path: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.accelerator = accelerator
        self.config = config
        self.output_dir = output_dir
        self.model_type = model_type
        self._original_model_path = original_model_path

        # setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        ) = accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader if self.eval_dataloader else [],
        )

        # after accelerator.prepare(), ensure LoRA parameters still have gradients
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                logger.info(
                    f"Re-enabled gradients for LoRA parameter after accelerator.prepare(): {name}"
                )

        # setup lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )

        self.global_step = 0
        self.best_eval_loss = float("inf")

    def train(self):
        """Run the training loop."""
        logger.info("Starting PPO training...")

        for epoch in range(self.config.num_train_epochs):
            self.model.train()

            for _, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    # Debug: check batch for NaN
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                            logger.warning(f"NaN in batch[{key}]: {value}")

                    # forward pass through policy model
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    loss_components = {
                        "policy_loss": 0.0,
                        "value_loss": 0.0,
                        "entropy_loss": 0.0,
                        "total_loss": 0.0,
                    }

                    if self.model_type == ModelType.POLICY:
                        # === POLICY MODEL TRAINING ===
                        # Skip batch if NaN logits detected
                        if torch.isnan(outputs.logits).any():
                            logger.warning("NaN in policy logits, skipping batch")
                            continue

                        # Compute log probabilities for generated responses
                        response_mask = batch.get("response_mask")
                        policy_seq_logps = get_sequence_logprobs(
                            outputs.logits,
                            batch["input_ids"],
                            batch["attention_mask"],
                            response_mask=response_mask,
                        )
                        ref_seq_logps = get_sequence_logprobs(
                            stored_logprobs=batch["stored_logprobs"],
                            attention_mask=batch["attention_mask"],
                            response_mask=response_mask,
                        )

                        # get value estimates for advantage computation
                        values = (
                            batch.get("predicted_values")
                            if batch.get("predicted_values") is not None
                            else torch.full_like(
                                batch["rewards"], batch["rewards"].mean()
                            )
                        )

                        # compute PPO policy loss (no value loss for policy training)
                        loss, loss_components = compute_policy_loss(
                            policy_logprobs=policy_seq_logps,
                            ref_logprobs=ref_seq_logps,
                            rewards=batch["rewards"],
                            values=values,
                            dones=batch.get("dones"),
                            clip_epsilon=self.config.clip_epsilon,
                        )

                        # add components for consistent logging interface
                        loss_components.update(
                            {
                                "value_loss": 0.0,
                                "entropy_loss": 0.0,
                                "total_loss": loss.item(),
                            }
                        )

                        if torch.isnan(loss):
                            logger.debug("NaN in total loss!")
                            logger.debug(
                                f"policy_loss: {loss_components['policy_loss']}"
                            )
                            logger.debug(f"value_loss: {loss_components['value_loss']}")
                            logger.debug(f"rewards: {batch['rewards']}")
                            logger.debug(f"values: {values}")
                            logger.debug(f"dones: {batch.get('dones')}")
                    else:  # ModelType.VALUE
                        # === VALUE MODEL TRAINING ===
                        # train to generate correct value estimate text using teacher forcing

                        # forward pass through the model
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )

                        # compute cross-entropy loss on response tokens only
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = batch["input_ids"][..., 1:].contiguous()
                        shift_attention_mask = batch["attention_mask"][
                            ..., 1:
                        ].contiguous()

                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        loss_per_token = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        ).view(shift_labels.shape)

                        # apply attention mask and response mask
                        loss_per_token = loss_per_token * shift_attention_mask
                        if batch.get("response_mask") is not None:
                            response_token_mask = batch["response_mask"][
                                ..., 1:
                            ].float()
                            loss_per_token = loss_per_token * response_token_mask
                            valid_count = (
                                shift_attention_mask * response_token_mask
                            ).sum()
                        else:
                            valid_count = shift_attention_mask.sum()

                        loss = loss_per_token.sum() / (valid_count + 1e-8)

                        # logging components
                        loss_components = {
                            "policy_loss": 0.0,
                            "value_loss": loss.item(),
                            "entropy_loss": 0.0,
                            "total_loss": loss.item(),
                        }

                    # backward pass
                    self.accelerator.backward(loss)

                    # clip gradients
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # memory cleanup
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

                self.global_step += 1

                # logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.model_type == ModelType.POLICY:
                        metrics = {
                            "loss": loss.item(),
                            "policy_loss": loss_components["policy_loss"],
                            "value_loss": loss_components["value_loss"],
                            "entropy_loss": loss_components["entropy_loss"],
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "global_step": self.global_step,
                        }
                        logger.info(
                            f"Step {self.global_step}: total_loss={loss.item():.4f}, "
                            f"policy_loss={loss_components['policy_loss']:.4f}, "
                            f"value_loss={loss_components['value_loss']:.4f}, "
                            f"entropy_loss={loss_components['entropy_loss']:.4f}, "
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}, "
                            f"epoch={epoch}"
                        )
                    else:  # ModelType.VALUE
                        metrics = {
                            "loss": loss.item(),
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "global_step": self.global_step,
                        }
                        logger.info(
                            f"Step {self.global_step}: value_loss={loss.item():.4f}, "
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}, "
                            f"epoch={epoch}"
                        )

                    self.accelerator.log(metrics)
                    # also log to wandb with model type
                    log_training_metrics(metrics, model_type=self.model_type.value)

                # evaluation
                if (
                    self.eval_dataloader
                    and self.global_step % self.config.eval_steps == 0
                ):
                    eval_loss, eval_kl_div = self.evaluate()
                    if self.model_type == ModelType.POLICY:
                        eval_metrics = {
                            "eval_loss": eval_loss,
                            "eval_kl_divergence": eval_kl_div,
                        }
                        logger.info(
                            f"Eval Step {self.global_step}: eval_loss={eval_loss:.4f}, "
                            f"eval_kl_div={eval_kl_div:.4f}"
                        )
                    else:  # ModelType.VALUE
                        eval_metrics = {
                            "eval_loss": eval_loss,
                        }
                        logger.info(
                            f"Eval Step {self.global_step}: eval_value_loss={eval_loss:.4f}"
                        )

                    self.accelerator.log(eval_metrics)
                    # also log to wandb with model type
                    log_training_metrics(eval_metrics, model_type=self.model_type.value)

                    # save best model
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_model("best")

                # save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f"step_{self.global_step}")

        # save final model to base output directory
        self.save_model()
        logger.info("PPO training completed")

    def evaluate(self) -> tuple[float, float]:
        """Run evaluation. Returns (avg_loss, avg_kl_divergence_or_zero)."""
        if not self.eval_dataloader:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_kl_div = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )

                if self.model_type == ModelType.POLICY:
                    # policy model evaluation with PPO loss
                    policy_seq_logps = get_sequence_logprobs(
                        outputs.logits, batch["input_ids"], batch["attention_mask"]
                    )
                    ref_seq_logps = get_sequence_logprobs(
                        stored_logprobs=batch["stored_logprobs"],
                        attention_mask=batch["attention_mask"],
                    )

                    # use same baseline as training
                    baseline = batch["rewards"].mean()
                    values = torch.full_like(batch["rewards"], baseline)
                    loss, _ = compute_policy_loss(
                        policy_logprobs=policy_seq_logps,
                        ref_logprobs=ref_seq_logps,
                        rewards=batch["rewards"],
                        values=values,
                        dones=batch.get("dones"),
                        clip_epsilon=self.config.clip_epsilon,
                    )

                    # compute KL divergence for logging (sequence level)
                    kl_div = (policy_seq_logps - ref_seq_logps).mean().item()
                else:  # ModelType.VALUE
                    # value model evaluation with MSE loss
                    # generate text and extract predicted values
                    value_outputs = self.model.generate(  # type: ignore[operator]
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,  # type: ignore[attr-defined]
                    )
                    generated_texts = self.tokenizer.batch_decode(  # type: ignore[attr-defined]
                        value_outputs, skip_special_tokens=True
                    )

                    predicted_values = []
                    for text in generated_texts:
                        import re

                        score_match = re.search(r"(-?\d+\.?\d*)", text)
                        if score_match:
                            try:
                                value = float(score_match.group(1))
                                predicted_values.append(np.clip(value, -1.0, 1.0))
                            except (ValueError, IndexError):
                                predicted_values.append(0.0)
                        else:
                            predicted_values.append(0.0)

                    predicted_values_tensor = torch.tensor(
                        predicted_values,
                        dtype=torch.float32,
                        device=self.accelerator.device,
                    )
                    target_values = batch["rewards"]
                    loss = value_loss_fn(predicted_values_tensor, target_values)
                    # not meaningful for value model
                    kl_div = 0.0

                total_loss += loss.item()
                total_kl_div += kl_div
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_kl_div = total_kl_div / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_kl_div

    def save_model(self, suffix: str = ""):
        """Save model checkpoint."""
        output_path = (
            os.path.join(self.output_dir, suffix) if suffix else self.output_dir
        )
        os.makedirs(output_path, exist_ok=True)

        logger.info(f"Attempting to save model to {output_path}")

        # save model; try unwrapping first, fallback to direct save
        model_saved = False
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_path)
            model_saved = True
            logger.info(f"Successfully saved unwrapped model to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save unwrapped model: {e}")
            try:
                # fallback: try saving directly
                self.model.save_pretrained(output_path)
                model_saved = True
                logger.info(f"Successfully saved model directly to {output_path}")
            except Exception as e2:
                logger.error(f"Failed to save model directly: {e2}")

        # save tokenizer as well
        tokenizer_saved = False
        if hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(output_path)  # type: ignore
                tokenizer_saved = True
                logger.info(f"Successfully saved tokenizer to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save tokenizer: {e}")

        # fix README.md base_model metadata if it exists
        if model_saved:
            readme_path = os.path.join(output_path, "README.md")
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, encoding="utf-8") as f:
                        content = f.read()

                    # replace any base_model that looks like a temp path, checkpoint path, or other invalid path
                    invalid_base_model_patterns = [
                        # /tmp/ paths
                        r'(base_model:\s*)(/tmp/[^"\s]+)',
                        # checkpoint paths
                        r'(base_model:\s*)(\./checkpoints/[^"\s]+)',
                        # quoted /tmp/ paths
                        r"(base_model:\s*)(/tmp/[^'\s]+)",
                        # quoted checkpoint paths
                        r"(base_model:\s*)(\./checkpoints/[^'\s]+)",
                        # double quoted temp paths
                        r'(base_model:\s*)"([^"]*tmp/[^"]*)"',
                        # single quoted temp paths
                        r"(base_model:\s*)'([^']*tmp/[^']*)'",
                    ]

                    fixed = False
                    for pattern in invalid_base_model_patterns:
                        if re.search(pattern, content):
                            # always use the base LLaMA model for Isabella
                            base_model = "EleutherAI/llemma_7b"
                            content = re.sub(pattern, rf'\1"{base_model}"', content)
                            fixed = True
                            break

                    if fixed:
                        with open(readme_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        logger.info(
                            "fixed base_model in README.md to EleutherAI/llemma_7b"
                        )
                    else:
                        logger.info("no invalid base_model path found in README.md")
                except Exception as e:
                    logger.warning(f"failed to fix README.md base_model: {e}")

        if model_saved and tokenizer_saved:
            logger.info(
                f"Successfully saved complete model checkpoint to {output_path}"
            )
        else:
            logger.error(
                f"Failed to save model checkpoint to {output_path} - model_saved: {model_saved}, tokenizer_saved: {tokenizer_saved}"
            )


def collate_mcts_batch(batch):
    """Collate function that handles variable-length sequences properly for MCTS training.

    This function must be at module level to be pickleable for multiprocessing.
    """
    # find max length in this batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    # pad all sequences to max length
    padded_batch = {}
    for key in batch[0]:
        if key in ["input_ids", "attention_mask", "response_mask"]:
            # pad sequences
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.size(0) < max_len:
                    # pad with zeros for input_ids, attention_mask, response_mask
                    pad_value = 0
                    padding = torch.full(
                        (max_len - tensor.size(0),), pad_value, dtype=tensor.dtype
                    )
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    padded_tensor = tensor[:max_len]  # Truncate if too long
                padded_tensors.append(padded_tensor)
            padded_batch[key] = torch.stack(padded_tensors)
        elif key == "stored_logprobs":
            # stored_logprobs has shape [seq_len-1], need to pad to [max_len-1]
            max_logprob_len = max_len - 1
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.size(0) < max_logprob_len:
                    # pad with zeros (neutral logprob)
                    padding = torch.zeros(
                        max_logprob_len - tensor.size(0), dtype=tensor.dtype
                    )
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    # truncate if too long
                    padded_tensor = tensor[:max_logprob_len]
                padded_tensors.append(padded_tensor)
            padded_batch[key] = torch.stack(padded_tensors)
        else:
            # stack other tensors normally (rewards, predicted_values, dones)
            padded_batch[key] = torch.stack([item[key] for item in batch])

    return padded_batch


def create_ppo_trainer(
    model_path: str,
    train_nodes: list[MCTSNode],
    eval_nodes: list[MCTSNode] | None = None,
    model_type: ModelType = ModelType.POLICY,
    output_dir: str = "./checkpoints",
    config: PPOConfig | None = None,
) -> PPOTrainer:
    """Create a PPO trainer with pure Accelerate."""

    if config is None:
        config = PPOConfig()

    # initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    # load model and tokenizer with memory optimizations
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        # automatically distribute across CPU/GPU
        device_map="auto",
        # use fp16 for base dtype
        dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # set padding side to 'left' for decoder-only models to avoid warnings
    tokenizer.padding_side = "left"

    # configure LoRA for training quantized models
    # use different target modules based on model architecture
    model_name = model_path.lower()
    if "gpt" in model_name or "dialo" in model_name:
        target_modules = ["c_attn", "c_proj"]
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=64,
        # for 4xA100 can increase to 128
        lora_alpha=128,
        # for 4xA100 can increase to 256
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # apply LoRA to the model
    # only if not already a PEFT model
    if not hasattr(model, "peft_config"):
        model = get_peft_model(model, lora_config)  # type: ignore[assignment]
        model.print_trainable_parameters()  # type: ignore[operator]

        # enable input gradients for PEFT models (important for training)
        model.enable_input_require_grads()  # type: ignore[operator]

        # ensure LoRA parameters require gradients
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                logger.info(f"Enabled gradients for LoRA parameter: {name}")
    else:
        logger.info("Model already has PEFT config, skipping LoRA application")

    # enable gradient checkpointing
    # disable caching to avoid conflicts
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore[operator]
        # disable caching when using gradient checkpointing to avoid conflicts
        model.config.use_cache = False  # type: ignore[attr-defined]

    # create datasets and dataloaders
    train_dataset = MCTSNodeDataset(train_nodes, tokenizer, model_type, max_length=4096)
    logger.info(
        f"Created dataset with {len(train_dataset)} samples from {len(train_nodes)} nodes"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_mcts_batch,
    )

    eval_dataloader = None
    if eval_nodes:
        eval_dataset = MCTSNodeDataset(
            eval_nodes, tokenizer, model_type, max_length=4096
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_mcts_batch,
        )

    return PPOTrainer(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator,
        config=config,
        output_dir=output_dir,
        model_type=model_type,
        original_model_path=model_path,
    )


def train_ppo(
    model_path: str,
    train_nodes: list[MCTSNode],
    eval_nodes: list[MCTSNode] | None = None,
    model_type: ModelType = ModelType.POLICY,
    output_dir: str = "./checkpoints",
    config: PPOConfig | None = None,
) -> PPOTrainer:
    """Train a model using PPO with stored reference logprobs."""

    logger.info(f"Starting PPO training for {model_type.value} model")
    logger.info(f"Training on {len(train_nodes)} nodes")
    if eval_nodes:
        logger.info(f"Evaluating on {len(eval_nodes)} nodes")

    # create trainer
    trainer = create_ppo_trainer(
        model_path=model_path,
        train_nodes=train_nodes,
        eval_nodes=eval_nodes,
        model_type=model_type,
        output_dir=output_dir,
        config=config,
    )

    # train
    trainer.train()

    logger.info("PPO training completed")
    return trainer
