from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset

from mcts.search import MCTSNode


class ModelType(Enum):
    POLICY = "policy"
    VALUE = "value"


class MCTSNodeDataset(Dataset):
    """dataset for training from mcts node"""

    def __init__(
        self,
        nodes: list[MCTSNode],
        tokenizer,
        model_type: ModelType,
        max_length=256,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.data = self._prepare_data()

    def __len__(self) -> int:
        """return the number of items in the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """get a single item from the dataset"""
        return self.data[idx]

    def _prepare_data(self):
        data = []
        all_targets = []

        # collect all target values for normalization
        for node in self.nodes:
            if self.model_type == ModelType.POLICY and node.command_generation_response:
                target = self._compute_policy_reward(node)
                all_targets.append(target)
            elif self.model_type == ModelType.VALUE and node.value_estimate_response:
                # target value for TD learning
                target = self._compute_value_reward(node)
                all_targets.append(target)

        # compute target normalization stats
        if all_targets:
            target_mean = np.mean(all_targets)
            # avoid division by zero (epsilon regularization)
            target_std = np.std(all_targets) + 1e-8
        else:
            target_mean, target_std = 0.0, 1.0

        # create data with normalized targets
        for node in self.nodes:
            match self.model_type:
                case ModelType.POLICY:
                    if node.command_generation_response is None:
                        continue
                    prompt = node.command_generation_response["full_prompt"]
                    response = node.command_generation_response["full_response"]
                    raw_target = self._compute_policy_reward(node)
                    # normalizing
                    target = (raw_target - target_mean) / target_std

                case ModelType.VALUE:
                    if node.value_estimate_response is None:
                        continue
                    prompt = node.value_estimate_response["full_prompt"]
                    response = node.value_estimate_response["full_response"]
                    # TD target value
                    raw_target = self._compute_value_reward(node)
                    # normalize
                    target = (raw_target - target_mean) / target_std

            # tokenize input prompt
            prompt_encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            prompt_input_ids = prompt_encoded["input_ids"][0]
            prompt_attention_mask = prompt_encoded["attention_mask"][0]

            # tokenize response
            response_encoded = self.tokenizer(
                response,
                return_tensors="pt",
                # not adding BOS/EOS to continuation
                add_special_tokens=False,
            )
            response_input_ids = response_encoded["input_ids"][0]
            response_attention_mask = response_encoded["attention_mask"][0]

            combined_input_ids = torch.cat([prompt_input_ids, response_input_ids])
            combined_attention_mask = torch.cat(
                [prompt_attention_mask, response_attention_mask]
            )

            # truncate if too long, prioritizing response tokens
            if len(combined_input_ids) > self.max_length:
                # keep full prompt; truncate response if needed
                prompt_len = len(prompt_input_ids)
                max_response_len = self.max_length - prompt_len
                if max_response_len > 0:
                    response_input_ids = response_input_ids[:max_response_len]
                    response_attention_mask = response_attention_mask[:max_response_len]
                else:
                    # skip if prompt alone exceeds max length
                    continue
                combined_input_ids = torch.cat([prompt_input_ids, response_input_ids])
                combined_attention_mask = torch.cat(
                    [prompt_attention_mask, response_attention_mask]
                )

            # pad to max_length
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            padding_length = self.max_length - len(combined_input_ids)

            if padding_length > 0:
                combined_input_ids = torch.cat(
                    [
                        combined_input_ids,
                        torch.full((padding_length,), pad_token_id, dtype=torch.long),
                    ]
                )
                combined_attention_mask = torch.cat(
                    [
                        combined_attention_mask,
                        torch.zeros(padding_length, dtype=torch.long),
                    ]
                )

            # create response mask: 1 for response tokens, 0 for prompt and padding
            # -> only keeping response
            response_mask = torch.zeros(self.max_length, dtype=torch.long)
            prompt_len = len(prompt_input_ids)
            response_len = len(response_input_ids)
            response_mask[prompt_len : prompt_len + response_len] = 1

            input_ids = combined_input_ids
            attention_mask = combined_attention_mask

            # get stored logprobs; these should align with response tokens
            if self.model_type == ModelType.POLICY and node.command_generation_response:
                stored_logprobs = node.command_generation_response["logprobs"]
            elif self.model_type == ModelType.VALUE and node.value_estimate_response:
                stored_logprobs = node.value_estimate_response["logprobs"]
            else:
                stored_logprobs = []

            # ensure logprobs align with response tokens
            if isinstance(stored_logprobs, list) and len(stored_logprobs) > 0:
                # logprobs should match response token count; note: logprobs are for tokens 1 to N, so N-1 values
                # -1 because logprobs are shifted
                expected_logprob_len = len(response_input_ids) - 1
                if len(stored_logprobs) != expected_logprob_len:
                    # pad or truncate to match expected length
                    if len(stored_logprobs) < expected_logprob_len:
                        stored_logprobs.extend(
                            [0.0] * (expected_logprob_len - len(stored_logprobs))
                        )
                    else:
                        stored_logprobs = stored_logprobs[:expected_logprob_len]
            else:
                # no stored logprobs; using zeros
                stored_logprobs = [0.0] * max(0, len(response_input_ids) - 1)

            # create full logprob tensor aligned with the combined sequence
            # logprobs correspond to positions 1 to N in the input_ids, so we need max_length-1 positions
            full_logprobs = torch.zeros(self.max_length - 1, dtype=torch.float32)

            # response logprobs start after the prompt tokens
            # prompt_input_ids has length P, so logprobs for response start at position P-1
            # note. since logprob[0] is for token[1], logprob[P-1] is for token[P]
            prompt_len = len(prompt_input_ids)
            response_logprob_start = prompt_len - 1

            # copy stored logprobs to the correct position
            if len(stored_logprobs) > 0 and response_logprob_start >= 0:
                # making sure we don't exceed the full_logprobs array bounds
                available_space = len(full_logprobs) - response_logprob_start
                copy_len = min(len(stored_logprobs), available_space)
                if copy_len > 0:
                    full_logprobs[
                        response_logprob_start : response_logprob_start + copy_len
                    ] = torch.tensor(stored_logprobs[:copy_len], dtype=torch.float32)

            # extract predicted value for value models
            predicted_value = None
            if self.model_type == ModelType.VALUE and node.value_estimate_response:
                predicted_value = node.value_estimate_response.get("value", 0.0)

            # determine if this is a terminal state
            is_done = node.is_terminal

            data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "response_mask": response_mask,
                    "rewards": torch.tensor(target, dtype=torch.float32),
                    "stored_logprobs": full_logprobs,
                    "predicted_values": (
                        torch.tensor(predicted_value, dtype=torch.float32)
                        if predicted_value is not None
                        else torch.tensor(0.0, dtype=torch.float32)
                    ),
                    "dones": torch.tensor(float(is_done), dtype=torch.float32),
                }
            )

        return data

    def _compute_policy_reward(self, node: MCTSNode) -> float:
        """compute reward for single command generation based on command quality"""
        if node.is_terminal:
            # terminal nodes: use the actual outcome reward
            if node.terminal_reward > 0.5:
                # proof completed successfully
                return 1.0
            elif node.terminal_reward < -0.1:
                # command caused an error
                return -0.5
            else:
                # timeout or other failure
                return 0.0
        else:
            # non-terminal nodes: evaluate command quality based on MCTS feedback
            if node.visit_count > 0:
                # use MCTS value as quality indicator, normalized to [-0.5, 0.5]
                mcts_value = node.average_value
                return float(np.clip(mcts_value, -0.5, 0.5))
            else:
                return 0.0

    def _compute_value_reward(self, node: MCTSNode) -> float:
        """compute target value for value model training (temporal difference learning)"""
        if node.is_terminal:
            # terminal nodes: use the terminal reward as the true value
            return node.terminal_reward
        else:
            # non-terminal nodes: use MCTS value estimate as approximation of true value
            # this is the expected future reward from this state
            return node.average_value
