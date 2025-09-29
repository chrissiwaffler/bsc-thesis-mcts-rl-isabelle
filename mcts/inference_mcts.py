import math
from typing import cast

import numpy as np
from langchain_core.output_parsers import XMLOutputParser

from mcts.inference_loader import ModelInferenceManager
from mcts.logging_utils import MCTSLogger
from mcts.shared_types import CommandGenerationResponse, ValueEstimateResponse


class MCTSInference:
    """inference wrapper that the MCTS search will use
    - custom templates for commands generation and value estimation
    - uses the light-weight ModelInferenceManager that is a wrapper around 2 vllm servers
    """

    def __init__(
        self,
        model_manager: ModelInferenceManager,
        temperature_policy: float,
        temperature_value: float,
    ) -> None:
        self.model_manager = model_manager
        self.xml_parser_command = XMLOutputParser(tags=["response"])
        self.xml_parser_value = XMLOutputParser(tags=["response"])
        self.logger = MCTSLogger.get_logger("inference_mcts")
        self.temperature_policy = temperature_policy
        self.temperature_value = temperature_value

    def _prompt_template_policy(
        self,
        theorem_statement: str,
        proof_history: list[tuple[str, str]],
        current_state: str,
        available_lemmas: list[str] | None,
        proof_mode: str = "prove",
    ) -> str:
        """prompt template for the policy model with xml tags"""
        history_text = ""
        if proof_history:
            for i, (cmd, result) in enumerate(proof_history):
                history_text += f"Step {i + 1}:\n  Command: {cmd}\n  Result: {result}\n"

        lemmas_text = ""
        if available_lemmas is not None:
            lemmas_text = "Available lemmas:\n"
            for lemma in available_lemmas:
                lemmas_text += f"  - {lemma}\n"

        return f"""
Example:
THEOREM: theorem plus_zero: "n + 0 = n"

PROOF HISTORY:

CURRENT STATE:
1. n + 0 = n

Available lemmas:
  - add_0
  - add_succ
MODE: prove
<response>
<think>
We need to prove n + 0 = n. This follows by induction on n.
</think>
<command>
apply (induct n)
</command>
</response>

Example:
THEOREM: theorem plus_succ: "n + Suc m = Suc (n + m)"

PROOF HISTORY:

CURRENT STATE:
1. n + Suc m = Suc (n + m)

Available lemmas:
  - add_0
  - add_succ
MODE: prove
<response>
<think>
This requires induction on m to handle the successor on the right side of the equation.
</think>
<command>
apply (induct m)
</command>
</response>

Your turn:
THEOREM: {theorem_statement}

PROOF HISTORY:
{history_text}

CURRENT STATE:
{current_state}

{lemmas_text}
MODE: {proof_mode}
"""

    def _prompt_template_value(
        self,
        theorem_statement: str,
        command: str,
        execution_result: str,
        proof_history: list[tuple[str, str]],
        proof_mode: str = "prove",
    ) -> str:
        """prompt template for the value model to evaluate the command that has the given execution result (updated subgoals)
        simplified to just output a score value for better training
        """
        history_text = ""
        if proof_history:
            for i, (cmd, result) in enumerate(proof_history):
                history_text += f"Step {i + 1}:\n  Command: {cmd}\n  Result: {result}\n"

        return f"""Evaluate the progress of this Isabelle proof step. Output only a score between -1.0 and 1.0 indicating how much closer this brings us to completing the proof.

THEOREM: {theorem_statement}

PROOF HISTORY:
{history_text}

EXECUTED COMMAND:
{command}

RESULTING SUBGOALS:
{execution_result}

MODE: {proof_mode}

Score:"""

    def generate_commands(
        self,
        theorem_statement: str,
        proof_mode: str,
        proof_history: list[tuple[str, str]] | None,
        current_state: str,
        available_lemmas: list[str] | None,
        n: int,
    ) -> list[CommandGenerationResponse] | None:
        """generate k commands with thinking for mcts expansion"""
        formatted_input = self._prompt_template_policy(
            theorem_statement=theorem_statement,
            proof_history=proof_history or [],
            current_state=current_state,
            proof_mode=proof_mode,
            available_lemmas=available_lemmas,
        )
        self.logger.info(
            f"[mcts-inference] sending prompt (first 200 chars): {formatted_input[:200]}..."
        )

        # check if servers are running before making request
        if not self.model_manager.are_servers_running():
            self.logger.error("[mcts-inference] vLLM servers not running!")
            return None

        raw_response = self.model_manager.generate_commands(
            prompt=formatted_input,
            stop=["</response>"],
            n=n,
            temperature=self.temperature_policy,
        )
        self.logger.info(
            f"[mcts-inference] raw_response received: {len(raw_response) if raw_response else 0} items"
        )

        if not raw_response:
            self.logger.warning(
                f"[mcts-inference] no commands response generated for command {proof_history}"
            )
            return None

        commands: list[CommandGenerationResponse] = []
        self.logger.info(
            f"[mcts-inference] processing {len(raw_response)} raw responses"
        )
        for i, (response_text, logprobs) in enumerate(raw_response):
            self.logger.debug(f"[mcts-inference] raw response {i}: '{response_text}'")
            # check if response is empty or whitespace-only
            if not response_text or not response_text.strip():
                self.logger.warning(
                    f"[mcts-inference] empty response {i} from model, skipping"
                )
                continue

            try:
                # since we're using </response> as a stop token, we need to add it back for proper xml parsing
                cleaned_response = response_text.strip()
                if not cleaned_response.endswith("</response>"):
                    cleaned_response += "</response>"

                # parse xml response to extract response element
                try:
                    parsed_response = self.xml_parser_command.parse(cleaned_response)
                    self.logger.debug(
                        f"[mcts-inference] xml parsing result: {parsed_response}"
                    )
                except Exception as parse_error:
                    self.logger.warning(
                        f"[mcts-inference] XML parsing failed for response '{cleaned_response}': {parse_error}"
                    )
                    continue

                # extract response content (handle both string and list)
                response_data = parsed_response.get("response", [])
                thinking_text = ""
                command_text = ""

                if isinstance(response_data, list) and len(response_data) > 0:
                    # the parser returns a list of dicts for flat tags
                    for item in response_data:
                        if isinstance(item, dict):
                            if "thinking" in item:
                                thinking_text = item.get("thinking", "").strip()
                            elif "command" in item:
                                command_text = item.get("command", "").strip()

                # fallback: if xml parsing failed, try to extract content manually
                if not command_text and "<command>" in cleaned_response:
                    try:
                        command_start = cleaned_response.find("<command>") + len(
                            "<command>"
                        )
                        command_end = cleaned_response.find("</command>", command_start)
                        if command_end == -1:
                            command_end = len(cleaned_response)
                        command_text = cleaned_response[
                            command_start:command_end
                        ].strip()
                    except Exception:
                        pass

                if not thinking_text and "<think>" in cleaned_response:
                    try:
                        thinking_start = cleaned_response.find("<think>") + len(
                            "<think>"
                        )
                        thinking_end = cleaned_response.find("</think>", thinking_start)
                        if thinking_end == -1:
                            thinking_end = len(cleaned_response)
                        thinking_text = cleaned_response[
                            thinking_start:thinking_end
                        ].strip()
                    except Exception:
                        pass

                if command_text:
                    self.logger.debug(
                        f"[mcts-inference] xml parsing found command: {command_text}"
                    )
                    if thinking_text:
                        self.logger.debug(f"[mcts-inference] thinking: {thinking_text}")

                    commands.append(
                        {
                            "command": command_text,
                            "thinking": thinking_text,
                            "full_prompt": formatted_input,
                            "full_response": response_text,
                            "logprobs": logprobs,
                        }
                    )
                else:
                    self.logger.warning(
                        f"[mcts-inference] no command found in xml response: {response_text}"
                    )
                    continue
            except Exception as e:
                self.logger.warning(
                    f"[mcts-inference] error parsing response '{response_text}': {e}"
                )
                continue

        return commands

    def estimate_value(
        self,
        theorem_statement: str,
        command: str,
        execution_result: str,
        proof_history: list[tuple[str, str]] | None,
        proof_mode: str,
    ) -> ValueEstimateResponse | None:
        """estimate value of proof state for mcts"""
        formatted_input = self._prompt_template_value(
            theorem_statement=theorem_statement,
            command=command,
            execution_result=execution_result,
            proof_history=proof_history or [],
            proof_mode=proof_mode,
        )

        # Check if servers are running before making request
        if not self.model_manager.are_servers_running():
            self.logger.error(
                "[mcts-inference] vLLM servers not running for value estimation!"
            )
            return None

        raw_response = self.model_manager.estimate_value(
            prompt=formatted_input,
            stop=["\n"],
            temperature=self.temperature_value,
        )
        self.logger.info(
            f"[mcts-inference] value raw_response received: {raw_response is not None}"
        )

        if not raw_response:
            self.logger.warning(
                f"[mcts-inference] no value estimation response generated for command {command}"
            )
            return None

        response_text, logprobs = raw_response
        if not response_text or not response_text.strip():
            self.logger.warning(
                "[mcts-inference] empty response from value model, returning neutral value"
            )
            # ensure logprobs is not NaN
            if isinstance(logprobs, float) and math.isnan(logprobs):
                logprobs = 0.0

            return cast(
                ValueEstimateResponse,
                {
                    # neutral value
                    "value": 0.0,
                    "full_prompt": formatted_input,
                    "full_response": response_text,
                    "logprobs": (
                        [
                            float(lp)
                            for lp in logprobs
                            if isinstance(lp, (int, float)) and not math.isnan(lp)
                        ]
                        if isinstance(logprobs, list)
                        else (
                            [float(logprobs)]
                            if isinstance(logprobs, (int, float))
                            and not math.isnan(logprobs)
                            else []
                        )
                    ),
                },
            )

        try:
            cleaned_response = response_text.strip()

            # extract numeric value from the response
            # clean the response to keep only numeric characters, decimal point, and minus sign
            value_cleaned = "".join(
                c for c in cleaned_response if c.isdigit() or c == "." or c == "-"
            )

            if value_cleaned:
                value = float(value_cleaned)
                # check for NaN and replace with 0.0
                if math.isnan(value):
                    value_clipped = 0.0
                else:
                    value_clipped = np.clip(value, -1.0, 1.0)
                self.logger.debug(
                    f"[mcts-inference] parsed value score: {value_clipped}"
                )
            else:
                self.logger.warning(
                    f"[mcts-inference] no numeric value found in response: '{cleaned_response}', using neutral value"
                )
                value_clipped = 0.0

        except Exception as e:
            self.logger.warning(
                f"[mcts-inference] error parsing value response '{response_text}': {e}, using neutral value"
            )
            value_clipped = 0.0

        # ensure logprobs is valid
        if isinstance(logprobs, list):
            # clean any NaN values in the list
            logprobs = [
                (
                    float(lp)
                    if isinstance(lp, (int, float)) and not math.isnan(lp)
                    else 0.0
                )
                for lp in logprobs
            ]
        else:
            # fallback for legacy single float
            logprobs = [
                (
                    float(logprobs)
                    if isinstance(logprobs, (int, float)) and not math.isnan(logprobs)
                    else 0.0
                )
            ]

        return cast(
            ValueEstimateResponse,
            {
                "value": value_clipped,
                "full_prompt": formatted_input,
                "full_response": response_text,
                "logprobs": logprobs,
            },
        )


if __name__ == "__main__":
    # info: update config accordingly to your current hardware
    manager = ModelInferenceManager(
        "EleutherAI/llemma_7b",
        "Qwen/Qwen3-0.6B",
        policy_vllm_kwargs={
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "max_model_len": 16384,
        },
        value_vllm_kwargs={
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.2,
            "max_model_len": 8192,
        },
    )
    manager.start_vllm_server()

    # create mcts inference wrapper
    mcts_inference = MCTSInference(manager, 1.0, 0.3)

    # test data
    theorem_statement = 'theorem add_comm: "∀n m. n + m = m + n"'
    current_state = "1. n + m = m + n"
    proof_history = []
    available_lemmas = ["add_0", "add_succ"]

    print("=== testing command generation ===")
    commands = mcts_inference.generate_commands(
        theorem_statement=theorem_statement,
        proof_mode="prove",
        proof_history=proof_history,
        current_state=current_state,
        available_lemmas=available_lemmas,
        n=3,
    )

    if commands:
        print(f"generated {len(commands)} commands:")
        for i, cmd in enumerate(commands):
            print(f"  {i + 1}. command: {cmd['command']}")
            print(f"     thinking: {cmd['thinking']}")
    else:
        print("no commands generated")

    print("\n=== testing value estimation ===")
    if commands:
        test_command = commands[0]["command"]
        execution_result = "2. n + 0 = n"

        value_estimate = mcts_inference.estimate_value(
            theorem_statement=theorem_statement,
            command=test_command,
            execution_result=execution_result,
            proof_history=proof_history,
            proof_mode="prove",
        )

        if value_estimate:
            print(f"estimated value: {value_estimate['value']}")
        else:
            print("no value estimate generated")

    print("\n=== testing engine status ===")
    print(f"servers running: {manager.are_servers_running()}")

    print("\n=== shutting down servers ===")
    manager.shutdown_servers()
    print("=== done ===")
