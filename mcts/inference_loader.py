import gc
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from typing import Any

import psutil
import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from mcts.logging_utils import MCTSLogger


class ModelInferenceManager:
    """light wrapper around 2 vllm instances
    - manages vllm servers
    - model reloading for updated weights
    - starts 2 separate subprocesses for each model, so they don't interfere each other
    """

    def _merge_lora_weights(self, adapter_path: str) -> str:
        """Merge LoRA weights into base model and return path to merged model"""
        try:
            # get base model name
            base_model_name = self._get_base_model_from_adapter(adapter_path)

            self.logger.info(
                f"Merging LoRA weights from {adapter_path} into {base_model_name}"
            )

            # load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float16, device_map="auto"
            )

            # load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            # load LoRA model
            lora_model = PeftModel.from_pretrained(base_model, adapter_path)

            # merge weights
            merged_model: PreTrainedModel = lora_model.merge_and_unload()  # type: ignore

            # create temporary directory for merged model
            temp_dir = tempfile.mkdtemp(prefix="merged_model_")

            # save merged model
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # fix README.md base_model metadata
            # needed for HF model upload
            readme_path = os.path.join(temp_dir, "README.md")
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, encoding="utf-8") as f:
                        content = f.read()

                    # Replace any base_model that looks like a temp path, checkpoint path, or other invalid path
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
                            content = re.sub(
                                pattern, rf'\1"{base_model_name}"', content
                            )
                            fixed = True
                            break

                    if fixed:
                        with open(readme_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        self.logger.info(
                            f"Fixed base_model in merged model README.md to {base_model_name}"
                        )
                    else:
                        self.logger.info(
                            "No invalid base_model path found in merged model README.md"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to fix merged model README.md base_model: {e}"
                    )

            self.logger.info(f"Merged model saved to {temp_dir}")
            return temp_dir

        except Exception as e:
            self.logger.error(f"Failed to merge LoRA weights: {e}")
            raise

    def _resolve_model_path(self, path: str) -> str:
        """resolve model path - handle both Hugging Face names and local paths"""
        # check for invalid paths
        if path is None or not isinstance(path, str) or path.strip() == "":
            self.logger.warning(
                f"Invalid model path {path}, falling back to base model"
            )
            return "EleutherAI/llemma_7b"

        # check if it's a local file/directory that exists
        try:
            if os.path.exists(path):
                # if it's a LoRA checkpoint, merge the weights
                if self._is_lora_checkpoint(path):
                    self.logger.info(
                        f"Detected LoRA checkpoint at {path}, merging weights..."
                    )
                    return self._merge_lora_weights(path)
                else:
                    return os.path.abspath(path)
        except (TypeError, ValueError):
            self.logger.warning(
                f"Error checking path existence for {path}, falling back to base model"
            )
            return "EleutherAI/llemma_7b"

        # if it's a local path that doesn't exist and looks like a checkpoint path,
        # fall back to the base model
        if path.startswith("./checkpoints/") and not os.path.exists(path):
            self.logger.warning(
                f"Checkpoint path {path} does not exist, falling back to base model"
            )
            return "EleutherAI/llemma_7b"

        # otherwise, assume it's a Hugging Face model name and leave as-is
        return path

    def __init__(
        self,
        initial_policy_path: str,
        initial_value_path: str,
        base_port=8000,
        policy_cuda_visible_devices: str = "",
        value_cuda_visible_devices: str = "",
        policy_vllm_kwargs: dict[str, int | float | str | bool] | None = None,
        value_vllm_kwargs: dict[str, int | float | str | bool] | None = None,
    ) -> None:
        # for first iteration, paths are hugging face model names; not local files
        # for subsequent iterations, paths are local model checkpoints
        self.policy_path = self._resolve_model_path(initial_policy_path)
        self.value_path = self._resolve_model_path(initial_value_path)

        self.policy_port = base_port
        self.value_port = base_port + 1

        self.policy_cuda_visible_devices = policy_cuda_visible_devices
        self.value_cuda_visible_devices = value_cuda_visible_devices

        self.policy_llm_kwargs = policy_vllm_kwargs or {}
        self.value_llm_kwargs = value_vllm_kwargs or {}

        self.policy_process = None
        self.value_process = None
        self.using_existing_server = False
        self.logger = MCTSLogger.get_logger("inference_loader")

    def _check_model_health(self) -> bool:
        """Check if models have valid weights by making a simple test request"""
        try:
            # test policy server
            if self.policy_process and self.policy_process.poll() is None:
                response = requests.post(
                    f"http://localhost:{self.policy_port}/v1/completions",
                    json={
                        "model": self.policy_path,
                        "prompt": "test",
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=10,
                )
                if response.status_code != 200:
                    return False

            # test value server
            if self.value_process and self.value_process.poll() is None:
                response = requests.post(
                    f"http://localhost:{self.value_port}/v1/completions",
                    json={
                        "model": self.value_path,
                        "prompt": "test",
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=10,
                )
                if response.status_code != 200:
                    return False

            return True
        except Exception as e:
            self.logger.warning(f"Model health check failed: {e}")
            return False

    def _deep_clean_nans(self, obj):
        """Aggressively clean all nan and inf values from nested data structures"""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        elif isinstance(obj, dict):
            return {k: self._deep_clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_clean_nans(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._deep_clean_nans(item) for item in obj)
        else:
            return obj

    def _is_lora_checkpoint(self, path: str) -> bool:
        """Check if the given path contains LoRA adapter files"""
        if not os.path.exists(path):
            return False

        # check for LoRA adapter files
        adapter_config = os.path.join(path, "adapter_config.json")
        adapter_model = os.path.join(path, "adapter_model.safetensors")

        return os.path.exists(adapter_config) and os.path.exists(adapter_model)

    def _get_base_model_from_adapter(self, adapter_path: str) -> str:
        """Extract base model name from LoRA adapter config"""
        try:
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                import json

                with open(adapter_config_path) as f:
                    config = json.load(f)
                    return config.get("base_model_name_or_path", "EleutherAI/llemma_7b")
        except Exception as e:
            self.logger.warning(f"Failed to read base model from adapter config: {e}")

        # fallback to default base model
        return "EleutherAI/llemma_7b"

    def _build_vllm_command(
        self,
        model_path: str,
        port: int,
        kwargs: dict[str, Any],
    ) -> list[str]:
        """build vllm server command with given parameters"""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--port",
            str(port),
        ]

        # model path has already been resolved (LoRA weights merged if needed)
        cmd.extend(["--model", model_path])

        # add additional kwargs as command line arguments
        for key, value in kwargs.items():
            # convert underscore to dash for cli arguments
            cli_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_key)
            else:
                cmd.extend([cli_key, str(value)])

        return cmd

    def _wait_for_servers(self, timeout: int = 240) -> None:
        """wait for both servers to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            policy_ready = self._check_server_ready(self.policy_port)
            value_ready = self._check_server_ready(self.value_port)

            self.logger.info(
                f"server status - policy: {policy_ready}, value: {value_ready}"
            )

            if policy_ready and value_ready:
                return

            # check if processes are still running
            if self.policy_process and self.policy_process.poll() is not None:
                self.logger.error("policy server process has terminated")
                self._log_process_output("policy")
            if self.value_process and self.value_process.poll() is not None:
                self.logger.error("value server process has terminated")
                self._log_process_output("value")

            self.logger.info("waiting for servers to be ready...")
            time.sleep(2)

        # log process outputs before timeout
        self._log_process_output("policy")
        self._log_process_output("value")
        raise TimeoutError("vllm servers did not start within timeout period")

    def _log_process_output(self, server_type: str) -> None:
        """log stdout/stderr from a process for debugging"""
        if server_type == "policy" and self.policy_process:
            process = self.policy_process
            port = self.policy_port
        elif server_type == "value" and self.value_process:
            process = self.value_process
            port = self.value_port
        else:
            return

        stdout, stderr = process.communicate()

        self.logger.info(f"=== {server_type.upper()} server (port {port}) output ===")
        if stdout:
            self.logger.info(f"stdout:\n{stdout.decode()}")
        if stderr:
            self.logger.info(f"stderr:\n{stderr.decode()}")
        self.logger.info(f"=== end {server_type.upper()} server output ===")

    def _check_server_ready(self, port: int, timeout: int = 5) -> bool:
        """check if a vllm server is ready on the given port"""
        try:
            # first check if the HTTP server is up and get available models
            response = requests.get(
                f"http://localhost:{port}/v1/models", timeout=timeout
            )
            if response.status_code != 200:
                return False

            models_data = response.json()
            available_models = [model["id"] for model in models_data.get("data", [])]

            if not available_models:
                return False

            # use the first available model name for testing
            model_name = available_models[0]

            # now make a small test completion request to ensure model is loaded
            test_payload = {
                "model": model_name,
                "prompt": "test",
                "max_tokens": 1,
                "temperature": 0.0,
            }

            response = requests.post(
                f"http://localhost:{port}/v1/completions",
                json=test_payload,
                # use a longer timeout for completion requests
                timeout=60,
            )
            return response.status_code == 200

        except (requests.RequestException, ConnectionError):
            return False

    def start_vllm_server(self):
        """start both vllm servers via cli"""
        self.logger.info("starting vllm servers via cli...")

        # build command for policy server
        policy_cmd = self._build_vllm_command(
            self.policy_path,
            self.policy_port,
            self.policy_llm_kwargs,
        )

        # build command for value server
        value_cmd = self._build_vllm_command(
            self.value_path,
            self.value_port,
            self.value_llm_kwargs,
        )

        self.logger.info(f"policy command: {' '.join(policy_cmd)}")
        self.logger.info(f"value command: {' '.join(value_cmd)}")

        try:
            # start policy server
            self.logger.info(f"starting policy server on port {self.policy_port}")
            policy_env = os.environ.copy()
            if self.policy_cuda_visible_devices:
                policy_env["CUDA_VISIBLE_DEVICES"] = self.policy_cuda_visible_devices
                self.logger.info(
                    f"policy CUDA_VISIBLE_DEVICES: {self.policy_cuda_visible_devices}"
                )

            self.policy_process = subprocess.Popen(
                policy_cmd,
                env=policy_env,
            )

            time.sleep(10)

            # start value server
            self.logger.info(f"starting value server on port {self.value_port}")
            value_env = os.environ.copy()
            if self.value_cuda_visible_devices:
                value_env["CUDA_VISIBLE_DEVICES"] = self.value_cuda_visible_devices
                self.logger.info(
                    f"value CUDA_VISIBLE_DEVICES: {self.value_cuda_visible_devices}"
                )

            self.value_process = subprocess.Popen(
                value_cmd,
                env=value_env,
            )

            # wait for servers to be ready
            self._wait_for_servers()

            self.logger.info("vllm servers ready!")
            self.using_existing_server = False
        except Exception as e:
            self.logger.error(f"failed to start vllm servers: {e}")
            self.shutdown_servers()
            raise

    def _kill_vllm_processes(self):
        """nuclear option: kill all vllm processes"""
        killed_pids = []
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["cmdline"] and any(
                        "vllm" in str(arg) for arg in proc.info["cmdline"]
                    ):
                        proc.kill()
                        killed_pids.append(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if killed_pids:
                self.logger.info(
                    f"nuclear kill: eliminated vllm processes {killed_pids}"
                )
                # let them die
                time.sleep(5)
        except Exception as e:
            self.logger.error(f"nuclear kill failed: {e}")

    def shutdown_servers(self):
        """shutdown both vllm servers with safe memory cleanup"""
        if self.using_existing_server:
            self.logger.info("using existing servers - not shutting down")
            return

        self.logger.info("shutting down vllm servers...")

        # graceful shutdown
        for process, name in [
            (self.policy_process, "policy"),
            (self.value_process, "value"),
        ]:
            if process:
                self.logger.info(f"terminating {name} server...")
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(
                            f"{name} server did not terminate gracefully, killing..."
                        )
                        process.kill()
                        process.communicate()

        self.policy_process = None
        self.value_process = None

        self._kill_vllm_processes()

        # safe memory cleanup
        self.logger.info("starting safe memory cleanup after vllm shutdown")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(1)

            for _ in range(3):
                gc.collect()

            self.logger.info("safe cuda cleanup complete")

        self.logger.info("vllm shutdown and cleanup complete")

    def are_servers_running(self):
        """check if both vllm servers are running and responsive"""
        if not self.policy_process or not self.value_process:
            return False

        # check if processes are still running
        policy_running = self.policy_process.poll() is None
        value_running = self.value_process.poll() is None

        if not policy_running or not value_running:
            return False

        # check if servers are responsive
        policy_ready = self._check_server_ready(self.policy_port)
        value_ready = self._check_server_ready(self.value_port)

        return policy_ready and value_ready

    def reload_models(self, new_policy_path, new_value_path):
        """reload models with new checkpoints"""
        self.logger.info(f"reloading models: {new_policy_path}, {new_value_path}")

        self.shutdown_servers()

        # additional aggressive cleanup before starting new servers
        self.logger.info(
            "Performing additional GPU memory cleanup before starting vLLM servers..."
        )
        # force multiple garbage collections
        for _ in range(5):
            gc.collect()

        if torch.cuda.is_available():
            # empty cuda cache multiple times
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # small delay to ensure cleanup completes
            time.sleep(2)

        self.logger.info("GPU memory cleanup completed")

        # wait for GPU memory to be fully freed; can take a few seconds
        self.logger.info("Waiting for GPU memory to be fully freed...")
        max_wait_time = 30  # seconds
        wait_start = time.time()

        while time.time() - wait_start < max_wait_time:
            if torch.cuda.is_available():
                try:
                    # get actual number of available GPUs
                    num_gpus = torch.cuda.device_count()

                    # policy server GPU
                    policy_gpu = (
                        0
                        if self.policy_cuda_visible_devices == ""
                        else int(self.policy_cuda_visible_devices)
                    )
                    # value server GPU
                    value_gpu = (
                        1
                        if self.value_cuda_visible_devices == ""
                        else int(self.value_cuda_visible_devices)
                    )

                    # ensure GPUs are valid
                    if policy_gpu >= num_gpus:
                        self.logger.warning(
                            f"Policy GPU {policy_gpu} not available, using GPU 0"
                        )
                        policy_gpu = 0
                    if value_gpu >= num_gpus:
                        self.logger.warning(
                            f"Value GPU {value_gpu} not available, using GPU 0"
                        )
                        value_gpu = 0

                    # check memory on policy GPU
                    policy_free = (
                        torch.cuda.get_device_properties(policy_gpu).total_memory
                        - torch.cuda.memory_allocated(policy_gpu)
                    ) / 1024**3

                    # check memory on value GPU (might be same as policy GPU)
                    value_free = (
                        torch.cuda.get_device_properties(value_gpu).total_memory
                        - torch.cuda.memory_allocated(value_gpu)
                    ) / 1024**3

                    # wait until both GPUs have enough free memory
                    min_required = 20.0  # GiB (reduced for safety)
                    if policy_free >= min_required and value_free >= min_required:
                        self.logger.info(
                            f"GPU memory freed: Policy GPU {policy_gpu} has {policy_free:.1f} GiB, Value GPU {value_gpu} has {value_free:.1f} GiB"
                        )
                        break

                    self.logger.info(
                        f"Waiting for GPU memory... Policy GPU {policy_gpu}: {policy_free:.1f} GiB, Value GPU {value_gpu}: {value_free:.1f} GiB"
                    )
                    time.sleep(2.0)

                except Exception as e:
                    self.logger.warning(
                        f"Error checking GPU memory: {e}, continuing..."
                    )
                    break
            else:
                break

        # resolve model paths
        self.policy_path = self._resolve_model_path(new_policy_path)
        self.value_path = self._resolve_model_path(new_value_path)

        self.start_vllm_server()

    def _make_vllm_request(
        self,
        port: int,
        prompt: str,
        stop: list[str],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_logprobs: bool = True,
    ) -> list[tuple[str, list[float]]]:
        """make a request to vllm server and return generated text with per-token logprobs"""
        # check model health before making request
        if not self._check_model_health():
            self.logger.error("Model contains NaN weights, reloading...")
            return [("", [0.0])]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"http://localhost:{port}/v1/completions"
                headers = {"Content-Type": "application/json"}

                # get the correct model name for this port
                if port == self.value_port:
                    model_name = self.value_path
                else:
                    model_name = self.policy_path

                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "n": n,
                }

                # only request logprobs if requested -> avoid NaN issues
                if request_logprobs:
                    payload["logprobs"] = 1

                self.logger.debug(f"making request to {url} with payload: {payload}")
                response = requests.post(url, headers=headers, json=payload, timeout=60)

                if response.status_code != 200:
                    # if we get a 500 error and we requested logprobs, try again without logprobs
                    if response.status_code == 500 and request_logprobs:
                        self.logger.warning(
                            f"Got 500 error with logprobs, retrying without logprobs for port {port}"
                        )
                        payload_no_logprobs = payload.copy()
                        del payload_no_logprobs["logprobs"]
                        response = requests.post(
                            url, headers=headers, json=payload_no_logprobs, timeout=60
                        )
                        if response.status_code != 200:
                            # check if this is a NaN error
                            if (
                                "nan" in response.text.lower()
                                or "Out of range float values" in response.text
                            ):
                                self.logger.warning(
                                    f"NaN error persists even without logprobs for port {port}, returning default"
                                )
                                return [("", [0.0])]
                            self.logger.error(
                                f"http {response.status_code} from port {port} (retry without logprobs): {response.text}"
                            )
                            response.raise_for_status()
                    else:
                        # check if this is a NaN error
                        if (
                            "nan" in response.text.lower()
                            or "Out of range float values" in response.text
                        ):
                            self.logger.warning(
                                f"NaN error for port {port}, returning default"
                            )
                            return [("", [0.0])]
                        self.logger.error(
                            f"http {response.status_code} from port {port}: {response.text}"
                        )
                        response.raise_for_status()

                result = response.json()
                result = self._deep_clean_nans(result)
                self.logger.debug(f"response from port {port}: {result}")

                # ensure result is still a dict after cleaning
                if not isinstance(result, dict):
                    self.logger.error(
                        f"Unexpected response type after cleaning: {type(result)}"
                    )
                    return [("", [0.0])]

                outputs = result.get("choices", [])

                # ensure outputs is a list
                if not isinstance(outputs, list):
                    self.logger.error(f"Unexpected outputs type: {type(outputs)}")
                    return [("", [0.0])]

                if not outputs:
                    return [("", [0.0])]

                generated_texts_with_logprobs = []
                for choice in outputs:
                    # ensure choice is a dict
                    if not isinstance(choice, dict):
                        self.logger.error(f"Unexpected choice type: {type(choice)}")
                        continue

                    text_raw = choice.get("text", "")
                    text = str(text_raw).strip() if text_raw is not None else ""

                    # apply stop sequences
                    for stop_seq in stop:
                        if stop_seq in text:
                            text = text.split(stop_seq)[0]
                            break

                    # calculate per-token logprobs for the generated sequence
                    if request_logprobs and "logprobs" in choice:
                        logprobs_data = choice.get("logprobs", {})
                        # ensure logprobs_data is a dict
                        if not isinstance(logprobs_data, dict):
                            token_logprobs = []
                        else:
                            token_logprobs = logprobs_data.get("token_logprobs", [])
                            # ensure token_logprobs is a list
                            if not isinstance(token_logprobs, list):
                                token_logprobs = []

                        # filter out NaN values to prevent JSON serialization errors
                        valid_logprobs = [
                            lp
                            for lp in token_logprobs
                            if not (isinstance(lp, float) and math.isnan(lp))
                        ]
                    else:
                        valid_logprobs = [0.0]  # fallback single logprob
                        self.logger.debug(
                            f"Using default logprob value 0.0 for port {port} (logprobs not available)"
                        )

                    # ensure all logprobs are valid floats
                    per_token_logprobs = []
                    for lp in valid_logprobs:
                        if isinstance(lp, (int, float)) and not math.isnan(lp):
                            per_token_logprobs.append(float(lp))
                        else:
                            per_token_logprobs.append(0.0)

                    generated_texts_with_logprobs.append((text, per_token_logprobs))

                return generated_texts_with_logprobs

            except Exception as e:
                if attempt == max_retries - 1:
                    # check if this is a NaN error
                    if "nan" in str(e).lower() or "Out of range float values" in str(e):
                        self.logger.warning(
                            f"NaN error detected on attempt {attempt + 1}"
                        )
                        return [("", [0.0])]
                    self.logger.error(f"Final attempt failed for port {port}: {e}")
                    return [("", [0.0])]
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for port {port}: {e}"
                )
                time.sleep(0.1 * (attempt + 1))

        # if all retries failed
        return [("", [0.0])]

    def generate_commands(
        self,
        prompt: str,
        stop: list[str],
        n: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """generate command using policy model"""
        if not self.are_servers_running():
            self.logger.error("policy server requested but not initialized")
            return

        try:
            self.logger.debug(
                f"sending to vllm - prompt length: {len(prompt)}, stop: {stop}, n: {n}"
            )

            result = self._make_vllm_request(
                port=self.policy_port,
                prompt=prompt,
                stop=stop,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                request_logprobs=True,
            )

            self.logger.debug(f"vllm response: {result}")
            return result

        except Exception as e:
            self.logger.error(f"error generating commands: {e}")

    def estimate_value(
        self,
        prompt: str,
        stop: list[str],
        n: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ):
        """estimate value using value model"""
        if not self.are_servers_running():
            self.logger.error("value server requested but not initialized")
            return

        try:
            result = self._make_vllm_request(
                port=self.value_port,
                prompt=prompt,
                stop=stop,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                request_logprobs=True,
            )

            return result[0] if result else None

        except Exception as e:
            self.logger.error(f"error estimating value: {e}")


if __name__ == "__main__":
    logger = MCTSLogger.get_logger("inference_loader_test")
    # test with custom kwargs
    policy_kwargs = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.4,
        "max_model_len": 16384,
    }

    value_kwargs = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.2,
        "max_model_len": 8192,
    }

    manager = ModelInferenceManager(
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        policy_vllm_kwargs=policy_kwargs,
        value_vllm_kwargs=value_kwargs,
    )
    manager.start_vllm_server()

    # test command generation
    test_prompt = "what is the next step in this proof? n+0 = n"
    test_stop = ["\n", "</command>"]

    logger.info("\n=== testing command generation ===")
    commands = manager.generate_commands(
        prompt=test_prompt,
        stop=test_stop,
        n=3,
        max_tokens=100,
        temperature=0.7,
    )
    logger.info(f"generated commands: {commands}")

    # test value estimation
    logger.info("\n=== testing value estimation ===")
    value = manager.estimate_value(
        prompt=f"rate this command {commands} for this proof: {test_prompt}",
        stop=test_stop,
        n=1,
        max_tokens=50,
        temperature=0.3,
    )
    logger.info(f"estimated value: {value}")

    # test server status
    logger.info("\n=== testing server status ===")
    logger.info(f"servers running: {manager.are_servers_running()}")

    # cleanup

    manager.shutdown_servers()
    logger.info("=== done ===")
