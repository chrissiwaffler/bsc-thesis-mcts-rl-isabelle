from concurrent.futures import ThreadPoolExecutor

import pytest
import ray

from mcts.isabelle_interface import IsabelleInterface, get_port_manager_status


@pytest.fixture(scope="module")
def ray_initialized():
    """Initialize Ray for tests"""
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "excludes": [
                    "_unsloth_temporary_saved_buffers/**",
                    "wandb/**",
                    "xformers/**",
                    "atropos/**",
                    "prime-rl/**",
                    "SkyRL/**",
                    "step_by_step_proof_old/**",
                    "*.pt",
                    "*.pth",
                    "*.bin",
                    "*.gguf",
                    "__pycache__/**",
                    ".git/**",
                    "*.log",
                    "*.wandb",
                    "node_modules/**",
                    ".venv/**",
                    "dist/**",
                    "build/**",
                    "outputs/**",
                    "logs/**",
                    "checkpoints/**",
                ]
            }
        )
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def sample_theorems():
    """Sample theorems for testing"""
    theorem1 = r"""
    theory test
    imports Main
    begin

    theorem a: "\<forall> n \<in> \<nat>. n * 0 = 0"
    sorry

    end
    """

    theorem2 = r"""
    theory test
    imports Main
    begin

    theorem a: "\<forall> n \<in> \<nat>. n + 0 = n"
    sorry

    end
    """
    return theorem1, theorem2


def test_small_parallel_isabelle_sessions(ray_initialized, sample_theorems):
    """Test creating 8 Isabelle sessions in parallel using ThreadPoolExecutor"""
    theorem1, theorem2 = sample_theorems
    max_workers = 8

    def create_and_start_interface(worker_id, theorem_content, theorem_name):
        """Helper function to create interface and start proof"""
        interface = IsabelleInterface(f"small_parallel_worker_{worker_id}")
        with interface:
            print(f"Interface {worker_id}: {interface.get_worker_info()}")
            result = interface.start_proof(
                theorem_content=theorem_content,
                theorem_name=theorem_name,
            )
            print(
                f"Interface {worker_id} result successfully init: {result.success}; whole object: {result}"
            )
            return result, interface.get_worker_info()

    # Create 8 interfaces in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(8):
            # Alternate between the two theorems
            theorem_content = theorem1 if i % 2 == 0 else theorem2
            theorem_name = f"theorem_{i}"
            future = executor.submit(
                create_and_start_interface, i, theorem_content, theorem_name
            )
            futures.append(future)

        # Get all results
        results = []
        infos = []
        for future in futures:
            result, info = future.result()
            results.append(result)
            infos.append(info)

    print("port status: ", get_port_manager_status())

    # Verify all interfaces were successful
    success_count = sum(1 for result in results if result.success)
    print(f"Successful interfaces: {success_count}/8")

    for i, result in enumerate(results):
        assert result.success, f"Interface {i} failed: {result}"

    # Verify all ports are unique
    ports = [info["port"] for info in infos]
    unique_ports = set(ports)
    assert len(unique_ports) == 8, f"Expected 8 unique ports, got {len(unique_ports)}"
    print(f"Successfully allocated {len(unique_ports)} unique ports")


if __name__ == "__main__":
    # Run tests directly if needed
    pytest.main([__file__, "-v"])
