from concurrent.futures import ThreadPoolExecutor

import pytest
import ray

from mcts.isabelle_interface import (
    IsabelleInterface,
    check_ray_compatibility,
    get_port_manager_status,
)


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


def test_synchronous_isabelle_sessions(ray_initialized, sample_theorems):
    """Test initializing two Isabelle sessions synchronously"""
    theorem1, theorem2 = sample_theorems

    # test basic allocation
    interface1 = IsabelleInterface("test_worker_1")
    interface2 = IsabelleInterface()

    with interface1:
        print(f"Interface 1: {interface1.get_worker_info()}")
        res1 = interface1.start_proof(
            theorem_content=theorem1,
            theorem_name="theorem1",
        )
        print(
            f"Interface 2 result successfully init: {res1.success}; whole object: {res1}"
        )

        # parallel usage -> interface 2 should use a different port compared to interface 1
        with interface2:
            print(f"Interface 2: {interface2.get_worker_info()}")
            res2 = interface2.start_proof(
                theorem_content=theorem2,
                theorem_name="theorem2",
            )
            print(
                f"Interface 2 result successfully init: {res2.success}; whole object: {res2}"
            )

            # Get port info while interfaces are still active
            info1 = interface1.get_worker_info()
            info2 = interface2.get_worker_info()
            port1 = info1["port"]
            port2 = info2["port"]

    print("port status: ", get_port_manager_status())

    # Verify both interfaces were successful
    assert res1.success, f"Interface 1 failed: {res1}"
    assert res2.success, f"Interface 2 failed: {res2}"

    # Verify different ports were used (using captured port values)
    assert port1 is not None, "Interface 1 has no port assigned"
    assert port2 is not None, "Interface 2 has no port assigned"
    assert port1 != port2, f"Both interfaces used the same port: {port1}"


def test_parallel_isabelle_sessions(ray_initialized, sample_theorems):
    """Test creating 16 Isabelle sessions in parallel using ThreadPoolExecutor (config supports up to 512)"""
    theorem1, theorem2 = sample_theorems
    max_workers = 16

    def create_and_start_interface(worker_id, theorem_content, theorem_name):
        """Helper function to create interface and start proof"""
        interface = IsabelleInterface(f"parallel_worker_{worker_id}")
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

    # Create 512 interfaces in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(16):
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
    print(f"Successful interfaces: {success_count}/16")

    for i, result in enumerate(results):
        assert result.success, f"Interface {i} failed: {result}"

    # Verify all ports are unique
    ports = [info["port"] for info in infos]
    unique_ports = set(ports)
    assert len(unique_ports) == 16, f"Expected 16 unique ports, got {len(unique_ports)}"
    print(f"Successfully allocated {len(unique_ports)} unique ports")


def test_ray_compatibility(ray_initialized):
    """Test Ray compatibility check"""
    assert check_ray_compatibility(), "Ray compatibility check failed"


if __name__ == "__main__":
    # Run tests directly if needed
    pytest.main([__file__, "-v"])
