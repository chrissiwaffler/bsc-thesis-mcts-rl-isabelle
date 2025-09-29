import asyncio
import atexit
import contextlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import ray
from pydantic import BaseModel

from mcts.logging_utils import MCTSLogger
from mcts.utils import extract_imports, extract_theorem_statement
from qisabelle.client.session import QIsabelleSession as _QIsabelleSession

# Global semaphore to limit concurrent theorem initializations to 4
init_semaphore = threading.Semaphore(4)


# global port manager
@ray.remote(num_cpus=0, lifetime="detached", max_restarts=-1)  # type: ignore
class IsabellePortManager:
    """ray remote actor for managing qisabelle port allocation across workers"""

    def __init__(self, min_port: int = 17000, max_port: int = 17511) -> None:
        self.min_port = min_port
        self.max_port = max_port
        self.allocated_ports: set[int] = set()
        self.port_assignments: dict[str, int] = {}
        self.lock = threading.Lock()

    def __getstate__(self):
        """custom serialization to handle lock object"""
        state = self.__dict__.copy()
        # don't serialize the lock object
        del state["lock"]
        return state

    def __setstate__(self, state):
        """custom deserialization to recreate lock object"""
        self.__dict__.update(state)
        # recreate the lock object
        self.lock = threading.Lock()

    def allocate_port(self, worker_id: str) -> int | None:
        """allocate a free port for the given worker"""
        with self.lock:
            # check if worker already has a port
            if worker_id in self.port_assignments:
                return self.port_assignments[worker_id]

            # find free port
            for port in range(self.min_port, self.max_port + 1):
                if port not in self.allocated_ports:
                    self.allocated_ports.add(port)
                    self.port_assignments[worker_id] = port
                    logger = MCTSLogger.get_logger("isabelle_interface")
                    logger.info(f"allocated port {port} to worker {worker_id}")
                    return port

            print(f"no port available for worker {worker_id}")
            return None

    def release_port(self, worker_id: str) -> bool:
        """release port allocated to worker"""
        with self.lock:
            if worker_id in self.port_assignments:
                port = self.port_assignments[worker_id]
                self.allocated_ports.remove(port)
                del self.port_assignments[worker_id]
                print(f"released port {port} from worker {worker_id}")
                return True

            return False

    def get_port_status(self) -> dict:
        """get current port allocation status over all ports"""
        with self.lock:
            return {
                "allocated_ports": list(self.allocated_ports),
                "available_ports": [
                    p
                    for p in range(self.min_port, self.max_port + 1)
                    if p not in self.allocated_ports
                ],
                "assignments": dict(self.port_assignments),
            }


# singleton pattern for port manager access
_port_manager = None
# use ray's actor locking mechanism instead of threading.lock
_port_manager_lock = None


def get_port_manager():
    """get or create the global port manager"""
    global _port_manager, _port_manager_lock

    if _port_manager is None:
        try:
            _port_manager = IsabellePortManager.remote()  # type: ignore[attr-defined]
        except Exception as e:
            # if actor already exists, try to get it
            if "already exists" in str(e):
                try:
                    _port_manager = ray.get_actor("port_manager")
                except Exception as get_error:
                    logger = MCTSLogger.get_logger("isabelle_interface")
                    logger.error(f"failed to get existing port manager: {get_error}")
                    # if we can't get it either, re-raise the original error
                    raise e
            else:
                logger = MCTSLogger.get_logger("isabelle_interface")
                logger.error(f"failed to create port manager: {e}")
                raise e
    return _port_manager


class QIsabelleSession(_QIsabelleSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state_name = "init"


class SledgehammerManager:
    def __init__(self, session: QIsabelleSession, max_workers: int) -> None:
        """Initialize Sledgehammer Tool Calling for the given session with a given maximum number of threads ouused for Sledgehammer"""
        self.session = session
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: dict[str, asyncio.Future] = {}
        # thread safety
        self._futures_lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def get_sledgehammer_tactic(self, goal_state: str) -> str | None:
        """Get sledgehammer tactic for the given goal state if it succeeds"""
        try:
            # TODO: we could pass down added_facts into the sledgehammer call
            proof = self.session.hammer(goal_state)
            print(f"\nDEBUG: Sledgehammer raw output for {goal_state}:")
            print(f"  Type: {type(proof)}")
            print(f"  Content: {proof!r}")
            print(f"  Length: {len(proof) if proof else 0}")
            return proof
        except Exception as e:
            print(f"\nDEBUG: Sledgehammer failed for {goal_state}: {e}")
            return None

    async def start_sledgehammer_async(self, goal_state: str) -> None:
        """Start sledgehammer in background - non-blocking"""
        # check if state is already running
        with self._futures_lock:
            if goal_state in self.futures:
                return

            # create a future for this sledgehammer task
            loop = asyncio.get_event_loop()

            # run in thread pool to avoid blocking
            future = loop.run_in_executor(
                self.executor,
                lambda: self.get_sledgehammer_tactic(goal_state),
            )
            self.futures[goal_state] = future

        # set up callback to log results when done
        def log_results(fut):
            try:
                if fut.cancelled():
                    print(f"Sledgehammer cancelled for {goal_state}")
                    return
                tactic = fut.result()
                if tactic:
                    print(f"Sledgehammer completed for {goal_state}: found tactic")
            except asyncio.CancelledError:
                print(f"Sledgehammer cancelled for {goal_state}")
            except Exception as e:
                print(f"Sledgehammer failed for {goal_state}: {e}")
            finally:
                with self._futures_lock:
                    if goal_state in self.futures:
                        del self.futures[goal_state]

        future.add_done_callback(log_results)

    def check_sledgehammer_result(self, goal_state: str) -> list[str] | None:
        """Check if sledgehammer result are ready (non-blocking)

        Returns:
            None if still running
            [] if not running (no tactic found)
            [tactic] if finished and tactic found
        """
        with self._futures_lock:
            if goal_state not in self.futures:
                return []

            future = self.futures[goal_state]
            if not future.done():
                return None

            try:
                tactic = future.result()
                del self.futures[goal_state]
                return [tactic] if tactic else []
            except Exception as e:
                print(f"Error getting sledgehammer result for {goal_state}: {e}")
                del self.futures[goal_state]
                return []

    async def wait_for_all_results(
        self, timeout: float = 80.0, return_partial: bool = True
    ) -> dict[str, str | None]:
        """Wait for all pending sledgehammer results with timeout

        Returns:
            Dict mapping goal_state to tactic (or None if no tactic found)
        """
        if not self.futures:
            return {}

        print(f"Waiting for {len(self.futures)} pending sledgehammer results...")

        # create a copy of futures to work with
        pending_futures = dict(self.futures)
        results: dict[str, str | None] = {}

        try:
            done, pending = await asyncio.wait(
                pending_futures.values(),
                timeout=timeout,
                return_when=(
                    asyncio.ALL_COMPLETED
                    if not return_partial
                    else asyncio.FIRST_EXCEPTION
                ),
            )
            print(
                f"{len(done)} sledgehammer tasks completed, {len(pending)} timed out/pending"
            )

            for future in done:
                for goal_state, fut in pending_futures.items():
                    if fut is future:
                        try:
                            tactic = future.result()
                            results[goal_state] = tactic
                            if tactic:
                                print(f"Sledgehammer found tactic for {goal_state}")
                            else:
                                print(f"Sledgehammer found no tactic for {goal_state}")
                        except Exception as e:
                            print(f"Sledgehammer error for {goal_state}: {e}")
                            results[goal_state] = None
                        break

            # handle timed out futures
            for future in pending:
                future.cancel()
                for goal_state, fut in pending_futures.items():
                    if fut is future:
                        print(f"Sledgehammer timeout for {goal_state}")
                        if return_partial:
                            results[goal_state] = None
                        break

        except Exception as e:
            print(f"Error waiting for sledgehammer results: {e}")
            # cancel all pending ones
            for future in self.futures.values():
                if not future.done():
                    future.cancel()

        return results

    async def wait_for_specific_results(
        self, goal_states: list[str], timeout: float = 80.0
    ) -> dict[str, str | None]:
        """Wait for specific sledgehammer results

        Returns:
            Dict mapping goal_state to tactic for the requested states
        """
        # filter to only wait for requested states
        futures_to_wait = {
            state: future
            for state, future in self.futures.items()
            if state in goal_states
        }

        if not futures_to_wait:
            return {}

        results: dict[str, str | None] = {}

        try:
            done, _pending = await asyncio.wait(
                futures_to_wait.values(), timeout=timeout
            )

            # process results
            for state, future in futures_to_wait.items():
                if future in done:
                    try:
                        results[state] = future.result()
                    except Exception:
                        results[state] = None
                else:
                    # timed out
                    future.cancel()
                    results[state] = None

        except Exception as e:
            print(f"Error waiting for specific results: {e}")

        return results

    def get_pending_count(self) -> int:
        """Get number of pending sledgehammer tasks"""
        return len(self.futures)

    def get_pending_states(self) -> list[str]:
        """Get list of goal states with pending sledgehammer tasks"""
        return list(self.futures.keys())

    def cancel_all(self) -> int:
        """Cancel all pending sledgehammer tasks

        Returns:
            Number of tasks cancelled
        """
        cancelled = 0
        for future in self.futures.values():
            if not future.done():
                future.cancel()
                cancelled += 1
        return cancelled

    def cleanup(self) -> None:
        """Clean up resources"""
        # cancel any pending sledgehammer tasks
        self.cancel_all()
        with self._futures_lock:
            self.futures.clear()

        # shutdown thread pool
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            print(f"Error shutting down executor: {e}")
            # force shutdown if graceful shutdown fails
            self.executor.shutdown(wait=False)


class IsabelleSuccessResult(BaseModel):
    """successful isabelle command execution result"""

    success: bool = True
    is_done: bool
    result: str
    state_name: str


class IsabelleErrorResult(BaseModel):
    """failed isabelle command execution result"""

    success: bool = False
    is_done: bool
    state_name: str
    error: str


IsabelleResult = IsabelleSuccessResult | IsabelleErrorResult


class IsabelleInterface:
    """thread-safe isabelle interface for parallel execution"""

    def __init__(self, worker_id: str | None = None):
        # todo: set max workers? or limit cpu usage
        self.worker_id = worker_id or str(uuid.uuid4())
        self.port: None | int = None
        self._session: QIsabelleSession | None = None

        self._sledgehammer_manager: SledgehammerManager | None = None

        # using the stringified version of this number to store new states
        self.state_counter = 0

        # str(state_counter) -> (is_done, result)
        self.active_states: dict[str, tuple[bool, str]] = {}
        self._initialized = False
        self._cleanup_registered = False

    def _ensure_initialized(self) -> bool:
        """lazy initialization of isabelle session with port allocation"""
        if self._initialized:
            return True

        try:
            # get port from global manager
            port_manager = get_port_manager()
            self.port = ray.get(port_manager.allocate_port.remote(self.worker_id))  # type: ignore

            if self.port is None:
                print(f"failed to allocate port for worker {self.worker_id}")
                return False

            # init session with allocated port
            print(
                f"initializing isabelle session on port {self.port} for worker {self.worker_id}"
            )
            # use base hol session - storage isolation prevents conflicts
            session_name = "HOL"
            self._session = QIsabelleSession(
                session_name=session_name, session_roots=[], port=self.port, debug=False
            )

            self._sledgehammer_manager = SledgehammerManager(
                self._session, max_workers=2
            )
            self._initialized = True

            # register cleanup with ray if not already registered
            if not self._cleanup_registered:
                # register cleanup using atexit (simpler approach)
                atexit.register(self._ray_cleanup_hook)
                self._cleanup_registered = True

            print(f"isabelle interface initialized on port {self.port}")
            return True

        except Exception as e:
            logger = MCTSLogger.get_logger("isabelle_interface")
            logger.error(f"Failed to initialize Isabelle interface: {e}")
            if self.port:
                # release port
                try:
                    port_manager = get_port_manager()
                    ray.get(port_manager.release_port.remote(self.worker_id))  # type: ignore
                except Exception as port_error:
                    logger.error(
                        f"Failed to release port during initialization error: {port_error}"
                    )
                finally:
                    self.port = None
            return False

    def _ray_cleanup_hook(self):
        """Ray shutdown hook"""
        self.cleanup()

    def cleanup(self):
        print(f"cleaning up Isabelle interface for worker {self.worker_id}")

        self.active_states = {}
        self.state_counter = 0

        if self._sledgehammer_manager:
            self._sledgehammer_manager.cleanup()
            self._sledgehammer_manager = None

        if self._session:
            with contextlib.suppress(Exception):
                self._session.__exit__(None, None, None)
                print("Closed Isabelle session")

        # release port with robust error handling
        if self.port:
            logger = MCTSLogger.get_logger("isabelle_interface")
            try:
                port_manager = get_port_manager()
                ray.get(port_manager.release_port.remote(self.worker_id))  # type: ignore
                print(f"released port {self.port} for worker {self.worker_id}")
            except Exception as e:
                logger.error(
                    f"failed to release port {self.port} for worker {self.worker_id}: {e}"
                )
                # try to kill and recreate the port manager if it's dead
                try:
                    global _port_manager
                    if _port_manager is not None:
                        ray.kill(_port_manager)
                        _port_manager = None
                        logger.info("Killed dead port manager actor")
                except Exception as kill_error:
                    logger.error(f"Failed to kill port manager: {kill_error}")
            finally:
                self.port = None

        self._initialized = False

    def start_proof(self, theorem_name: str, theorem_content: str) -> IsabelleResult:
        """init a new isabelle session with the given theorem"""
        with init_semaphore:
            if not self._ensure_initialized():
                return IsabelleErrorResult(
                    error="Failed to initialize Isabelle session",
                    state_name="error",
                    is_done=True,
                )
            if not self._session:
                return IsabelleErrorResult(
                    error="Isabelle session not initialized",
                    state_name="error",
                    is_done=True,
                )

            theorem_statement = extract_theorem_statement(theorem_content)
            imports = extract_imports(theorem_content)

            try:
                self._session.new_theory(
                    theory_name=f"Test_{theorem_name}",
                    new_state_name=self._session.initial_state_name,
                    imports=imports,
                    only_import_from_session_heap=False,
                )
            except Exception:
                print(f"Failed to start session with imports {imports}")
                print("Falling back to Main import...")
                self._session.new_theory(
                    theory_name=f"Test_{theorem_name}",
                    new_state_name=self._session.initial_state_name,
                    imports=["Main"],
                    only_import_from_session_heap=False,
                )

            return self.next_step(self._session.initial_state_name, theorem_statement)

    def next_step(self, current_state_name: str, command: str) -> IsabelleResult:
        """executes the tactic and returns result"""
        if not self._session:
            return IsabelleErrorResult(
                error="No Isabelle session available",
                state_name="error",
                is_done=True,
            )

        new_state_name = str(self.state_counter)
        self.state_counter += 1

        try:
            is_done, result = self._session.execute(
                current_state_name,
                command,
                new_state_name,
            )
            self.active_states[new_state_name] = (is_done, result)
            return IsabelleSuccessResult(
                is_done=is_done,
                result=result,
                state_name=new_state_name,
            )
        except Exception as e:
            error_msg = str(e).split("Traceback:")[0].strip()
            return IsabelleErrorResult(
                state_name=new_state_name,
                error=error_msg,
                is_done=True,
            )

    def __enter__(self):
        if not self._ensure_initialized():
            raise RuntimeError("Failed to initialize Isabelle interface")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def get_worker_info(self) -> dict:
        """get worker information for debuggin"""
        return {
            "worker_id": self.worker_id,
            "port": self.port,
            "initialized": self._initialized,
            "active_states": len(self.active_states),
            "state_counter": self.state_counter,
        }


# helper functions for SkyRL environment usage
def get_port_manager_status() -> dict:
    """get curretn port allocation status"""
    try:
        port_manager = get_port_manager()
        return ray.get(port_manager.get_port_status.remote())  # type: ignore
    except Exception as e:
        return {"error": str(e)}


def get_port_manager_health() -> dict:
    """get port manager health status"""
    try:
        port_manager = get_port_manager()
        return ray.get(port_manager.health_check.remote())  # type: ignore
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ray compatibility checker
def check_ray_compatibility() -> bool:
    """check if ray is properly initialized"""
    try:
        ray.cluster_resources()
        print("ray cluster is available")
        return True
    except Exception as e:
        print(f"ray cluster not available: {e}")
        return False


if __name__ == "__main__":
    # test the port manager functionality
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

    if check_ray_compatibility():
        print("testing port alloc ...")

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

        print("port status: ", get_port_manager_status())

    else:
        print("please init ray first: ray.init()")


__all__ = [
    "IsabelleErrorResult",
    "IsabelleInterface",
    "IsabelleResult",
    "IsabelleSuccessResult",
    "check_ray_compatibility",
    "get_port_manager_health",
    "get_port_manager_status",
]
