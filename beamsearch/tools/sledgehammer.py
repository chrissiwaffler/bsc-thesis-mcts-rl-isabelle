import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ..core import QIsabelleSession


class SledgehammerManager:
    def __init__(self, session: QIsabelleSession, max_workers: int) -> None:
        """Initialize Sledgehammer Tool Calling for the given session with a given maximum number of threads ouused for Sledgehammer"""
        self.session = session
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: dict[str, asyncio.Future] = {}
        self._futures_lock = Lock()  # Add thread safety

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
        # Filter to only wait for requested states
        futures_to_wait = {
            state: future
            for state, future in self.futures.items()
            if state in goal_states
        }

        if not futures_to_wait:
            return {}

        results: dict[str, str | None] = {}

        try:
            done, pending = await asyncio.wait(
                futures_to_wait.values(), timeout=timeout
            )

            # Process results
            for state, future in futures_to_wait.items():
                if future in done:
                    try:
                        results[state] = future.result()
                    except Exception:
                        results[state] = None
                else:
                    # Timed out
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
        # Cancel any pending sledgehammer tasks
        self.cancel_all()
        with self._futures_lock:
            self.futures.clear()

        # Shutdown thread pool
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            print(f"Error shutting down executor: {e}")
            # Force shutdown if graceful shutdown fails
            self.executor.shutdown(wait=False)
