"""
Centralized WandB Manager for MCTS Training

This module provides a clean, centralized interface for all WandB operations
in the MCTS training pipeline. It handles initialization, logging, and step
management in a consistent way.
"""

import os
from typing import Any, cast

from mcts.logging_utils import MCTSLogger

logger = MCTSLogger.get_logger("wandb_manager")


class WandbManager:
    """Centralized manager for WandB operations in MCTS training."""

    def __init__(self):
        self._run = None
        self._initialized = False
        self._step_counters = {
            "mcts": 0,
            "training": 0,
            "evaluation": 0,
        }

    @property
    def run(self):
        """Get the current wandb run, or None if not initialized."""
        return self._run

    @property
    def is_initialized(self) -> bool:
        """Check if wandb is properly initialized."""
        return self._initialized and self._run is not None

    def initialize(
        self,
        project: str = "mcts-training",
        name: str | None = None,
        config: dict[str, Any] | Any | None = None,
        **kwargs,
    ) -> bool:
        """
        Initialize wandb run.

        Args:
            project: WandB project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary
            **kwargs: Additional wandb.init arguments

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.warning("WandB already initialized, skipping")
            return True

        try:
            # check if wandb is disabled via environment
            if os.getenv("WANDB_DISABLED", "false").lower() == "true":
                logger.info("WandB disabled via WANDB_DISABLED environment variable")
                return False

            import wandb

            # Convert config object to dict if it has attributes
            config_dict = {}
            if config is not None:
                if hasattr(config, "__dict__"):
                    # Handle dataclass or object with attributes
                    config_dict = {
                        k: v
                        for k, v in config.__dict__.items()
                        if not k.startswith("_")
                    }
                elif isinstance(config, dict):
                    # Handle existing dict
                    config_dict = config
                else:
                    # Handle other types by converting to string representation
                    config_dict = {"config": str(config)}

            # initialize wandb
            self._run = wandb.init(
                project=project, name=name, config=config_dict, reinit=True, **kwargs
            )

            # define metrics with step management
            self._run.define_metric("mcts/*", step_metric="mcts_step")
            self._run.define_metric("training/*", step_metric="training_step")
            self._run.define_metric("evaluation/*", step_metric="evaluation_step")
            self._run.define_metric(
                "final_evaluation/*", step_metric="final_evaluation_step"
            )

            self._initialized = True
            logger.info(
                f"Initialized WandB run: {self._run.name} in project: {project}"
            )
            return True

        except ImportError:
            logger.warning("WandB not available, logging disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            return False

    def log_mcts_metrics(
        self, metrics: dict[str, Any], step: int | None = None
    ) -> None:
        """Log MCTS-specific metrics."""
        if not self.is_initialized:
            return

        run = cast(
            Any, self._run
        )  # Type assertion since is_initialized ensures _run is not None

        if step is None:
            step = self._step_counters["mcts"]
            self._step_counters["mcts"] += 1

        # add step information
        log_data = {f"mcts/{k}": v for k, v in metrics.items()}
        log_data["mcts_step"] = step

        run.log(log_data)

    def log_training_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        model_type: str | None = None,
    ) -> None:
        """Log training-specific metrics."""
        if not self.is_initialized:
            return

        run = cast(
            Any, self._run
        )  # Type assertion since is_initialized ensures _run is not None

        if step is None:
            step = self._step_counters["training"]
            self._step_counters["training"] += 1

        # create prefix based on model type
        if model_type:
            prefix = f"{model_type}_training"
        else:
            prefix = "training"

        # add step information
        log_data = {f"{prefix}/{k}": v for k, v in metrics.items()}
        log_data[f"{prefix}_step"] = step

        run.log(log_data)

    def log_evaluation_metrics(
        self, metrics: dict[str, Any], step: int | None = None
    ) -> None:
        """Log evaluation-specific metrics."""
        if not self.is_initialized:
            return

        run = cast(
            Any, self._run
        )  # Type assertion since is_initialized ensures _run is not None

        if step is None:
            step = self._step_counters["evaluation"]
            self._step_counters["evaluation"] += 1

        # add step information
        log_data = {f"evaluation/{k}": v for k, v in metrics.items()}
        log_data["evaluation_step"] = step

        run.log(log_data)

    def log_final_evaluation_metrics(self, metrics: dict[str, Any]) -> None:
        """Log final evaluation metrics (single point, no stepping)."""
        if not self.is_initialized:
            return

        run = cast(
            Any, self._run
        )  # Type assertion since is_initialized ensures _run is not None

        log_data = {f"final_evaluation/{k}": v for k, v in metrics.items()}
        run.log(log_data)

    def log_config(self, config: dict[str, Any]) -> None:
        """Update the run configuration."""
        if not self.is_initialized:
            return

        run = cast(
            Any, self._run
        )  # Type assertion since is_initialized ensures _run is not None

        run.config.update(config)

    def finish(self) -> None:
        """Finish the wandb run."""
        if not self.is_initialized:
            return

        run = cast(
            Any, self._run
        )  # Type assertion since is_initialized ensures _run is not None

        try:
            run.finish()
            logger.info("Finished WandB run")
        except Exception as e:
            logger.error(f"Error finishing WandB run: {e}")
        finally:
            self._initialized = False
            self._run = None

    def get_step(self, phase: str) -> int:
        """Get current step for a phase."""
        return self._step_counters.get(phase, 0)

    def set_step(self, phase: str, step: int) -> None:
        """Set step for a phase."""
        self._step_counters[phase] = step


# Global instance for easy access
_wandb_manager = WandbManager()


def get_wandb_manager() -> WandbManager:
    """Get the global wandb manager instance."""
    return _wandb_manager


def init_wandb(
    project: str = "mcts-training",
    name: str | None = None,
    config: Any | None = None,
    **kwargs,
) -> bool:
    """Initialize wandb with the global manager."""
    return _wandb_manager.initialize(project, name, config, **kwargs)


def log_mcts_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log MCTS metrics with the global manager."""
    _wandb_manager.log_mcts_metrics(metrics, step)


def log_training_metrics(
    metrics: dict[str, Any], step: int | None = None, model_type: str | None = None
) -> None:
    """Log training metrics with the global manager."""
    _wandb_manager.log_training_metrics(metrics, step, model_type)


def log_evaluation_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log evaluation metrics with the global manager."""
    _wandb_manager.log_evaluation_metrics(metrics, step)


def log_final_evaluation_metrics(metrics: dict[str, Any]) -> None:
    """Log final evaluation metrics with the global manager."""
    _wandb_manager.log_final_evaluation_metrics(metrics)


def finish_wandb() -> None:
    """Finish wandb run with the global manager."""
    _wandb_manager.finish()
