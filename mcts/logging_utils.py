import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import ClassVar


class MCTSLogger:
    """Centralized logging configuration for MCTS accelerate module"""

    _loggers: ClassVar[dict[str, logging.Logger]] = {}
    _log_file: ClassVar[str | None] = None
    _console_handler: ClassVar[logging.Handler | None] = None
    _file_handler: ClassVar[logging.Handler | None] = None

    @classmethod
    def setup_logging(
        cls,
        log_file: str | None = None,
        log_level: str = "INFO",
        console_output: bool = True,
        file_output: bool = True,
    ) -> None:
        """Setup centralized logging configuration

        Args:
            log_file: Path to log file. If None, creates timestamped log in logs/ directory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_output: Whether to output to console
            file_output: Whether to output to file
        """
        # create logs directory if it doesn't exist
        if file_output:
            if log_file is None:
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = logs_dir / f"mcts_run_{timestamp}.log"
            else:
                log_file_path = Path(log_file)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            cls._log_file = str(log_file_path)

        # configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # removing existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # setup console handler
        if console_output:
            cls._console_handler = logging.StreamHandler(sys.stdout)
            cls._console_handler.setLevel(getattr(logging, log_level.upper()))
            console_formatter = logging.Formatter(
                "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            cls._console_handler.setFormatter(console_formatter)
            root_logger.addHandler(cls._console_handler)

        # setup file handler
        if file_output and cls._log_file:
            cls._file_handler = logging.FileHandler(cls._log_file, mode="w")
            cls._file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(
                "%(asctime)s - [%(name)s] - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            cls._file_handler.setFormatter(file_formatter)
            root_logger.addHandler(cls._file_handler)

        # log init
        logger = cls.get_logger(__name__)
        logger.info(f"Logging initialized - Level: {log_level}")
        if file_output and cls._log_file:
            logger.info(f"Log file: {cls._log_file}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance for the given name"""
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]

    @classmethod
    def get_log_file_path(cls) -> str | None:
        """Get the path to the current log file"""
        return cls._log_file

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown logging and close all handlers"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        cls._loggers.clear()


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger instance"""
    return MCTSLogger.get_logger(name)
