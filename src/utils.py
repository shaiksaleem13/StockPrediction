"""Utility helpers for EAATS: Logging, Seed reproducibility, and Caching."""

import logging
import random
import sys
import time
from functools import wraps
from typing import Any, Callable
import numpy as np

import config


def setup_logger(name: str = "eaats") -> logging.Logger:
    """Configures and returns a centralized logger writing to both file and console."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler
    log_file = config.LOGS_DIR / "eaats.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger("eaats")


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def timed_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log function execution time."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
        return result

    return wrapper
