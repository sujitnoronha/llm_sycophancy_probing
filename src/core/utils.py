"""Shared utilities: seeding, logging, checkpointing, GPU management."""

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger


def set_all_seeds(seed: int) -> None:
    """Set seeds for all RNG sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure structured logging with loguru."""
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level=log_level)
    if log_file:
        logger.add(log_file, format=fmt, level=log_level, rotation="50 MB")


def get_config_hash(config) -> str:
    """SHA256 hash of serialized config for artifact provenance tracking."""
    config_str = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def get_checkpoint_path(output_dir: str, module_name: str, model_name: str) -> Path:
    """Get the checkpoint file path for a given module and model."""
    return Path(output_dir) / "checkpoints" / f"{module_name}_{model_name}.done"


def check_checkpoint(output_dir: str, module_name: str, model_name: str) -> bool:
    """Check if a checkpoint exists (i.e., this module+model already completed)."""
    return get_checkpoint_path(output_dir, module_name, model_name).exists()


def save_checkpoint(output_dir: str, module_name: str, model_name: str) -> None:
    """Mark a module+model combination as completed."""
    path = get_checkpoint_path(output_dir, module_name, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    logger.info(f"Checkpoint saved: {module_name}/{model_name}")


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"{prefix}GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def ensure_output_dir(output_dir: str, module_name: str) -> Path:
    """Create and return the output directory for a module."""
    path = Path(output_dir) / module_name
    path.mkdir(parents=True, exist_ok=True)
    return path
