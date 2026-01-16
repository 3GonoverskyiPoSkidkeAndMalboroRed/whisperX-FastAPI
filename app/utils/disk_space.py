"""Utility functions for checking disk space availability."""

import os
import shutil
from pathlib import Path

from app.core.logging import logger


def get_free_space(path: str) -> int:
    """
    Get free disk space in bytes for the given path.

    Args:
        path: Path to check free space for

    Returns:
        Free space in bytes, or 0 if unable to determine
    """
    try:
        stat = shutil.disk_usage(path)
        return stat.free
    except Exception as e:
        logger.warning("Failed to get disk space for %s: %s", path, str(e))
        return 0


def check_available_space(path: str, required_bytes: int) -> tuple[bool, int, int]:
    """
    Check if there is enough free space at the given path.

    Args:
        path: Path to check free space for
        required_bytes: Required space in bytes

    Returns:
        Tuple of (has_enough_space, free_bytes, required_bytes)
    """
    free_bytes = get_free_space(path)
    has_enough = free_bytes >= required_bytes

    if not has_enough:
        logger.warning(
            "Insufficient disk space at %s: required %d MB, available %d MB",
            path,
            required_bytes / (1024 * 1024),
            free_bytes / (1024 * 1024),
        )

    return has_enough, free_bytes, required_bytes


def get_huggingface_cache_dir() -> str:
    """
    Get the HuggingFace cache directory path.

    Returns:
        Path to HuggingFace cache directory
    """
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        return cache_dir

    cache_dir = os.environ.get("XDG_CACHE_HOME")
    if cache_dir:
        return os.path.join(cache_dir, "huggingface")

    # Default to ~/.cache/huggingface
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "huggingface")


def check_model_download_space(model_size_mb: float, cache_dir: str | None = None) -> bool:
    """
    Check if there is enough space to download a model.

    Args:
        model_size_mb: Model size in megabytes
        cache_dir: Optional cache directory path (defaults to HuggingFace cache)

    Returns:
        True if there is enough space, False otherwise
    """
    if cache_dir is None:
        cache_dir = get_huggingface_cache_dir()

    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    required_bytes = int(model_size_mb * 1024 * 1024)
    has_enough, free_bytes, required = check_available_space(cache_dir, required_bytes)

    if not has_enough:
        logger.error(
            "Not enough space to download model. Required: %.2f MB, Available: %.2f MB, Path: %s",
            required / (1024 * 1024),
            free_bytes / (1024 * 1024),
            cache_dir,
        )

    return has_enough
