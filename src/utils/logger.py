"""
Logging configuration using Loguru.

Provides a pre-configured logger for the entire application with
both console and file output.
"""

import sys

from loguru import logger

from src.utils.config import BASE_DIR, settings


def setup_logger() -> None:
    """Configure loguru logger with console and file outputs."""
    # Remove default handler
    logger.remove()

    # Console output — colored, human-readable
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File output — structured, with rotation
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",       # Rotate when file reaches 10 MB
        retention="30 days",    # Keep logs for 30 days
        compression="zip",      # Compress rotated logs
        encoding="utf-8",
    )

    logger.info("Logger initialized — level: {}", settings.log_level)


# Initialize on import
setup_logger()