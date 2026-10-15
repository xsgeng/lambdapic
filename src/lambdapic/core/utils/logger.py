import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from .terminal import is_terminal

LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"


def configure_logger(
    level: str = DEFAULT_LOG_LEVEL,
    sink: str | None = None,
    format_str: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                     "<level>{level: <8}</level> | "
                     "<level>{message}</level>",
    colorize: bool | None = None,
    serialize: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
    truncate_existing: bool = True,
):
    """Configure the global logger instance.

    Args:
        level: Minimum logging level (default: INFO)
        sink: Path to the main log file. If ``None``, no log file is created
            and only the console sink is used.
        format_str: Log message format
        colorize: Whether to add colors to output. If None, auto-detect based on terminal.
        serialize: Whether to output as JSON
        backtrace: Whether to show exception backtrace
        diagnose: Whether to show variable values in backtrace
        truncate_existing: Whether to truncate existing log file
    """
    colorize = colorize if colorize is not None else is_terminal()
    
    logger.remove()
    
    # Register custom levels first
    try:
        logger.level("SUCCESS")
    except ValueError:
        logger.level("SUCCESS", no=25, color="<bold><green>")
    
    # Set up console sink (stderr)
    console_sink = sys.stderr
    
    # Set log level from environment if present
    env_level = os.getenv("LAMBDAPIC_LOG_LEVEL", "").upper()
    if env_level in LOG_LEVELS:
        level = env_level
    
    # Add file sink only if provided
    if sink is not None:
        # Only truncate if requested and file exists
        if truncate_existing and isinstance(sink, (str, os.PathLike)) and os.path.exists(sink):
            with open(sink, "w") as f:
                f.truncate()
        logger.add(
            sink,
            format=format_str,
            level=level,
            colorize=False,
            serialize=serialize,
            backtrace=backtrace,
            diagnose=diagnose
        )

    # Add console sink
    logger.add(
        console_sink,
        format=format_str,
        level="INFO",
        colorize=colorize,
        serialize=False,
        backtrace=backtrace,
        diagnose=diagnose
    )

def rank_log(msg: str, comm):
    rank = comm.Get_rank()
    if rank == 0:
        logger.info(msg)
    else:
        logger.debug(f"Rank {rank}: {msg}")