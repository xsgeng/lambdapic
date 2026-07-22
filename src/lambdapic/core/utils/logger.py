import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from .terminal import is_terminal

LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"


def timer_sink_path(sink: str) -> str:
    """Derive the dedicated timer log file path from the main log file path.

    Parameters:
        sink: Path to the main log file.

    Returns:
        Path to the timer log file. For example ``log.txt`` becomes
        ``log.timer.txt``.
    """
    p = Path(sink)
    return str(p.with_name(p.stem + ".timer" + p.suffix))

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
    enable_timer: bool = False
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
        enable_timer: If ``True``, TIMER records are written to a separate file
            derived from ``sink`` (e.g. ``log.txt`` -> ``log.timer.txt``). If
            ``sink`` is ``None``, timer records are discarded. Defaults to ``False``.
    """
    colorize = colorize if colorize is not None else is_terminal()
    
    logger.remove()
    
    # Register custom levels first
    try:
        logger.level("SUCCESS")
    except ValueError:
        logger.level("SUCCESS", no=25, color="<bold><green>")
    
    try:
        logger.level("TIMER")
    except ValueError:
        logger.level("TIMER", no=22, color="<bold><cyan>")
    
    # Get TIMER level number for filtering
    timer_level = logger.level("TIMER").no

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
            filter=lambda record: record["level"].name != "TIMER",
            colorize=False,
            serialize=serialize,
            backtrace=backtrace,
            diagnose=diagnose
        )

        # Dedicated timer file sink
        if enable_timer and isinstance(sink, (str, os.PathLike)):
            timer_sink = timer_sink_path(str(sink))
            if truncate_existing and os.path.exists(timer_sink):
                with open(timer_sink, "w") as f:
                    f.truncate()
            logger.add(
                timer_sink,
                format=format_str,
                level="TIMER",
                filter=lambda record: record["level"].name == "TIMER",
                colorize=False,
                serialize=False,
                backtrace=False,
                diagnose=False
            )

    # Add console sink - exclude TIMER level
    logger.add(
        console_sink,
        format=format_str,
        level="INFO",
        filter=lambda record: record["level"].no != timer_level,  # Exclude TIMER
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