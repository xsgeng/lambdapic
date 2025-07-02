import os
import sys
from datetime import datetime
from loguru import logger
from typing import Optional

LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"

def configure_logger(
    level: str = DEFAULT_LOG_LEVEL,
    sink=None,
    format_str: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                     "<level>{level: <8}</level> | "
                     "<level>{message}</level>",
    colorize: bool = True,
    serialize: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
    truncate_existing: bool = True
):
    """Configure the global logger instance
    
    Args:
        level: Minimum logging level (default: INFO)
        sink: Where to send logs - None for auto filename (default: logs/lambdapic_YYYYMMDD_HHMMSS.log)
        format_str: Log message format
        colorize: Whether to add colors to output
        serialize: Whether to output as JSON
        backtrace: Whether to show exception backtrace
        diagnose: Whether to show variable values in backtrace
        truncate_existing: Whether to truncate existing log file
    """
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
        if truncate_existing and isinstance(sink, str) and os.path.exists(sink):
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

