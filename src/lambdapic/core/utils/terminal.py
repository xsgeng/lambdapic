"""Terminal detection utilities."""

import sys


def is_terminal() -> bool:
    """Check if the current process is running in a terminal.

    Returns True if stdout is attached to a terminal (TTY), False otherwise.
    This is useful for disabling progress bars and spinners in non-terminal
    environments like logs, pipes, or batch job systems.

    Returns:
        bool: True if running in a terminal, False otherwise.
    """
    return sys.stdout.isatty()
