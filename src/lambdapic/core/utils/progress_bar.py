from io import StringIO

from tqdm import tqdm

from .logger import logger
from .terminal import is_terminal


class ProgressBar:
    """A progress bar that logs timing info when not in a terminal.

    This class wraps tqdm to provide progress display in terminals and
    structured logging when running in non-terminal environments (e.g.,
    batch jobs, logs, pipes).

    Parameters
    ----------
    total : int
        Total number of iterations.
    initial : int, optional
        Initial counter value. Default: 0.
    desc : str, optional
        Description prefix. Default: "Progress".
    disable : bool, optional
        Force disable progress bar and logging. Default: None (auto-detect).
    progress_interval : int, optional
        Log progress every N iterations when not in terminal. Default: min(100, total//10).
    position : int, optional
        Line offset for tqdm display. Default: None.

    Example
    -------
    >>> with ProgressBar(total=100, desc="Simulation") as pbar:
    ...     for i in range(100):
    ...         pbar.update(1)
    """

    def __init__(
        self,
        total: int,
        initial: int = 0,
        desc: str = "Progress",
        disable: bool | None = None,
        progress_interval: int | None = None,
        position: int | None = None,
    ):
        self.total = total
        self.initial = initial
        self.desc = desc
        self._n = initial
        self.progress_interval = progress_interval if progress_interval is not None else min(100, max(1, total // 10))

        self._is_terminal = is_terminal()
        self.disable = disable

        # Create tqdm instance with appropriate output
        if self._is_terminal and not self.disable:
            # Terminal: normal display
            self.pbar = tqdm(
                total=total,
                initial=initial,
                disable=False,
                desc=desc,
                position=position,
            )
        elif not self.disable:
            # Non-terminal: redirect to StringIO so tqdm updates internal state
            # but doesn't display. This preserves EMA smoothing for rate calc.
            self._dummy_file = StringIO()
            self.pbar = tqdm(
                total=total,
                initial=initial,
                disable=False,
                file=self._dummy_file,
                desc=desc,
                position=position,
            )
            self._last_log_step = initial
        else:
            # Disabled: no tqdm at all
            self.pbar = None

    def update(self, n: int = 1):
        """Update progress by n steps."""
        if self.disable or self.pbar is None:
            self._n += n
            return

        self._n += n
        self.pbar.update(n)

        # Non-terminal: log progress at intervals using tqdm's format_dict
        if not self._is_terminal:
            if (self._n - self._last_log_step) >= self.progress_interval:
                self._log_progress()
                self._last_log_step = self._n

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS.

        Parameters
        ----------
        seconds : float
            Time in seconds.

        Returns
        -------
        str
            Formatted time string as HH:MM:SS.
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_rate(rate: float) -> str:
        """Format rate adaptively as steps/s or s/step.

        Parameters
        ----------
        rate : float
            Rate in steps per second.

        Returns
        -------
        str
            Formatted rate string with appropriate unit.
        """
        if rate <= 0:
            return "-- step/s"
        if rate >= 1:
            return f"{rate:.2f} step/s"
        else:
            return f"{1 / rate:.2f} s/step"

    def _log_progress(self):
        """Log current progress with timing info from tqdm."""
        if self.pbar is None:
            return

        fmt = self.pbar.format_dict
        rate = fmt.get("rate") or 0
        elapsed = fmt.get("elapsed", 0)
        remaining = (self.total - self._n) / rate if rate > 0 else 0

        logger.info(
            f"{self.desc}: {self._n}/{self.total} "
            f"({100 * self._n / self.total:.1f}%) | "
            f"Elapsed: {self._format_time(elapsed)} | "
            f"Remaining: {self._format_time(remaining)} | "
            f"Speed: {self._format_rate(rate)}"
        )

    def close(self):
        """Close the progress bar and log final status."""
        if self.disable or self.pbar is None:
            return

        if not self._is_terminal and self._n > self._last_log_step:
            self._log_progress()
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def n(self) -> int:
        """Current iteration count."""
        return self._n
