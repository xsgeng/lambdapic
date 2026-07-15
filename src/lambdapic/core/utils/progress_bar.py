import os
import subprocess
import time
from io import StringIO

from tqdm import tqdm

from .logger import logger
from .terminal import is_terminal


def _update_slurm_comment(job_id: str, progress_pct: float) -> None:
    procid = os.environ.get("SLURM_PROCID")
    if procid is not None and procid != "0":
        return
    try:
        subprocess.run(
            ["scontrol", "update", f"job={job_id}", f"comment={progress_pct:.1f}%"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except Exception:
        pass


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
    maxinterval : float, optional
        Maximum seconds between tqdm display updates. Default: None (tqdm default 60.0).
    slurm : bool, optional
        Update SLURM job comment with progress percentage. Default: None (auto-detect from
        SLURM_JOB_ID environment variable).
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
        maxinterval: float | None = None,
        position: int | None = None,
        slurm: bool | None = None,
    ):
        self.total = total
        self.initial = initial
        self.desc = desc
        self._n = initial
        self.progress_interval = progress_interval if progress_interval is not None else min(100, max(1, total // 10))
        self.maxinterval = maxinterval if maxinterval is not None else 60.0

        self._is_terminal = is_terminal()
        self.disable = disable

        self._slurm = slurm if slurm is not None else os.environ.get("SLURM_JOB_ID") is not None
        self._job_id = os.environ.get("SLURM_JOB_ID") if self._slurm else None
        if slurm is True and self._job_id is None:
            logger.warning("SLURM_JOB_ID not found in environment, disabling SLURM progress updates")

        # Create tqdm instance with appropriate output
        tqdm_kwargs = {
            "total": total,
            "initial": initial,
            "disable": False,
            "desc": desc,
            "position": position,
        }
        if maxinterval is not None:
            tqdm_kwargs["maxinterval"] = maxinterval

        if self._is_terminal and not self.disable:
            # Terminal: normal display
            self.pbar = tqdm(**tqdm_kwargs)
        elif not self.disable:
            # Non-terminal: redirect to StringIO so tqdm updates internal state
            # but doesn't display. This preserves EMA smoothing for rate calc.
            self._dummy_file = StringIO()
            self.pbar = tqdm(**tqdm_kwargs, file=self._dummy_file)
        else:
            # Disabled: no tqdm at all
            self.pbar = None

        self._last_log_step = initial
        self._last_log_time = time.monotonic()

    def update(self, n: int = 1):
        """Update progress by n steps."""
        if self.disable or self.pbar is None:
            self._n += n
            return

        self._n += n
        self.pbar.update(n)

        now = time.monotonic()
        step_trigger = (self._n - self._last_log_step) >= self.progress_interval
        time_trigger = (now - self._last_log_time) >= self.maxinterval
        if step_trigger or time_trigger:
            if not self._is_terminal:
                self._log_progress()
            if self._slurm and self._job_id is not None:
                _update_slurm_comment(self._job_id, 100 * self._n / self.total)
            self._last_log_step = self._n
            self._last_log_time = now

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
        if self._slurm and self._job_id is not None:
            _update_slurm_comment(self._job_id, 100 * self._n / self.total)
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def n(self) -> int:
        """Current iteration count."""
        return self._n


class ProgressBarFloat:
    def __init__(
        self,
        total: float,
        initial: float = 0,
        desc: str = "Progress",
        disable: bool | None = None,
        progress_interval: float | None = None,
        maxinterval: float | None = None,
        position: int | None = None,
        bar_format: str | None = None,
        unit: str = "step",
        slurm: bool | None = None,
    ):
        self.total = float(total)
        self.initial = float(initial)
        self.desc = desc
        self._n = float(initial)
        self.unit = unit
        self.progress_interval = (
            progress_interval if progress_interval is not None else self.total / 10.0
        )
        self.maxinterval = maxinterval if maxinterval is not None else 60.0

        self._is_terminal = is_terminal()
        self.disable = disable

        self._slurm = slurm if slurm is not None else os.environ.get("SLURM_JOB_ID") is not None
        self._job_id = os.environ.get("SLURM_JOB_ID") if self._slurm else None
        if slurm is True and self._job_id is None:
            logger.warning("SLURM_JOB_ID not found in environment, disabling SLURM progress updates")

        tqdm_kwargs = {
            "total": total,
            "initial": initial,
            "disable": False,
            "desc": desc,
            "position": position,
        }
        if maxinterval is not None:
            tqdm_kwargs["maxinterval"] = maxinterval
        if bar_format is not None:
            tqdm_kwargs["bar_format"] = bar_format

        if self._is_terminal and not self.disable:
            self.pbar = tqdm(**tqdm_kwargs)
        elif not self.disable:
            self._dummy_file = StringIO()
            self.pbar = tqdm(**tqdm_kwargs, file=self._dummy_file)
        else:
            self.pbar = None

        self._last_log_step = float(initial)
        self._last_log_time = time.monotonic()

    def update(self, n: float = 1):
        if self.disable or self.pbar is None:
            self._n += n
            return

        self._n += n
        self.pbar.update(n)

        now = time.monotonic()
        step_trigger = (self._n - self._last_log_step) >= self.progress_interval
        time_trigger = (now - self._last_log_time) >= self.maxinterval
        if step_trigger or time_trigger:
            if not self._is_terminal:
                self._log_progress()
            if self._slurm and self._job_id is not None:
                _update_slurm_comment(self._job_id, 100 * self._n / self.total)
            self._last_log_step = self._n
            self._last_log_time = now

    def set_description(self, desc: str):
        self.desc = desc
        if self.pbar is not None:
            self.pbar.set_description(desc)

    @staticmethod
    def _format_time(seconds: float) -> str:
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_rate(rate: float, unit: str = "step") -> str:
        if rate <= 0:
            return f"-- {unit}/s"
        if rate >= 1:
            return f"{rate:.2f} {unit}/s"
        return f"{rate:.2e} {unit}/s"

    def _log_progress(self):
        if self.pbar is None:
            return

        fmt = self.pbar.format_dict
        rate = fmt.get("rate") or 0
        elapsed = fmt.get("elapsed", 0)
        remaining = (self.total - self._n) / rate if rate > 0 else 0
        progress_pct = 100 * self._n / self.total

        logger.info(
            f"{self.desc}: {self._n:.2e}/{self.total:.2e} "
            f"({progress_pct:.1f}%) | "
            f"Elapsed: {self._format_time(elapsed)} | "
            f"Remaining: {self._format_time(remaining)} | "
            f"Speed: {self._format_rate(rate, self.unit)}"
        )

    def close(self):
        if self.disable or self.pbar is None:
            return

        if not self._is_terminal and self._n > self._last_log_step:
            self._log_progress()
        if self._slurm and self._job_id is not None:
            _update_slurm_comment(self._job_id, 100 * self._n / self.total)
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def n(self) -> float:
        return self._n
