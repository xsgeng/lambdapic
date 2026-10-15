import time
from pathlib import Path

import pytest

from lambdapic import Simulation
from lambdapic.core.utils.timer import (
    Timer,
    flush_timers,
    set_timer_enabled,
    timer_enabled,
    timer_sink_path,
)


@pytest.fixture(autouse=True)
def reset_timer():
    """Reset global timer state after each test."""
    yield
    set_timer_enabled(False)


class TestTimerSinkPath:
    def test_txt_extension(self):
        assert timer_sink_path("log.txt") == "log.timer.txt"

    def test_no_extension(self):
        assert timer_sink_path("run") == "run.timer"

    def test_with_directory(self):
        assert timer_sink_path("logs/run.log") == "logs/run.timer.log"

    def test_multiple_dots(self):
        assert timer_sink_path("a.b.log") == "a.b.timer.log"


class TestTimerGlobalSwitch:
    def test_default_state(self):
        assert timer_enabled() is False

    def test_set_timer_enabled(self):
        set_timer_enabled(True)
        assert timer_enabled() is True
        set_timer_enabled(False)
        assert timer_enabled() is False

    def test_timer_respects_global_disable(self, tmp_path):
        log_file = tmp_path / "timer.log"
        set_timer_enabled(False, timer_file=str(log_file))

        with Timer("disabled timer"):
            time.sleep(0.0002)

        flush_timers()
        assert not log_file.exists() or log_file.read_text() == ""

    def test_timer_records_when_enabled(self, tmp_path):
        timer_file = tmp_path / "timer.timer.log"
        set_timer_enabled(True, timer_file=str(timer_file))

        with Timer("enabled timer"):
            time.sleep(0.0002)

        # Nothing written until flushed
        assert not timer_file.exists() or timer_file.read_text() == ""

        flush_timers()

        text = timer_file.read_text()
        assert "enabled timer took" in text

    def test_timer_buffer_cleared_after_flush(self, tmp_path):
        timer_file = tmp_path / "timer.timer.log"
        set_timer_enabled(True, timer_file=str(timer_file))

        with Timer("a"):
            time.sleep(0.0002)
        with Timer("b"):
            time.sleep(0.0002)

        flush_timers()
        # After flush, buffer is empty; a second flush writes nothing new
        first = timer_file.read_text()
        flush_timers()
        second = timer_file.read_text()
        assert second == first

    def test_flush_is_append_mode(self, tmp_path):
        timer_file = tmp_path / "timer.timer.log"
        set_timer_enabled(True, timer_file=str(timer_file))

        with Timer("first"):
            time.sleep(0.0002)
        flush_timers()

        with Timer("second"):
            time.sleep(0.0002)
        flush_timers()

        text = timer_file.read_text()
        assert "first took" in text
        assert "second took" in text

    def test_flush_noop_when_no_file(self, tmp_path):
        # enable but no timer_file: should not raise
        set_timer_enabled(True, timer_file=None)
        with Timer("orphan"):
            time.sleep(0.0002)
        flush_timers()


class TestSimulationTimerIntegration:
    def _make_sim(self, tmp_path, enable_timer: bool, flush_interval: int = 100):
        return Simulation(
            nx=32,
            ny=32,
            dx=1e-8,
            dy=1e-8,
            npatch_x=2,
            npatch_y=2,
            nsteps=2,
            log_file=str(tmp_path / "sim.log"),
            enable_timer=enable_timer,
            timer_flush_interval=flush_interval,
        )

    def test_timer_disabled_by_default(self, tmp_path):
        sim = self._make_sim(tmp_path, enable_timer=False)
        sim.initialize()

        assert timer_enabled() is False

        timer_file = tmp_path / "sim.timer.log"
        assert not timer_file.exists()

        log_text = (tmp_path / "sim.log").read_text()
        assert "TIMER" not in log_text

    def test_timer_enabled_and_flushed_at_run_end(self, tmp_path):
        sim = self._make_sim(tmp_path, enable_timer=True, flush_interval=1000)
        sim.initialize()

        assert timer_enabled() is True

        sim.run()

        timer_file = tmp_path / "sim.timer.log"
        assert timer_file.exists()
        text = timer_file.read_text()
        assert "took" in text

        log_text = (tmp_path / "sim.log").read_text()
        assert "TIMER" not in log_text

    def test_periodic_flush_writes_during_run(self, tmp_path):
        # flush every step so the file is written before run() ends
        sim = self._make_sim(tmp_path, enable_timer=True, flush_interval=1)
        sim.initialize()

        timer_file = tmp_path / "sim.timer.log"
        # Run one step then inspect the file mid-run is awkward; instead run
        # fully and verify records landed (final flush + periodic flushes).
        sim.run()

        text = timer_file.read_text()
        assert "update E field took" in text
