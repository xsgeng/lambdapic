import time
from pathlib import Path

import pytest

from lambdapic import Simulation
from lambdapic.core.utils.logger import configure_logger, timer_sink_path
from lambdapic.core.utils.timer import Timer, set_timer_enabled, timer_enabled


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
        configure_logger(sink=str(log_file), enable_timer=False)
        set_timer_enabled(False)

        with Timer("disabled timer"):
            time.sleep(0.0002)

        log_text = log_file.read_text()
        assert "TIMER" not in log_text
        assert "disabled timer" not in log_text

    def test_timer_logs_when_enabled(self, tmp_path):
        log_file = tmp_path / "timer.log"
        configure_logger(sink=str(log_file), enable_timer=True)
        set_timer_enabled(True)

        with Timer("enabled timer"):
            time.sleep(0.0002)

        timer_file = tmp_path / "timer.timer.log"
        assert timer_file.exists()
        timer_text = timer_file.read_text()
        assert "TIMER" in timer_text
        assert "enabled timer took" in timer_text

        log_text = log_file.read_text()
        assert "TIMER" not in log_text
        assert "enabled timer" not in log_text


class TestSimulationTimerIntegration:
    def _make_sim(self, tmp_path, enable_timer: bool):
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
        )

    def test_timer_disabled_by_default(self, tmp_path):
        sim = self._make_sim(tmp_path, enable_timer=False)
        sim.initialize()

        assert timer_enabled() is False

        timer_file = tmp_path / "sim.timer.log"
        assert not timer_file.exists()

        log_text = (tmp_path / "sim.log").read_text()
        assert "TIMER" not in log_text

    def test_timer_enabled(self, tmp_path):
        sim = self._make_sim(tmp_path, enable_timer=True)
        sim.initialize()

        assert timer_enabled() is True

        with Timer("integration timer"):
            time.sleep(0.0002)

        timer_file = tmp_path / "sim.timer.log"
        assert timer_file.exists()
        timer_text = timer_file.read_text()
        assert "TIMER" in timer_text
        assert "integration timer took" in timer_text

        log_text = (tmp_path / "sim.log").read_text()
        assert "TIMER" not in log_text
