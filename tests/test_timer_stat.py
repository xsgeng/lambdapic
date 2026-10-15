from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from lambdapic.cli.main import app
from lambdapic.cli.stat import calculate_percentages, parse_log_file

runner = CliRunner()


PLAIN = """\
Rank 0 update E field took 1.2ms
Rank 0 update E field took 3.4ms
Rank 0 update B field took 5.0ms
Rank 1 update E field took 2.0ms
"""

LEGACY = (
    "2024-01-01 00:00:00.000 | TIMER    | Rank 0 update E field took 1.2ms\n"
    "2024-01-01 00:00:00.001 | TIMER    | Rank 0 update B field took 4.8ms\n"
)

MIXED = (
    "Rank 0 update E field took 1.0ms\n"
    "2024-01-01 00:00:00.000 | TIMER    | Rank 0 update E field took 2.0ms\n"
    "some unrelated log line\n"
    "Rank 1 update B field took 6.0ms\n"
)


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


class TestParseLogFile:
    def test_plain_format(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", PLAIN)
        ct, cc = parse_log_file(str(f))
        assert ct["update E field"] == pytest.approx(1.2 + 3.4 + 2.0)
        assert ct["update B field"] == pytest.approx(5.0)
        assert cc["update E field"] == 3
        assert cc["update B field"] == 1

    def test_legacy_loguru_format(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", LEGACY)
        ct, cc = parse_log_file(str(f))
        assert ct["update E field"] == pytest.approx(1.2)
        assert ct["update B field"] == pytest.approx(4.8)
        assert cc["update E field"] == 1
        assert cc["update B field"] == 1

    def test_mixed_format(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", MIXED)
        ct, cc = parse_log_file(str(f))
        assert ct["update E field"] == pytest.approx(3.0)
        assert ct["update B field"] == pytest.approx(6.0)
        assert cc["update E field"] == 2
        assert cc["update B field"] == 1

    def test_multi_rank(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", PLAIN)
        ct, cc = parse_log_file(str(f))
        # rank is part of the line but category excludes rank number
        assert "update E field" in ct
        assert "update B field" in ct

    def test_empty_file(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", "")
        ct, cc = parse_log_file(str(f))
        assert dict(ct) == {}
        assert dict(cc) == {}

    def test_no_matching_lines(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", "nothing here\nanother line\n")
        ct, cc = parse_log_file(str(f))
        assert dict(ct) == {}

    def test_missing_file_exits(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(typer.Exit) as exc:
            parse_log_file(str(missing))
        assert exc.value.exit_code == 1


class TestCalculatePercentages:
    def test_basic(self):
        ct = {"a": 30.0, "b": 70.0}
        cc = {"a": 3, "b": 7}
        pct, avg, total = calculate_percentages(ct, cc)
        assert total == pytest.approx(100.0)
        assert pct["a"] == pytest.approx(30.0)
        assert pct["b"] == pytest.approx(70.0)
        assert avg["a"] == pytest.approx(10.0)
        assert avg["b"] == pytest.approx(10.0)

    def test_single_category_is_100pct(self):
        ct = {"only": 42.0}
        cc = {"only": 2}
        pct, avg, total = calculate_percentages(ct, cc)
        assert total == pytest.approx(42.0)
        assert pct["only"] == pytest.approx(100.0)
        assert avg["only"] == pytest.approx(21.0)

    def test_empty(self):
        ct = {}
        cc = {}
        pct, avg, total = calculate_percentages(ct, cc)
        assert total == 0.0
        assert pct == {}
        assert avg == {}


class TestTimerStatCli:
    def _run(self, args):
        return runner.invoke(app, ["timer-stat", *args])

    def test_plain_file(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", PLAIN)
        r = self._run([str(f)])
        assert r.exit_code == 0
        out = r.stdout
        assert "TIMER CATEGORY ANALYSIS" in out
        assert "update E field" in out
        assert "update B field" in out
        assert "TOTAL" in out

    def test_auto_detect_sibling_timer_file(self, tmp_path):
        # main log file given, but the timer sibling exists -> used automatically
        write(tmp_path, "run.timer.txt", PLAIN)
        main = write(tmp_path, "run.txt", "regular log content\n")
        r = self._run([str(main)])
        assert r.exit_code == 0
        assert "Using timer file" in r.stdout
        assert "update E field" in r.stdout

    def test_no_sibling_uses_given_file(self, tmp_path):
        # when no timer sibling exists, the given file is parsed directly
        main = write(tmp_path, "solo.txt", PLAIN)
        r = self._run([str(main)])
        assert r.exit_code == 0
        assert "update E field" in r.stdout

    def test_sort_by_name(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", PLAIN)
        r = self._run([str(f), "--sort", "name"])
        assert r.exit_code == 0
        lines = [ln for ln in r.stdout.splitlines() if "update" in ln]
        names = [ln.split()[0] for ln in lines]
        assert names == sorted(names)

    def test_sort_by_time_desc_default(self, tmp_path):
        f = write(tmp_path, "t.timer.txt", PLAIN)
        r = self._run([str(f)])
        assert r.exit_code == 0
        # update E field total (6.6) > update B field (5.0); desc sort puts E first
        out_lines = r.stdout.splitlines()
        idx_b = next(i for i, ln in enumerate(out_lines) if "update B field" in ln)
        idx_e = next(i for i, ln in enumerate(out_lines) if "update E field" in ln)
        assert idx_e < idx_b

    def test_missing_file_exit_code(self, tmp_path):
        r = self._run([str(tmp_path / "nope.txt")])
        assert r.exit_code == 1
