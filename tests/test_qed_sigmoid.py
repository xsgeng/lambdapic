import os
import sys
import subprocess

import pytest


@pytest.mark.slow
def test_qed_sigmoid_sampling():
    """Run the full QED test suite under LAMBDAPIC_USE_SIGMOID_SAMPLING=1.

    The env var is read at module-import time (optical_depth.py), so the
    sigmoid table path must be exercised in a fresh subprocess where the
    numba-jit kernels bind to the sigmoid lookup tables from import.
    """
    env = os.environ.copy()
    env["LAMBDAPIC_USE_SIGMOID_SAMPLING"] = "1"

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_radiation.py", "tests/test_pair_production.py",
            "-o", "addopts=",
            "-q",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"sigmoid sampling path failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
