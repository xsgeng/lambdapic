"""Tests for RestartDump checkpoint save/load."""
import tempfile

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, e, m_e, pi

from lambdapic import Electron, Simulation
from lambdapic.callback.restart import RestartDump

l0 = 0.8e-6
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2


@pytest.mark.unit
class TestRestartTimeSync:
    """Verify sim.time stays consistent with sim.itime across a restart."""

    def test_time_stays_in_sync_after_load(self) -> None:
        nx = 64
        ny = 64
        dx = l0 / 20
        dy = l0 / 20

        sim = Simulation(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            npatch_x=4,
            npatch_y=4,
            boundary_conditions={
                "xmin": "periodic",
                "xmax": "periodic",
                "ymin": "periodic",
                "ymax": "periodic",
            },
        )

        ne = 0.01 * nc
        ele = Electron(density=lambda x, y: ne, ppc=4)
        sim.add_species([ele])
        sim.initialize()

        # Advance the simulation a few steps so itime/time move off zero.
        nsteps = 5
        sim.run(nsteps=nsteps)

        assert sim.itime == nsteps
        np.testing.assert_allclose(sim.time, sim.itime * sim.dt, rtol=1e-15)

        # Save a checkpoint at the current state, then load it.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = RestartDump(tmp, interval=nsteps)
            ckpt._call(sim)
            ckpt_dir = ckpt._ckpt_dir(sim.itime)

            loaded = RestartDump.load(ckpt_dir)

        # load() fast-forwards itime by 1 (restart is called before the loop's
        # itime increment). time must track itime after the fast-forward.
        assert loaded.itime == sim.itime + 1
        np.testing.assert_allclose(
            loaded.time, loaded.itime * loaded.dt, rtol=1e-15
        )
