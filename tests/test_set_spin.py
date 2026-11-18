"""Tests for the SetSpin init callback (spatially varying initial spin)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c, e, epsilon_0, m_e, pi

from lambdapic import Electron, Simulation, SetSpin

l0 = 0.8e-6
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2


def _gather_spin(sim, species):
    """Collect alive-particle positions and spin across all patches."""
    ispec = species.ispec
    keys = ("x", "y", "z", "sx", "sy", "sz")
    out = {k: [] for k in keys}
    for p in sim.patches:
        part = p.particles[ispec]
        alive = part.is_alive
        if not alive.any():
            continue
        for k in keys:
            out[k].append(getattr(part, k)[alive])
    for k in keys:
        out[k] = np.concatenate(out[k]) if out[k] else np.empty(0)
    return out


def _make_sim_2d(polarization=(0.0, 0.0, 0.0)):
    sim = Simulation(
        nx=32,
        ny=32,
        dx=l0 / 20,
        dy=l0 / 20,
        npatch_x=2,
        npatch_y=2,
    )
    ele = Electron(
        density=lambda x, y: 0.01 * nc,
        ppc=4,
        polarization=polarization,
        pusher="boris+tbmt",
    )
    sim.add_species([ele])
    sim.initialize()
    return sim, ele


@pytest.mark.unit
class TestSetSpin2D:
    """2D: spatially varying functions, constants, defaults."""

    def test_spatial_functions(self):
        """All three components from vectorized f(x, y).

        The plasma is at rest with no fields, so the spin stays frozen
        after the init stage.
        """
        sim, ele = _make_sim_2d()
        set_spin = SetSpin(
            ele,
            sx=lambda x, y: np.cos(x / l0),
            sy=lambda x, y: np.sin(x / l0) * np.cos(y / l0),
            sz=lambda x, y: 0.5 * np.ones_like(x),
        )
        sim.run(1, callbacks=[set_spin])

        d = _gather_spin(sim, ele)
        assert d["sx"].size > 0
        assert_allclose(d["sx"], np.cos(d["x"] / l0), rtol=1e-12)
        assert_allclose(d["sy"], np.sin(d["x"] / l0) * np.cos(d["y"] / l0), rtol=1e-12)
        assert_allclose(d["sz"], 0.5 * np.ones_like(d["x"]), rtol=1e-12)

    def test_constants_and_mixed(self):
        """Constant, callable and unset (defaults to 0) components."""
        sim, ele = _make_sim_2d()
        x0 = 16 * l0 / 20  # domain center
        set_spin = SetSpin(
            ele,
            sx=0.3,
            sy=lambda x, y: np.exp(-((x - x0) ** 2 + (y - x0) ** 2) / (2 * l0**2)),
        )
        sim.run(1, callbacks=[set_spin])

        d = _gather_spin(sim, ele)
        assert_allclose(d["sx"], 0.3, rtol=1e-15)
        assert_allclose(
            d["sy"],
            np.exp(-((d["x"] - x0) ** 2 + (d["y"] - x0) ** 2) / (2 * l0**2)),
            rtol=1e-12,
        )
        # unset component defaults to 0
        assert_allclose(d["sz"], 0.0, atol=1e-15)

    def test_unset_components_override_polarization(self):
        """Unset components are zeroed, overriding the uniform polarization init."""
        sim, ele = _make_sim_2d(polarization=(1.0, 0.0, 0.0))
        set_spin = SetSpin(ele, sz=lambda x, y: 0.25 * np.ones_like(x))
        sim.run(1, callbacks=[set_spin])

        d = _gather_spin(sim, ele)
        assert_allclose(d["sx"], 0.0, atol=1e-15)
        assert_allclose(d["sy"], 0.0, atol=1e-15)
        assert_allclose(d["sz"], 0.25, rtol=1e-15)

    def test_dead_particles_untouched(self):
        """Only alive particles are updated; dead ones keep their spin."""
        sim, ele = _make_sim_2d(polarization=(0.7, 0.0, 0.0))
        for p in sim.patches:
            part = p.particles[ele.ispec]
            part.is_dead[::2] = True

        set_spin = SetSpin(ele, sx=lambda x, y: 0.25 * np.ones_like(x))
        set_spin(sim)  # itime == 0, interval triggered

        for p in sim.patches:
            part = p.particles[ele.ispec]
            dead = part.is_dead
            alive = part.is_alive
            if dead.any():
                # dead particles keep the polarization-initialized value
                assert_allclose(part.sx[dead], 0.7, rtol=1e-15)
            if alive.any():
                assert_allclose(part.sx[alive], 0.25, rtol=1e-15)


@pytest.mark.unit
class TestSetSpin3D:
    """3D: spatially varying functions of (x, y, z)."""

    def test_spatial_functions(self):
        from lambdapic import Simulation3D

        sim = Simulation3D(
            nx=16,
            ny=16,
            nz=16,
            dx=l0 / 20,
            dy=l0 / 20,
            dz=l0 / 20,
            npatch_x=2,
            npatch_y=2,
            npatch_z=2,
        )
        ele = Electron(
            density=lambda x, y, z: 0.01 * nc,
            ppc=1,
            polarization=(0.0, 0.0, 0.0),
            pusher="boris+tbmt",
        )
        sim.add_species([ele])
        sim.initialize()

        set_spin = SetSpin(
            ele,
            sx=lambda x, y, z: np.cos(x / l0) * np.cos(y / l0) * np.cos(z / l0),
            sy=lambda x, y, z: np.sin(x / l0) * np.sin(z / l0),
            sz=0.1,
        )
        sim.run(1, callbacks=[set_spin])

        d = _gather_spin(sim, ele)
        assert d["sx"].size > 0
        assert_allclose(
            d["sx"],
            np.cos(d["x"] / l0) * np.cos(d["y"] / l0) * np.cos(d["z"] / l0),
            rtol=1e-12,
        )
        assert_allclose(
            d["sy"],
            np.sin(d["x"] / l0) * np.sin(d["z"] / l0),
            rtol=1e-12,
        )
        assert_allclose(d["sz"], 0.1, rtol=1e-15)


@pytest.mark.unit
class TestSetSpinErrors:
    """Error handling for species without spin arrays."""

    def test_raises_without_polarization(self):
        sim = Simulation(
            nx=16,
            ny=16,
            dx=l0 / 20,
            dy=l0 / 20,
            npatch_x=2,
            npatch_y=2,
        )
        ele = Electron(density=lambda x, y: 0.01 * nc, ppc=2)  # no polarization
        sim.add_species([ele])
        sim.initialize()

        with pytest.raises(TypeError, match="spin arrays"):
            sim.run(1, callbacks=[SetSpin(ele, sx=1.0)])
