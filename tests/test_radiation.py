import numpy as np
import pytest

from lambdapic.core.fields import Fields2D
from lambdapic.core.patch.patch import Patch2D, Patches
from lambdapic.core.qed.radiation import NonlinearComptonLCFA
from lambdapic.core.species import Electron, Photon


@pytest.fixture
def radiation():
    """Build a 2D NonlinearComptonLCFA on a 2x2 patch grid with electrons + photons."""
    nc = 1.74e27

    dx = 1e-6
    dy = 1e-6

    nx = 128
    ny = 128

    npatch_x = 2
    npatch_y = 2

    nx_per_patch = nx // npatch_x
    ny_per_patch = ny // npatch_y

    Lx = nx * dx
    Ly = ny * dy

    n_guard = 3
    patches = Patches(dimension=2)
    for j in range(npatch_y):
        for i in range(npatch_x):
            index = i + j * npatch_x
            p = Patch2D(
                rank=0,
                index=index,
                ipatch_x=i,
                ipatch_y=j,
                x0=i * Lx / npatch_x,
                y0=j * Ly / npatch_y,
                nx=nx_per_patch,
                ny=ny_per_patch,
                dx=dx,
                dy=dy,
            )
            f = Fields2D(
                nx=nx_per_patch, ny=ny_per_patch, dx=dx, dy=dy,
                x0=i * Lx / npatch_x, y0=j * Ly / npatch_y, n_guard=n_guard,
            )
            p.set_fields(f)

            if i > 0:
                p.set_neighbor_index(xmin=(i - 1) + j * npatch_x)
            if i < npatch_x - 1:
                p.set_neighbor_index(xmax=(i + 1) + j * npatch_x)
            if j > 0:
                p.set_neighbor_index(ymin=i + (j - 1) * npatch_x)
            if j < npatch_y - 1:
                p.set_neighbor_index(ymax=i + (j + 1) * npatch_x)

            patches.append(p)

    def density(x, y):
        n0 = 0.01 * nc
        return n0

    ele = Electron(density=density, ppc=1, radiation="photons")
    pho = Photon()

    ele.set_photon(pho)

    patches.add_species(ele)
    ele.ispec = 0
    patches.add_species(pho)
    pho.ispec = 1

    rng = np.random.default_rng(42)
    patches.fill_particles(rng)

    patches.update_lists()

    for patch in patches:
        p = patch.particles[0]
        p.ux.fill(10)
        p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2)) ** -0.5
        p.chi.fill(0.1)
        p.is_dead[:] = rng.uniform(size=p.is_dead.size) < 0.1

    return NonlinearComptonLCFA(patches, 0)


def test_chi(radiation):
    patches = radiation.patches
    for patch in patches:
        p = patch.particles[0]
        p.ey_part.fill(1e12)
        p.chi.fill(0)

    radiation.update_chi()
    for ipatch in range(patches.npatches):
        chi = radiation.chi_list[ipatch]
        is_dead = radiation.is_dead_list[ipatch]
        alive = ~is_dead
        assert np.all(chi[alive] > 0), f"patch {ipatch}: chi not positive for alive particles"
        assert np.all(chi[~alive] == 0), f"patch {ipatch}: chi not zero for dead particles"

    chi1_all = [c.copy() for c in radiation.chi_list]
    for patch in patches:
        patch.particles[0].ey_part.fill(2e12)
    radiation.update_chi()
    for ipatch in range(patches.npatches):
        alive = ~radiation.is_dead_list[ipatch]
        np.testing.assert_allclose(
            radiation.chi_list[ipatch][alive] / chi1_all[ipatch][alive],
            2.0, rtol=1e-10,
            err_msg=f"patch {ipatch}: chi not proportional to E"
        )


def test_event(radiation):
    radiation.event(dt=0.1)
    total_events = sum(e.sum() for e in radiation.event_list)
    assert total_events > 0, "no events occurred"
    for ipatch in range(radiation.patches.npatches):
        event = radiation.event_list[ipatch]
        is_dead = radiation.is_dead_list[ipatch]
        assert event.dtype == bool
        assert event.shape == is_dead.shape
        assert event.sum() <= np.logical_not(is_dead).sum(), (
            f"patch {ipatch}: events on dead particles"
        )


def test_create_particles(radiation):
    assert radiation.x_pho_list[0].size == 0
    radiation.event(dt=0.1)
    radiation.create_particles()
    assert radiation.x_pho_list[0].size > 0


def test_reaction(radiation):
    radiation.event(dt=0.1)
    total_events = sum(e.sum() for e in radiation.event_list)
    assert total_events > 0, "no events occurred"
    delta_all = [d.copy() for d in radiation.delta_list]
    radiation.reaction()
    for ipatch in range(radiation.patches.npatches):
        ux = radiation.ux_list[ipatch]
        delta = delta_all[ipatch]
        is_dead = radiation.is_dead_list[ipatch]
        alive = ~is_dead
        np.testing.assert_allclose(
            ux[alive], (1 - delta[alive]) * 10.0, rtol=1e-14,
            err_msg=f"patch {ipatch}: recoil not consistent"
        )
