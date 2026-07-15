import numpy as np
import pytest

from lambdapic.core.fields import Fields2D
from lambdapic.core.patch.patch import Patch2D, Patches
from lambdapic.core.qed.pair_production import NonlinearPairProductionLCFA
from lambdapic.core.species import Electron, Photon, Positron


@pytest.fixture
def pair_production():
    """Build a 2D NonlinearPairProductionLCFA on a 2x2 patch grid with photons -> e-/e+."""
    nc = 1.74e27

    dx = 1e-6
    dy = 1e-6

    nx = 32
    ny = 32

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

    pho = Photon(density=density, ppc=1)
    ele = Electron()
    pos = Positron()

    pho.set_bw_pair(electron=ele, positron=pos)

    patches.add_species(pho)
    pho.ispec = 0
    patches.add_species(ele)
    ele.ispec = 1
    patches.add_species(pos)
    pos.ispec = 2

    rng = np.random.default_rng(42)
    patches.fill_particles(rng)

    patches.update_lists()

    for patch in patches:
        p = patch.particles[0]
        p.ux.fill(10)
        p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2)) ** -0.5
        p.chi.fill(0.1)
        p.is_dead[:] = rng.uniform(size=p.is_dead.size) < 0.1

    return NonlinearPairProductionLCFA(patches, 0)


def test_chi(pair_production):
    patches = pair_production.patches
    for patch in patches:
        p = patch.particles[0]
        p.ey_part.fill(1e12)
        p.chi.fill(0)

    pair_production.update_chi()
    for ipatch in range(patches.npatches):
        chi = pair_production.chi_list[ipatch]
        is_dead = pair_production.is_dead_list[ipatch]
        alive = ~is_dead
        assert np.all(chi[alive] > 0), f"patch {ipatch}: chi not positive for alive particles"
        assert np.all(chi[~alive] == 0), f"patch {ipatch}: chi not zero for dead particles"

    chi1_all = [c.copy() for c in pair_production.chi_list]
    for patch in patches:
        patch.particles[0].ey_part.fill(2e12)
    pair_production.update_chi()
    for ipatch in range(patches.npatches):
        alive = ~pair_production.is_dead_list[ipatch]
        np.testing.assert_allclose(
            pair_production.chi_list[ipatch][alive] / chi1_all[ipatch][alive],
            2.0, rtol=1e-10,
            err_msg=f"patch {ipatch}: chi not proportional to E"
        )


def test_event(pair_production):
    pair_production.event(dt=100)
    for ipatch in range(pair_production.patches.npatches):
        event = pair_production.event_list[ipatch]
        is_dead = pair_production.is_dead_list[ipatch]
        assert event.dtype == bool
        assert event.shape == is_dead.shape
        assert event.sum() == np.logical_not(is_dead).sum(), (
            f"patch {ipatch}: not all alive photons have events"
        )


def test_create_particles(pair_production):
    assert pair_production.x_ele_list[0].size == 0
    assert pair_production.x_pos_list[0].size == 0
    pair_production.event(dt=0.1)
    pair_production.create_particles()
    assert pair_production.x_ele_list[0].size > 0
    assert pair_production.x_pos_list[0].size > 0


def test_reaction(pair_production):
    patches = pair_production.patches
    pair_production.event(dt=100)
    pre_is_dead = [d.copy() for d in pair_production.is_dead_list]
    pair_production.reaction()
    for ipatch in range(patches.npatches):
        post_is_dead = pair_production.is_dead_list[ipatch]
        event = pair_production.event_list[ipatch]
        np.testing.assert_array_equal(
            post_is_dead, pre_is_dead[ipatch] | event,
            err_msg=f"patch {ipatch}: reaction should kill event photons"
        )
        ele = patches[ipatch].particles[pair_production.electron_ispec]
        pos = patches[ipatch].particles[pair_production.positron_ispec]
        assert not ele.is_dead.any(), f"patch {ipatch}: electrons affected by reaction"
        assert not pos.is_dead.any(), f"patch {ipatch}: positrons affected by reaction"
