"""Bitwise reproducibility of the compiled collision kernels across
OpenMP thread counts.

The C ports replaced numba prange with ``#pragma omp parallel for`` and one
PCG32 stream per patch, promising that results are independent of the OMP
thread count (each thread only touches its own patch's stream). These tests
lock that property in by re-running the same seeds under different
``omp_set_num_threads`` values and comparing the particle arrays bitwise.

The thread count is switched at runtime through libgomp (the extensions link
it), so the tests run in-process without env-var re-exec gymnastics.
"""

import ctypes

import numpy as np
import pytest

from lambdapic.core.collision.cpu import (
    inter_collision_patches,
    intra_collision_patches,
    seed_rng_states,
)
from lambdapic.core.particles import ParticlesBase

m_e = 9.1093837015e-31
eV = 1.602176634e-19

cell_vol = 1e-18
dt = 1e-15
npatches = 8
npp = 128  # particles per patch per species

try:
    ctypes.CDLL("libgomp.so.1")
    _HAS_LIBGOMP = True
except OSError:
    _HAS_LIBGOMP = False


def _omp_lib():
    return ctypes.CDLL("libgomp.so.1")


def _set_threads(n: int) -> None:
    _omp_lib().omp_set_num_threads(n)


def _patch_arrays(n: int, seed: int) -> list:
    """One random-but-seeded particle set per patch, as plain buffers."""
    out = []
    for ipatch in range(npatches):
        rng = np.random.default_rng(seed + ipatch)
        p = ParticlesBase()
        p.initialize(n)
        p.ux[:] = rng.normal(0.0, 0.05, n)
        p.uy[:] = rng.normal(0.0, 0.05, n)
        p.uz[:] = rng.normal(0.0, 0.05, n)
        p.inv_gamma[:] = 1.0 / np.sqrt(1.0 + p.ux**2 + p.uy**2 + p.uz**2)
        p.w[:] = 1e29 * cell_vol / n
        p.is_dead[:] = False
        out.append(p)
    return out


def _bmin() -> list:
    # one cell per patch, starting at the first slot
    return [np.array([0], dtype=np.int64) for _ in range(npatches)]


def _bmax(n: int) -> list:
    # ...covering the whole patch: [0, n)
    return [np.array([n], dtype=np.int64) for _ in range(npatches)]


def _snapshot(parts) -> list:
    return [p.ux.copy() for p in parts] + [p.uy.copy() for p in parts] + [p.uz.copy() for p in parts]


def _run_intra(parts, nthreads: int) -> list:
    _set_threads(nthreads)
    intra_collision_patches(
        parts, m_e, -eV,
        _bmin(), _bmax(npp),
        2.0,  # lnLambda
        [np.zeros(1) for _ in range(npatches)],  # debye (unused, constant lnL)
        cell_vol, dt,
        seed_rng_states(np.arange(npatches, dtype=np.int64)),
    )
    return _snapshot(parts)


def _run_inter(parts1, parts2, nthreads: int) -> list:
    _set_threads(nthreads)
    inter_collision_patches(
        parts1, m_e, -eV,
        _bmin(), _bmax(npp),
        parts2, m_e, -eV,
        _bmin(), _bmax(npp),
        2.0,
        [np.zeros(1) for _ in range(npatches)],
        cell_vol, dt,
        seed_rng_states(np.arange(npatches, dtype=np.int64)),
    )
    return _snapshot(parts1) + _snapshot(parts2)


@pytest.mark.skipif(not _HAS_LIBGOMP, reason="libgomp not available")
@pytest.mark.parametrize("nthreads", [2, 4, 8])
def test_intra_collision_thread_count_reproducibility(nthreads):
    ref = _run_intra(_patch_arrays(npp, 123), 1)
    out = _run_intra(_patch_arrays(npp, 123), nthreads)
    for a, b in zip(ref, out):
        np.testing.assert_array_equal(a, b)


def test_intra_collision_deflects():
    """Companion to the reproducibility tests: assert the setup actually
    collides, so the bitwise comparison above is not vacuous (it once was:
    bmin == bmax made every cell empty)."""
    parts = _patch_arrays(npp, 123)
    before = _snapshot(parts)
    _run_intra(parts, 1)
    after = _snapshot(parts)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after)), \
        "expected collision momenta to change"


@pytest.mark.skipif(not _HAS_LIBGOMP, reason="libgomp not available")
@pytest.mark.parametrize("nthreads", [2, 4, 8])
def test_inter_collision_thread_count_reproducibility(nthreads):
    ref = _run_inter(_patch_arrays(npp, 5), _patch_arrays(npp, 6), 1)
    out = _run_inter(_patch_arrays(npp, 5), _patch_arrays(npp, 6), nthreads)
    for a, b in zip(ref, out):
        np.testing.assert_array_equal(a, b)


def test_inter_collision_deflects():
    """Companion to the reproducibility tests: assert the setup actually
    collides, so the bitwise comparison above is not vacuous."""
    parts1 = _patch_arrays(npp, 5)
    parts2 = _patch_arrays(npp, 6)
    before = _snapshot(parts1) + _snapshot(parts2)
    _run_inter(parts1, parts2, 1)
    after = _snapshot(parts1) + _snapshot(parts2)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after)), \
        "expected collision momenta to change"
