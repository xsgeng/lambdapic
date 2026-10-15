"""Particle injection from domain boundaries.

The :class:`ParticleInjector` callback is a boundary flux source for an
existing species. On every call it samples macroparticles in the one-cell
slab at a domain face, pre-drifts them by one timestep along their sampled
momenta (plus one cell outward, so the layer straddles the boundary), and
keeps only the particles that land inside the domain. The surviving
population then tiles the boundary region with the exact analytical flux
``n * v * A * dt`` per step.

Profiles are given over the *transverse* coordinates of the injection face
(in ascending x/y/z order), optionally followed by the time ``t``:

===========  ==================  ==================
side         3D signature        2D signature
===========  ==================  ==================
xmin/xmax    f(y, z[, t])        f(y[, t])
ymin/ymax    f(x, z[, t])        f(x[, t])
zmin/zmax    f(x, y[, t])        --
===========  ==================  ==================

Scalars are broadcast to the whole face.
"""
import inspect
from typing import Callable

import numpy as np
from scipy.constants import c, e

from ..core.patch.patch import Patch
from ..core.species import Species
from ..simulation import Simulation
from .callback import Callback
from .utils import SetTemperature

_SIDES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
_AXIS_NAME = ("x", "y", "z")


class ParticleInjector(Callback):
    """Continuously inject particles of an existing species from a boundary face.

    Parameters
    ----------
    species : Species
        Target species. Its ``density`` may be None; all of its particles can
        then come from this injector.
    density : float | Callable
        Number density in SI units (lab frame), as a scalar or a profile
        ``f(⊥[, t])`` over the transverse coordinates of the face.
    side : str
        Boundary face to inject from: one of "xmin", "xmax", "ymin", "ymax",
        "zmin", "zmax".
    ux, uy, uz : float | Callable
        Local mean normalized momentum ``u = p / (m_e c)`` in the lab frame,
        as scalars or profiles ``f(⊥[, t])``. For a cold beam (``temperature=0``)
        every particle carries exactly this momentum. Cells whose drift points
        out of the domain inject nothing (the injector is a one-way flux source).
    temperature : float | Callable
        Temperature in eV (proper, rest-frame temperature), scalar or profile.
        With a nonzero temperature the momentum is sampled from a
        Maxwell-Jüttner distribution around the local drift.
    ppc : int
        Macroparticles sampled per slab cell.
    position_initialization : str
        "random" (default) or "centered" transverse placement. The position
        along the injection axis is always sampled uniformly: this is what
        makes the pre-drift filter reproduce the correct boundary flux.
    interval : int | float | Callable
        Callback interval (timesteps, seconds, or predicate), see :any:`Callback`.

    Notes
    -----
    The injection happens at the ``start`` stage, so injected particles are
    pushed and deposit current in the same timestep. Cross-patch migration of
    the injected particles is handled by the regular particle sync.

    The injection slab is the first (or last) cell of the boundary patches.
    On a side with a PML boundary the PML layer is a particle-absorbing zone,
    so the slab is the first (or last) cell *inside* the PML layer and the
    plasma enters directly at the interior edge.

    Examples
    --------
    >>> injector = ParticleInjector(
    ...     beam, density=lambda y: 0.1 * nc, side="xmin", ux=3.0, ppc=8,
    ... )
    >>> sim.run(1000, callbacks=[injector])
    """

    DEFAULT_STAGE = "start"

    def __init__(
        self,
        species: Species,
        density: float | Callable,
        side: str = "xmin",
        ux: float | Callable = 0.0,
        uy: float | Callable = 0.0,
        uz: float | Callable = 0.0,
        temperature: float | Callable = 0.0,
        ppc: int = 1,
        position_initialization: str = "random",
        interval: int | float | Callable = 1,
    ) -> None:
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
        if position_initialization not in ("random", "centered"):
            raise ValueError(
                "position_initialization must be 'random' or 'centered', "
                f"got {position_initialization!r}"
            )
        if int(ppc) != ppc or ppc < 1:
            raise ValueError(f"ppc must be a positive integer, got {ppc}")
        if species.mass == 0 and not self._is_zero(temperature):
            raise ValueError(
                f"species {species.name!r} is massless: thermal injection "
                "is not defined, use a cold beam (temperature=0)"
            )

        self.species = species
        self.density = density
        self.side = side
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.temperature = temperature
        self.ppc = int(ppc)
        self.position_initialization = position_initialization
        self.stage = self.DEFAULT_STAGE
        self.interval = interval

    @staticmethod
    def _is_zero(value: float | Callable) -> bool:
        return not callable(value) and value == 0

    @staticmethod
    def sample_drifting_maxwell_juttner(
        size: int,
        theta: float,
        drift_momentum: tuple[float, float, float],
        rand_gen: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample momenta of a drifting Maxwell-Jüttner distribution.

        Rest-frame isotropic Maxwell-Jüttner samples are boosted to the lab
        frame with the drift velocity and thinned with the acceptance
        probability ``p = (eps/eps*) / (gamma_d (1 + beta_d))``, which loads
        the exact Jüttner-Synge distribution (lab density ``gamma_d n*``,
        bulk velocity exactly ``v_d``) instead of the naive boosted sample.
        The naive boost under-weights particles moving along the drift by
        ``eps*/eps`` and badly distorts relativistically drifted beams.

        Parameters
        ----------
        size : int
            Number of samples to draw.
        theta : float
            Proper (rest-frame) temperature in units of ``m c**2``.
        drift_momentum : tuple[float, float, float]
            Drift as normalized momentum ``u_d = p / (m c)``; the sampling
            distribution has bulk velocity ``v_d = u_d c / gamma_d``.
        rand_gen : np.random.Generator, optional
            Source of randomness.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            Normalized momenta ``(ux, uy, uz)`` of length ``size``.

        Notes
        -----
        The acceptance-rejection scheme follows S. Zenitani,
        Phys. Plasmas 22, 042116 (2015).
        """
        if rand_gen is None:
            rand_gen = np.random.default_rng()

        ud = np.asarray(drift_momentum, dtype=float)
        ud2 = float((ud**2).sum())
        if ud2 == 0.0:
            return SetTemperature.sample_maxwell_juttner(size, theta, rand_gen)

        gamma_d = np.sqrt(1.0 + ud2)
        beta_vec = ud / gamma_d  # drift velocity / c
        beta = float(np.sqrt((beta_vec**2).sum()))
        betahat = beta_vec / beta
        acc_norm = 1.0 / (gamma_d * (1.0 + beta))

        ux = np.empty(size)
        uy = np.empty(size)
        uz = np.empty(size)
        filled = 0
        while filled < size:
            n_missing = size - filled
            sx, sy, sz = SetTemperature.sample_maxwell_juttner(
                n_missing, theta, rand_gen
            )
            s = np.stack((sx, sy, sz), axis=1)
            eps_star = np.sqrt(1.0 + (s**2).sum(axis=1))

            # general Lorentz boost with velocity beta_vec
            par = s @ betahat
            u = s + np.outer(
                par * (gamma_d - 1.0) + gamma_d * eps_star * beta, betahat
            )
            eps = gamma_d * (eps_star + par * beta)

            accept = rand_gen.uniform(size=n_missing) < (eps / eps_star) * acc_norm
            u_acc = u[accept]
            k = len(u_acc)
            ux[filled : filled + k] = u_acc[:, 0]
            uy[filled : filled + k] = u_acc[:, 1]
            uz[filled : filled + k] = u_acc[:, 2]
            filled += k

        return ux, uy, uz

    def _call(self, sim: Simulation) -> None:
        axis = _AXIS_NAME.index(self.side[0])
        at_max = self.side[1:] == "max"

        if axis >= sim.dimension:
            raise ValueError(
                f"side {self.side!r} is only valid in a 3D simulation"
            )

        if sim.dimension == 2:
            npatch_counts = (sim.npatch_x, sim.npatch_y)
        else:
            npatch_counts = (sim.npatch_x, sim.npatch_y, sim.npatch_z)

        trans_axes = [a for a in range(sim.dimension) if a != axis]
        axis_name = _AXIS_NAME[axis]
        ispec = self.species.ispec

        any_injected = False
        for p in sim.patches:
            if getattr(p, f"ipatch_{axis_name}") != (
                npatch_counts[axis] - 1 if at_max else 0
            ):
                continue

            injected = self._inject_patch(sim, p, axis, at_max, trans_axes, ispec)
            any_injected = any_injected or injected

        if any_injected:
            sim.update_lists()

    def _inject_patch(
        self,
        sim: Simulation,
        p: Patch,
        axis: int,
        at_max: bool,
        trans_axes: list[int],
        ispec: int,
    ) -> bool:
        """Sample, pre-drift, filter and append particles for one patch.

        Returns True if any particle was appended.
        """
        # slab geometry: one cell thick at the domain face. The patch
        # xmin/xmax properties already skip the PML layer, so on a PML face
        # the slab is the first interior cell (particles entering the PML
        # layer are absorbed by the particle sync).
        axis_name = _AXIS_NAME[axis]
        axis_arr = getattr(p, axis_name + "axis")
        axis_dl = getattr(p, "d" + axis_name)
        face = getattr(p, axis_name + ("max" if at_max else "min"))
        x_b = face
        cell_volume = p.dx * p.dy * p.dz if sim.dimension == 3 else p.dx * p.dy

        # transverse cell-center mesh over the patch face
        trans_arrs = [getattr(p, _AXIS_NAME[a] + "axis") for a in trans_axes]
        mesh = np.meshgrid(*trans_arrs, indexing="ij")
        shape = mesh[0].shape

        t = sim.time
        dens = self._evaluate_profile(self.density, mesh, t, "density")
        mask = dens > 0.0
        n_cells = int(mask.sum())
        if n_cells == 0:
            return False

        ux = self._evaluate_profile(self.ux, mesh, t, "ux")
        uy = self._evaluate_profile(self.uy, mesh, t, "uy")
        uz = self._evaluate_profile(self.uz, mesh, t, "uz")

        n_part = n_cells * self.ppc

        # per-particle cell values: each dense cell contributes `ppc` particles
        w_cell = dens[mask] * cell_volume / self.ppc
        w_p = np.repeat(w_cell, self.ppc)

        gen = sim.rand_gen.spawn(1)[0]

        if self._is_zero(self.temperature):
            # cold beam: every particle carries the drift momentum exactly
            ux_p = np.repeat(ux[mask], self.ppc)
            uy_p = np.repeat(uy[mask], self.ppc)
            uz_p = np.repeat(uz[mask], self.ppc)
        else:
            # thermal beam: drifting Maxwell-Jüttner sampling per cell
            temp = self._evaluate_profile(self.temperature, mesh, t, "temperature")
            theta_cell = temp[mask] * e / (self.species.m * c**2)

            params = np.stack(
                (theta_cell, ux[mask], uy[mask], uz[mask]), axis=1
            )
            unique_params, cell_group = np.unique(params, axis=0, return_inverse=True)

            ux_p = np.empty(n_part)
            uy_p = np.empty(n_part)
            uz_p = np.empty(n_part)
            # particle slots of dense cell j are j*ppc ... (j+1)*ppc-1, so the
            # per-particle group id is the per-cell group id repeated ppc times
            particle_group = np.repeat(cell_group, self.ppc)
            for g, (theta, *drift) in enumerate(unique_params):
                slots = np.flatnonzero(particle_group == g)
                gx, gy, gz = self.sample_drifting_maxwell_juttner(
                    slots.size, float(theta), tuple(map(float, drift)), gen
                )
                ux_p[slots] = gx
                uy_p[slots] = gy
                uz_p[slots] = gz

        inv_gamma = 1.0 / np.sqrt(1.0 + ux_p**2 + uy_p**2 + uz_p**2)

        # positions: normal coordinate is always uniform within the slab cell
        normal_pos = x_b + gen.uniform(-axis_dl / 2, axis_dl / 2, n_part)

        trans_pos: list[np.ndarray] = []
        for i, (a, arr) in enumerate(zip(trans_axes, trans_arrs)):
            centers = np.repeat(mesh[i][mask], self.ppc)
            if self.position_initialization == "random":
                dl = getattr(p, "d" + _AXIS_NAME[a])
                centers = centers + gen.uniform(-dl / 2, dl / 2, n_part)
            trans_pos.append(centers)

        # pre-drift: free flight over one timestep plus one cell outward
        dt = sim.dt
        velocities = (ux_p * inv_gamma * c, uy_p * inv_gamma * c, uz_p * inv_gamma * c)
        shift = axis_dl if at_max else -axis_dl
        normal_pos = normal_pos + velocities[axis] * dt + shift
        trans_pos = [
            pos + velocities[a] * dt for a, pos in zip(trans_axes, trans_pos)
        ]

        # keep only particles that land inside the domain, on the interior
        # side of the face cell edge (matches the absorbing bound used by
        # the particle sync, which also accounts for the PML layer)
        if at_max:
            keep = normal_pos <= face + axis_dl / 2
        else:
            keep = normal_pos >= face - axis_dl / 2

        n_keep = int(keep.sum())
        if n_keep == 0:
            return False

        part = p.particles[ispec]
        part.extend(n_keep)
        sl = slice(part.npart - n_keep, part.npart)

        part.w[sl] = w_p[keep]
        part.ux[sl] = ux_p[keep]
        part.uy[sl] = uy_p[keep]
        part.uz[sl] = uz_p[keep]
        part.inv_gamma[sl] = inv_gamma[keep]

        coords = [None, None, None]
        coords[axis] = normal_pos[keep]
        for a, pos in zip(trans_axes, trans_pos):
            coords[a] = pos[keep]
        for a in range(sim.dimension):
            arr = getattr(part, _AXIS_NAME[a])
            arr[sl] = coords[a]
        if sim.dimension == 2:
            part.z[sl] = 0.0

        part.is_dead[sl] = False
        return True

    def _evaluate_profile(
        self,
        profile: float | Callable,
        mesh: list[np.ndarray],
        t: float,
        name: str,
    ) -> np.ndarray:
        """Evaluate a scalar or profile on the transverse cell-center mesh."""
        shape = mesh[0].shape
        if not callable(profile):
            return np.full(shape, float(profile))

        n_trans = len(mesh)
        arity = self._positional_arity(profile, name)
        if arity not in (n_trans, n_trans + 1):
            raise ValueError(
                f"{name} profile {profile} must accept {n_trans} transverse "
                f"coordinate(s) and an optional trailing time argument t"
            )
        args = list(mesh) + ([t] if arity == n_trans + 1 else [])

        values = None
        try:
            values = np.asarray(profile(*args), dtype=float)
            if values.shape not in (shape, ()):
                values = None
        except Exception:
            values = None

        if values is None:
            # scalar-only profiles (e.g. plain if/else): loop elementwise
            values = np.asarray(np.vectorize(profile, otypes=[float])(*args), dtype=float)

        if values.shape == ():
            values = np.full(shape, float(values))
        if values.shape != shape:
            raise ValueError(f"{name} profile returned shape {values.shape}, expected {shape}")
        return values

    @staticmethod
    def _positional_arity(func: Callable, name: str) -> int:
        params = inspect.signature(func).parameters.values()
        n = 0
        for prm in params:
            if prm.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                n += 1
            elif prm.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ValueError(f"{name} profile {func} must not use *args")
        return n
