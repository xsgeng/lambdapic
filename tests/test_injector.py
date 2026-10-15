"""Tests for the ParticleInjector callback.

The injector is a boundary flux source: each call it samples macroparticles
in the one-cell slab at a domain face, pre-drifts them by one timestep along
their sampled momentum (plus one cell outward), and keeps the particles that
land inside the domain. For a cold beam this leaves a uniform layer of
thickness ``v*dt`` at the boundary whose total weight equals the analytical
flux ``n * v * A * dt``.
"""
import numpy as np
import pytest
from scipy.constants import c, epsilon_0, e, m_e, pi

from lambdapic import Electron, Photon, Simulation
from lambdapic.callback.injector import ParticleInjector


def test_top_level_export() -> None:
    """Examples import ParticleInjector from the lambdapic root."""
    import lambdapic

    assert lambdapic.ParticleInjector is ParticleInjector
    assert "ParticleInjector" in lambdapic.__all__

l0 = 0.8e-6
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2


def _gather_species(sim, species, attrs):
    """Collect alive-particle attributes across all local patches."""
    ispec = species.ispec
    out = {a: [] for a in attrs}
    for p in sim.patches:
        part = p.particles[ispec]
        alive = part.is_alive
        if not alive.any():
            continue
        for a in attrs:
            out[a].append(getattr(part, a)[alive])
    return {a: np.concatenate(v) if v else np.array([]) for a, v in out.items()}


def _setup_sim_2d() -> Simulation:
    sim = Simulation(
        nx=64,
        ny=64,
        dx=l0 / 20,
        dy=l0 / 20,
        npatch_x=4,
        npatch_y=4,
        boundary_conditions={
            "xmin": "periodic",
            "xmax": "periodic",
            "ymin": "periodic",
            "ymax": "periodic",
        },
        random_seed=42,
    )
    return sim


@pytest.mark.unit
class TestColdBeamInjection2D:
    """Cold beam injected from xmin: count, weights, momenta, positions."""

    def setup_method(self) -> None:
        self.nx = 64
        self.ny = 64
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = _setup_sim_2d()

        # Empty target species: all of its particles come from the injector.
        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_cold_beam_from_xmin(self) -> None:
        n0 = 0.1 * nc
        u0 = 1.0
        ppc = 8

        injector = ParticleInjector(
            self.beam, side="xmin", density=n0, ux=u0, ppc=ppc
        )
        assert injector.stage == "start"

        injector(self.sim)

        data = _gather_species(
            self.sim, self.beam, ["x", "y", "w", "ux", "uy", "uz", "inv_gamma"]
        )
        n_injected = data["x"].size

        # --- injected macro-particle count --------------------------------
        # Per transverse cell, ``ppc`` particles are created in the first cell
        # slab; each survives the pre-drift filter with probability v*dt/dx,
        # so the total count is Binomial(ny*ppc, v*dt/dx).
        gamma0 = np.sqrt(1.0 + u0**2)
        v = u0 / gamma0 * c
        p_surv = v * self.sim.dt / self.dx
        n_created = self.ny * ppc

        assert 0.0 < p_surv < 1.0
        expected = n_created * p_surv
        sigma = np.sqrt(n_created * p_surv * (1.0 - p_surv))
        assert abs(n_injected - expected) < 6 * sigma, (
            f"injected {n_injected}, expected {expected:.1f} +/- {sigma:.1f}"
        )

        # newly created slots hold no dead particles
        total_npart = sum(
            p.particles[self.beam.ispec].npart for p in self.sim.patches
        )
        assert total_npart == n_injected

        # --- weights: lab-frame density * cell volume / particles per cell --
        np.testing.assert_allclose(data["w"], n0 * self.dx * self.dy / ppc, rtol=1e-12)

        # --- cold momenta and inv_gamma ------------------------------------
        np.testing.assert_allclose(data["ux"], u0, rtol=1e-12)
        np.testing.assert_allclose(data["uy"], 0.0, atol=1e-12)
        np.testing.assert_allclose(data["uz"], 0.0, atol=1e-12)
        np.testing.assert_allclose(data["inv_gamma"], 1.0 / gamma0, rtol=1e-12)

        # --- positions ------------------------------------------------------
        # Cold beam: survivors land in [xmin_global, xmin_global + v*dt].
        xmin_global = self.sim.patches.xmin_global
        assert data["x"].min() >= xmin_global - 1e-9 * self.dx
        # min of ~N uniform samples sits close to the lower edge
        assert data["x"].min() <= xmin_global + 0.2 * v * self.sim.dt
        assert data["x"].max() <= xmin_global + v * self.sim.dt + 1e-9 * self.dx

        # y is uniform within the transverse cells of the boundary patches
        assert data["y"].min() >= -self.dy / 2
        assert data["y"].max() <= self.sim.Ly - self.dy / 2

    def test_centered_transverse_positions(self) -> None:
        """centered mode places particles at transverse cell centers exactly."""
        injector = ParticleInjector(
            self.beam,
            density=0.1 * nc,
            side="xmin",
            ux=1.0,
            ppc=4,
            position_initialization="centered",
        )
        injector(self.sim)

        data = _gather_species(self.sim, self.beam, ["x", "y"])
        assert data["x"].size > 0

        # the normal coordinate is still uniformly sampled (layer of width v*dt)
        u0 = 1.0
        v = u0 / np.sqrt(1.0 + u0**2) * c
        assert data["x"].max() >= self.sim.patches.xmin_global
        assert (data["x"].max() - data["x"].min()) > 0.4 * v * self.sim.dt

        # every y is exactly a transverse cell center of a boundary patch
        centers = np.concatenate(
            [p.yaxis for p in self.sim.patches if p.ipatch_x == 0]
        )
        assert np.all(np.isin(data["y"], centers))

    def test_zero_drift_injects_nothing(self) -> None:
        """A cold stationary beam has zero influx: nothing may be injected."""
        injector = ParticleInjector(
            self.beam, density=0.1 * nc, side="xmin", ux=0.0, ppc=8
        )
        injector(self.sim)

        for p in self.sim.patches:
            assert p.particles[self.beam.ispec].npart == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "side, momentum, trans_axis",
    [
        ("xmin", {"ux": +1.0}, "y"),
        ("xmax", {"ux": -1.0}, "y"),
        ("ymin", {"uy": +1.0}, "x"),
        ("ymax", {"uy": -1.0}, "x"),
    ],
)
class TestInjectionSides2D:
    """Same physics from every face: mirrored layer geometry and profiles.

    A density gate over the upper half of the *transverse* axis doubles as a
    check that the profile signature maps to the right coordinate for each side.
    """

    def setup_method(self) -> None:
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = _setup_sim_2d()

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_side_geometry_and_transverse_profile(
        self, side: str, momentum: dict, trans_axis: str
    ) -> None:
        axis = side[0]
        at_max = side[1:] == "max"

        u0 = abs(next(iter(momentum.values())))
        gamma0 = np.sqrt(1.0 + u0**2)
        v = u0 / gamma0 * c
        ppc = 8
        n0 = 0.1 * nc
        d_axis = getattr(self, f"d{axis}")

        # density gate over the upper half of the transverse axis
        half = getattr(self.sim, f"L{trans_axis}") / 2
        if trans_axis == "y":
            def density(y):
                return n0 if y >= half else 0.0
        else:
            def density(x):
                return n0 if x >= half else 0.0

        injector = ParticleInjector(
            self.beam, density=density, side=side, ppc=ppc, **momentum
        )
        injector(self.sim)

        data = _gather_species(
            self.sim, self.beam, ["x", "y", "w", "ux", "uy", "uz", "inv_gamma"]
        )
        n_injected = data["x"].size
        assert n_injected > 0, side

        # count: half of the transverse cells are gated on
        p_surv = v * self.sim.dt / d_axis
        n_created = (getattr(self.sim, f"n{trans_axis}") // 2) * ppc
        expected = n_created * p_surv
        sigma = np.sqrt(n_created * p_surv * (1.0 - p_surv))
        assert abs(n_injected - expected) < 6 * sigma, (
            f"{side}: injected {n_injected}, expected {expected:.1f} +/- {sigma:.1f}"
        )

        np.testing.assert_allclose(data["w"], n0 * self.dx * self.dy / ppc, rtol=1e-12)

        # particles may only land in patches on the injection face (a face
        # patch can still be empty if its cells are gated off by the profile)
        npatch_axis = getattr(self.sim, f"npatch_{axis}")
        for p in self.sim.patches:
            on_face = getattr(p, f"ipatch_{axis}") == (npatch_axis - 1 if at_max else 0)
            if not on_face:
                assert p.particles[self.beam.ispec].npart == 0, (
                    f"{side}: patch {p.index} is not on the face"
                )

        # mirrored layer geometry: [face, face + v*dt] measured inward
        bound = getattr(
            self.sim.patches,
            f"{axis}min_global" if not at_max else f"{axis}max_global",
        )
        normal = data[axis]
        layer_width = v * self.sim.dt
        if at_max:
            assert normal.min() >= bound - layer_width - 1e-9 * d_axis
            assert normal.max() <= bound + 1e-9 * d_axis
        else:
            assert normal.min() >= bound - 1e-9 * d_axis
            assert normal.max() <= bound + layer_width + 1e-9 * d_axis

        # cold momenta and inv_gamma are exact
        comp = next(iter(momentum))
        np.testing.assert_allclose(data[comp], momentum[comp], rtol=1e-12)
        for other in ("ux", "uy", "uz"):
            if other != comp:
                np.testing.assert_allclose(data[other], 0.0, atol=1e-12)
        np.testing.assert_allclose(data["inv_gamma"], 1.0 / gamma0, rtol=1e-12)

        # the gate ran on the transverse axis: particles only in the upper half
        # (cell centers >= half, sampled positions may jitter by half a cell)
        d_trans = getattr(self.sim, f"d{trans_axis}")
        assert (data[trans_axis] >= half - d_trans / 2).all(), (
            f"{side}: transverse gate leaked into the lower half"
        )


@pytest.mark.unit
class TestShearedColdFlow2D:
    """Per-cell drift profiles: each transverse band injects with its own flux."""

    def setup_method(self) -> None:
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = _setup_sim_2d()

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_shear_bands_have_band_specific_flux(self) -> None:
        u_lo, u_hi = 1.0, 2.0
        half = self.sim.Ly / 2
        n0 = 0.1 * nc
        ppc = 8

        def ux(y):
            return u_lo if y < half else u_hi

        injector = ParticleInjector(
            self.beam, density=n0, side="xmin", ux=ux, ppc=ppc
        )
        injector(self.sim)

        data = _gather_species(self.sim, self.beam, ["x", "y", "ux", "w"])
        assert data["x"].size > 0

        # cold beams carry their band momentum exactly
        lo = data["ux"] == u_lo
        hi = data["ux"] == u_hi
        assert lo.sum() > 0 and hi.sum() > 0

        # each band obeys its own Binomial(n_band_cells*ppc, v_i*dt/dx) law:
        # local flux = n * v(y) * A * dt, independent of the neighbouring band
        dt = self.sim.dt
        n_cells_per_band = self.sim.ny // 2
        for mask, u0 in ((lo, u_lo), (hi, u_hi)):
            v = u0 / np.sqrt(1.0 + u0**2) * c
            p_surv = v * dt / self.dx
            n_created = n_cells_per_band * ppc
            expected = n_created * p_surv
            sigma = np.sqrt(n_created * p_surv * (1.0 - p_surv))
            got = int(mask.sum())
            assert abs(got - expected) < 6 * sigma, (
                f"band u={u0}: injected {got}, expected {expected:.1f} +/- {sigma:.1f}"
            )

            # and its own layer thickness v*dt at the boundary
            layer = data["x"][mask]
            assert layer.min() >= self.sim.patches.xmin_global - 1e-9 * self.dx
            assert layer.max() <= self.sim.patches.xmin_global + v * dt + 1e-9 * self.dx

        # weights are still density * cell volume / ppc everywhere
        np.testing.assert_allclose(data["w"], n0 * self.dx * self.dy / ppc, rtol=1e-12)


@pytest.mark.unit
class TestTimeDependentInjection2D:
    """Profiles with a trailing time argument gate the injection in time."""

    def setup_method(self) -> None:
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = _setup_sim_2d()

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_density_time_gate(self) -> None:
        t_start = 10 * self.sim.dt
        n0 = 0.1 * nc

        def density(y, t):
            return n0 if t >= t_start else 0.0

        injector = ParticleInjector(
            self.beam, density=density, side="xmin", ux=3.0, ppc=8
        )

        # before start time: nothing
        self.sim.time = 0.0
        injector(self.sim)
        assert sum(
            p.particles[self.beam.ispec].npart for p in self.sim.patches
        ) == 0

        # after start time: injection resumes with the full flux
        self.sim.time = t_start
        injector(self.sim)
        assert sum(
            p.particles[self.beam.ispec].npart for p in self.sim.patches
        ) > 0

    def test_momentum_time_ramp(self) -> None:
        """A velocity ramp changes the injected layer thickness over time."""
        t_ramp = 20 * self.sim.dt
        n0 = 0.1 * nc

        def ux(y, t):
            return 3.0 if t >= t_ramp else 0.5

        injector = ParticleInjector(
            self.beam, density=n0, side="xmin", ux=ux, ppc=8
        )

        u_low = 0.5
        v_low = u_low / np.sqrt(1.0 + u_low**2) * c
        self.sim.time = 0.0
        injector(self.sim)
        x_first = _gather_species(self.sim, self.beam, ["x"])["x"]
        assert x_first.max() <= self.sim.patches.xmin_global + v_low * self.sim.dt + 1e-9 * self.dx

        u_high = 3.0
        v_high = u_high / np.sqrt(1.0 + u_high**2) * c
        self.sim.time = 20 * self.sim.dt + self.sim.dt
        injector(self.sim)
        x_all = _gather_species(self.sim, self.beam, ["x"])["x"]
        # the fast injection reaches deeper than the slow layer ever could
        assert x_all.max() >= self.sim.patches.xmin_global + 0.9 * v_high * self.sim.dt
        assert x_all.max() <= self.sim.patches.xmin_global + 1.01 * v_high * self.sim.dt


@pytest.mark.unit
class TestInjectorValidation:
    """Constructor and call-time argument validation."""

    def setup_method(self) -> None:
        self.sim = _setup_sim_2d()
        self.beam = Electron(name="beam", density=None)
        self.photon = Photon(name="gamma", density=None)
        self.sim.add_species([self.beam, self.photon])
        self.sim.initialize()

    def test_invalid_side(self) -> None:
        with pytest.raises(ValueError, match="side"):
            ParticleInjector(self.beam, density=1e27, side="left")

    def test_invalid_position_initialization(self) -> None:
        with pytest.raises(ValueError, match="position_initialization"):
            ParticleInjector(
                self.beam, density=1e27, position_initialization="lattice"
            )

    def test_invalid_ppc(self) -> None:
        with pytest.raises(ValueError, match="ppc"):
            ParticleInjector(self.beam, density=1e27, ppc=0)
        with pytest.raises(ValueError, match="ppc"):
            ParticleInjector(self.beam, density=1e27, ppc=2.5)

    def test_thermal_photon_rejected(self) -> None:
        with pytest.raises(ValueError, match="massless"):
            ParticleInjector(self.photon, density=1e27, ux=2.0, temperature=10.0)

    def test_cold_photon_allowed(self) -> None:
        injector = ParticleInjector(self.photon, density=1e27, ux=2.0)
        assert injector.species is self.photon

    def test_3d_side_rejected_in_2d(self) -> None:
        injector = ParticleInjector(self.beam, density=1e27, side="zmin", ux=1.0)
        with pytest.raises(ValueError, match="3D"):
            injector(self.sim)

    def test_profile_arity_rejected(self) -> None:
        # 2D xmin side: profiles take f(y) or f(y, t) -- not three arguments
        injector = ParticleInjector(
            self.beam, density=lambda x, y, z: 1e27, side="xmin", ux=1.0
        )
        with pytest.raises(ValueError, match="transverse"):
            injector(self.sim)

    def test_thermal_injector_runs(self) -> None:
        """Placeholder removed: thermal injection is implemented and tested
        in TestThermalBeamInjection2D / TestDriftingMaxwellJuttnerSampler."""


@pytest.mark.unit
class TestRunLoopIntegration2D:
    """Full ``sim.run`` loop: the injector feeds the pusher/deposition/sync chain.

    With uniform injection over N steps the total beam weight must equal
    ``N * n0 * v * Ly * dt`` (boundary flux), and every injected particle
    must have been pushed by the Boris pusher at least once.
    """

    NSTEPS = 8

    def setup_method(self) -> None:
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = _setup_sim_2d()

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_continuous_injection_over_run(self) -> None:
        import os

        os.environ["LAMBDAPIC_CHECK_UPDATE"] = "0"

        n0 = 0.01 * nc
        u0 = 1.0
        gamma0 = np.sqrt(1.0 + u0**2)
        v = u0 / gamma0 * c
        ppc = 8

        injector = ParticleInjector(
            self.beam, density=n0, side="xmin", ux=u0, ppc=ppc
        )

        self.sim.run(self.NSTEPS, callbacks=[injector])

        data = _gather_species(
            self.sim, self.beam, ["x", "y", "w", "ux", "uy", "uz", "inv_gamma"]
        )
        assert data["x"].size > 0

        # --- total weight: N steps of analytical boundary flux ---------------
        w_macro = n0 * self.dx * self.dy / ppc
        expected_w_total = self.NSTEPS * n0 * v * self.sim.Ly * self.sim.dt
        sigma_w_total = (
            self.NSTEPS**0.5 * w_macro
            * np.sqrt(self.sim.ny * ppc * (v * self.sim.dt / self.dx)
                      * (1.0 - v * self.sim.dt / self.dx))
        )
        w_total = data["w"].sum()
        assert abs(w_total - expected_w_total) < 3 * sigma_w_total, (
            f"total weight {w_total:.3e}, expected {expected_w_total:.3e} "
            f"± {3 * sigma_w_total:.1e}"
        )

        # uniform macro weights
        np.testing.assert_allclose(data["w"], w_macro, rtol=1e-12)

        # --- all injected particles were pushed -------------------------------
        # The pre-drift itself accounts for one window of streaming (the
        # crossing that happened during the previous step), and every step
        # then pushes all particles once more. The oldest layer (injected at
        # the start of step 0) therefore streams (NSTEPS+1)*v*dt in total.
        edge = self.sim.patches.xmin_global
        assert data["x"].min() >= edge + 0.8 * v * self.sim.dt
        assert data["x"].max() >= edge + (self.NSTEPS + 0.9) * v * self.sim.dt
        # and nothing outruns a free-streaming beam
        assert data["x"].max() <= edge + (self.NSTEPS + 1.05) * v * self.sim.dt

        # --- the beam stays (approximately) cold -------------------------------
        np.testing.assert_allclose(data["ux"].mean(), u0, rtol=0.05)
        assert data["ux"].std() < 0.05
        expected_inv_gamma = 1.0 / np.sqrt(1.0 + data["ux"]**2 + data["uy"]**2 + data["uz"]**2)
        np.testing.assert_allclose(data["inv_gamma"], expected_inv_gamma, rtol=1e-9)

        # every particle is alive: no losses through periodic wrap or sorters
        for p in self.sim.patches:
            part = p.particles[self.beam.ispec]
            assert part.is_dead.sum() == 0


@pytest.mark.unit
class TestDriftingMaxwellJuttnerSampler:
    """Statistical validation of the drifting MJ sampler at its public seam.

    Independent ground truth from the literature moments of the
    Juttner-Synge distribution: mean energy per particle
    ``<gamma> = gamma_d * (K1(1/theta)/K2(1/theta) + 3*theta)`` and bulk
    velocity exactly the drift velocity ``beta_d``.
    """

    RNG_SEED = 20260930
    N_SAMPLES = 200_000

    @staticmethod
    def _moments(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> tuple:
        gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        inv_gamma = 1.0 / gamma
        return (
            gamma.mean(),
            (ux * inv_gamma).mean(),
            (uy * inv_gamma).mean(),
            (uz * inv_gamma).mean(),
        )

    def test_zero_drift_reduces_to_thermal_maxwell_juttner(self) -> None:
        from scipy.special import kn

        theta = 0.1
        gen = np.random.default_rng(self.RNG_SEED)
        ux, uy, uz = ParticleInjector.sample_drifting_maxwell_juttner(
            self.N_SAMPLES, theta, (0.0, 0.0, 0.0), gen
        )
        mean_gamma, mean_vx, mean_vy, mean_vz = self._moments(ux, uy, uz)

        expected_gamma = kn(1, 1 / theta) / kn(2, 1 / theta) + 3 * theta
        np.testing.assert_allclose(mean_gamma, expected_gamma, atol=0.01)
        # isotropic rest frame: no bulk motion in any direction
        assert abs(mean_vx) < 0.005
        assert abs(mean_vy) < 0.005
        assert abs(mean_vz) < 0.005

    def test_relativistic_drift_energy_and_bulk_velocity(self) -> None:
        from scipy.special import kn

        # gamma_d = sqrt(10), beta_d = 3/sqrt(10): a hot, fast drift that a
        # naive momentum addition would badly distort
        theta = 0.5
        drift = (3.0, 0.0, 0.0)
        gamma_d = np.sqrt(1.0 + 3.0**2)
        beta_d = 3.0 / gamma_d

        gen = np.random.default_rng(self.RNG_SEED)
        ux, uy, uz = ParticleInjector.sample_drifting_maxwell_juttner(
            self.N_SAMPLES, theta, drift, gen
        )
        mean_gamma, mean_vx, mean_vy, mean_vz = self._moments(ux, uy, uz)

        # Rest-frame moments of the MJ distribution (all from Bessel functions):
        #   <eps*> = 3*theta + K1(x)/K2(x),   <1/eps*> = N(x)/D(x) with
        #   N(x) = K1(x)/x = int sqrt(e^2-1) e^{-x e} de,
        #   D(x) = -N'(x) = (K0(x)+K2(x))/(2x) + K1(x)/x^2
        x = 1.0 / theta
        mean_eps_star = 3 * theta + kn(1, x) / kn(2, x)
        n_int = kn(1, x) / x
        d_int = (kn(0, x) + kn(2, x)) / (2 * x) + kn(1, x) / x**2
        h = mean_eps_star - n_int / d_int  # <p*^2/eps*> by isotropy

        # Mean lab energy of the Juttner-Synge distribution:
        #   <eps> = gamma_d (<eps*> + beta_d^2 h / 3).
        # A naive boost (no acceptance correction) yields gamma_d*<eps*> here
        # (~6.49) instead of the correct ~7.91, so this discriminates sharply.
        expected_gamma = gamma_d * (mean_eps_star + beta_d**2 * h / 3)
        np.testing.assert_allclose(mean_gamma, expected_gamma, atol=0.15)

        # bulk velocity is exactly the drift velocity (Juttner-Synge property)
        np.testing.assert_allclose(mean_vx, beta_d, atol=0.01)
        assert abs(mean_vz) < 0.01
        # transverse drift zero: no transverse bulk motion
        assert abs(mean_vy) < 0.01

    def test_extreme_drift_loop_terminates(self) -> None:
        theta = 0.05
        drift = (10.0, 0.0, 0.0)  # gamma_d ~ 10: lowest acceptance regime
        gamma_d = np.sqrt(1.0 + 100.0)
        beta_d = 10.0 / gamma_d

        gen = np.random.default_rng(self.RNG_SEED)
        n = 20_000
        ux, uy, uz = ParticleInjector.sample_drifting_maxwell_juttner(
            n, theta, drift, gen
        )
        assert ux.shape == (n,)
        _, mean_vx, _, _ = self._moments(ux, uy, uz)
        np.testing.assert_allclose(mean_vx, beta_d, atol=0.01)

    def test_transverse_drift(self) -> None:
        theta = 0.2
        drift = (1.0, -2.0, 0.5)
        gamma_d = np.sqrt(1.0 + 1.0 + 4.0 + 0.25)

        gen = np.random.default_rng(self.RNG_SEED)
        ux, uy, uz = ParticleInjector.sample_drifting_maxwell_juttner(
            self.N_SAMPLES // 2, theta, drift, gen
        )
        _, mean_vx, mean_vy, mean_vz = self._moments(ux, uy, uz)
        np.testing.assert_allclose(mean_vx, 1.0 / gamma_d, atol=0.01)
        np.testing.assert_allclose(mean_vy, -2.0 / gamma_d, atol=0.01)
        np.testing.assert_allclose(mean_vz, 0.5 / gamma_d, atol=0.01)


@pytest.mark.unit
class TestThermalBeamInjection2D:
    """Thermal beam with drift through the callback itself."""

    def setup_method(self) -> None:
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = _setup_sim_2d()

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_thermal_beam_from_xmin(self) -> None:
        n0 = 0.01 * nc
        ppc = 8
        kT_eV = 1.0e4  # 10 keV ~ theta = 0.02 for electrons

        injector = ParticleInjector(
            self.beam,
            density=n0,
            side="xmin",
            ux=1.0,
            temperature=kT_eV,
            ppc=ppc,
        )
        injector(self.sim)

        data = _gather_species(
            self.sim, self.beam, ["x", "y", "ux", "uy", "uz", "inv_gamma", "w"]
        )
        n_injected = data["x"].size
        assert n_injected > 0

        # a thermal stream enters (up to v < c), never further than one cell
        edge = self.sim.patches.xmin_global
        assert data["x"].min() >= edge - 1e-9 * self.dx
        assert data["x"].max() <= edge + c * self.sim.dt

        # weights are unaffected by the momentum sampling
        np.testing.assert_allclose(data["w"], n0 * self.dx * self.dy / ppc, rtol=1e-12)

        # momenta have thermal spread around the drift
        assert data["ux"].std() > 1e-6
        np.testing.assert_allclose(data["ux"].mean(), 1.0, rtol=0.15)
        # the pre-drift filter is symmetric in the transverse directions
        sigma_uy = data["uy"].std()
        tol = 5 * sigma_uy / np.sqrt(n_injected)
        np.testing.assert_allclose(data["uy"].mean(), 0.0, atol=max(tol, 1e-6))

        # inv_gamma is consistent with the stored momenta
        expected = 1.0 / np.sqrt(1.0 + data["ux"]**2 + data["uy"]**2 + data["uz"]**2)
        np.testing.assert_allclose(data["inv_gamma"], expected, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "side, momentum, trans_gates",
    [
        # (side, drift, gates selecting half the domain along each transverse axis)
        ("xmin", {"ux": 1.0}, (None, 0.5)),  # gate on z only, y open
        ("zmin", {"uz": 1.0}, (0.5, 0.5)),   # gates on x and y
    ],
)
class TestInjection3D:
    """3D faces: two transverse profile coordinates f(y, z[, t])."""

    def _setup(self) -> None:
        from lambdapic import Simulation3D

        self.dx = l0 / 20
        self.dy = l0 / 20
        self.dz = l0 / 20

        self.sim = Simulation3D(
            nx=32,
            ny=32,
            nz=32,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            npatch_x=2,
            npatch_y=2,
            npatch_z=2,
            boundary_conditions={
                "xmin": "periodic",
                "xmax": "periodic",
                "ymin": "periodic",
                "ymax": "periodic",
                "zmin": "periodic",
                "zmax": "periodic",
            },
            random_seed=42,
        )

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_3d_face_injection(self, side: str, momentum: dict, trans_gates) -> None:
        self._setup()

        axis = side[0]
        at_max = side[1:] == "max"
        u0 = abs(next(iter(momentum.values())))
        gamma0 = np.sqrt(1.0 + u0**2)
        v = u0 / gamma0 * c
        ppc = 4
        n0 = 0.1 * nc

        # transverse coordinate order follows the face:
        #   xmin -> f(y, z), zmin -> f(x, y)
        trans_names = [a for a in "xyz" if a != axis]
        L = [getattr(self.sim, f"L{a}") for a in trans_names]
        gate_lo = dict(zip(trans_names, trans_gates))
        if len(trans_names) == 2:
            a1, a2 = trans_names
            o1, o2 = gate_lo[a1], gate_lo[a2]
            if o1 is None:
                def density(c1, c2):
                    return n0 if c2 >= o2 * L[1] else 0.0
            else:
                def density(c1, c2):
                    return (
                        n0
                        if (c1 >= o1 * L[0] and c2 >= o2 * L[1])
                        else 0.0
                    )
        else:
            raise AssertionError("3D faces have two transverse coordinates")

        injector = ParticleInjector(
            self.beam, density=density, side=side, ppc=ppc, **momentum
        )
        injector(self.sim)

        data = _gather_species(
            self.sim, self.beam, ["x", "y", "z", "w", "ux", "uy", "uz", "inv_gamma"]
        )
        n_injected = data["x"].size
        assert n_injected > 0, side

        # counts follow the same survival law over the gated face cells
        p_surv = v * self.sim.dt / getattr(self, f"d{axis}")
        n_cells = 1
        for name, g in zip(trans_names, trans_gates):
            n_axis = getattr(self.sim, f"n{name}")
            n_cells *= n_axis // 2 if g is not None else n_axis
        n_created = n_cells * ppc
        expected = n_created * p_surv
        sigma = np.sqrt(n_created * p_surv * (1.0 - p_surv))
        assert abs(n_injected - expected) < 6 * sigma, (
            f"{side}: injected {n_injected}, expected {expected:.1f} +/- {sigma:.1f}"
        )

        dV = self.dx * self.dy * self.dz
        np.testing.assert_allclose(data["w"], n0 * dV / ppc, rtol=1e-12)

        # layer at the face, inside the domain
        bound = getattr(
            self.sim.patches, f"{axis}min_global" if not at_max else f"{axis}max_global"
        )
        normal = data[axis]
        assert normal.min() >= bound - 1e-9 * self.dx
        assert normal.max() <= bound + v * self.sim.dt + 1e-9 * self.dx

        # cold momenta exact
        comp = next(iter(momentum))
        np.testing.assert_allclose(data[comp], momentum[comp], rtol=1e-12)
        np.testing.assert_allclose(data["inv_gamma"], 1.0 / gamma0, rtol=1e-12)

        # transverse gates hold on both transverse coordinates
        for name, g in gate_lo.items():
            if g is None:
                continue
            half = g * getattr(self.sim, f"L{name}")
            d_trans = getattr(self, f"d{name}")
            assert (data[name] >= half - d_trans / 2).all(), (
                f"{side}: gate leaked on {name}"
            )


@pytest.mark.unit
class TestInjectionWithPML2D:
    """Injection from a PML face: the slab must sit inside the interior edge.

    Particles entering the PML layer are absorbed by the particle sync, so
    the injector has to place its slab at the first interior cell (the
    ``patch.xmin/xmax`` property already accounts for the PML thickness).
    """

    def setup_method(self) -> None:
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = Simulation(
            nx=64,
            ny=64,
            dx=self.dx,
            dy=self.dy,
            npatch_x=4,
            npatch_y=4,
            boundary_conditions={
                "xmin": "pml",
                "xmax": "pml",
                "ymin": "periodic",
                "ymax": "periodic",
            },
            random_seed=42,
        )

        self.beam = Electron(name="beam", density=None)
        self.sim.add_species([self.beam])
        self.sim.initialize()

    def test_beam_survives_particle_sync(self) -> None:
        n0 = 0.1 * nc
        u0 = 1.0
        gamma0 = np.sqrt(1.0 + u0**2)
        v = u0 / gamma0 * c
        ppc = 8

        injector = ParticleInjector(
            self.beam, density=n0, side="xmin", ux=u0, ppc=ppc
        )
        injector(self.sim)

        # particles sit past the PML layer, at the interior face
        pml_thickness = self.sim.cpml_thickness
        interior_face = pml_thickness * self.dx
        data = _gather_species(self.sim, self.beam, ["x", "w", "ux", "inv_gamma"])
        n_injected = data["x"].size
        assert n_injected > 0
        assert data["x"].min() >= interior_face - self.dx / 2 - 1e-9 * self.dx
        assert data["x"].max() <= interior_face - self.dx / 2 + v * self.sim.dt + 1e-9 * self.dx
        np.testing.assert_allclose(data["w"], n0 * self.dx * self.dy / ppc, rtol=1e-12)
        np.testing.assert_allclose(data["ux"], u0, rtol=1e-12)

        # the very same particles are still alive after a particle sync:
        # nothing may fall inside the absorbing PML layer
        self.sim.patches.sync_particles()

        alive = 0
        for p in self.sim.patches:
            part = p.particles[self.beam.ispec]
            alive += int((~part.is_dead).sum())

        assert alive == n_injected, "injected particles were absorbed by the sync"
