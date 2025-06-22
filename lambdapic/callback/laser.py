import numpy as np

from lambdapic.core.boundary.cpml import PMLXmin
from lambdapic.core.utils.logger import logger
from ..core.fields import Fields
from ..core.patch.patch import Patch
from numba import njit, prange, typed
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
from numpy.typing import NDArray

from ..simulation import Simulation

@njit(parallel=True, cache=True)
def _update_laser_bfields_2d(
    laserpos,
    ex, ey, ez,
    bx, by, bz,
    jx, jy, jz, 
    dx, dy, nx, ny, dt,
    ey_source: NDArray[np.float64], ez_source: NDArray[np.float64], 
):
    for iy in prange(ny):
        bx[laserpos-1, iy] = bx[0, iy]
    for iy in prange(ny):
        bz[laserpos-1, iy] = 1 / ((c*dt / dx + 1)*c) * (
            + 4 * ey_source[iy]
            + 2 * (ey[0, iy] + c * 0.5*(bz[0, iy] + bz[-1, iy]))
            - 2 * ey[laserpos, iy]
            + dt/epsilon_0 * jy[laserpos, iy]
            + (c*dt / dx - 1)*c * bz[laserpos, iy]
        )
        by[laserpos-1, iy] = 1 / ((c*dt / dx + 1)*c) * (
            - 4 * ez_source[iy]
            - 2 * (ez[0, iy] - c * 0.5*(by[0, iy] + by[-1, iy]))
            + 2 * ez[laserpos, iy]
            - (dt*c**2) * (bx[laserpos, iy] - bx[laserpos, iy-1])/dy
            - dt/epsilon_0 * jz[laserpos, iy]
            + (c*dt / dx - 1)*c * by[laserpos, iy]
        )

@njit(parallel=True, cache=True)
def _update_laser_bfields_3d(
    laserpos,
    ex, ey, ez,
    bx, by, bz,
    jx, jy, jz, 
    dx, dy, dz, nx, ny, nz, dt,
    ey_source: NDArray[np.float64], ez_source: NDArray[np.float64], 
):
    for iy in prange(ny):
        bx[laserpos-1, iy, :] = bx[0, iy, :]
    for iy in prange(ny):
        bz[laserpos-1, iy, :] = 1 / ((c*dt / dx + 1)*c) * (
            + 4 * ey_source[iy, :]
            + 2 * (ey[0, iy, :] + c * 0.5*(bz[0, iy, :] + bz[-1, iy, :]))
            - 2 * ey[laserpos, iy, :]
            + dt/epsilon_0 * jy[laserpos, iy, :]
            + (c*dt / dx - 1)*c * bz[laserpos, iy, :]
        )
        by[laserpos-1, iy, :] = 1 / ((c*dt / dx + 1)*c) * (
            - 4 * ez_source[iy, :]
            - 2 * (ez[0, iy, :] - c * 0.5*(by[0, iy, :] + by[-1, iy, :]))
            + 2 * ez[laserpos, iy, :]
            - (dt*c**2) * (bx[laserpos, iy, :] - bx[laserpos, iy, :]-1)/dy
            - dt/epsilon_0 * jz[laserpos, iy, :]
            + (c*dt / dx - 1)*c * by[laserpos, iy, :]
        )

        
class Laser:
    staget = "_laser"
    disabled = False
    def _get_r(self, sim: Simulation, patch: Patch) -> NDArray[np.float64]:
        """Calculate the radial distance from the center of the laser beam."""
        raise NotImplementedError
    
    def _update_bfields(self, laserpos: int, f: Fields, ey_source: NDArray[np.float64], ez_source: NDArray[np.float64], dt: float):
        raise NotImplementedError
    
class Laser2D(Laser):
    def _get_r(self, sim: Simulation, patch: Patch) -> NDArray[np.float64]:
        f = patch.fields
        r = abs(f.yaxis[0, :] - sim.dy/2 - sim.Ly/2)
        return r

    def _update_bfields(self, laserpos: int, f: Fields, ey_source: NDArray[np.float64], ez_source: NDArray[np.float64], dt: float):
        _update_laser_bfields_2d(
            laserpos,
            f.ex, f.ey, f.ez,
            f.bx, f.by, f.bz,
            f.jx, f.jy, f.jz,
            f.dx, f.dy, f.nx, f.ny, dt,
            ey_source,
            ez_source
        )

class Laser3D(Laser):
    def _get_r(self, sim: Simulation, patch: Patch) -> NDArray[np.float64]:
        f = patch.fields
        r = ((f.yaxis[0, :, :] - sim.dy/2 - sim.Ly/2)**2 + (f.zaxis[0, :, :] - sim.dz/2 - sim.Lz/2)**2)**0.5
        return r
    
    def _update_bfields(self, laserpos: int, f: Fields, ey_source: NDArray[np.float64], ez_source: NDArray[np.float64], dt: float):
        _update_laser_bfields_3d(
            laserpos,
            f.ex, f.ey, f.ez,
            f.bx, f.by, f.bz,
            f.jx, f.jy, f.jz,
            f.dx, f.dy, f.dz, f.nx, f.ny, f.nz, dt,
            ey_source,
            ez_source
        )
    
class SimpleLaser(Laser):
    """
    A simple laser pulse implementation with basic spatial and temporal profiles.
    This class provides a straightforward way to inject a laser pulse into the simulation
    from the left boundary with a Gaussian transverse profile and a smooth temporal envelope.
    
    Attributes:
        a0 (float): Normalized vector potential amplitude
        l0 (float): Laser wavelength
        omega0 (float): Laser angular frequency (2πc/λ)
        w0 (float): Laser waist size (determines transverse width of the beam)
        ctau (float): Pulse duration (c*tau)
        E0 (float): Peak electric field amplitude, derived from a0
        pol_angle (float): Polarization angle in radians, 0 for z-polarization, π/2 for y-polarization
    
    Note:
        This is a simplified laser implementation suitable for basic simulations.
        For more accurate physics including proper beam evolution, wavefront curvature,
        and Gouy phase, use the GaussianLaser class instead.
    """
    def __init__(self, a0: float, w0: float, ctau: float, tstop: float|None=None, pol_angle: float = 0.0, l0: float=0.8e-6, side="xmin"):
        """
        Initialize the SimpleLaser with given parameters.
        
        Args:
            sim: Simulation object that this laser will be injected into
            a0: Normalized vector potential amplitude
            w0: Laser waist size
            ctau: Pulse duration (c*tau)
            pol_angle: Polarization angle in radians (default: 0.0 for z-polarization)
            l0: Laser wavelength (default: 800nm)
        
        Raises:
            ValueError: If parameters are invalid (negative or zero values)
        """
        # Parameter validation
        if any(p <= 0 for p in [a0, l0, w0, ctau]):
            raise ValueError("All parameters (a0, l0, w0, ctau) must be positive")
        
        if side not in ["xmin"]:
            raise ValueError("Invalid side: only 'xmin' is supported.")
            
        self.a0 = a0
        self.l0 = l0
        self.omega0 = 2 * pi * c / l0
        self.w0 = w0
        self.ctau = ctau
        self.tstop = 2*self.ctau or tstop
        self.E0 = a0 * m_e * c * self.omega0 / e
        self.pol_angle = pol_angle
        self.side = side


    def __call__(self, sim: Simulation):
        if self.disabled:
            return
        
        time = sim.itime * sim.dt
        # Stop injecting after twice the pulse duration for smooth falloff
        if c*time >= self.tstop:
            self.disabled = True
            return

        # Calculate temporal profile (sin² envelope for smooth turn-on/off)
        tprof = np.sin(c*time/(2*self.ctau)*pi)**2 * (c*time < 2*self.ctau)
        
        if self.side == "xmin":
            ipatch_x = 0
            laserpos = sim.cpml_thickness + 2
            patches = list(filter(lambda p: p.ipatch_x == ipatch_x, sim.patches))
            n_pmlxmin = sum(isinstance(pml, PMLXmin) for p in patches for pml in p.pml_boundary)
            if n_pmlxmin < len(patches):
                logger.info("Disabling laser for lacking of PML. Maybe due to MovingWindow starts.")
                self.disabled = True
                return
        else:
            raise ValueError("Invalid side: only 'xmin' is supported.")

        # Inject the laser from the left boundary
        for p in sim.patches:
            # Only inject from the leftmost patch
            if p.ipatch_x == ipatch_x:
                # r is 2D or 3D depending on the simulation dimension
                r = self._get_r(sim, p)
                # Calculate base field amplitude with:
                # - Gaussian transverse profile: exp(-r²/w0²)
                # - Temporal oscillation: sin(ω₀t)
                # - Smooth temporal envelope: tprof
                efield = self.E0 * np.exp(-r**2/self.w0**2) * np.sin(self.omega0 * time) * tprof
                f = p.fields
                self._update_bfields(laserpos, f, ey_source=efield * np.cos(self.pol_angle), ez_source=efield * np.sin(self.pol_angle), dt=sim.dt)

class SimpleLaser2D(Laser2D, SimpleLaser):
    ...

class SimpleLaser3D(Laser3D, SimpleLaser):
    ...


class GaussianLaser(Laser):
    """
    Implementation of a proper Gaussian laser beam with full physics including:
    
    - Gaussian temporal and spatial profiles
    - Proper beam waist evolution (:math:`w(z) = w_0\sqrt{1 + (z/z_R)^2}`)
    - Gouy phase (:math:`tan^{-1}(z/z_R)`)
    - Wavefront curvature (:math:`R(z) = z(1 + (z_R/z)^2)`)
    - Correct phase evolution including propagation and curvature terms
    
    Attributes:
        a0 (float): Normalized vector potential amplitude
        l0 (float): Laser wavelength
        omega0 (float): Laser angular frequency (2πc/λ)
        w0 (float): Laser waist size at focus
        ctau (float): Pulse duration (c*tau)
        x0 (float): Pulse center position (default: 3*ctau)
        tstop (float): Time to stop injection (default: 6*ctau)
        E0 (float): Peak electric field amplitude, derived from a0
        pol_angle (float): Polarization angle in radians, 0 for z-polarization, π/2 for y-polarization
        focus_position (float): Position of laser focus relative to boundary
        zR (float): Rayleigh length (πw₀²/λ)
        side (str): Injection boundary ('xmin' or 'xmax')
    
    Note:
        This implementation provides more accurate physics than SimpleLaser,
        including proper beam evolution and phase effects. Use this for
        realistic simulations where these effects matter.
    """
    def __init__(self, a0: float, l0: float, w0: float, ctau: float, 
                 x0: float=None, tstop: float=None, pol_angle: float = 0.0, focus_position: float = 0.0, side: str = "xmin"):
        """
        Initialize the GaussianLaser with given parameters.
        
        Args:
            a0: Normalized vector potential amplitude
            l0: Laser wavelength
            w0: Waist size at focus
            ctau: Pulse duration (c*tau)
            x0: Pulse center position (default: 3*ctau)
            tstop: Time to stop injection (default: 6*ctau)
            pol_angle: Polarization angle in radians (default: 0.0 for z-polarization)
            focus_position: Position of laser focus relative to boundary (default: 0.0)
            side: Injection boundary ('xmin' or 'xmax') (default: 'xmin')
        
        Raises:
            ValueError: If parameters are invalid (negative or zero values)
        """
        # Parameter validation
        if any(p <= 0 for p in [a0, l0, w0, ctau]):
            raise ValueError("All parameters (a0, l0, w0, ctau) must be positive")
        
        if side not in ["xmin"]:
            raise ValueError("Invalid side: only 'xmin' is supported.")
            
        self.a0 = a0
        self.l0 = l0
        self.omega0 = 2 * pi * c / l0
        self.k0 = self.omega0 / c
        self.w0 = w0
        self.ctau = ctau
        self.x0 = 3*ctau if x0 is None else x0
        self.tstop = 6*ctau if tstop is None else tstop
        self.E0 = a0 * m_e * c * self.omega0 / e
        self.pol_angle = pol_angle
        self.focus_position = focus_position
        self.side = side
        
        # Derived parameters
        self.zR = pi * w0**2 / l0  # Rayleigh length
        
    def _gaussian_beam_params(self, z):
        """Calculate Gaussian beam parameters at position z"""
        # Normalized distance from focus
        z = z - self.focus_position
        
        # Beam radius
        w = self.w0 * np.sqrt(1 + (z/self.zR)**2)
        
        # Radius of curvature
        R = z * (1 + (self.zR/z)**2) if abs(z) > 1e-10 else np.inf
        
        # Gouy phase
        psi = np.arctan(z/self.zR)
        
        return w, R, psi
    
    def __call__(self, sim: Simulation):
        """
        Inject the Gaussian laser pulse into the simulation.
        """
        time = sim.itime * sim.dt
        if c*time >= self.tstop:
            return
            
        # Temporal envelope (Gaussian)
        tprof = np.exp(-(c*time - self.x0)**2 / self.ctau**2)
        
        # Calculate boundary parameters
        x_rel = sim.cpml_thickness * sim.dx
        boundary_w, boundary_R, boundary_psi = self._gaussian_beam_params(x_rel)
        
        if self.side == "xmin":
            ipatch_x = 0
            laserpos = sim.cpml_thickness + 2
            patches = list(filter(lambda p: p.ipatch_x == ipatch_x, sim.patches))
            n_pmlxmin = sum(isinstance(pml, PMLXmin) for p in patches for pml in p.pml_boundary)
            if n_pmlxmin < len(patches):
                logger.info("Disabling laser for lacking of PML. Maybe due to MovingWindow starts.")
                self.disabled = True
        else:
            raise ValueError("Invalid side: only 'xmin' is supported.")

        for p in sim.patches:
            if p.ipatch_x == ipatch_x:
                # r is 2D or 3D depending on the simulation dimension
                r = self._get_r(sim, p)
                
                # Calculate amplitude and phase
                amp = self.E0 * (self.w0/boundary_w) * np.exp(-r**2/boundary_w**2)
                phase_curv = self.k0 * r**2/(2*boundary_R)
                phase = (self.omega0 * time -          # Oscillation
                        self.k0 * x_rel -             # Propagation
                        phase_curv -                  # Curvature
                        boundary_psi)                 # Gouy phase
                
                # Set fields based on polarization
                efield = amp * np.sin(phase) * tprof
                f = p.fields
                self._update_bfields(laserpos, f, ey_source=efield * np.cos(self.pol_angle), ez_source=efield * np.sin(self.pol_angle), dt=sim.dt)

class GaussianLaser2D(Laser2D, GaussianLaser):
    ...

class GaussianLaser3D(Laser3D, GaussianLaser):
    ...
