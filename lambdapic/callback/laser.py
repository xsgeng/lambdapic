import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from ..simulation import Simulation


class SimpleLaser:
    """
    A simple laser pulse implementation with basic spatial and temporal profiles.
    This class provides a straightforward way to inject a laser pulse into the simulation
    from the left boundary with a Gaussian transverse profile and a smooth temporal envelope.
    
    Attributes:
        stage (str): Defines when the callback should be executed ('start')
        sim (Simulation): Reference to the simulation object
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
    stage = "start"
    def __init__(self, sim: Simulation, a0: float, l0: float, w0: float, ctau: float, pol_angle: float = 0.0):
        """
        Initialize the SimpleLaser with given parameters.
        
        Args:
            sim: Simulation object that this laser will be injected into
            a0: Normalized vector potential amplitude
            l0: Laser wavelength
            w0: Laser waist size
            ctau: Pulse duration (c*tau)
            pol_angle: Polarization angle in radians (default: 0.0 for z-polarization)
        
        Raises:
            ValueError: If parameters are invalid (negative or zero values)
        """
        # Parameter validation
        if any(p <= 0 for p in [a0, l0, w0, ctau]):
            raise ValueError("All parameters (a0, l0, w0, ctau) must be positive")
            
        self.sim = sim
        self.a0 = a0
        self.l0 = l0
        self.omega0 = 2 * pi * c / l0
        self.w0 = w0
        self.ctau = ctau
        self.E0 = a0 * m_e * c * self.omega0 / e
        self.pol_angle = pol_angle
    

    def __call__(self):
        """
        Inject the laser pulse into the simulation.
        This method is called at each timestep and updates the electromagnetic fields
        at the left boundary to create the laser pulse.
        
        The laser has:
        - Smooth temporal envelope: sin²(πct/2ctau) for t < 2ctau
        - Gaussian transverse profile: exp(-r²/w0²)
        - Sinusoidal oscillation at the laser frequency
        - Arbitrary polarization in the y-z plane determined by pol_angle
        
        The sin² envelope ensures smooth turn-on and turn-off of the pulse,
        reducing numerical artifacts compared to a sharp cutoff.
        """
        time = self.sim.itime * self.sim.dt
        # Stop injecting after twice the pulse duration for smooth falloff
        if c*time >= 2*self.ctau:
            return

        # Calculate temporal profile (sin² envelope for smooth turn-on/off)
        tprof = np.sin(c*time/(2*self.ctau)*pi)**2 * (c*time < 2*self.ctau)
        
        # Inject the laser from the left boundary
        for p in self.sim.patches:
            # Only inject from the leftmost patch
            if p.ipatch_x > 0:
                continue
            f = p.fields
            # Calculate radial distance from the center of the simulation box
            r = p.yaxis - self.sim.dy/2 - self.sim.Ly/2
            
            # Calculate base field amplitude with:
            # - Gaussian transverse profile: exp(-r²/w0²)
            # - Temporal oscillation: sin(ω₀t)
            # - Smooth temporal envelope: tprof
            field = self.E0 / c * np.exp(-r**2/self.w0**2) * np.sin(self.omega0 * time) * tprof
            
            # Update By and Bz fields at the left boundary (CPML layer) based on polarization angle
            f.by[self.sim.cpml_thickness, :] += field * np.sin(self.pol_angle)
            f.bz[self.sim.cpml_thickness, :] += field * np.cos(self.pol_angle)


class GaussianLaser:
    """
    Implementation of a proper Gaussian laser beam with:
    - Gaussian temporal and spatial profiles
    - Proper beam waist evolution
    - Gouy phase
    - Wavefront curvature
    """
    stage = "start"
    
    def __init__(self, sim: Simulation, a0: float, l0: float, w0: float, ctau: float, 
                 polarization: str = 'z', focus_position: float = 0.0):
        """
        Parameters:
        -----------
        sim : Simulation
            Simulation object
        a0 : float
            Normalized vector potential amplitude
        l0 : float
            Laser wavelength
        w0 : float
            Waist size at focus
        ctau : float
            Pulse duration (c*tau)
        polarization : str
            Laser polarization ('z' or 'y')
        focus_position : float
            Position of the laser focus relative to the left boundary
        """
        self.sim = sim
        self.a0 = a0
        self.l0 = l0
        self.omega0 = 2 * pi * c / l0
        self.k0 = self.omega0 / c
        self.w0 = w0
        self.ctau = ctau
        self.E0 = a0 * m_e * c * self.omega0 / e
        self.polarization = polarization
        self.focus_position = focus_position
        
        # Derived parameters
        self.zR = pi * w0**2 / l0  # Rayleigh length
        
        # Pre-calculate fixed parameters at the boundary
        x_rel = self.sim.cpml_thickness * self.sim.dx
        self.boundary_w, self.boundary_R, self.boundary_psi = self.gaussian_beam_params(x_rel)
        self.boundary_x_rel = x_rel
        
        # Pre-calculate transverse coordinates and spatial amplitude profile
        for p in self.sim.patches:
            if p.ipatch_x == 0:  # Only for leftmost patch
                r = p.yaxis - self.sim.dy/2 - self.sim.Ly/2
                self.boundary_amp = self.E0 * (self.w0/self.boundary_w) * np.exp(-r**2/self.boundary_w**2)
                self.boundary_phase_curv = self.k0 * r**2/(2*self.boundary_R)
                break
        
    def gaussian_beam_params(self, z):
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
    
    def __call__(self):
        time = self.sim.itime * self.sim.dt
        if c*time >= 2*self.ctau:  # Allow for 2*ctau to ensure smooth falloff
            return
            
        # Temporal envelope (Gaussian)
        tenv = np.exp(-(c*time - self.ctau)**2 / (2*(self.ctau/4)**2))
        
        for p in self.sim.patches:
            if p.ipatch_x > 0:
                continue
                
            f = p.fields
            
            # Phase terms (only time-dependent parts need to be calculated here)
            phase = (self.omega0 * time -                # Oscillation
                    self.k0 * self.boundary_x_rel -      # Propagation
                    self.boundary_phase_curv -           # Curvature (pre-calculated)
                    self.boundary_psi)                   # Gouy phase (pre-calculated)
            
            # Set fields based on polarization using pre-calculated amplitude
            if self.polarization == 'z':
                f.bz[self.sim.cpml_thickness, :] += self.boundary_amp/c * np.sin(phase) * tenv
            else:  # y-polarization
                f.by[self.sim.cpml_thickness, :] += self.boundary_amp/c * np.sin(phase) * tenv
