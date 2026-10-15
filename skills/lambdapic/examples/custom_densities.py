#!/usr/bin/env python3
"""
lambdapic Custom Density Profile Examples

This file demonstrates two approaches for defining custom density profiles
in lambdapic simulations:

1. Simple closure-based density (2D circular target)
   - Uses Python closures to create parameterized density functions
   - Suitable for simple geometries with moderate performance requirements

2. Numba-accelerated density (1D layered target)
   - Uses @njit decorators for high-performance density evaluation
   - Required for complex simulations with many particles
   - Supports multi-layer targets with different densities

Physics Context:
--------------
The second example models a laser-solid interaction target with:
- CNF (Carbon Nano Foam): 0.5 nc (underdense, laser propagation region)
- DLC (Diamond-Like Carbon): 150 nc (overdense, main target)
- Rear H (Hydrogen): 10 nc (optional proton acceleration layer)

Where nc is the critical density for the laser wavelength.

Key Points:
-----------
- Density functions receive (x, y) for 2D or (x, y, z) for 3D simulations
- Return electron density in m^-3
- Numba-jitted functions are required for performance with high particle counts
- Use closures or global variables to parameterize density profiles
"""

from numba import njit
import numpy as np
from scipy.constants import c, pi, epsilon_0, m_e, e

# ============================================================================
# SIMULATION PARAMETERS (shared across examples)
# ============================================================================

um = 1e-6                      # Micrometer to meter conversion
l0 = 0.8 * um                  # Laser wavelength: 800 nm (Ti:Sapphire)
t0 = l0 / c                    # Normalized time unit
omega0 = 2 * pi * c / l0       # Laser angular frequency
nc = epsilon_0 * m_e * omega0**2 / e**2  # Critical density

# Grid parameters
nx = 1024
ny = 1024
dx = l0 / 50
dy = l0 / 50

Lx = nx * dx  # Domain size in x
Ly = ny * dy  # Domain size in y


# ============================================================================
# EXAMPLE 1: SIMPLE CLOSURE-BASED DENSITY (2D CIRCULAR TARGET)
# ============================================================================

def density(n0):
    """
    Create a 2D circular density profile using closures.
    
    Returns a density function that is uniform (n0) within a 5 μm radius
    circle centered in the simulation domain, and zero outside.
    
    Args:
        n0: Peak electron density in m^-3
    
    Returns:
        Function f(x, y) -> density at position (x, y)
    
    Usage:
        ne = 100 * nc
        ele = Electron(density=density(ne), ppc=10)
    """
    def _density(x, y):
        ne = 0.0
        # Calculate squared distance from domain center
        r2 = (x - Lx/2)**2 + (y - Ly/2)**2
        # Return n0 inside 5 μm radius, zero outside
        if r2 <= (5*um)**2:
            return n0
        return ne
    return _density


# ============================================================================
# EXAMPLE 2: NUMBA-ACCELERATED DENSITY (1D LAYERED TARGET)
# ============================================================================

# Helper function for range checking - MUST be njit'd for performance
@njit
def in_range(x, xs, xe):
    """Check if x is within the range [xs, xe]."""
    return (x >= xs) and (x <= xe)


# Layer boundary positions (in meters) - customize these for your target geometry
# These would typically be set based on your specific target design
x_cnf_start = 10 * um   # CNF layer start
x_cnf_end = 15 * um     # CNF layer end
x_dlc_front = 15 * um   # DLC layer start
x_dlc_back = 16 * um    # DLC layer end (1 μm thick)
x_H_front = 16 * um     # Hydrogen layer start
x_H_back = 16.1 * um    # Hydrogen layer end (0.1 μm thick)

# Density values for each layer
rear_H_density = 10.0 * nc  # Hydrogen layer density


@njit(cache=False)
def ne_profile(x, y, z):
    """
    Numba-accelerated 3D electron density profile for layered targets.
    
    Models a laser-solid interaction target with three layers:
    1. CNF (Carbon Nano Foam): 0.5 nc - underdense, allows laser propagation
    2. DLC (Diamond-Like Carbon): 150 nc - overdense main target
    3. Rear H (Hydrogen): 10 nc - proton acceleration layer (optional)
    
    The profile varies only in x (1D), uniform in y and z.
    
    Args:
        x, y, z: Position coordinates in meters
    
    Returns:
        Electron density in m^-3
    
    Note:
        - This function must be @njit decorated for performance
        - cache=False allows dynamic modification of global parameters
        - Use closures if you need runtime parameter changes
    
    Usage:
        ele = Electron(density=ne_profile, ppc=10)
        # or with functools.partial for parameterization:
        # from functools import partial
        # ele = Electron(density=partial(ne_profile, y=0, z=0), ppc=10)
    """
    # CNF layer: underdense plasma for laser propagation
    if in_range(x, x_cnf_start, x_cnf_end):
        return 0.5 * nc
    
    # DLC layer: overdense main target
    if in_range(x, x_dlc_front, x_dlc_back):
        return 150.0 * nc
    
    # Rear hydrogen layer: for proton acceleration
    if in_range(x, x_H_front, x_H_back):
        return rear_H_density
    
    # Vacuum everywhere else
    return 0.0

