from math import sqrt
from typing import List, overload

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import numba
from numba.experimental import jitclass
from scipy.constants import c, epsilon_0, pi

from ..particles import ParticlesBase
from ..utils.jit_spinner import jit_spinner

@jitclass
class CollisionData:
    """Data class for collision parameters."""
    px1: float
    py1: float
    pz1: float
    gamma1: float
    vx1: float
    vy1: float
    vz1: float
    m1: float
    q1: float
    w1: float

    px2: float
    py2: float
    pz2: float
    gamma2: float
    vx2: float
    vy2: float
    vz2: float
    m2: float
    q2: float
    w2: float
    
    vx_com: float
    vy_com: float
    vz_com: float
    gamma_com: float
    v_com_square: float
    
    px1_com: float
    py1_com: float
    pz1_com: float
    p1_com: float
    p_perp: float
    
    gamma1_com: float
    gamma2_com: float
    
    def __init__(
        self,
        ux1: float, uy1: float, uz1: float, inv_gamma1: float, w1: float,
        m1: float, q1: float, 
        ux2: float, uy2: float, uz2: float, inv_gamma2: float, w2: float,
        m2: float, q2: float
    ) -> None:
        self.px1, self.py1, self.pz1, self.gamma1 = ux1*m1*c, uy1*m1*c, uz1*m1*c, 1/inv_gamma1
        self.m1, self.q1, self.w1 = m1, q1, w1
        
        self.px2, self.py2, self.pz2, self.gamma2 = ux2*m2*c, uy2*m2*c, uz2*m2*c, 1/inv_gamma2
        self.m2, self.q2, self.w2 = m2, q2, w2
        
        self.vx1 = ux1 * inv_gamma1 * c
        self.vy1 = uy1 * inv_gamma1 * c
        self.vz1 = uz1 * inv_gamma1 * c
        
        self.vx2 = ux2 * inv_gamma2 * c
        self.vy2 = uy2 * inv_gamma2 * c
        self.vz2 = uz2 * inv_gamma2 * c
        
        # speed of COM
        self.vx_com = vx_com = (self.px1 + self.px2) / (self.gamma1*m1 + self.gamma2*m2)
        self.vy_com = vy_com = (self.py1 + self.py2) / (self.gamma1*m1 + self.gamma2*m2)
        self.vz_com = vz_com = (self.pz1 + self.pz2) / (self.gamma1*m1 + self.gamma2*m2)
        v_com_square = vx_com**2 + vy_com**2 + vz_com**2
        self.gamma_com = gamma_com = 1.0 / sqrt(1 - v_com_square/c**2 )
        self.v_com_square = v_com_square = vx_com**2 + vy_com**2 + vz_com**2
        
        fac = (gamma_com-1)/v_com_square if v_com_square > 0 else 0.0
        self.px1_com = self.px1 + (fac * (self.vx1*vx_com + self.vy1*vy_com + self.vz1*vz_com) - gamma_com) * m1*self.gamma1*vx_com
        self.py1_com = self.py1 + (fac * (self.vx1*vx_com + self.vy1*vy_com + self.vz1*vz_com) - gamma_com) * m1*self.gamma1*vy_com
        self.pz1_com = self.pz1 + (fac * (self.vx1*vx_com + self.vy1*vy_com + self.vz1*vz_com) - gamma_com) * m1*self.gamma1*vz_com
        self.p1_com = sqrt(self.px1_com**2 + self.py1_com**2 + self.pz1_com**2)
        
        # p2_com = -p1_com
        
        self.gamma1_com = (1 - (vx_com*self.vx1 + vy_com*self.vy1 + vz_com*self.vz1) / c**2) * gamma_com*self.gamma1
        self.gamma2_com = (1 - (vx_com*self.vx2 + vy_com*self.vy2 + vz_com*self.vz2) / c**2) * gamma_com*self.gamma2
        
        self.p_perp = sqrt(self.px1_com**2 + self.py1_com**2)


@jitclass
class ParticleData:
    """Data class for particle properties."""
    x: numba.float64[:]
    y: numba.float64[:]
    z: numba.float64[:]
    ux: numba.float64[:]
    uy: numba.float64[:]
    uz: numba.float64[:]
    inv_gamma: numba.float64[:]
    w: numba.float64[:]
    is_dead: numba.types.bool_[:]
    m: float
    q: float

    def __init__(
        self,
        x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64], 
        ux: NDArray[np.float64], uy: NDArray[np.float64], uz: NDArray[np.float64], inv_gamma: NDArray[np.float64],
        w: NDArray[np.float64], is_dead: NDArray[np.bool_], m: float, q: float
    ) -> None:
        self.x, self.y, self.z = x, y, z
        self.ux, self.uy, self.uz = ux, uy, uz
        self.inv_gamma, self.w, self.is_dead = inv_gamma, w, is_dead
        self.m, self.q = m, q

@overload
def pack_particle_data(particles: ParticlesBase, m: float, q: float) -> ParticleData: ...
@overload
def pack_particle_data(particles: List[ParticlesBase], m: float, q: float) -> List[ParticleData]: ...

def pack_particle_data(particles: List[ParticlesBase] | ParticlesBase, m: float, q: float) -> List[ParticleData] | ParticleData:
    if isinstance(particles, ParticlesBase):
        return ParticleData(
            particles.x, particles.y, particles.z, 
            particles.ux, particles.uy, particles.uz, particles.inv_gamma, particles.w, particles.is_dead, m, q
        )
    elif len(particles):
        particle_data: List[ParticleData] = []
        for part in particles:
            particle_data.append(ParticleData(
                part.x, part.y, part.z, 
                part.ux, part.uy, part.uz, part.inv_gamma, part.w, part.is_dead, m, q
            ))
        return particle_data
    else:
        raise ValueError("particles must be a ParticlesBase instance or a list of ParticlesBase instances.")
    
@njit(inline='always', cache=True)
def coulomb_scattering(data: CollisionData, 
                       cell_vol: float, dt: float, lnLambda: float, 
                       gen: np.random.Generator) -> tuple[float, float, float]:
    """
    Coulomb scattering in the center of mass frame.
    
    Parameters:
        data (CollisionData): Collision parameters.
        dx,dy,dz (float): Spatial step sizes.
        dt (float): Time step scaled by dt_corr.
        lnLambda (float): Coulomb logarithm.
        gen (np.random.Generator): Random number generator.
        
    Returns:
        p_new (tuple[float, float, float]): New momentum components in the center of mass frame.
    """
    if data.p1_com == 0:
        return data.px1_com, data.py1_com, data.pz1_com
    
    w_max = max(data.w1, data.w2)
    q1, q2 = data.q1, data.q2
    m1, m2 = data.m1, data.m2
    
    gamma1 = data.gamma1
    gamma2 = data.gamma2
    
    px1_com = data.px1_com
    py1_com = data.py1_com
    pz1_com = data.pz1_com
    p1_com = data.p1_com
    
    gamma1_com = data.gamma1_com
    gamma2_com = data.gamma2_com
    
    gamma_com = data.gamma_com
    
    # dt = dt * dt_corr
    s = w_max/cell_vol *dt * (lnLambda * (q1*q2)**2) / (4*pi*epsilon_0**2*c**4 * m1*gamma1 * m2*gamma2) \
            *(gamma_com * p1_com)/(m1*gamma1 + m2*gamma2) * (m1*gamma1_com*m2*gamma2_com/p1_com**2 * c**2 + 1)**2
    
    U = gen.uniform()
    
    if s < 4:
        alpha = 0.37*s - 0.005*s**2 - 0.0064*s**3
        sin2X2 = alpha * U / np.sqrt( (1-U) + alpha*alpha*U )
        cosX = 1. - 2.*sin2X2
        sinX = 2.*np.sqrt( sin2X2 *(1.-sin2X2) )
    else:
        cosX = 2.*U - 1.
        sinX = np.sqrt( 1. - cosX*cosX )

    phi = gen.uniform(0, 2*pi)
        
    return rotate_momentum(px1_com, py1_com, pz1_com, p1_com, data.p_perp, 
                           cosX, sinX, phi)
    


@njit(inline='always', cache=True)
def rotate_momentum(
    px: float, py: float, pz: float, p: float, p_perp: float,
    cosX: float, sinX: float, phi: float
) -> tuple[float, float, float]:
    sinXcosPhi = sinX * np.cos(phi)
    sinXsinPhi = sinX * np.sin(phi)
    
    if p_perp > 1.e-10 * p:
        px_new = (px * pz * sinXcosPhi - py * p * sinXsinPhi) / p_perp + px * cosX
        py_new = (py * pz * sinXcosPhi + px * p * sinXsinPhi) / p_perp + py * cosX
        pz_new = -p_perp * sinXcosPhi + pz * cosX
    else:
        px_new = p * sinXcosPhi
        py_new = p * sinXsinPhi
        pz_new = p * cosX
    
    return px_new, py_new, pz_new


@njit(inline='always', cache=True)
def boost_to_lab(
    px_com: float, py_com: float, pz_com: float, gamma_com_particle: float, m: float, # of the particle
    vx_com: float, vy_com: float, vz_com: float, v_com_square: float, gamma_com: float, # of the COM frame
):
    """
    Boost from the center of mass frame to the laboratory frame.
    
    Parameters:
        px_com,py_com,pz_com (float): Momentum components in the COM frame.
        gamma_com_particle (float): Lorentz factor of the particle in the COM frame.
        m (float): Mass of the particle.
        vx_com,vy_com,vz_com (float): Velocity components in the COM frame.
        v_com_square (float): Velocity square in the COM frame.
        gamma_com (float): Lorentz factor of the COM frame.
    
    Returns:
        p (tuple[float, float, float]): Momentum components in the laboratory frame.
    """
    vcom_dot_p = vx_com*px_com + vy_com*py_com + vz_com*pz_com
    fac = (gamma_com-1)/v_com_square if v_com_square > 0 else 0.0
    px = px_com + vx_com * (fac * vcom_dot_p + m*gamma_com_particle*gamma_com)
    py = py_com + vy_com * (fac * vcom_dot_p + m*gamma_com_particle*gamma_com)
    pz = pz_com + vz_com * (fac * vcom_dot_p + m*gamma_com_particle*gamma_com)
    
    return px, py, pz