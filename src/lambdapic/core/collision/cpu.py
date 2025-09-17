import math
from math import sqrt
from typing import List

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.constants import c, epsilon_0, k, pi, h

from ..utils.jit_spinner import jit_spinner
from .utils import CollisionData, ParticleData, boost_to_lab, coulomb_scattering


@njit
def self_pairing(dead, ip_start, ip_end, random_gen):
    nbuf = ip_end - ip_start
    npart = nbuf - dead[ip_start:ip_end].sum()
    if npart < 2:
        return

    idx = np.arange(nbuf) + ip_start
    random_gen.shuffle(idx)

    npairs = (npart + 1) // 2

    even = (npart % 2) == 0
    odd = not even

    ip1 = -1
    ip2 = -1
    for ipair in range(npairs):
        for ip1 in range(ip2+1, nbuf):
            if not dead[idx[ip1]]:
                break
        # even
        if even:
            for ip2 in range(ip1+1, nbuf):
                if not dead[idx[ip2]]:
                    break
        # odd
        else:
            # before last pair
            if ipair < npairs - 1:
                for ip2 in range(ip1+1, nbuf):
                    if not dead[idx[ip2]]:
                        break
            # last pair
            else:
                for ip2 in range(ip1):
                    if not dead[idx[ip2]]:
                        break

        # the first pariticle is splitted into two pairs
        w_corr = 1.0
        if odd:
            if ipair == 0:
                w_corr = 0.5
            elif ipair == npairs - 1:
                w_corr = 0.5
            
        yield ipair, idx[ip1], idx[ip2], w_corr

@njit
def pairing(
    dead1, ip_start1, ip_end1,
    dead2, ip_start2, ip_end2,
    random_gen
):
    nbuf1 = ip_end1 - ip_start1
    nbuf2 = ip_end2 - ip_start2

    npart1 = nbuf1 - dead1[ip_start1:ip_end1].sum()
    npart2 = nbuf2 - dead2[ip_start2:ip_end2].sum()

    if npart1 == 0 or npart2 == 0:
        return

    if npart1 >= npart2:
        npairs = npart1
        npairs_not_repeated = npart2
        shuffled_idx = np.arange(nbuf1) + ip_start1
    else:
        npairs = npart2
        npairs_not_repeated = npart1
        shuffled_idx = np.arange(nbuf2) + ip_start2

    random_gen.shuffle(shuffled_idx)

    # indices will be offsetted by ip_start later
    ip1 = -1
    ip2 = -1
    if npart1 >= npart2:
        for ipair in range(npairs):
            for ip1 in range(ip1+1, nbuf1):
                if not dead1[shuffled_idx[ip1]]:
                    break
            if ipair % npart2 == 0:
                ip2 = -1
            for ip2 in range((ip2+1) % nbuf2, nbuf2):
                if not dead2[ip_start2+ip2]:
                    break
            if (ipair % npairs_not_repeated) < (npairs % npairs_not_repeated):
                w_corr = 1. / ( npart1 // npart2 + 1 )
            else:
                w_corr = 1. / ( npart1 // npart2 )
            yield ipair, shuffled_idx[ip1], ip_start2 + ip2, w_corr
    else:
        for ipair in range(npairs):
            for ip2 in range(ip2+1, nbuf2):
                if not dead2[shuffled_idx[ip2]]:
                    break
            if ipair % npart1 == 0:
                ip1 = -1
            for ip1 in range((ip1+1) % nbuf1, nbuf1):
                if not dead1[ip_start1 + ip1]:
                    break
                    
            if (ipair % npairs_not_repeated) < (npairs % npairs_not_repeated):
                w_corr = 1. / ( npart2 // npart1 + 1 )
            else:
                w_corr = 1. / ( npart2 // npart1 )
                
            yield ipair, ip_start1 + ip1, shuffled_idx[ip2], w_corr

@jit_spinner
@njit(parallel=True, cache=True)
def debye_length_patches(
    part_list: List[ParticleData],
    bucket_bound_min_list: List[NDArray[np.int64]],
    bucket_bound_max_list: List[NDArray[np.int64]],
    cell_vol: float,
    debye_length_inv_sqare_list: List[NDArray[np.float64]],
    total_density_list: List[NDArray[np.float64]],
    reset: bool=False
):
    for ipatch in prange(len(part_list)):
        if reset:
            debye_length_inv_sqare_list[ipatch].fill(0)
        for icell in range(bucket_bound_min_list[ipatch].size):
            ip_start = bucket_bound_min_list[ipatch].flat[icell]
            ip_end   = bucket_bound_max_list[ipatch].flat[icell]
            inv_d2, n = debye_length_cell(part_list[ipatch], ip_start, ip_end, cell_vol)
            if n > 0:
                debye_length_inv_sqare_list[ipatch].flat[icell] += inv_d2
                total_density_list[ipatch].flat[icell] += n

@njit(cache=True)
def debye_length_patch(
    part: ParticleData,
    bucket_bound_min: NDArray[np.int64],
    bucket_bound_max: NDArray[np.int64],
    cell_vol: float,
    debye_length_inv_sqare: NDArray[np.float64],
):
    for icell in range(bucket_bound_min.size):
            ip_start = bucket_bound_min.flat[icell]
            ip_end   = bucket_bound_max.flat[icell]
            inv_d2, n = debye_length_cell(part, ip_start, ip_end, cell_vol)
            if n > 0:
                debye_length_inv_sqare.flat[icell] += inv_d2
            

@njit(cache=True,inline='always')
def debye_length_cell(
    part: ParticleData,
    ip_start: np.int64, ip_end: np.int64,
    cell_vol: float
) -> tuple[float, float]:
    density = 0.0
    kT_mc2 = 0.0

    is_dead = part.is_dead
    inv_gamma = part.inv_gamma
    w = part.w
    m = part.m
    q = part.q
    for ip in range(ip_start, ip_end):
        if is_dead[ip]: 
            continue
        density += w[ip]

        gamma = 1 / inv_gamma[ip]
        u2 = gamma**2 - 1

        # T = <v*p> / 3
        kT_mc2 += w[ip] * u2 / sqrt(1 + u2) / 3


    if density > 0:
        kT_mc2 /= density
        density /= cell_vol

        kT = kT_mc2 * m * c**2

        if kT > 0:
            debye_length_inv_sqare = density * q**2 / (epsilon_0 * kT)
        else:
            debye_length_inv_sqare = math.inf
    else:
        debye_length_inv_sqare = -1.0

    return debye_length_inv_sqare, density

@jit_spinner
@njit(parallel=True, cache=True)
def constrain_debye_length_patches(
    debye_length_inv_sqare_list: List[NDArray[np.float64]],
    total_density_list: List[NDArray[np.float64]],
):
    for ipatch in prange(len(debye_length_inv_sqare_list)):
        for icell in range(debye_length_inv_sqare_list[ipatch].size):
            inv_d2 = debye_length_inv_sqare_list[ipatch].flat[icell]
            if inv_d2 <= 0:
                # will be handled in varying_lnLambda
                continue
            
            d2 = 1/inv_d2

            nmax = total_density_list[ipatch].flat[icell]
            rmin2 = (4*pi*nmax/3)**(-2/3)

            if d2 < rmin2:
                debye_length_inv_sqare_list[ipatch].flat[icell] = 1/rmin2

@njit(cache=True)
def varying_lnLambda(d: CollisionData, debye_length_inv_sqare: np.float64|float) -> float:
    """
    https://doi.org/10.1063/1.4742167

    $b_{0} = \frac{q_{1}q_{2}}{4\pi\epsilon_{0}c^{2}}\frac{\gamma c}{m_{1}\gamma_{1} + m_{2}\gamma_{2}}\left(\frac{m_{1}\gamma_{1}^{\star}m_{2}\gamma_{2}^{\star}}{p_{1}^{\star 2}} c^{2} + 1\right)^{2} \tag{22}$
    """
    m1 = d.m1
    m2 = d.m2
    gamma1_com = d.gamma1_com
    gamma2_com = d.gamma2_com
    gamma_com = d.gamma_com
    p1_com = d.p1_com

    q1q2 = abs(d.q1*d.q2)

    b0 = q1q2 / (4*pi*epsilon_0*c**2) * gamma_com / (m1*gamma1_com + m2*gamma2_com) * ((m1*gamma1_com*m2*gamma2_com)/p1_com**2*c**2 + 1)
    bmin = max(h/2/p1_com, b0)

    if debye_length_inv_sqare > 0:
        lambdaD2 = 1 / debye_length_inv_sqare
        lnLambda = max(2.0, 0.5*np.log(1 + lambdaD2/bmin**2))
    else:
        lnLambda = 2.0

    return lnLambda


@jit_spinner
@njit(parallel=True, cache=True)
def intra_collision_patches(
    part_list: List[ParticleData],
    bucket_bound_min_list: List[NDArray[np.int64]],
    bucket_bound_max_list: List[NDArray[np.int64]],
    lnLambda: float, debye_length_inv_sqare_list: List[NDArray[np.float64]],
    cell_vol: float, dt: float,
    gen_list: List[np.random.Generator]
):
    for ipatch in prange(len(part_list)):
        for icell in range(bucket_bound_min_list[ipatch].size):
            ip_start = bucket_bound_min_list[ipatch].flat[icell]
            ip_end   = bucket_bound_max_list[ipatch].flat[icell]
            
            intra_collision_cell(
                part_list[ipatch], ip_start, ip_end, 
                lnLambda, debye_length_inv_sqare_list[ipatch].flat[icell], 
                cell_vol, dt, 
                gen_list[ipatch]
            )


@njit(cache=True)
def intra_collision_cell(
    part: ParticleData, ip_start: np.int64, ip_end: np.int64,
    lnLambda: float, debye_length_inv_sqare: np.float64|float,
    cell_vol: float, dt: float,
    gen: np.random.Generator
):
    dead = part.is_dead

    m, q = part.m, part.q

    nbuf = ip_end - ip_start
    npart = nbuf - dead[ip_start:ip_end].sum()
    if npart < 2: 
        return

    npairs = (npart + 1 ) // 2
    dt_corr = 2*npairs - 1

    # loop pairs
    for ipair, ip1, ip2, w_corr in self_pairing(dead, ip_start, ip_end, gen):

        w1_corr = part.w[ip1] * w_corr
        w2_corr = part.w[ip2] * w_corr
        w_max = max(w1_corr, w2_corr)

        d = CollisionData(
            part.ux[ip1], part.uy[ip1], part.uz[ip1], part.inv_gamma[ip1], w1_corr,
            m, q,
            part.ux[ip2], part.uy[ip2], part.uz[ip2], part.inv_gamma[ip2], w2_corr,
            m, q,
        )

        vx_com, vy_com, vz_com = d.vx_com, d.vy_com, d.vz_com
        gamma_com = d.gamma_com
        v_com_square = d.v_com_square
        
        gamma1_com = d.gamma1_com
        gamma2_com = d.gamma2_com
        
        if lnLambda > 0:
            lnLambda_ = lnLambda
        else:
            lnLambda_ = varying_lnLambda(d, debye_length_inv_sqare)

        px1_com_new, py1_com_new, pz1_com_new = coulomb_scattering(d, cell_vol, dt*dt_corr,
                                                                   lnLambda_, gen)

        U = gen.uniform()
        if w2_corr / w_max > U:
            px1_new, py1_new, pz1_new = boost_to_lab(
                px1_com_new, py1_com_new, pz1_com_new, gamma1_com, m, 
                vx_com, vy_com, vz_com, v_com_square, gamma_com
            )
            part.ux[ip1], part.uy[ip1], part.uz[ip1] = px1_new / m / c, py1_new / m / c, pz1_new / m / c
            part.inv_gamma[ip1] = 1/sqrt(part.ux[ip1]**2 + part.uy[ip1]**2 + part.uz[ip1]**2 + 1)
        if w1_corr / w_max > U:
            px2_new, py2_new, pz2_new = boost_to_lab(
                -px1_com_new, -py1_com_new, -pz1_com_new, gamma2_com, m, 
                vx_com, vy_com, vz_com, v_com_square, gamma_com
            )
            part.ux[ip2], part.uy[ip2], part.uz[ip2] = px2_new / m / c, py2_new / m / c, pz2_new / m / c
            part.inv_gamma[ip2] = 1/sqrt(part.ux[ip2]**2 + part.uy[ip2]**2 + part.uz[ip2]**2 + 1)
            

@jit_spinner
@njit(parallel=True, cache=True)
def inter_collision_patches(
    part1_list: List[ParticleData], bucket_bound_min1_list: List[NDArray[np.int64]], bucket_bound_max1_list: List[NDArray[np.int64]],
    part2_list: List[ParticleData], bucket_bound_min2_list: List[NDArray[np.int64]], bucket_bound_max2_list: List[NDArray[np.int64]],
    npatches: int, 
    lnLambda: float, debye_length_inv_sqare_list: List[NDArray[np.float64]],
    cell_vol: float, dt: float,
    gen_list: List[np.random.Generator]
):
    for ipatch in prange(npatches):
        for icell in range(bucket_bound_min1_list[ipatch].size):
            ip_start1 = bucket_bound_min1_list[ipatch].flat[icell]
            ip_end1   = bucket_bound_max1_list[ipatch].flat[icell]
            ip_start2 = bucket_bound_min2_list[ipatch].flat[icell]
            ip_end2   = bucket_bound_max2_list[ipatch].flat[icell]

            inter_collision_cell(
                part1_list[ipatch], ip_start1, ip_end1,
                part2_list[ipatch], ip_start2, ip_end2,
                lnLambda, debye_length_inv_sqare_list[ipatch].flat[icell],
                cell_vol, dt,
                gen_list[ipatch]
            )

@njit(inline='always')
def inter_collision_cell(
    part1: ParticleData, ip_start1: np.int64, ip_end1: np.int64,
    part2: ParticleData, ip_start2: np.int64, ip_end2: np.int64,
    lnLambda: float, debye_length_inv_sqare: np.float64|float,
    cell_vol: float, dt: float,
    gen: np.random.Generator
):
    dead1 = part1.is_dead
    dead2 = part2.is_dead

    m1 = part1.m
    m2 = part2.m
    
    nbuf1 = ip_end1 - ip_start1
    npart1 = nbuf1 - dead1[ip_start1:ip_end1].sum()

    nbuf2 = ip_end2 - ip_start2
    npart2 = nbuf2 - dead2[ip_start2:ip_end2].sum()
    if npart1 == 0 or npart2 == 0: 
        return

    npairs = max(npart1, npart2)
    dt_corr = npairs

    # loop pairs
    for ipair, ip1, ip2, w_corr in pairing(
        dead1, ip_start1, ip_end1, 
        dead2, ip_start2, ip_end2, gen
    ):
        w1_corr = part1.w[ip1] * w_corr
        w2_corr = part2.w[ip2] * w_corr
        w_max = max(w1_corr, w2_corr)
        
        d = CollisionData(
            part1.ux[ip1], part1.uy[ip1], part1.uz[ip1], part1.inv_gamma[ip1], w1_corr,
            part1.m, part1.q,
            part2.ux[ip2], part2.uy[ip2], part2.uz[ip2], part2.inv_gamma[ip2], w2_corr,
            part2.m, part2.q,
        )
        
        vx_com, vy_com, vz_com = d.vx_com, d.vy_com, d.vz_com
        gamma_com = d.gamma_com
        v_com_square = d.v_com_square
        
        gamma1_com = d.gamma1_com
        gamma2_com = d.gamma2_com
        
        if lnLambda > 0:
            lnLambda_ = lnLambda
        else:
            lnLambda_ = varying_lnLambda(d, debye_length_inv_sqare)

        px1_com_new, py1_com_new, pz1_com_new = coulomb_scattering(d, cell_vol, dt*dt_corr,
                                                                   lnLambda_, gen)

        U = gen.uniform()
        if w2_corr / w_max > U:
            px1_new, py1_new, pz1_new = boost_to_lab(
                px1_com_new, py1_com_new, pz1_com_new, gamma1_com, m1, 
                vx_com, vy_com, vz_com, v_com_square, gamma_com
            )
            part1.ux[ip1], part1.uy[ip1], part1.uz[ip1] = px1_new / m1 / c, py1_new / m1 / c, pz1_new / m1 / c
            part1.inv_gamma[ip1] = 1/sqrt(part1.ux[ip1]**2 + part1.uy[ip1]**2 + part1.uz[ip1]**2 + 1)
        if w1_corr / w_max > U:
            px2_new, py2_new, pz2_new = boost_to_lab(
                -px1_com_new, -py1_com_new, -pz1_com_new, gamma2_com, m2, 
                vx_com, vy_com, vz_com, v_com_square, gamma_com
            )
            part2.ux[ip2], part2.uy[ip2], part2.uz[ip2] = px2_new / m2 / c, py2_new / m2 / c, pz2_new / m2 / c
            part2.inv_gamma[ip2] = 1/sqrt(part2.ux[ip2]**2 + part2.uy[ip2]**2 + part2.uz[ip2]**2 + 1)