from numba import njit, prange
from math import sqrt
from scipy.constants import c, pi, epsilon_0
import numpy as np

@njit(cache=True)
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

@njit(cache=True)
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

    if npart1 > npart2:
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

# def debye_length_cell(
#     ux, uy, uz, inv_gamma, w, dead,
#     m, q, lnLambda,
#     ip_start, ip_end,
#     dx, dy, dz, dt,
#     debye_length
# ):
#     nbuf = ip_end - ip_start
#     npart = nbuf - dead[ip_start:ip_end].sum()
#     if npart == 0:
#         return

#     density = 0.0
#     kT = 0.0 # in mc2
#     mean_charge = 0.0
#     for ip in range(ip_start, ip_end):
#         if dead[ip]: 
#             continue
#         px = ux[ip] * m * c
#         py = uy[ip] * m * c
#         pz = uz[ip] * m * c
#         p2 = px**2 + py**2 + pz**2
#         kT += p2 * inv_gamma[ip]
#         density += w[ip] / (dx*dy*dz)
#         mean_charge += w[ip] * q

@njit(cache=True)
def self_collision_cell(
    ux, uy, uz, inv_gamma, w, dead,
    m, q, lnLambda,
    ip_start, ip_end,
    dx, dy, dz, dt,
    random_gen
):
    """
    Self collision
    
    Args:
        ux,uy,uz (np.ndarray): particle velocity
        inv_gamma (np.ndarray): 1 / gamma
        w (np.ndarray): particle weight
        dead (np.ndarray): dead particle flag
        m (float): particle mass
        q (float): particle charge
        lnLambda (float): Coulomb logarithm
        ip_start (int): start index of particle buffer
        ip_end (int): end index of particle buffer
        dx,dy,dz (float): cell size, set `dz = 1` for 2D
        dt (float): time step
        random_gen (np.random.Generator): random number generator
    """
    nbuf = ip_end - ip_start
    npart = nbuf - dead[ip_start:ip_end].sum()
    if npart < 2: 
        return

    npairs = (npart + 1 ) // 2
    dt_corr = 2*npairs - 1

    # loop pairs
    for ipair, ip1, ip2, w_corr in self_pairing(dead, ip_start, ip_end, random_gen):

        w1 = w[ip1] * w_corr
        w2 = w[ip2] * w_corr
        w_max = max(w1, w2)

        gamma1 = 1/inv_gamma[ip1]
        gamma2 = 1/inv_gamma[ip2]

        p1x = ux[ip1]*m*c
        p1y = uy[ip1]*m*c
        p1z = uz[ip1]*m*c

        p2x = ux[ip2]*m*c
        p2y = uy[ip2]*m*c
        p2z = uz[ip2]*m*c

        v1x = p1x / gamma1 / m
        v1y = p1y / gamma1 / m
        v1z = p1z / gamma1 / m

        v2x = p2x / gamma2 / m
        v2y = p2y / gamma2 / m
        v2z = p2z / gamma2 / m

        vx_com = (p1x + p2x) / (gamma1*m + gamma2*m)
        vy_com = (p1y + p2y) / (gamma1*m + gamma2*m)
        vz_com = (p1z + p2z) / (gamma1*m + gamma2*m)
        gamma_com = 1.0 / sqrt(1 - (vx_com**2 + vy_com**2 + vz_com**2)/c**2 )
        v_com_square = vx_com**2 + vy_com**2 + vz_com**2

        # _, p1x_com, p1y_com, p1z_com = lorentz_boost(gamma1*m1*c, p1x, p1y, p1z, vx_com, vy_com, vz_com, gamma_com)
        p1x_com = p1x + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m*gamma1*vx_com
        p1y_com = p1y + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m*gamma1*vy_com
        p1z_com = p1z + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m*gamma1*vz_com
        p1_com = np.sqrt(p1x_com**2 + p1y_com**2 + p1z_com**2)

        gamma1_com = (1 - (vx_com*v1x + vy_com*v1y + vz_com*v1z) / c**2) * gamma_com*gamma1
        gamma2_com = (1 - (vx_com*v2x + vy_com*v2y + vz_com*v2z) / c**2) * gamma_com*gamma2

        
        s = w_max/(dx*dy*dz) *dt * (lnLambda * (q*q)**2) / (4*pi*epsilon_0**2*c**4 * m*gamma1 * m*gamma2) \
                *(gamma_com * p1_com)/(m*gamma1 + m*gamma2) * (m*gamma1_com*m*gamma2_com/p1_com**2 * c**2 + 1)**2
        s *= dt_corr

        U1 = random_gen.uniform()
        U2 = random_gen.uniform()
        if s < 4:
            alpha = 0.37*s - 0.005*s**2 - 0.0064*s**3
            sin2X2 = alpha * U1 / np.sqrt( (1-U1) + alpha*alpha*U1 )
            cosX = 1. - 2.*sin2X2
            sinX = 2.*np.sqrt( sin2X2 *(1.-sin2X2) )
        else:
            cosX = 2.*U1 - 1.
            sinX = np.sqrt( 1. - cosX*cosX )

        phi = np.random.uniform(0, 2*pi)
        sinXcosPhi = sinX*np.cos( phi )
        sinXsinPhi = sinX*np.sin( phi )

        p_perp = sqrt( p1x_com**2 + p1y_com**2 )
        # make sure p_perp is not too small
        if p_perp > 1.e-10*p1_com:
            p1x_com_new = ( p1x_com * p1z_com * sinXcosPhi - p1y_com * p1_com * sinXsinPhi ) / p_perp + p1x_com * cosX
            p1y_com_new = ( p1y_com * p1z_com * sinXcosPhi + p1x_com * p1_com * sinXsinPhi ) / p_perp + p1y_com * cosX
            p1z_com_new = -p_perp * sinXcosPhi + p1z_com * cosX
        # if p_perp is too small, we use the limit px->0, py=0
        else:
            p1x_com_new = p1_com * sinXcosPhi
            p1y_com_new = p1_com * sinXsinPhi
            p1z_com_new = p1_com * cosX

        vcom_dot_p = vx_com*p1x_com_new + vy_com*p1y_com_new + vz_com*p1z_com_new
        if w2/w_max > U2:
            p1x_new = p1x_com_new + vx_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m*gamma1_com*gamma_com)
            p1y_new = p1y_com_new + vy_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m*gamma1_com*gamma_com)
            p1z_new = p1z_com_new + vz_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m*gamma1_com*gamma_com)
            ux[ip1] = p1x_new / m / c
            uy[ip1] = p1y_new / m / c
            uz[ip1] = p1z_new / m / c
            inv_gamma[ip1] = 1/sqrt(ux[ip1]**2 + uy[ip1]**2 + uz[ip1]**2 + 1)
        if w1/w_max > U2:
            p2x_new = -p1x_com_new + vx_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m*gamma2_com*gamma_com)
            p2y_new = -p1y_com_new + vy_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m*gamma2_com*gamma_com)
            p2z_new = -p1z_com_new + vz_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m*gamma2_com*gamma_com)
            ux[ip2] = p2x_new / m / c
            uy[ip2] = p2y_new / m / c
            uz[ip2] = p2z_new / m / c
            inv_gamma[ip2] = 1/sqrt(ux[ip2]**2 + uy[ip2]**2 + uz[ip2]**2 + 1)

@njit(cache=True)
def inter_collision_cell(
    ux1, uy1, uz1, inv_gamma1, w1, dead1, ip_start1, ip_end1,
    ux2, uy2, uz2, inv_gamma2, w2, dead2, ip_start2, ip_end2,
    m1, q1, m2, q2,
    lnLambda,
    dx, dy, dz, dt,
    random_gen
):
    """
    Inter collision
    
    Args:
        ux1,uy1,uz1 (np.ndarray): momentum of species 1
        inv_gamma1 (np.ndarray): 1 / gamma of species 1
        w1 (np.ndarray): weight of species 1
        dead1 (np.ndarray): dead particle flag of species 1
        ip_start1 (int): start index of particle buffer of species 1
        ip_end1 (int): end index of particle buffer of species 1
        m1 (float): mass of species 1
        q1 (float): charge of species 1
        lnLambda (float): Coulomb logarithm
        dx,dy,dz (float): cell size, set `dz = 1` for 2D
        dt (float): time step
        random_gen (np.random.Generator): random number generator
    """
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
        dead2, ip_start2, ip_end2, random_gen
    ):
        w1_ = w1[ip1] * w_corr
        w2_ = w2[ip2] * w_corr
        w_max = max(w1_, w2_)
        
        U1 = random_gen.uniform()
        U2 = random_gen.uniform()

        gamma1 = 1/inv_gamma1[ip1]
        gamma2 = 1/inv_gamma2[ip2]

        p1x = ux1[ip1]*m1*c
        p1y = uy1[ip1]*m1*c
        p1z = uz1[ip1]*m1*c

        p2x = ux2[ip2]*m2*c
        p2y = uy2[ip2]*m2*c
        p2z = uz2[ip2]*m2*c

        v1x = p1x / gamma1 / m1
        v1y = p1y / gamma1 / m1
        v1z = p1z / gamma1 / m1

        v2x = p2x / gamma2 / m2
        v2y = p2y / gamma2 / m2
        v2z = p2z / gamma2 / m2

        vx_com = (p1x + p2x) / (gamma1*m1 + gamma2*m2)
        vy_com = (p1y + p2y) / (gamma1*m1 + gamma2*m2)
        vz_com = (p1z + p2z) / (gamma1*m1 + gamma2*m2)
        gamma_com = 1.0 / sqrt(1 - (vx_com**2 + vy_com**2 + vz_com**2)/c**2 )
        v_com_square = vx_com**2 + vy_com**2 + vz_com**2

        # _, p1x_com, p1y_com, p1z_com = lorentz_boost(gamma1*m1*c, p1x, p1y, p1z, vx_com, vy_com, vz_com, gamma_com)
        p1x_com = p1x + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m1*gamma1*vx_com
        p1y_com = p1y + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m1*gamma1*vy_com
        p1z_com = p1z + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m1*gamma1*vz_com

        # p2x_com = p2x + ((gamma_com-1)/v_com_square * (v2x*vx_com + v2y*vy_com + v2z*vz_com) - gamma_com) * m2*gamma2*vx_com
        # p2y_com = p2y + ((gamma_com-1)/v_com_square * (v2x*vx_com + v2y*vy_com + v2z*vz_com) - gamma_com) * m2*gamma2*vy_com
        # p2z_com = p2z + ((gamma_com-1)/v_com_square * (v2x*vx_com + v2y*vy_com + v2z*vz_com) - gamma_com) * m2*gamma2*vz_com

        p1_com = np.sqrt(p1x_com**2 + p1y_com**2 + p1z_com**2)

        gamma1_com = (1 - (vx_com*v1x + vy_com*v1y + vz_com*v1z) / c**2) * gamma_com*gamma1
        gamma2_com = (1 - (vx_com*v2x + vy_com*v2y + vz_com*v2z) / c**2) * gamma_com*gamma2

        s = w_max/(dx*dy*dz) *dt * (lnLambda * (q1*q2)**2) / (4*pi*epsilon_0**2*c**4 * m1*gamma1 * m2*gamma2) \
                *(gamma_com * p1_com)/(m1*gamma1 + m2*gamma2) * (m1*gamma1_com*m2*gamma2_com/p1_com**2 * c**2 + 1)**2
        s *= dt_corr

        
        if s < 4:
            alpha = 0.37*s - 0.005*s**2 - 0.0064*s**3
            sin2X2 = alpha * U1 / np.sqrt( (1-U1) + alpha*alpha*U1 )
            cosX = 1. - 2.*sin2X2
            sinX = 2.*np.sqrt( sin2X2 *(1.-sin2X2) )
        else:
            cosX = 2.*U1 - 1.
            sinX = np.sqrt( 1. - cosX*cosX )

        phi = np.random.uniform(0, 2*pi)
        sinXcosPhi = sinX*np.cos( phi )
        sinXsinPhi = sinX*np.sin( phi )

        p_perp = sqrt( p1x_com**2 + p1y_com**2 )
        # make sure p_perp is not too small
        if p_perp > 1.e-10*p1_com:
            p1x_com_new = ( p1x_com * p1z_com * sinXcosPhi - p1y_com * p1_com * sinXsinPhi ) / p_perp + p1x_com * cosX
            p1y_com_new = ( p1y_com * p1z_com * sinXcosPhi + p1x_com * p1_com * sinXsinPhi ) / p_perp + p1y_com * cosX
            p1z_com_new = -p_perp * sinXcosPhi + p1z_com * cosX
        # if p_perp is too small, we use the limit px->0, py=0
        else:
            p1x_com_new = p1_com * sinXcosPhi
            p1y_com_new = p1_com * sinXsinPhi
            p1z_com_new = p1_com * cosX

        vcom_dot_p = vx_com*p1x_com_new + vy_com*p1y_com_new + vz_com*p1z_com_new
        if w2_ / w_max > U2:
            p1x_new = p1x_com_new + vx_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m1*gamma1_com*gamma_com)
            p1y_new = p1y_com_new + vy_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m1*gamma1_com*gamma_com)
            p1z_new = p1z_com_new + vz_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m1*gamma1_com*gamma_com)
            ux1[ip1] = p1x_new / m1 / c
            uy1[ip1] = p1y_new / m1 / c
            uz1[ip1] = p1z_new / m1 / c
            inv_gamma1[ip1] = 1/sqrt(ux1[ip1]**2 + uy1[ip1]**2 + uz1[ip1]**2 + 1)
        if w1_ / w_max > U2:
            p2x_new = -p1x_com_new + vx_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m2*gamma2_com*gamma_com)
            p2y_new = -p1y_com_new + vy_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m2*gamma2_com*gamma_com)
            p2z_new = -p1z_com_new + vz_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m2*gamma2_com*gamma_com)
            ux2[ip2] = p2x_new / m2 / c
            uy2[ip2] = p2y_new / m2 / c
            uz2[ip2] = p2z_new / m2 / c
            inv_gamma2[ip2] = 1/sqrt(ux2[ip2]**2 + uy2[ip2]**2 + uz2[ip2]**2 + 1)

@njit(parallel=True, cache=True)
def self_collision_parallel(
    cell_bound_min, cell_bound_max, 
    nx, ny, dx, dy, dz, dt,
    ux, uy, uz, inv_gamma, w, dead,
    m, q, lnLambda,
    random_gen
):
    for icell in prange(nx*ny):
        ix = icell // ny
        iy = icell % ny
        ip_start = cell_bound_min[ix,iy]
        ip_end = cell_bound_max[ix,iy]
        self_collision_cell(
            ux, uy, uz, inv_gamma, w, dead,
            m, q, lnLambda,
            ip_start, ip_end,
            dx, dy, dz, dt,
            random_gen
        )

@njit(cache=True)
def self_collision(
    cell_bound_min, cell_bound_max, 
    nx, ny, dx, dy, dz, dt,
    ux, uy, uz, inv_gamma, w, dead,
    m, q, lnLambda,
    random_gen
):
    for ix in range(nx):
        for iy in range(ny):
            ip_start = cell_bound_min[ix,iy]
            ip_end = cell_bound_max[ix,iy]
            self_collision_cell(
                ux, uy, uz, inv_gamma, w, dead,
                m, q, lnLambda,
                ip_start, ip_end,
                dx, dy, dz, dt,
                random_gen
            )


@njit(parallel=True, cache=True)
def inter_collision_parallel(
    cell_bound_min1, cell_bound_max1, 
    cell_bound_min2, cell_bound_max2, 
    nx, ny, dx, dy, dz, dt,
    ux1, uy1, uz1, inv_gamma1, w1, dead1,
    ux2, uy2, uz2, inv_gamma2, w2, dead2,
    m1, q1, m2, q2,
    lnLambda,
    random_gen
):
    for icell in prange(nx*ny):
        ix = icell // ny
        iy = icell % ny
        
        ip_start1 = cell_bound_min1[ix,iy]
        ip_end1 = cell_bound_max1[ix,iy]

        ip_start2 = cell_bound_min2[ix,iy]
        ip_end2 = cell_bound_max2[ix,iy]

        inter_collision_cell(
            ux1, uy1, uz1, inv_gamma1, w1, dead1, ip_start1, ip_end1,
            ux2, uy2, uz2, inv_gamma2, w2, dead2, ip_start2, ip_end2,
            m1, q1, m2, q2,
            lnLambda,
            dx, dy, dz, dt,
            random_gen
        )

@njit(cache=True)
def inter_collision(
    cell_bound_min1, cell_bound_max1, 
    cell_bound_min2, cell_bound_max2, 
    nx, ny, dx, dy, dz, dt,
    ux1, uy1, uz1, inv_gamma1, w1, dead1,
    ux2, uy2, uz2, inv_gamma2, w2, dead2,
    m1, q1, m2, q2,
    lnLambda,
    random_gen
):
    for ix in range(nx):
        for iy in range(ny):
            ip_start1 = cell_bound_min1[ix,iy]
            ip_end1 = cell_bound_max1[ix,iy]

            ip_start2 = cell_bound_min2[ix,iy]
            ip_end2 = cell_bound_max2[ix,iy]

            inter_collision_cell(
                ux1, uy1, uz1, inv_gamma1, w1, dead1, ip_start1, ip_end1,
                ux2, uy2, uz2, inv_gamma2, w2, dead2, ip_start2, ip_end2,
                m1, q1, m2, q2,
                lnLambda,
                dx, dy, dz, dt,
                random_gen
            )