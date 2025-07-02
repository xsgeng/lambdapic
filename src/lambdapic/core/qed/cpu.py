import numpy as np
from numba import boolean, float64, int64, njit, prange, void

from .inline import (
    calculate_chi_inline,
    create_pair_inline_2d,
    create_pair_inline_3d,
    create_photon_inline_2d,
    create_photon_inline_3d,
    find_event_index_inline,
)
from .optical_depth import update_tau_e, update_tau_gamma

calculate_chi_cpu = njit(calculate_chi_inline)


@njit(parallel=True, cache=True)
def update_chi_patches(
    ux_list, uy_list, uz_list, inv_gamma_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    is_dead_list,
    npatches,
    chi_list,
) -> None:
    for ipatch in prange(npatches):
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]

        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        is_dead = is_dead_list[ipatch]
        npart = len(is_dead)

        chi = chi_list[ipatch]

        for ip in range(npart):
            if is_dead[ip]:
                continue
            chi[ip] = calculate_chi_cpu(
                ex[ip], ey[ip], ez[ip],
                bx[ip], by[ip], bz[ip],
                ux[ip], uy[ip], uz[ip],
                inv_gamma[ip]
            )



@njit(parallel=True, cache=True)
def radiation_event_patches(
    tau_list,
    chi_list,
    inv_gamma_list,
    is_dead_list,
    npatches,
    dt,
    event_list,
    delta_list,
    integral_photon_prob_along_delta, photon_prob_rate_total_table
):
    for ipatch in prange(npatches):
        inv_gamma = inv_gamma_list[ipatch]

        tau_e = tau_list[ipatch]
        chi_e = chi_list[ipatch]
        event = event_list[ipatch]
        delta = delta_list[ipatch]

        is_dead = is_dead_list[ipatch]
        npart = len(is_dead)
        update_tau_e(
            tau_e, inv_gamma, chi_e, dt,
            npart, is_dead, event, delta,
            integral_photon_prob_along_delta, photon_prob_rate_total_table
        )


@njit(parallel=True, cache=True)
def pairproduction_event_patches(
    tau_list,
    chi_list,
    inv_gamma_list,
    is_dead_list,
    npatches,
    dt,
    event_list,
    delta_list,
    integral_pair_prob_along_delta, pair_prob_rate_total_table
):
    for ipatch in prange(npatches):
        inv_gamma = inv_gamma_list[ipatch]
        tau_gamma = tau_list[ipatch]
        chi_gamma = chi_list[ipatch]
        event = event_list[ipatch]
        delta = delta_list[ipatch]

        is_dead = is_dead_list[ipatch]
        npart = len(is_dead)
        update_tau_gamma(
            tau_gamma, inv_gamma, chi_gamma, dt,
            npart, is_dead, event, delta,
            integral_pair_prob_along_delta, pair_prob_rate_total_table
        )
        
create_photon_2d = njit(create_photon_inline_2d)
find_event_index = njit(find_event_index_inline)

@njit(parallel=True, cache=True)
def create_photon_patches_2d(
    x_ele_list, y_ele_list, ux_ele_list, uy_ele_list, uz_ele_list, w_ele_list, is_dead_ele_list,
    x_pho_list, y_pho_list, ux_pho_list, uy_pho_list, uz_pho_list, inv_gamma_pho_list, w_pho_list, is_dead_pho_list, 
    delta_list, event_list,
    npatches,
):
    for ipatch in prange(npatches):
        x_ele = x_ele_list[ipatch]
        y_ele = y_ele_list[ipatch]
        ux_ele = ux_ele_list[ipatch]
        uy_ele = uy_ele_list[ipatch]
        uz_ele = uz_ele_list[ipatch]
        w_ele = w_ele_list[ipatch]
        is_dead_ele = is_dead_ele_list[ipatch]

        x_pho = x_pho_list[ipatch]
        y_pho = y_pho_list[ipatch]
        ux_pho = ux_pho_list[ipatch]
        uy_pho = uy_pho_list[ipatch]
        uz_pho = uz_pho_list[ipatch]
        w_pho = w_pho_list[ipatch]
        inv_gamma_pho = inv_gamma_pho_list[ipatch]
        is_dead_pho = is_dead_pho_list[ipatch]

        delta = delta_list[ipatch]
        event = event_list[ipatch]

        event_index = find_event_index(event, is_dead_ele)

        create_photon_2d(
            x_ele, y_ele, ux_ele, uy_ele, uz_ele, w_ele, 
            x_pho, y_pho, ux_pho, uy_pho, uz_pho, inv_gamma_pho, w_pho, is_dead_pho, 
            event_index, delta,
        )

create_pair_2d = njit(create_pair_inline_2d)

@njit(parallel=True, cache=True)
def create_pair_patches_2d(
    x_pho_list, y_pho_list, ux_pho_list, uy_pho_list, uz_pho_list, w_pho_list, is_dead_pho_list,
    x_ele_list, y_ele_list, ux_ele_list, uy_ele_list, uz_ele_list, inv_gamma_ele_list, w_ele_list, is_dead_ele_list,
    x_pos_list, y_pos_list, ux_pos_list, uy_pos_list, uz_pos_list, inv_gamma_pos_list, w_pos_list, is_dead_pos_list,
    delta_list, event_list,
    npatches,
):
    for ipatch in prange(npatches):
        x_pho = x_pho_list[ipatch]
        y_pho = y_pho_list[ipatch]
        ux_pho = ux_pho_list[ipatch]
        uy_pho = uy_pho_list[ipatch]
        uz_pho = uz_pho_list[ipatch]
        w_pho = w_pho_list[ipatch]
        is_dead_pho = is_dead_pho_list[ipatch]

        x_ele = x_ele_list[ipatch]
        y_ele = y_ele_list[ipatch]
        ux_ele = ux_ele_list[ipatch]
        uy_ele = uy_ele_list[ipatch]
        uz_ele = uz_ele_list[ipatch]
        inv_gamma_ele = inv_gamma_ele_list[ipatch]
        w_ele = w_ele_list[ipatch]
        is_dead_ele = is_dead_ele_list[ipatch]

        x_pos = x_pos_list[ipatch]
        y_pos = y_pos_list[ipatch]
        ux_pos = ux_pos_list[ipatch]
        uy_pos = uy_pos_list[ipatch]
        uz_pos = uz_pos_list[ipatch]
        inv_gamma_pos = inv_gamma_pos_list[ipatch]
        w_pos = w_pos_list[ipatch]
        is_dead_pos = is_dead_pos_list[ipatch]

        delta = delta_list[ipatch]
        event = event_list[ipatch]
        event_index = find_event_index(event, is_dead_pho)

        create_pair_2d(
            event_index, 
            x_pho, y_pho, ux_pho, uy_pho, uz_pho, w_pho, delta,
            x_ele, y_ele, ux_ele, uy_ele, uz_ele, inv_gamma_ele, w_ele, is_dead_ele, 
            x_pos, y_pos, ux_pos, uy_pos, uz_pos, inv_gamma_pos, w_pos, is_dead_pos, 
        )

create_photon_3d = njit(create_photon_inline_3d)
find_event_index = njit(find_event_index_inline)

@njit(parallel=True, cache=True)
def create_photon_patches_3d(
    x_ele_list, y_ele_list, z_ele_list, ux_ele_list, uy_ele_list, uz_ele_list, w_ele_list, is_dead_ele_list,
    x_pho_list, y_pho_list, z_pho_list, ux_pho_list, uy_pho_list, uz_pho_list, inv_gamma_pho_list, w_pho_list, is_dead_pho_list, 
    delta_list, event_list,
    npatches,
):
    for ipatch in prange(npatches):
        x_ele = x_ele_list[ipatch]
        y_ele = y_ele_list[ipatch]
        z_ele = z_ele_list[ipatch]
        ux_ele = ux_ele_list[ipatch]
        uy_ele = uy_ele_list[ipatch]
        uz_ele = uz_ele_list[ipatch]
        w_ele = w_ele_list[ipatch]
        is_dead_ele = is_dead_ele_list[ipatch]

        x_pho = x_pho_list[ipatch]
        y_pho = y_pho_list[ipatch]
        z_pho = z_pho_list[ipatch]
        ux_pho = ux_pho_list[ipatch]
        uy_pho = uy_pho_list[ipatch]
        uz_pho = uz_pho_list[ipatch]
        w_pho = w_pho_list[ipatch]
        inv_gamma_pho = inv_gamma_pho_list[ipatch]
        is_dead_pho = is_dead_pho_list[ipatch]

        delta = delta_list[ipatch]
        event = event_list[ipatch]

        event_index = find_event_index(event, is_dead_ele)

        create_photon_3d(
            x_ele, y_ele, z_ele, ux_ele, uy_ele, uz_ele, w_ele, 
            x_pho, y_pho, z_pho, ux_pho, uy_pho, uz_pho, inv_gamma_pho, w_pho, is_dead_pho, 
            event_index, delta,
        )

create_pair_3d = njit(create_pair_inline_3d)

@njit(parallel=True, cache=True)
def create_pair_patches_3d(
    x_pho_list, y_pho_list, z_pho_list, ux_pho_list, uy_pho_list, uz_pho_list, w_pho_list, is_dead_pho_list,
    x_ele_list, y_ele_list, z_ele_list, ux_ele_list, uy_ele_list, uz_ele_list, inv_gamma_ele_list, w_ele_list, is_dead_ele_list,
    x_pos_list, y_pos_list, z_pos_list, ux_pos_list, uy_pos_list, uz_pos_list, inv_gamma_pos_list, w_pos_list, is_dead_pos_list,
    delta_list, event_list,
    npatches,
):
    for ipatch in prange(npatches):
        x_pho = x_pho_list[ipatch]
        y_pho = y_pho_list[ipatch]
        z_pho = z_pho_list[ipatch]
        ux_pho = ux_pho_list[ipatch]
        uy_pho = uy_pho_list[ipatch]
        uz_pho = uz_pho_list[ipatch]
        w_pho = w_pho_list[ipatch]
        is_dead_pho = is_dead_pho_list[ipatch]

        x_ele = x_ele_list[ipatch]
        y_ele = y_ele_list[ipatch]
        z_ele = z_ele_list[ipatch]
        ux_ele = ux_ele_list[ipatch]
        uy_ele = uy_ele_list[ipatch]
        uz_ele = uz_ele_list[ipatch]
        inv_gamma_ele = inv_gamma_ele_list[ipatch]
        w_ele = w_ele_list[ipatch]
        is_dead_ele = is_dead_ele_list[ipatch]

        x_pos = x_pos_list[ipatch]
        y_pos = y_pos_list[ipatch]
        z_pos = z_pos_list[ipatch]
        ux_pos = ux_pos_list[ipatch]
        uy_pos = uy_pos_list[ipatch]
        uz_pos = uz_pos_list[ipatch]
        inv_gamma_pos = inv_gamma_pos_list[ipatch]
        w_pos = w_pos_list[ipatch]
        is_dead_pos = is_dead_pos_list[ipatch]

        delta = delta_list[ipatch]
        event = event_list[ipatch]
        event_index = find_event_index(event, is_dead_pho)

        create_pair_3d(
            event_index, 
            x_pho, y_pho, z_pho, ux_pho, uy_pho, uz_pho, w_pho, delta,
            x_ele, y_ele, z_ele, ux_ele, uy_ele, uz_ele, inv_gamma_ele, w_ele, is_dead_ele, 
            x_pos, y_pos, z_pos, ux_pos, uy_pos, uz_pos, inv_gamma_pos, w_pos, is_dead_pos, 
        )

@njit(parallel=True, cache=True)
def photon_recoil_patches(
    ux_list, uy_list, uz_list, inv_gamma_list,
    event_list, delta_list, is_dead_list,
    npatches,
):
    for ipatch in prange(npatches):
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]
        event = event_list[ipatch]
        delta = delta_list[ipatch]
        is_dead = is_dead_list[ipatch]
        npart = len(is_dead)
        for ip in range(npart):
            if is_dead[ip]:
                continue
            if event[ip]:
                ux[ip] *= 1 - delta[ip]
                uy[ip] *= 1 - delta[ip]
                uz[ip] *= 1 - delta[ip]
                inv_gamma[ip] = (1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2) ** -0.5


@njit(parallel=True, cache=True)
def remove_photon_patches(
    event_list, is_dead_list, npatches
):
    for ipatch in prange(npatches):
        event = event_list[ipatch]
        is_dead = is_dead_list[ipatch]
        npart = len(is_dead)
        for ip in range(npart):
            if is_dead[ip]:
                continue
            if event[ip]:
                is_dead[ip] = True


@njit
def get_particle_extension_size(event, pho_is_dead):
    """
    Get the number to extend for the target particle.

    Parameters
    ----------
    event: bool array
        QED events of the source particle
    is_dead: bool array
        is_dead flag of the target particle.

    """
    npho = 0
    ndead = 0

    for event_ in event:
        npho += int(event_)
    for is_dead_ in pho_is_dead:
        ndead += int(is_dead_)

    if ndead < npho:
        return npho - ndead

    return 0

    

@njit(parallel=True)
def get_particle_extension_size_patches(event_list, is_dead_list, npatches):
    num_to_extend = np.zeros(npatches, dtype='int64')

    for ipatch in prange(npatches):
        event = event_list[ipatch]
        is_dead = is_dead_list[ipatch]

        # num_to_extend[ipatch] = get_particle_extension_size(event, is_dead)
        num_to_extend[ipatch] = max(0, event.sum() - is_dead.sum())

    return num_to_extend