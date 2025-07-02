import numpy as np
from scipy.constants import c, e, hbar, m_e


def calculate_chi_inline(Ex, Ey, Ez, Bx, By, Bz, ux, uy, uz, inv_gamma):
    factor = e*hbar / (m_e**2 * c**3)
    gamma = 1.0 / inv_gamma
    return factor * (
        (gamma*Ex + (uy*Bz - uz*By)*c)**2 +
        (gamma*Ey + (uz*Bx - ux*Bz)*c)**2 +
        (gamma*Ez + (ux*By - uy*Bx)*c)**2 -
        (ux*Ex + uy*Ey + uz*Ez)**2
    ) ** 0.5


def find_event_index_inline(event, is_dead):
    '''
    find an event index given an event array.
    
    Parameters
    ----------
    event: boolean array
        event array
    N_event: int64
        total number of events
    
    Returns
    -------
    event_index: int64 array
        1D array of event index
    '''

    n_event = 0
    for ip in range(len(event)):
        if is_dead[ip]:
            continue
        n_event += int(event[ip])

    event_index = np.zeros(n_event, dtype='i8')
    idx = 0
    for ip in range(len(event)):
        if is_dead[ip]:
            continue
        if event[ip]:
            event_index[idx] = ip
            idx += 1
    return event_index


def create_photon_inline_2d(
    x_ele, y_ele, ux_ele, uy_ele, uz_ele, w_ele, 
    x_pho, y_pho, ux_pho, uy_pho, uz_pho, inv_gamma_pho, w_pho, is_dead_pho, 
    event_index, delta,
):
    """
    create photon from electron to photon.
    
    Parameters
    ----------
    event_index: int64 array
        event index
    *_ele: float64 array
        electron
    *_pho: float64 array
        photon
    """

    idx_pho = 0
    for idx_ele in event_index:
        while not is_dead_pho[idx_pho]:
            idx_pho += 1
        x_pho[idx_pho] = x_ele[idx_ele]
        y_pho[idx_pho] = y_ele[idx_ele]
        
        ux_pho[idx_pho] = delta[idx_ele] * ux_ele[idx_ele]
        uy_pho[idx_pho] = delta[idx_ele] * uy_ele[idx_ele]
        uz_pho[idx_pho] = delta[idx_ele] * uz_ele[idx_ele]
        inv_gamma_pho[idx_pho] = (ux_pho[idx_pho]**2 + uy_pho[idx_pho]**2 + uz_pho[idx_pho]**2) ** -0.5
        w_pho[idx_pho] = w_ele[idx_ele]
        
        # mark created photon as existing
        is_dead_pho[idx_pho] = False

def create_photon_inline_3d(
    x_ele, y_ele, z_ele, ux_ele, uy_ele, uz_ele, w_ele, 
    x_pho, y_pho, z_pho, ux_pho, uy_pho, uz_pho, inv_gamma_pho, w_pho, is_dead_pho, 
    event_index, delta,
):
    """
    create photon from electron to photon.
    
    Parameters
    ----------
    event_index: int64 array
        event index
    *_ele: float64 array
        electron
    *_pho: float64 array
        photon
    """

    idx_pho = 0
    for idx_ele in event_index:
        while not is_dead_pho[idx_pho]:
            idx_pho += 1
        x_pho[idx_pho] = x_ele[idx_ele]
        y_pho[idx_pho] = y_ele[idx_ele]
        z_pho[idx_pho] = z_ele[idx_ele]
        
        ux_pho[idx_pho] = delta[idx_ele] * ux_ele[idx_ele]
        uy_pho[idx_pho] = delta[idx_ele] * uy_ele[idx_ele]
        uz_pho[idx_pho] = delta[idx_ele] * uz_ele[idx_ele]
        inv_gamma_pho[idx_pho] = (ux_pho[idx_pho]**2 + uy_pho[idx_pho]**2 + uz_pho[idx_pho]**2) ** -0.5
        w_pho[idx_pho] = w_ele[idx_ele]
        
        # mark created photon as existing
        is_dead_pho[idx_pho] = False
        
        
def create_pair_inline_2d(
    event_index, 
    x_pho, y_pho, ux_pho, uy_pho, uz_pho, w_pho, delta,
    x_ele, y_ele, ux_ele, uy_ele, uz_ele, inv_gamma_ele, w_ele, is_dead_ele, 
    x_pos, y_pos, ux_pos, uy_pos, uz_pos, inv_gamma_pos, w_pos, is_dead_pos, 
):
    idx_ele = 0
    idx_pos = 0
    for idx_pho in event_index:
        # electron
        while not is_dead_ele[idx_ele]:
            idx_ele += 1

        x_ele[idx_ele] = x_pho[idx_pho]
        y_ele[idx_ele] = y_pho[idx_pho]

        ux_ele[idx_ele] = delta[idx_pho] * ux_pho[idx_pho]
        uy_ele[idx_ele] = delta[idx_pho] * uy_pho[idx_pho]
        uz_ele[idx_ele] = delta[idx_pho] * uz_pho[idx_pho]
        inv_gamma_ele[idx_ele] = (1 + ux_ele[idx_ele]**2 + uy_ele[idx_ele]**2 + uz_ele[idx_ele]**2) ** -0.5
        w_ele[idx_ele] = w_pho[idx_pho]
        is_dead_ele[idx_ele] = False

        # positron
        while not is_dead_pos[idx_pos]:
            idx_pos += 1
        x_pos[idx_pos] = x_pho[idx_pho]
        y_pos[idx_pos] = y_pho[idx_pho]

        ux_pos[idx_pos] = (1 - delta[idx_pho]) * ux_pho[idx_pho]
        uy_pos[idx_pos] = (1 - delta[idx_pho]) * uy_pho[idx_pho]
        uz_pos[idx_pos] = (1 - delta[idx_pho]) * uz_pho[idx_pho]
        inv_gamma_pos[idx_pos] = (1 + ux_pos[idx_pos]**2 + uy_pos[idx_pos]**2 + uz_pos[idx_pos]**2) ** -0.5
        w_pos[idx_pos] = w_pho[idx_pho]
        is_dead_pos[idx_pos] = False

def create_pair_inline_3d(
    event_index, 
    x_pho, y_pho, z_pho, ux_pho, uy_pho, uz_pho, w_pho, delta,
    x_ele, y_ele, z_ele, ux_ele, uy_ele, uz_ele, inv_gamma_ele, w_ele, is_dead_ele, 
    x_pos, y_pos, z_pos, ux_pos, uy_pos, uz_pos, inv_gamma_pos, w_pos, is_dead_pos, 
):
    idx_ele = 0
    idx_pos = 0
    for idx_pho in event_index:
        # electron
        while not is_dead_ele[idx_ele]:
            idx_ele += 1

        x_ele[idx_ele] = x_pho[idx_pho]
        y_ele[idx_ele] = y_pho[idx_pho]
        z_ele[idx_ele] = z_pho[idx_pho]

        ux_ele[idx_ele] = delta[idx_pho] * ux_pho[idx_pho]
        uy_ele[idx_ele] = delta[idx_pho] * uy_pho[idx_pho]
        uz_ele[idx_ele] = delta[idx_pho] * uz_pho[idx_pho]
        inv_gamma_ele[idx_ele] = (1 + ux_ele[idx_ele]**2 + uy_ele[idx_ele]**2 + uz_ele[idx_ele]**2) ** -0.5
        w_ele[idx_ele] = w_pho[idx_pho]
        is_dead_ele[idx_ele] = False

        # positron
        while not is_dead_pos[idx_pos]:
            idx_pos += 1
        x_pos[idx_pos] = x_pho[idx_pho]
        y_pos[idx_pos] = y_pho[idx_pho]
        z_pos[idx_pos] = z_pho[idx_pho]

        ux_pos[idx_pos] = (1 - delta[idx_pho]) * ux_pho[idx_pho]
        uy_pos[idx_pos] = (1 - delta[idx_pho]) * uy_pho[idx_pho]
        uz_pos[idx_pos] = (1 - delta[idx_pho]) * uz_pho[idx_pho]
        inv_gamma_pos[idx_pos] = (1 + ux_pos[idx_pos]**2 + uy_pos[idx_pos]**2 + uz_pos[idx_pos]**2) ** -0.5
        w_pos[idx_pos] = w_pho[idx_pho]
        is_dead_pos[idx_pos] = False