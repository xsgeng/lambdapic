# print("Using optical depth method.")
import os
from math import isnan

from numba import boolean, float64, int64, njit, prange, void
from numpy import log, random

if os.environ.get("LIBPIC_USE_SIGMOID_SAMPLING") == "1":
    from .optical_depth_tables_sigmoid import (
        _integral_photon_prob_along_delta,
        _integral_pair_prob_along_delta,
        _log_chi_range,
        _photon_prob_rate_total_table,
        _pair_prob_rate_total_table,
        integ_pair_prob_rate_from_table,
        integ_photon_prob_rate_from_table,
        pair_delta_from_chi_delta_table,
        photon_delta_from_chi_delta_table,
    )
    print("Using sigmoid sampling.")
else:
    from .optical_depth_tables import (
        _integral_photon_prob_along_delta,
        _integral_pair_prob_along_delta,
        _log_chi_range,
        _photon_prob_rate_total_table,
        _pair_prob_rate_total_table,
        integ_pair_prob_rate_from_table,
        integ_photon_prob_rate_from_table,
        pair_delta_from_chi_delta_table,
        photon_delta_from_chi_delta_table,
    )


_chi_min = 10.0**_log_chi_range[0]
@njit( void(float64[:], float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:], float64[:, :], float64[:]),
    cache=False, inline="always")
def update_tau_e(tau_e, inv_gamma, chi_e, dt, npart, is_dead, event, delta, integral_photon_prob_along_delta, photon_prob_rate_total_table):
    '''
    update optical depth tau of electron
    
    Parameters
    ----------
    tau_e : array of floats
        tau of electron
    inv_gamma : array of floats
        inverse of gamma factor
    chi_e : array of floats
        chi of electron
    dt : float
        dt
    N : int
        total number of particles, is_dead included
    is_dead : array of booleans
        to be pruned flag
    event : array of booleans
        event flag
    delta : array of floats
        photon delta
    '''
    for ip in range(npart):
        if is_dead[ip] or chi_e[ip]  < _chi_min:
            event[ip] = False
            delta[ip] = 0.0
            continue
        integ_prob_rate = integ_photon_prob_rate_from_table(chi_e[ip], photon_prob_rate_total_table)
        dtau = dt * inv_gamma[ip]

        # reset if not set
        if tau_e[ip] == 0.0 or isnan(tau_e[ip]):
            tau_e[ip] = -log(1 - random.rand())

        tau_e[ip] -= integ_prob_rate * dtau

        if tau_e[ip] < 0:
            tau_e[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = photon_delta_from_chi_delta_table(chi_e[ip], integral_photon_prob_along_delta)
        else:
            event[ip] = False
            delta[ip] = 0.0

@njit(
    void(float64[:], float64[:], float64[:], float64, int64, boolean[:], boolean[:], float64[:], float64[:, :], float64[:]),
    cache=False
)
def update_tau_gamma(tau_gamma, inv_gamma, chi_gamma, dt, npart, is_dead, event, delta, integral_pair_prob_along_delta, pair_prob_rate_total_table):
    '''
    update optical depth tau of gamma photon
    
    Parameters
    ----------
    tau_gamma : array of floats
        tau of gamma photon
    inv_gamma : array of floats
        inverse of gamma factor
    chi_gamma : array of floats
        chi of gamma photon
    dt : float
        dt
    N : int
        total number of particles, is_dead included
    is_dead : array of booleans
        to be pruned flag
    event : array of booleans
        event flag
    delta : array of floats
        pair delta
    '''
    for ip in range(npart):
        if is_dead[ip] or chi_gamma[ip] < _chi_min:
            event[ip] = False
            delta[ip] = 0.0
            continue
        integ_prob_rate = integ_pair_prob_rate_from_table(chi_gamma[ip], pair_prob_rate_total_table)
        dtau = dt * inv_gamma[ip]

        # reset if not set
        if tau_gamma[ip] == 0.0 or isnan(tau_gamma[ip]):
            tau_gamma[ip] = -log(1 - random.rand())

        tau_gamma[ip] -= integ_prob_rate * dtau

        if tau_gamma[ip] <= 0:
            tau_gamma[ip] = -log(1 - random.rand())
            event[ip] = True
            delta[ip] = pair_delta_from_chi_delta_table(chi_gamma[ip], integral_pair_prob_along_delta)
        else:
            event[ip] = False
            delta[ip] = 0.0