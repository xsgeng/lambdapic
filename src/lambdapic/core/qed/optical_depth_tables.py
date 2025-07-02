
import math
import multiprocessing
import os

import numpy as np
from numba import cfunc, njit
from scipy.constants import alpha, c, hbar, m_e, pi
from scipy.integrate import quad
from scipy.special import airy

# built-in tables
table_path = os.path.join(os.path.dirname(__file__), 'optical_depth_tables.npz')
if os.path.exists(table_path) and __name__ == "lambdapic.core.qed.optical_depth_tables":
    f = np.load(table_path)
    # 1d
    _photon_prob_rate_total_table = f['photon_prob_rate_total'][()]
    _pair_prob_rate_total_table = f['pair_prob_rate_total'][()]
    # 2d
    _integral_photon_prob_along_delta = f['integral_photon_prob_along_delta'][()]
    _integral_pair_prob_along_delta = f['integral_pair_prob_along_delta'][()]

    _chi_N = f['chi_N']
    _log_chi_range = f['log_chi_range']
    _log_chi_delta = f['log_chi_delta']
    _delta_N = f['delta_N']
    _log_delta_range = f['log_delta_range']
    _log_delta_delta = f['log_delta_delta']

    del f

@njit
def _get_chi_idx(chi):
    log_chi = np.log10(chi)
    if log_chi < _log_chi_range[0]:
        return -1
    if log_chi > _log_chi_range[1]:
        idx = _chi_N - 1
    if _log_chi_range[0] <= log_chi <= _log_chi_range[1]:
        idx = math.floor((log_chi - _log_chi_range[0]) / _log_chi_delta)

    return idx

@njit
def _linear_interp1d(chi, table1d):
    idx = _get_chi_idx(chi)
    if idx == -1:
        return 0.0
    log_chi_left = _log_chi_range[0] + idx*_log_chi_delta
    # linear interp
    log_chi = np.log10(chi)
    k = (table1d[idx+1] - table1d[idx]) / _log_chi_delta
    prob_rate = table1d[idx] + k * (log_chi-log_chi_left)
    return prob_rate


@njit
def _bisect_interp(chi, table2d):
    low, high = 0, _delta_N-1
    chi_idx = _get_chi_idx(chi)
    
    # Calculate fractional position between chi grid points
    log_chi = np.log10(chi)
    log_chi_left = _log_chi_range[0] + chi_idx * _log_chi_delta
    t = (log_chi - log_chi_left) / _log_chi_delta
    
    # Bilinear interpolation for ymin/ymax
    ymin = table2d[chi_idx, 0] * (1 - t) + table2d[chi_idx+1, 0] * t
    ymax = table2d[chi_idx, -1] * (1 - t) + table2d[chi_idx+1, -1] * t

    r = np.random.rand() * (ymax-ymin) + ymin
    while low <= high:
        mid = int((low + high)/2)
        mid_delta = table2d[chi_idx, mid] * (1 - t) + table2d[chi_idx+1, mid] * t

        if mid_delta < r:
            low = mid + 1
        elif mid_delta > r:
            high = mid - 1
    
    # interp
    delta_idx = high
    
    y1 = table2d[chi_idx, delta_idx] * (1 - t) + table2d[chi_idx+1, delta_idx] * t
    y2 = table2d[chi_idx, delta_idx+1] * (1 - t) + table2d[chi_idx+1, delta_idx+1] * t
    k = _log_delta_delta / (y2 - y1)
    log_delta_left = _log_delta_range[0] + delta_idx*_log_delta_delta
    log_delta = log_delta_left + k * (r - y1)   

    return 10**log_delta
    

@njit
def integ_photon_prob_rate_from_table(chi_e, photon_prob_rate_total_table):
    return _linear_interp1d(chi_e, photon_prob_rate_total_table)


@njit
def integ_pair_prob_rate_from_table(chi_gamma, pair_prob_rate_total_table):
    return _linear_interp1d(chi_gamma, pair_prob_rate_total_table)


@njit
def photon_delta_from_chi_delta_table(chi_e, integral_photon_prob_along_delta):
    return _bisect_interp(chi_e, integral_photon_prob_along_delta)

@njit
def pair_delta_from_chi_delta_table(chi_gamma, integral_pair_prob_along_delta):
    return _bisect_interp(chi_gamma, integral_pair_prob_along_delta)

'''
table generations
'''
def Ai(z):
    return airy(z)[0]

def Aip(z):
    return airy(z)[1]


def int_Ai(z):
    return quad(Ai, z, np.inf)[0]

def gen_photon_prob_rate_for_delta(chi_e):
    factor = -alpha*m_e*c**2/hbar
    def prob_(delta):
        if delta == 1.0: return 0.0
        chi_gamma = delta * chi_e
        chi_ep = chi_e - chi_gamma
        z = (chi_gamma/chi_e/chi_ep)**(2/3)
        return factor*(int_Ai(z) + (2.0/z + chi_gamma*np.sqrt(z)) * Aip(z))

    return prob_

def gen_pair_prob_rate_for_delta(chi_gamma):
    factor = alpha*m_e*c**2/hbar
    def prob_(delta):
        if delta == 1.0: return 0.0
        chi_e = delta * chi_gamma
        chi_ep = chi_gamma - chi_e
        z = (chi_gamma/chi_e/chi_ep)**(2/3)
        return factor*(int_Ai(z) + (2.0/z - chi_gamma*np.sqrt(z)) * Aip(z))

    return prob_

def integral_photon_prob_over_delta(chi_e):
    P = gen_photon_prob_rate_for_delta(chi_e)
    prob_rate_total, _ = quad(P, 0, 1)
    return prob_rate_total

def integral_pair_prob_over_delta(chi_gamma):
    P = gen_pair_prob_rate_for_delta(chi_gamma)
    prob_rate_total, _ = quad(P, 0, 1)
    return prob_rate_total

def integral_photon_prob_along_delta(chi_e, delta_N, log_delta_min):
    P = gen_photon_prob_rate_for_delta(chi_e)
    delta = np.logspace(log_delta_min, 0, delta_N)
    integ = np.zeros(delta_N)
    # 积分从delta_min开始
    integ[0] = quad(P, 0, delta[0])[0]
    for i in range(1, delta_N):
        integ[i] = integ[i-1] + P(delta[i]) * (delta[i] - delta[i-1])
    return integ

def integral_pair_prob_along_delta(chi_gamma, delta_N, log_delta_min):
    P = gen_pair_prob_rate_for_delta(chi_gamma)
    delta = np.logspace(log_delta_min, 0, delta_N)
    integ = np.zeros(delta_N)
    # 积分从delta_min开始
    integ[0] = quad(P, 0, delta[0])[0]
    for i in range(1, delta_N):
        integ[i] = integ[i-1] + P(delta[i]) * (delta[i] - delta[i-1])
    return integ

def photon_prob_rate_total(chi_N=256, log_chi_min=-3, log_chi_max=2):
    with multiprocessing.Pool() as pool:
        data = pool.map(integral_photon_prob_over_delta, np.logspace(log_chi_min, log_chi_max, chi_N))
    return np.array(data)

def pair_prob_rate_total(chi_N=256, log_chi_min=-3, log_chi_max=2):
    with multiprocessing.Pool() as pool:
        data = pool.map(integral_pair_prob_over_delta, np.logspace(log_chi_min, log_chi_max, chi_N))
    return np.array(data)


def table_gen(
    table_path, 
    chi_N=256, log_chi_min=-3.0, log_chi_max=2.0, 
    delta_N=1024, log_delta_min=-3,
):

    print("计算 不同chi_e的总辐射概率")
    _photon_prob_rate_total = photon_prob_rate_total(chi_N, log_chi_min, log_chi_max)

    print("计算不同chi_gamma的总电子对概率")
    _pair_prob_rate_total = pair_prob_rate_total(chi_N, log_chi_min, log_chi_max)

    chi = np.logspace(log_chi_min, log_chi_max, chi_N)
    with multiprocessing.Pool() as pool:
        print("计算不同chi_e辐射概率的积分")
        _integral_photon_prob_along_delta = pool.starmap(integral_photon_prob_along_delta, zip(chi, [delta_N]*chi_N, [log_delta_min]*chi_N))
        print("计算不同chi_gamma电子对概率的积分")
        _integral_pair_prob_along_delta = pool.starmap(integral_pair_prob_along_delta, zip(chi, [delta_N]*chi_N, [log_delta_min]*chi_N))

    np.savez(
        os.path.join(table_path, 'optical_depth_tables.npz'),
        photon_prob_rate_total=_photon_prob_rate_total,
        pair_prob_rate_total=_pair_prob_rate_total,
        integral_photon_prob_along_delta=_integral_photon_prob_along_delta,
        integral_pair_prob_along_delta=_integral_pair_prob_along_delta,
        chi_N=chi_N,
        log_chi_range=(log_chi_min, log_chi_max),
        log_chi_delta=(log_chi_max - log_chi_min) / (chi_N - 1),
        delta_N=delta_N,
        log_delta_range=(log_delta_min, 0),
        log_delta_delta=(0 - log_delta_min) / (delta_N - 1),
    )


if __name__ == '__main__':
    table_gen(os.path.dirname(__file__))
