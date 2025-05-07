import numpy as np
from numba import njit, prange
from scipy.constants import c, epsilon_0, mu_0


@njit(cache=True)
def update_efield_2d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    dx, dy, dt, 
    nx, ny, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for i in range(nx):
        for j in range(ny):
            ex[i, j] += bfactor * ( (bz[i, j] - bz[i, j-1]) / dy) - jfactor * jx[i, j]
            ey[i, j] += bfactor * (-(bz[i, j] - bz[i-1, j]) / dx) - jfactor * jy[i, j]
            ez[i, j] += bfactor * ( (by[i, j] - by[i-1, j]) / dx - (bx[i, j] - bx[i, j-1]) / dy) - jfactor * jz[i, j]


@njit(cache=True)
def update_bfield_2d(
    ex, ey, ez, 
    bx, by, bz, 
    dx, dy, dt, 
    nx, ny, n_guard
):
    for i in range(nx):
        for j in range(ny):
            bx[i, j] -= dt * ( (ez[i, j+1] - ez[i, j]) / dy)
            by[i, j] -= dt * (-(ez[i+1, j] - ez[i, j]) / dx)
            bz[i, j] -= dt * ( (ey[i+1, j] - ey[i, j]) / dx - (ex[i, j+1] - ex[i, j]) / dy)

@njit(cache=True, parallel=True)
def update_efield_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]

        update_efield_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, dt, nx, ny, n_guard)


@njit(cache=True, parallel=True)
def update_bfield_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        update_bfield_2d(ex, ey, ez, bx, by, bz, dx, dy, dt, nx, ny, n_guard)


'''3D functions'''
@njit(cache=True)
def update_efield_3d(
    ex, ey, ez,
    bx, by, bz,
    jx, jy, jz,
    dx, dy, dz, dt,
    nx, ny, nz, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ex[i, j, k] += bfactor * ((bz[i, j, k] - bz[i, j-1, k]) / dy - (by[i, j, k] - by[i, j, k-1]) / dz) - jfactor * jx[i, j ,k]
                ey[i, j, k] += bfactor * ((bx[i, j, k] - bx[i, j, k-1]) / dz - (bz[i, j, k] - bz[i-1, j, k]) / dx) - jfactor * jy[i, j, k]
                ez[i, j, k] += bfactor * ((by[i, j, k] - by[i-1, j, k]) / dx - (bx[i, j, k] - bx[i, j-1, k]) / dy) - jfactor * jz[i, j, k]


@njit(cache=True)
def update_bfield_3d(
    ex, ey, ez, 
    bx, by, bz,
    dx, dy, dz, dt, 
    nx, ny, nz, n_guard
):
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                bx[i, j, k] -= dt * ((ez[i, j+1, k] - ez[i, j, k]) / dy - (ey[i, j, k+1] - ey[i, j, k]) / dz)
                by[i, j, k] -= dt * ((ex[i, j, k+1] - ex[i, j, k]) / dz - (ez[i+1, j, k] - ez[i, j, k]) / dx)
                bz[i, j, k] -= dt * ((ey[i+1, j, k] - ey[i, j, k]) / dx - (ex[i, j+1, k] - ex[i, j, k]) / dy)


@njit(cache=True, parallel=True)
def update_efield_patches_3d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    npatches,
    dx, dy, dz, dt,
    nx, ny, nz, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]

        update_efield_3d(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, dz, dt, nx, ny, nz, n_guard)


@njit(cache=True, parallel=True)
def update_bfield_patches_3d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches,
    dx, dy, dz, dt,
    nx, ny, nz, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        update_bfield_3d(ex, ey, ez, bx, by, bz, dx, dy, dz, dt, nx, ny, nz, n_guard)