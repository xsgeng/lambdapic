import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

""" Parallel functions for patches """
@njit(cache=False, parallel=True)
def get_num_macro_particles_2d(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc) -> NDArray[np.int64]:
    num_particles = np.zeros(npatches, dtype=np.int64)
    for ipatch in prange(npatches):
        ip = np.int64(ipatch)
        xaxis = xaxis_list[ip]
        yaxis = yaxis_list[ip]
        for x_grid in xaxis:
            for y_grid in yaxis:
                dens = density_func(x_grid, y_grid)
                if dens > dens_min:
                    num_particles[ipatch] += ppc
    return num_particles


@njit(cache=False, parallel=True)
def fill_particles_2d(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc, x_list, y_list, w_list):
    dx = xaxis_list[0][1] - xaxis_list[0][0]
    dy = yaxis_list[0][1] - yaxis_list[0][0]
    for ipatch in prange(npatches):
        ip = np.int64(ipatch)
        xaxis = xaxis_list[ip]
        yaxis = yaxis_list[ip]
        x = x_list[ip]
        y = y_list[ip]
        w = w_list[ip]
        ipart = 0
        for x_grid in xaxis:
            for y_grid in yaxis:
                dens = density_func(x_grid, y_grid)
                if dens > dens_min:
                    x[ipart:ipart+ppc] = np.random.uniform(-dx/2, dx/2, ppc) + x_grid
                    y[ipart:ipart+ppc] = np.random.uniform(-dy/2, dy/2, ppc) + y_grid
                    w[ipart:ipart+ppc] = dens*dx*dy / ppc
                    ipart += ppc


@njit(cache=False, parallel=True)
def get_num_macro_particles_3d(
    density_func, xaxis_list, yaxis_list, zaxis_list, npatches, dens_min, ppc
) -> NDArray[np.int64]:
    num_particles = np.zeros(npatches, dtype=np.int64)
    for ipatch in prange(npatches):
        ip = np.int64(ipatch)
        xaxis = xaxis_list[ip]
        yaxis = yaxis_list[ip]
        zaxis = zaxis_list[ip]
        for x_grid in xaxis:
            for y_grid in yaxis:
                for z_grid in zaxis:
                    dens = density_func(x_grid, y_grid, z_grid)
                    if dens > dens_min:
                        num_particles[ipatch] += ppc
    return num_particles


@njit(cache=False, parallel=True)
def fill_particles_3d(
    density_func, xaxis_list, yaxis_list, zaxis_list, npatches, dens_min, ppc, 
    x_list, y_list, z_list, w_list
):
    dx = xaxis_list[0][1] - xaxis_list[0][0]
    dy = yaxis_list[0][1] - yaxis_list[0][0]
    dz = zaxis_list[0][1] - zaxis_list[0][0]
    
    for ipatch in prange(npatches):
        ip = np.int64(ipatch)
        xaxis = xaxis_list[ip]
        yaxis = yaxis_list[ip]
        zaxis = zaxis_list[ip]
        x = x_list[ip]
        y = y_list[ip]
        z = z_list[ip]
        w = w_list[ip]
        ipart = 0
        
        for x_grid in xaxis:
            for y_grid in yaxis:
                for z_grid in zaxis:
                    dens = density_func(x_grid, y_grid, z_grid)
                    if dens > dens_min:
                        # Generate particles in 3D cell
                        x[ipart:ipart+ppc] = np.random.uniform(-dx/2, dx/2, ppc) + x_grid
                        y[ipart:ipart+ppc] = np.random.uniform(-dy/2, dy/2, ppc) + y_grid 
                        z[ipart:ipart+ppc] = np.random.uniform(-dz/2, dz/2, ppc) + z_grid
                        w[ipart:ipart+ppc] = dens*dx*dy*dz / ppc
                        ipart += ppc
