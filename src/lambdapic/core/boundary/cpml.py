import numpy as np
from numba import njit, prange
from scipy.constants import c, epsilon_0

from ..fields import Fields, Fields2D, Fields3D


class Boundary:
    ...

class PML(Boundary):
    """ A perfectly matched layer (PML)

    PML is an extended area between real simulation grids and ghost/guard cells

    """

    efield_start: int
    efield_end: int
    bfield_start: int
    bfield_end: int

    def __init__(
        self, 
        fields: Fields,
        thickness : int=6, 
        kappa_max : float=20.0, 
        a_max:float=0.15, 
        sigma_max:float=0.7
    ) -> None:
        """ Perfectly Matched Layer

        Args:
            dimensions: list of inteters
                the dimensions of the main grid
            thickness: 
                the thickness of the PML
            kappa_max: 
                the maximum value of the electric conductivity
            a_max: 
                the maximum value of the parameter a
            sigma_max: 
                the maximum value of the parameter sigma
        """

        self.fields = fields
        self.nx = fields.nx
        self.dx = fields.dx
        self.n_guard = fields.n_guard

        self.thickness = thickness
        self.kappa_max = kappa_max
        self.a_max = a_max
        self.sigma_max = sigma_max

        self.cpml_m: int = 3
        self.cpml_ma: int = 1
        self.sigma_maxval: float = sigma_max * c * 0.8* (self.cpml_m + 1.0) / self.dx


        if isinstance(fields, Fields2D):
            self.ny = fields.ny
            self.dy = fields.dy
            self.dimensions = (fields.nx, fields.ny)
            shapex = fields.nx
            shapey = fields.ny
            shapez = 0
        elif isinstance(fields, Fields3D):
            self.ny = fields.ny
            self.nz = fields.nz
            self.dy = fields.dy
            self.dz = fields.dz
            self.dimensions = (fields.nx, fields.ny, fields.nz)
            shapex = fields.nx
            shapey = fields.ny
            shapez = fields.nz
        else:
            raise NotImplementedError

        self.kappa_ex = np.ones(shapex)
        self.kappa_bx = np.ones(shapex)
        self.a_ex = np.zeros(shapex)
        self.a_bx = np.zeros(shapex)
        self.sigma_ex = np.zeros(shapex)
        self.sigma_bx = np.zeros(shapex)


        self.kappa_ey = np.ones(shapey)
        self.kappa_by = np.ones(shapey)
        self.a_ey = np.zeros(shapey)
        self.a_by = np.zeros(shapey)
        self.sigma_ey = np.zeros(shapey)
        self.sigma_by = np.zeros(shapey)

        self.kappa_ez = np.ones(shapez)
        self.kappa_bz = np.ones(shapez)
        self.a_ez = np.zeros(shapez)
        self.a_bz = np.zeros(shapez)
        self.sigma_ez = np.zeros(shapez)
        self.sigma_bz = np.zeros(shapez)

        self.init_parameters()


    def init_parameters(self) -> None:
        """
        Init parameters, will be called by inherted PMLs.
        """
        raise NotImplementedError

    def init_coefficents(self, pos: np.ndarray, s: slice, kappa: np.ndarray, sigma: np.ndarray, a: np.ndarray) -> None:
        pos_m = pos**self.cpml_m
        pos_ma = (1 - pos)**self.cpml_ma

        kappa[s] = 1 + (self.kappa_max - 1) * pos_m
        sigma[s] = self.sigma_maxval * pos_m
        a[s] = self.a_max * pos_ma

    def advance_e_currents(self, dt):
        """ Advance the CPML psi_e """
        raise NotImplementedError
    
    def advance_b_currents(self, dt):
        """ Advance the CPML psi_b """
        raise NotImplementedError

class PMLX(PML):
    def __init__(self, fields: Fields, thickness: int = 6, kappa_max: float = 20, a_max: float = 0.15, sigma_max: float = 0.7) -> None:
        super().__init__(fields, thickness, kappa_max, a_max, sigma_max)
        self.psi_ey_x = np.zeros(self.dimensions)
        self.psi_ez_x = np.zeros(self.dimensions)
        self.psi_by_x = np.zeros(self.dimensions)
        self.psi_bz_x = np.zeros(self.dimensions)

    def advance_e_currents(self, dt):
        if isinstance(self.fields, Fields3D):
            update_psi_x_and_e_3d(self.kappa_ex, self.sigma_ex, self.a_ex,
                                self.ny, self.nz, dt, self.dx,
                                self.efield_start, self.efield_end,
                                self.fields.by, self.fields.bz,
                                self.fields.ey, self.fields.ez,
                                self.psi_ey_x, self.psi_ez_x)
        else:
            update_psi_x_and_e_2d(self.kappa_ex, self.sigma_ex, self.a_ex, self.ny, dt, self.dx, self.efield_start, self.efield_end, 
                               self.fields.by, self.fields.bz, self.fields.ey, self.fields.ez, self.psi_ey_x, self.psi_ez_x)

    def advance_b_currents(self, dt):
        if isinstance(self.fields, Fields3D):
            update_psi_x_and_b_3d(self.kappa_bx, self.sigma_bx, self.a_bx,
                                self.ny, self.nz, dt, self.dx,
                                self.bfield_start, self.bfield_end,
                                self.fields.ey, self.fields.ez,
                                self.fields.by, self.fields.bz,
                                self.psi_by_x, self.psi_bz_x)
        else:
            update_psi_x_and_b_2d(self.kappa_bx, self.sigma_bx, self.a_bx, self.ny, dt, self.dx, self.bfield_start, self.bfield_end, 
                               self.fields.ey, self.fields.ez, self.fields.by, self.fields.bz, self.psi_by_x, self.psi_bz_x)

class PMLY(PML):
    def __init__(self, fields: Fields, thickness: int = 6, kappa_max: float = 20, a_max: float = 0.15, sigma_max: float = 0.7) -> None:
        super().__init__(fields, thickness, kappa_max, a_max, sigma_max)
        self.psi_ex_y = np.zeros(self.dimensions)
        self.psi_ez_y = np.zeros(self.dimensions)
        self.psi_bx_y = np.zeros(self.dimensions)
        self.psi_bz_y = np.zeros(self.dimensions)

    def advance_e_currents(self, dt):
        if isinstance(self.fields, Fields3D):
            update_psi_y_and_e_3d(
                self.kappa_ey, self.sigma_ey, self.a_ey,
                self.nx, self.nz, dt, self.dy,
                self.efield_start, self.efield_end,
                self.fields.bx, self.fields.bz,
                self.fields.ex, self.fields.ez,
                self.psi_ex_y, self.psi_ez_y
            )
        else:
            update_psi_y_and_e_2d(self.kappa_ey, self.sigma_ey, self.a_ey, self.nx, dt, self.dy, self.efield_start, self.efield_end, 
                           self.fields.bx, self.fields.bz, self.fields.ex, self.fields.ez, self.psi_ex_y, self.psi_ez_y)

    def advance_b_currents(self, dt):
        if isinstance(self.fields, Fields3D):
            update_psi_y_and_b_3d(
                self.kappa_by, self.sigma_by, self.a_by,
                self.nx, self.nz, dt, self.dy,
                self.bfield_start, self.bfield_end,
                self.fields.ex, self.fields.ez,
                self.fields.bx, self.fields.bz,
                self.psi_bx_y, self.psi_bz_y
            )
        else:
            update_psi_y_and_b_2d(self.kappa_by, self.sigma_by, self.a_by, self.nx, dt, self.dy, self.bfield_start, self.bfield_end, 
                           self.fields.ex, self.fields.ez, self.fields.bx, self.fields.bz, self.psi_bx_y, self.psi_bz_y)

class PMLZ(PML):
    def __init__(self, fields: Fields, thickness: int = 6, kappa_max: float = 20, 
                a_max: float = 0.15, sigma_max: float = 0.7) -> None:
        super().__init__(fields, thickness, kappa_max, a_max, sigma_max)
        self.psi_ex_z = np.zeros(self.dimensions)
        self.psi_ey_z = np.zeros(self.dimensions)
        self.psi_bx_z = np.zeros(self.dimensions)
        self.psi_by_z = np.zeros(self.dimensions)

    def advance_e_currents(self, dt):
        if isinstance(self.fields, Fields3D):
            update_psi_z_and_e_3d(self.kappa_ez, self.sigma_ez, self.a_ez, 
                                self.nx, self.ny, dt, self.dz,
                                self.efield_start, self.efield_end,
                                self.fields.bx, self.fields.by,
                                self.fields.ex, self.fields.ey,
                                self.psi_ex_z, self.psi_ey_z)

    def advance_b_currents(self, dt: float) -> None:
        if isinstance(self.fields, Fields3D):
            update_psi_z_and_b_3d(self.kappa_bz, self.sigma_bz, self.a_bz,
                                self.nx, self.ny, dt, self.dz,
                                self.bfield_start, self.bfield_end,
                                self.fields.ex, self.fields.ey,
                                self.fields.bx, self.fields.by,
                                self.psi_bx_z, self.psi_by_z)
                                
class PMLXmin(PMLX):
    def init_parameters(self):
        # runs from 1.0 to nearly 0.0 (actually 0.0 at cpml_thickness+1)
        pos = 1.0 - np.arange(self.thickness, dtype=float) / self.thickness
        cpml_slice = np.s_[:self.thickness]
        self.init_coefficents(pos, cpml_slice, self.kappa_ex, self.sigma_ex, self.a_ex)

        # runs from nearly 1.0 to nearly 0.0 on the half intervals
        # 1.0 at ix_glob=1-1/2 and 0.0 at ix_glob=cpml_thickness+1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5) / self.thickness
        self.init_coefficents(pos, cpml_slice, self.kappa_bx, self.sigma_bx, self.a_bx)
        
        # pml range
        self.efield_start = 0
        self.efield_end = self.thickness
        self.bfield_start = 0
        self.bfield_end = self.thickness


class PMLXmax(PMLX):
    def init_parameters(self):
        # runs from nearly 0.0 (actually 0.0 at cpml_thickness+1) to 1.0
        pos = 1.0 - np.arange(self.thickness, dtype=float)[::-1] / self.thickness
        cpml_slice = np.s_[self.nx-self.thickness : self.nx]
        self.init_coefficents(pos, cpml_slice, self.kappa_ex, self.sigma_ex, self.a_ex)

        # runs from nearly 0.0 to nearly 1.0 on the half intervals
        # 0.0 at ix_glob=cpml_thickness+1/2 and 1.0 at ix_glob=1-1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5)[::-1] / self.thickness
        cpml_slice = np.s_[self.nx-self.thickness-1 : self.nx-1]
        self.init_coefficents(pos, cpml_slice, self.kappa_bx, self.sigma_bx, self.a_bx)

        # pml range
        self.efield_start = self.nx - self.thickness
        self.efield_end = self.nx
        self.bfield_start = self.nx - self.thickness - 1
        self.bfield_end = self.nx - 1

class PMLYmin(PMLY):
    def init_parameters(self):
        # runs from 1.0 to nearly 0.0 (actually 0.0 at cpml_thickness+1)
        pos = 1.0 - np.arange(self.thickness, dtype=float) / self.thickness
        cpml_slice = np.s_[:self.thickness]
        self.init_coefficents(pos, cpml_slice, self.kappa_ey, self.sigma_ey, self.a_ey)

        # runs from nearly 1.0 to nearly 0.0 on the half intervals
        # 1.0 at iy_glob=1-1/2 and 0.0 at iy_glob=cpml_thickness+1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5) / self.thickness
        self.init_coefficents(pos, cpml_slice, self.kappa_by, self.sigma_by, self.a_by)
        
        # pml range
        self.efield_start = 0
        self.efield_end = self.thickness
        self.bfield_start = 0
        self.bfield_end = self.thickness

class PMLYmax(PMLY):
    def init_parameters(self):
        # runs from nearly 0.0 (actually 0.0 at cpml_thickness+1) to 1.0
        pos = 1.0 - np.arange(self.thickness, dtype=float)[::-1] / self.thickness
        cpml_slice = np.s_[self.ny-self.thickness : self.ny]
        self.init_coefficents(pos, cpml_slice, self.kappa_ey, self.sigma_ey, self.a_ey)

        # runs from nearly 0.0 to nearly 1.0 on the half intervals
        # 0.0 at iy_glob=cpml_thickness+1/2 and 1.0 at iy_glob=1-1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5)[::-1] / self.thickness
        cpml_slice = np.s_[self.ny-self.thickness-1 : self.ny-1]
        self.init_coefficents(pos, cpml_slice, self.kappa_by, self.sigma_by, self.a_by)

        # pml range
        self.efield_start = self.ny - self.thickness
        self.efield_end = self.ny
        self.bfield_start = self.ny - self.thickness - 1
        self.bfield_end = self.ny - 1

class PMLZmin(PMLZ):
    def init_parameters(self):
        pos = 1.0 - np.arange(self.thickness, dtype=float) / self.thickness
        cpml_slice = np.s_[:self.thickness]
        self.init_coefficents(pos, cpml_slice, self.kappa_ez, self.sigma_ez, self.a_ez)

        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5) / self.thickness
        cpml_slice = np.s_[:self.thickness]
        self.init_coefficents(pos, cpml_slice, self.kappa_bz, self.sigma_bz, self.a_bz)

        self.efield_start = 0
        self.efield_end = self.thickness
        self.bfield_start = 0
        self.bfield_end = self.thickness

class PMLZmax(PMLZ):
    def init_parameters(self):
        # runs from nearly 0.0 (actually 0.0 at cpml_thickness+1) to 1.0
        pos = 1.0 - np.arange(self.thickness, dtype=float)[::-1] / self.thickness
        cpml_slice = np.s_[self.nz-self.thickness : self.nz]
        self.init_coefficents(pos, cpml_slice, self.kappa_ez, self.sigma_ez, self.a_ez)

        # runs from nearly 0.0 to nearly 1.0 on the half intervals
        # 0.0 at iz_glob=cpml_thickness+1/2 and 1.0 at iz_glob=1-1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5)[::-1] / self.thickness
        cpml_slice = np.s_[self.nz-self.thickness-1 : self.nz-1]
        self.init_coefficents(pos, cpml_slice, self.kappa_bz, self.sigma_bz, self.a_bz)

        # pml range
        self.efield_start = self.nz - self.thickness
        self.efield_end = self.nz
        self.bfield_start = self.nz - self.thickness - 1
        self.bfield_end = self.nz - 1

@njit
def update_efield_cpml_2d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    kappa_ex, kappa_ey,
    dx, dy, dt, 
    nx, ny, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for i in range(nx):
        bfactor_x = bfactor / kappa_ex[i]
        for j in range(ny):
            bfactor_y = bfactor / kappa_ey[j]
            ex[i, j] += bfactor_y * ( (bz[i, j] - bz[i, j-1]) / dy) - jfactor * jx[i, j]
            ey[i, j] += bfactor_x * (-(bz[i, j] - bz[i-1, j]) / dx) - jfactor * jy[i, j]
            ez[i, j] += bfactor_x * ( (by[i, j] - by[i-1, j]) / dx) \
                      - bfactor_y * ( (bx[i, j] - bx[i, j-1]) / dy) - jfactor * jz[i, j]


@njit
def update_bfield_cpml_2d(
    ex, ey, ez, 
    bx, by, bz, 
    kappa_bx, kappa_by,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for i in range(nx):
        efactor_x = dt / kappa_bx[i]
        for j in range(ny):
            efactor_y = dt / kappa_by[j]
            bx[i, j] -= efactor_y * ( (ez[i, j+1] - ez[i, j]) / dy)
            by[i, j] -= efactor_x * (-(ez[i+1, j] - ez[i, j]) / dx)
            bz[i, j] -= efactor_x * ( (ey[i+1, j] - ey[i, j]) / dx) \
                      - efactor_y * ( (ex[i, j+1] - ex[i, j]) / dy)



@njit(cache=True, parallel=True)
def update_efield_cpml_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    kappa_ex_list, kappa_ey_list,
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
        kappa_ex = kappa_ex_list[ipatch]
        kappa_ey = kappa_ey_list[ipatch]

        update_efield_cpml_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, kappa_ex, kappa_ey, dx, dy, dt, nx, ny, n_guard)


@njit(cache=True, parallel=True)
def update_bfield_cpml_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    kappa_bx_list, kappa_by_list,
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
        kappa_bx = kappa_bx_list[ipatch]
        kappa_by = kappa_by_list[ipatch]

        update_bfield_cpml_2d(ex, ey, ez, bx, by, bz, kappa_bx, kappa_by, dx, dy, dt, nx, ny, n_guard)

@njit
def update_efield_cpml_3d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    kappa_ex, kappa_ey, kappa_ez,
    dx, dy, dz, dt, 
    nx, ny, nz, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for i in range(nx):
        bfactor_x = bfactor / kappa_ex[i]
        for j in range(ny):
            bfactor_y = bfactor / kappa_ey[j]
            for k in range(nz):
                bfactor_z = bfactor / kappa_ez[k]
                ex[i,j,k] += (bfactor_y*(bz[i,j,k] - bz[i,j-1,k])/dy 
                            - bfactor_z*(by[i,j,k] - by[i,j,k-1])/dz) - jfactor*jx[i,j,k]
                ey[i,j,k] += (bfactor_z*(bx[i,j,k] - bx[i,j,k-1])/dz 
                            - bfactor_x*(bz[i,j,k] - bz[i-1,j,k])/dx) - jfactor*jy[i,j,k]
                ez[i,j,k] += (bfactor_x*(by[i,j,k] - by[i-1,j,k])/dx 
                            - bfactor_y*(bx[i,j,k] - bx[i,j-1,k])/dy) - jfactor*jz[i,j,k]

@njit
def update_bfield_cpml_3d(
    ex, ey, ez, 
    bx, by, bz, 
    kappa_bx, kappa_by, kappa_bz,
    dx, dy, dz, dt, 
    nx, ny, nz, n_guard
):
    for i in range(nx):
        efactor_x = dt / kappa_bx[i]
        for j in range(ny):
            efactor_y = dt / kappa_by[j]
            for k in range(nz):
                efactor_z = dt / kappa_bz[k]
                bx[i,j,k] -= (efactor_y*(ez[i,j+1,k] - ez[i,j,k])/dy 
                            - efactor_z*(ey[i,j,k+1] - ey[i,j,k])/dz)
                by[i,j,k] -= (efactor_z*(ex[i,j,k+1] - ex[i,j,k])/dz 
                            - efactor_x*(ez[i+1,j,k] - ez[i,j,k])/dx)
                bz[i,j,k] -= (efactor_x*(ey[i+1,j,k] - ey[i,j,k])/dx 
                            - efactor_y*(ex[i,j+1,k] - ex[i,j,k])/dy)

@njit(cache=True, parallel=True)
def update_efield_cpml_patches_3d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    kappa_ex_list, kappa_ey_list, kappa_ez_list,
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
        kappa_ex = kappa_ex_list[ipatch]
        kappa_ey = kappa_ey_list[ipatch]
        kappa_ez = kappa_ez_list[ipatch]

        update_efield_cpml_3d(ex, ey, ez, bx, by, bz, jx, jy, jz, 
                            kappa_ex, kappa_ey, kappa_ez, 
                            dx, dy, dz, dt, nx, ny, nz, n_guard)

@njit(cache=True, parallel=True)
def update_bfield_cpml_patches_3d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    kappa_bx_list, kappa_by_list, kappa_bz_list,
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
        kappa_bx = kappa_bx_list[ipatch]
        kappa_by = kappa_by_list[ipatch]
        kappa_bz = kappa_bz_list[ipatch]

        update_bfield_cpml_3d(ex, ey, ez, bx, by, bz, 
                            kappa_bx, kappa_by, kappa_bz,
                            dx, dy, dz, dt, nx, ny, nz, n_guard)

@njit(cache=True)
def update_psi_x_and_e_2d(kappa, sigma, a, ny, dt, dx, start, stop, by, bz, ey, ez, psi_ey_x, psi_ez_x):
    fac = dt * c**2
    for ipos in range(start, stop):
        kappa_ = kappa[ipos]
        sigma_ = sigma[ipos]
        acoeff = a[ipos]
        bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
        ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dx
        for iy in range(ny):

            psi_ey_x[ipos, iy] = bcoeff * psi_ey_x[ipos, iy] \
                + ccoeff_d * (bz[ipos, iy] - bz[ipos-1, iy])
            psi_ez_x[ipos, iy] = bcoeff * psi_ez_x[ipos, iy] \
                + ccoeff_d * (by[ipos, iy] - by[ipos-1, iy])

            ey[ipos, iy] -= fac * psi_ey_x[ipos, iy]
            ez[ipos, iy] += fac * psi_ez_x[ipos, iy]

@njit(cache=True)
def update_psi_x_and_b_2d(kappa, sigma, a, ny, dt, dx, start, stop, ey, ez, by, bz, psi_by_x, psi_bz_x):
    fac = dt
    for ipos in range(start, stop):
        kappa_ = kappa[ipos]
        sigma_ = sigma[ipos]
        acoeff = a[ipos]
        bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
        ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dx
        for iy in range(ny):

            psi_by_x[ipos, iy] = bcoeff * psi_by_x[ipos, iy] \
                + ccoeff_d * (ez[ipos+1, iy] - ez[ipos, iy])
            psi_bz_x[ipos, iy] = bcoeff * psi_bz_x[ipos, iy] \
                + ccoeff_d * (ey[ipos+1, iy] - ey[ipos, iy])

            by[ipos, iy] += fac * psi_by_x[ipos, iy]
            bz[ipos, iy] -= fac * psi_bz_x[ipos, iy]

@njit(cache=True)
def update_psi_y_and_e_2d(kappa, sigma, a, nx, dt, dy, start, stop, bx, bz, ex, ez, psi_ex_y, psi_ez_y):
    fac = dt * c**2
    for ix in range(nx):
        for ipos in range(start, stop):
            kappa_ = kappa[ipos]
            sigma_ = sigma[ipos]
            acoeff = a[ipos]
            bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
            ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dy

            psi_ex_y[ix, ipos] = bcoeff * psi_ex_y[ix, ipos] \
                + ccoeff_d * (bz[ix, ipos] - bz[ix, ipos-1])
            psi_ez_y[ix, ipos] = bcoeff * psi_ez_y[ix, ipos] \
                + ccoeff_d * (bx[ix, ipos] - bx[ix, ipos-1])

            ex[ix, ipos] += fac * psi_ex_y[ix, ipos]
            ez[ix, ipos] -= fac * psi_ez_y[ix, ipos]

@njit(cache=True)
def update_psi_y_and_b_2d(kappa, sigma, a, nx, dt, dy, start, stop, ex, ez, bx, bz, psi_bx_y, psi_bz_y):
    fac = dt
    for ix in range(nx):
        for ipos in range(start, stop):
            kappa_ = kappa[ipos]
            sigma_ = sigma[ipos]
            acoeff = a[ipos]
            bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
            ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dy

            psi_bx_y[ix, ipos] = bcoeff * psi_bx_y[ix, ipos] \
                + ccoeff_d * (ez[ix, ipos+1] - ez[ix, ipos])
            psi_bz_y[ix, ipos] = bcoeff * psi_bz_y[ix, ipos] \
                + ccoeff_d * (ex[ix, ipos+1] - ex[ix, ipos])

            bx[ix, ipos] -= fac * psi_bx_y[ix, ipos]
            bz[ix, ipos] += fac * psi_bz_y[ix, ipos]

@njit(parallel=True, cache=True)
def update_psi_x_and_e_3d(kappa, sigma, a, ny, nz, dt, dx, start, stop, by, bz, ey, ez, psi_ey_x, psi_ez_x):
    fac = dt * c**2
    for ipos in range(start, stop):
        kappa_ = kappa[ipos]
        sigma_ = sigma[ipos]
        acoeff = a[ipos]
        bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
        ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dx
        for iy in prange(ny):
            for iz in range(nz):

                psi_ey_x[ipos, iy, iz] = bcoeff * psi_ey_x[ipos, iy, iz] \
                    + ccoeff_d * (bz[ipos, iy, iz] - bz[ipos-1, iy, iz])
                psi_ez_x[ipos, iy, iz] = bcoeff * psi_ez_x[ipos, iy, iz] \
                    + ccoeff_d * (by[ipos, iy, iz] - by[ipos-1, iy, iz])

                ey[ipos, iy, iz] -= fac * psi_ey_x[ipos, iy, iz]
                ez[ipos, iy, iz] += fac * psi_ez_x[ipos, iy, iz]

@njit(parallel=True, cache=True)
def update_psi_x_and_b_3d(kappa, sigma, a, ny, nz, dt, dx, start, stop, ey, ez, by, bz, psi_by_x, psi_bz_x):
    fac = dt
    for ipos in range(start, stop):
        kappa_ = kappa[ipos]
        sigma_ = sigma[ipos]
        acoeff = a[ipos]
        bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
        ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dx
        for iy in prange(ny):
            for iz in range(nz):

                psi_by_x[ipos, iy, iz] = bcoeff * psi_by_x[ipos, iy, iz] \
                    + ccoeff_d * (ez[ipos+1, iy, iz] - ez[ipos, iy, iz])
                psi_bz_x[ipos, iy, iz] = bcoeff * psi_bz_x[ipos, iy, iz] \
                    + ccoeff_d * (ey[ipos+1, iy, iz] - ey[ipos, iy, iz])

                by[ipos, iy, iz] += fac * psi_by_x[ipos, iy, iz]
                bz[ipos, iy, iz] -= fac * psi_bz_x[ipos, iy, iz]

@njit(parallel=True, cache=True)
def update_psi_y_and_e_3d(kappa, sigma, a, nx, nz, dt, dy, start, stop, bx, bz, ex, ez, psi_ex_y, psi_ez_y):
    fac = dt * c**2
    for ix in prange(nx):
        for ipos in range(start, stop):
            kappa_ = kappa[ipos]
            sigma_ = sigma[ipos]
            acoeff = a[ipos]
            bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
            ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dy
            for iz in range(nz):
                psi_ex_y[ix, ipos, iz] = bcoeff * psi_ex_y[ix, ipos, iz] \
                    + ccoeff_d * (bz[ix, ipos, iz] - bz[ix, ipos-1, iz])
                psi_ez_y[ix, ipos, iz] = bcoeff * psi_ez_y[ix, ipos, iz] \
                    + ccoeff_d * (bx[ix, ipos, iz] - bx[ix, ipos-1, iz])

                ex[ix, ipos, iz] += fac * psi_ex_y[ix, ipos, iz]
                ez[ix, ipos, iz] -= fac * psi_ez_y[ix, ipos, iz]

@njit(parallel=True, cache=True)
def update_psi_y_and_b_3d(kappa, sigma, a, nx, nz, dt, dy, start, stop, ex, ez, bx, bz, psi_bx_y, psi_bz_y):
    fac = dt
    for ix in prange(nx):
        for ipos in range(start, stop):
            kappa_ = kappa[ipos]
            sigma_ = sigma[ipos]
            acoeff = a[ipos]
            bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
            ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dy
            for iz in range(nz):
                psi_bx_y[ix, ipos, iz] = bcoeff * psi_bx_y[ix, ipos, iz] \
                    + ccoeff_d * (ez[ix, ipos+1, iz] - ez[ix, ipos, iz])
                psi_bz_y[ix, ipos, iz] = bcoeff * psi_bz_y[ix, ipos, iz] \
                    + ccoeff_d * (ex[ix, ipos+1, iz] - ex[ix, ipos, iz])

                bx[ix, ipos, iz] -= fac * psi_bx_y[ix, ipos, iz]
                bz[ix, ipos, iz] += fac * psi_bz_y[ix, ipos, iz]

@njit(parallel=True, cache=True)
def update_psi_z_and_e_3d(kappa, sigma, a, nx, ny, dt, dz, start, stop, bx, by, ex, ey, psi_ex_z, psi_ey_z):
    fac = dt * c**2
    for ix in prange(nx):
        for iy in range(ny):
            for ipos in range(start, stop):
                kappa_ = kappa[ipos]
                sigma_ = sigma[ipos]
                acoeff = a[ipos]
                bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
                ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dz

                psi_ex_z[ix, iy, ipos] = bcoeff * psi_ex_z[ix, iy, ipos] \
                    + ccoeff_d * (by[ix, iy, ipos] - by[ix, iy, ipos-1])
                psi_ey_z[ix, iy, ipos] = bcoeff * psi_ey_z[ix, iy, ipos] \
                    + ccoeff_d * (bx[ix, iy, ipos] - bx[ix, iy, ipos-1])

                ex[ix, iy, ipos] += fac * psi_ex_z[ix, iy, ipos]
                ey[ix, iy, ipos] -= fac * psi_ey_z[ix, iy, ipos]

@njit(parallel=True, cache=True)
def update_psi_z_and_b_3d(kappa, sigma, a, nx, ny, dt, dz, start, stop, ex, ey, bx, by, psi_bx_z, psi_by_z):
    fac = dt
    for ix in prange(nx):
        for iy in range(ny):
            for ipos in range(start, stop):
                kappa_ = kappa[ipos]
                sigma_ = sigma[ipos]
                acoeff = a[ipos]
                bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
                ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dz

                psi_bx_z[ix, iy, ipos] = bcoeff * psi_bx_z[ix, iy, ipos] \
                    + ccoeff_d * (ey[ix, iy, ipos+1] - ey[ix, iy, ipos])
                psi_by_z[ix, iy, ipos] = bcoeff * psi_by_z[ix, iy, ipos] \
                    + ccoeff_d * (ex[ix, iy, ipos+1] - ex[ix, iy, ipos])

                bx[ix, iy, ipos] += fac * psi_bx_z[ix, iy, ipos]
                by[ix, iy, ipos] -= fac * psi_by_z[ix, iy, ipos]

