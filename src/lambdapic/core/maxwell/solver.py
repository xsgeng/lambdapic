
import numpy as np
from numba import typed
from scipy.constants import c, e, epsilon_0, mu_0

from ..boundary.cpml import (PMLX, PMLY, PMLZ, update_bfield_cpml_patches_2d,
                             update_efield_cpml_patches_2d,
                             update_bfield_cpml_patches_3d,
                             update_efield_cpml_patches_3d)
from ..patch import Patches

from .cpu import (update_bfield_patches_2d, update_bfield_patches_3d,
                  update_efield_patches_2d, update_efield_patches_3d)


class MaxwellSolver:
    def __init__(self, patches: Patches) -> None:
        """
        Initialize Maxwell solver from Patches.

        Handles parallel execution of Maxwell solver of all patches.

        Parameters
        ----------
        patches : Patches
        """
        self.patches = patches
        self.npatches: int = patches.npatches

        self.dx: float = patches.dx
        self.nx: int = patches.nx

        self.n_guard: int = patches.n_guard

        self.generate_field_lists()

    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches for parallel execution.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        patches_non_boundary = [p for p in self.patches if len(p.pml_boundary) == 0]
        patches_pml_boundary = [p for p in self.patches if len(p.pml_boundary) > 0]
        
        self.patches_non_boundary = patches_non_boundary 
        self.patches_pml_boundary = patches_pml_boundary 

        self.ex_list = typed.List([p.fields.ex for p in patches_non_boundary])
        self.ey_list = typed.List([p.fields.ey for p in patches_non_boundary])
        self.ez_list = typed.List([p.fields.ez for p in patches_non_boundary])
        self.bx_list = typed.List([p.fields.bx for p in patches_non_boundary])
        self.by_list = typed.List([p.fields.by for p in patches_non_boundary])
        self.bz_list = typed.List([p.fields.bz for p in patches_non_boundary])
        self.jx_list = typed.List([p.fields.jx for p in patches_non_boundary])
        self.jy_list = typed.List([p.fields.jy for p in patches_non_boundary])
        self.jz_list = typed.List([p.fields.jz for p in patches_non_boundary])

        if patches_pml_boundary:
            self.ex_pml_list = typed.List([p.fields.ex for p in patches_pml_boundary])
            self.ey_pml_list = typed.List([p.fields.ey for p in patches_pml_boundary])
            self.ez_pml_list = typed.List([p.fields.ez for p in patches_pml_boundary])
            self.bx_pml_list = typed.List([p.fields.bx for p in patches_pml_boundary])
            self.by_pml_list = typed.List([p.fields.by for p in patches_pml_boundary])
            self.bz_pml_list = typed.List([p.fields.bz for p in patches_pml_boundary])
            self.jx_pml_list = typed.List([p.fields.jx for p in patches_pml_boundary])
            self.jy_pml_list = typed.List([p.fields.jy for p in patches_pml_boundary])
            self.jz_pml_list = typed.List([p.fields.jz for p in patches_pml_boundary])
            self.generate_kappa_lists()

    def generate_kappa_lists(self) -> None:
        self.kappa_ex_list = typed.List()
        self.kappa_ey_list = typed.List()
        self.kappa_ez_list = typed.List()
        self.kappa_bx_list = typed.List()
        self.kappa_by_list = typed.List()
        self.kappa_bz_list = typed.List()

        for p in self.patches_pml_boundary:
            kappa_ex = p.pml_boundary[0].kappa_ex
            kappa_ey = p.pml_boundary[0].kappa_ey
            kappa_ez = p.pml_boundary[0].kappa_ez
            kappa_bx = p.pml_boundary[0].kappa_bx
            kappa_by = p.pml_boundary[0].kappa_by
            kappa_bz = p.pml_boundary[0].kappa_bz

            # Populate kappa values from PML boundaries
            for pml in p.pml_boundary:
                if isinstance(pml, PMLX):
                    kappa_ex = pml.kappa_ex
                    kappa_bx = pml.kappa_bx
                elif isinstance(pml, PMLY):
                    kappa_ey = pml.kappa_ey
                    kappa_by = pml.kappa_by
                elif isinstance(pml, PMLZ):
                    kappa_ez = pml.kappa_ez
                    kappa_bz = pml.kappa_bz

            self.kappa_ex_list.append(kappa_ex)
            self.kappa_ey_list.append(kappa_ey)
            self.kappa_ez_list.append(kappa_ez)
            self.kappa_bx_list.append(kappa_bx)
            self.kappa_by_list.append(kappa_by)
            self.kappa_bz_list.append(kappa_bz)

    def update_efield(self, dt: float) -> None:
        """
        Update electric field of all patches.

        Parameters
        ----------
        dt : float
            Time step.
        """
        raise NotImplementedError

    def update_bfield(self, dt: float) -> None:
        """
        Update magnetic field of all patches.

        Parameters
        ----------
        dt : float
            Time step.
        """
        raise NotImplementedError

class MaxwellSolver2D(MaxwellSolver):
    def __init__(self, patches: Patches) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy
        self.ny: int = patches.ny

    def update_efield(self, dt: float) -> None:
        if self.patches_non_boundary:
            update_efield_patches_2d(
                self.ex_list, self.ey_list, self.ez_list,
                self.bx_list, self.by_list, self.bz_list,
                self.jx_list, self.jy_list, self.jz_list,
                len(self.patches_non_boundary),
                self.dx, self.dy, dt,
                self.nx, self.ny, self.n_guard,
            )
        # Maxwell equation with kappa of PML boundary
        if self.patches_pml_boundary:
            update_efield_cpml_patches_2d(
                self.ex_pml_list, self.ey_pml_list, self.ez_pml_list, 
                self.bx_pml_list, self.by_pml_list, self.bz_pml_list, 
                self.jx_pml_list, self.jy_pml_list, self.jz_pml_list, 
                self.kappa_ex_list, self.kappa_ey_list,
                len(self.patches_pml_boundary),
                self.dx, self.dy, dt,
                self.nx, self.ny, self.n_guard,
            )
            for p in self.patches_pml_boundary:
                for pml in p.pml_boundary:
                    pml.advance_e_currents(dt)

        
    def update_bfield(self, dt: float) -> None:
        if self.patches_non_boundary:
            update_bfield_patches_2d(
                self.ex_list, self.ey_list, self.ez_list,
                self.bx_list, self.by_list, self.bz_list,
                len(self.patches_non_boundary),
                self.dx, self.dy, dt,
                self.nx, self.ny, self.n_guard,
            )
        # Maxwell equation with kappa of PML boundary
        if self.patches_pml_boundary:
            update_bfield_cpml_patches_2d(
                self.ex_pml_list, self.ey_pml_list, self.ez_pml_list, 
                self.bx_pml_list, self.by_pml_list, self.bz_pml_list, 
                self.kappa_bx_list, self.kappa_by_list,
                len(self.patches_pml_boundary),
                self.dx, self.dy, dt,
                self.nx, self.ny, self.n_guard,
            )
            for p in self.patches_pml_boundary:
                for pml in p.pml_boundary:
                    pml.advance_b_currents(dt)


class MaxwellSolver3D(MaxwellSolver):
    def __init__(self, patches: Patches) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy
        self.dz: float = patches.dz
        self.ny: int = patches.ny
        self.nz: int = patches.nz

    def update_efield(self, dt: float) -> None:
        # Regular field update for non-PML patches
        if self.patches_non_boundary:
            update_efield_patches_3d(
                self.ex_list, self.ey_list, self.ez_list,
                self.bx_list, self.by_list, self.bz_list,
                self.jx_list, self.jy_list, self.jz_list,
                len(self.patches_non_boundary),
                self.dx, self.dy, self.dz, dt,
                self.nx, self.ny, self.nz, self.n_guard,
            )
        
        # CPML update for PML-boundary patches
        if self.patches_pml_boundary:
            update_efield_cpml_patches_3d(
                self.ex_pml_list, self.ey_pml_list, self.ez_pml_list,
                self.bx_pml_list, self.by_pml_list, self.bz_pml_list,
                self.jx_pml_list, self.jy_pml_list, self.jz_pml_list,
                self.kappa_ex_list, self.kappa_ey_list, self.kappa_ez_list,
                len(self.patches_pml_boundary),
                self.dx, self.dy, self.dz, dt,
                self.nx, self.ny, self.nz, self.n_guard,
            )
            # Advance PML auxiliary currents
            for p in self.patches_pml_boundary:
                for pml in p.pml_boundary:
                    pml.advance_e_currents(dt)

    def update_bfield(self, dt: float) -> None:
        # Regular field update for non-PML patches
        if self.patches_non_boundary:
            update_bfield_patches_3d(
                self.ex_list, self.ey_list, self.ez_list,
                self.bx_list, self.by_list, self.bz_list,
                len(self.patches_non_boundary),
                self.dx, self.dy, self.dz, dt,
                self.nx, self.ny, self.nz, self.n_guard,
            )
        
        # CPML update for PML-boundary patches
        if self.patches_pml_boundary:
            update_bfield_cpml_patches_3d(
                self.ex_pml_list, self.ey_pml_list, self.ez_pml_list,
                self.bx_pml_list, self.by_pml_list, self.bz_pml_list,
                self.kappa_bx_list, self.kappa_by_list, self.kappa_bz_list,
                len(self.patches_pml_boundary),
                self.dx, self.dy, self.dz, dt,
                self.nx, self.ny, self.nz, self.n_guard,
            )
            # Advance PML auxiliary currents
            for p in self.patches_pml_boundary:
                for pml in p.pml_boundary:
                    pml.advance_b_currents(dt)
