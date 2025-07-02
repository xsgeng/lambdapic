import numpy as np

from ..patch import Patches
from .cpu2d import current_deposition_cpu_2d
from deprecated import deprecated

class CurrentDeposition:
    """
    Current deposition class.

    Holds J, Rho fields and some particle attributes of all patches.

    """
    def __init__(self, patches: Patches) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        
        """
        self.patches = patches
        self.npatches: int = patches.npatches
        self.dx: float = patches.dx

        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.w_list = []
        self.ux_list = []
        self.uy_list = []
        self.uz_list = []
        self.inv_gamma_list = []
        self.is_dead_list = []

        self.q = []

        self.generate_field_lists()
        self.generate_particle_lists()

    def generate_particle_lists(self) -> None:
        """
        Add species to the current deposition class.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        for ispec, s in enumerate(self.patches.species):
            self.x_list.append([p.particles[ispec].x for p in self.patches])
            self.w_list.append([p.particles[ispec].w for p in self.patches])
            self.ux_list.append([p.particles[ispec].ux for p in self.patches])
            self.uy_list.append([p.particles[ispec].uy for p in self.patches])
            self.uz_list.append([p.particles[ispec].uz for p in self.patches])
            self.inv_gamma_list.append([p.particles[ispec].inv_gamma for p in self.patches])
            self.is_dead_list.append([p.particles[ispec].is_dead for p in self.patches])

            self.q.append(s.q)

    @deprecated(reason="No need to update particle for currentdeposition. It has no effect now.")
    def update_particle_lists(self, ipatch: int, ispec: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        particle : Particles
            Particle object in the patch.
        """
        pass
        # self.x_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].x
        # self.w_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].w
        # self.ux_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].ux
        # self.uy_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].uy
        # self.uz_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].uz
        # self.inv_gamma_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].inv_gamma
        # self.is_dead_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].is_dead

    
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.jx_list = [p.fields.jx for p in self.patches]
        self.jy_list = [p.fields.jy for p in self.patches]
        self.jz_list = [p.fields.jz for p in self.patches]
        self.rho_list = [p.fields.rho for p in self.patches]

        self.x0s = [p.x0 for p in self.patches]

    def update_patches(self) -> None:
        """
        Called when the arangement of patches changes.
        """
        self.generate_field_lists()
        self.generate_particle_lists()
        self.npatches = self.patches.npatches
        # raise NotImplementedError

    def reset(self) -> None:
        """
        Reset J and Rho to zero.
        """
        for ipatch in range(self.npatches):
            self.jx_list[ipatch].fill(0)
            self.jy_list[ipatch].fill(0)
            self.jz_list[ipatch].fill(0)
            self.rho_list[ipatch].fill(0)
        

    def __call__(self, ispec: int, dt: float) -> None:
        """
        Current deposition.

        Parameters
        ----------
        ispec : int
            Species index.
        dt : float
            Time step.
        """
        raise NotImplementedError


class CurrentDeposition2D(CurrentDeposition):
    def __init__(self, patches: Patches) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy


    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        for ispec, s in enumerate(self.patches.species):
            self.y_list.append([p.particles[ispec].y for p in self.patches])


    # def update_particle_lists(self, ipatch: int, ispec: int):
    #     super().update_particle_lists(ipatch, ispec)
    #     self.y_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].y


    def generate_field_lists(self) -> None:
        super().generate_field_lists()
        self.y0s = [p.y0 for p in self.patches]


    def __call__(self, ispec:int, dt: float) -> None:
        if self.q[ispec] != 0:
            current_deposition_cpu_2d(
                [p.fields for p in self.patches],
                [p.particles[ispec] for p in self.patches],
                self.npatches,
                dt, self.q[ispec]
            )

class CurrentDeposition3D(CurrentDeposition):
    ...
