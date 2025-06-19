
from numba import typed

from ..patch import Patches
from ..species import Electron, Photon, Positron, Species
from .cpu import (
    create_pair_patches_2d,
    create_pair_patches_3d,
    get_particle_extension_size_patches,
    pairproduction_event_patches,
    remove_photon_patches,
    update_chi_patches,
)


class PairProductionBase:
    """
    Radiation class handles creation of photons.

    QED processes are species-wised.
    """
    def __init__(self, patches: Patches, ispec: int) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        patches : Patches
            The patches.
        ispec : int
            The index of species.
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches = patches.npatches

        self.ispec = ispec


    def generate_particle_lists(self) -> None:
        ispec = self.ispec
        self.tau_list = typed.List([p.particles[ispec].tau for p in self.patches])
        self.chi_list = typed.List([p.particles[ispec].chi for p in self.patches])
        self.delta_list = typed.List([p.particles[ispec].delta for p in self.patches])

        self.ux_list = typed.List([p.particles[ispec].ux for p in self.patches])
        self.uy_list = typed.List([p.particles[ispec].uy for p in self.patches])
        self.uz_list = typed.List([p.particles[ispec].uz for p in self.patches])
        self.inv_gamma_list = typed.List([p.particles[ispec].inv_gamma for p in self.patches])


        self.ex_part_list = typed.List([p.particles[ispec].ex_part for p in self.patches])
        self.ey_part_list = typed.List([p.particles[ispec].ey_part for p in self.patches])
        self.ez_part_list = typed.List([p.particles[ispec].ez_part for p in self.patches])
        self.bx_part_list = typed.List([p.particles[ispec].bx_part for p in self.patches])
        self.by_part_list = typed.List([p.particles[ispec].by_part for p in self.patches])
        self.bz_part_list = typed.List([p.particles[ispec].bz_part for p in self.patches])

        self.event_list = typed.List([p.particles[ispec].event for p in self.patches])
        self.is_dead_list = typed.List([p.particles[ispec].is_dead for p in self.patches])

    def update_particle_lists(self, ipatch: int) -> None:
        particles = self.patches[ipatch].particles[self.ispec]
        self.tau_list[ipatch] = particles.tau
        self.chi_list[ipatch] = particles.chi
        self.delta_list[ipatch] = particles.delta

        self.ux_list[ipatch] = particles.ux
        self.uy_list[ipatch] = particles.uy
        self.uz_list[ipatch] = particles.uz
        self.inv_gamma_list[ipatch] = particles.inv_gamma


        self.ex_part_list[ipatch] = particles.ex_part
        self.ey_part_list[ipatch] = particles.ey_part
        self.ez_part_list[ipatch] = particles.ez_part
        self.bx_part_list[ipatch] = particles.bx_part
        self.by_part_list[ipatch] = particles.by_part
        self.bz_part_list[ipatch] = particles.bz_part

        self.event_list[ipatch] = particles.event
        self.is_dead_list[ipatch] = particles.is_dead

    def update_chi(self) -> None:
        update_chi_patches(
            self.ux_list, self.uy_list, self.uz_list,
            self.inv_gamma_list,
            self.ex_part_list, self.ey_part_list, self.ez_part_list,
            self.bx_part_list, self.by_part_list, self.bz_part_list,
            self.is_dead_list, self.npatches, self.chi_list,
        )

    def event(self, dt: float) -> None:
        raise NotImplementedError

    def filter(self) -> None:
        raise NotImplementedError

    def create_particles(self, extra_buff=0.25) -> None:
        raise NotImplementedError

    def reaction(self) -> None:
        raise NotImplementedError


class NonlinearPairProductionLCFA(PairProductionBase):
    def __init__(self, patches: Patches, ispec: int) -> None:
        super().__init__(patches, ispec)

        species = patches.species[ispec]
        assert isinstance(species, Photon), "Only support photon pair production. Please use `Photon` species."
        assert isinstance(species.electron, Electron), "Please use `Electron` species for pair production."
        assert isinstance(species.positron, Positron), "Please use `Positron` species for pair production."

        assert species.electron.ispec is not None
        assert species.positron.ispec is not None
        self.electron_ispec = species.electron.ispec
        self.positron_ispec = species.positron.ispec

        self.generate_particle_lists()

    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        # photons
        ispec = self.ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_list = typed.List([p.z for p in particles])
        self.w_list = typed.List([p.w for p in particles])

        # electrons
        ispec = self.electron_ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_ele_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_ele_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_ele_list = typed.List([p.z for p in particles])
        self.ux_ele_list = typed.List([p.ux for p in particles])
        self.uy_ele_list = typed.List([p.uy for p in particles])
        self.uz_ele_list = typed.List([p.uz for p in particles])
        self.inv_gamma_ele_list = typed.List([p.inv_gamma for p in particles])
        self.w_ele_list = typed.List([p.w for p in particles])

        self.is_dead_ele_list = typed.List([p.is_dead for p in particles])

        # positrons
        ispec = self.positron_ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_pos_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_pos_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_pos_list = typed.List([p.z for p in particles])
        self.ux_pos_list = typed.List([p.ux for p in particles])
        self.uy_pos_list = typed.List([p.uy for p in particles])
        self.uz_pos_list = typed.List([p.uz for p in particles])
        self.inv_gamma_pos_list = typed.List([p.inv_gamma for p in particles])
        self.w_pos_list = typed.List([p.w for p in particles])

        self.is_dead_pos_list = typed.List([p.is_dead for p in particles])
        


    def update_particle_lists(self, ipatch: int) -> None:
        super().update_particle_lists(ipatch)
        # photons
        pho = self.patches[ipatch].particles[self.ispec]
        self.x_list[ipatch] = pho.x
        if self.dimension >= 2:
            self.y_list[ipatch] = pho.y
        if self.dimension == 3:
            self.z_list[ipatch] = pho.z
        self.w_list[ipatch] = pho.w

        # electrons
        ele = self.patches[ipatch].particles[self.electron_ispec]
        self.x_ele_list[ipatch] = ele.x
        if self.dimension >= 2:
            self.y_ele_list[ipatch] = ele.y
        if self.dimension == 3:
            self.z_ele_list[ipatch] = ele.z

        self.ux_ele_list[ipatch] = ele.ux
        self.uy_ele_list[ipatch] = ele.uy
        self.uz_ele_list[ipatch] = ele.uz
        self.inv_gamma_ele_list[ipatch] = ele.inv_gamma
        self.w_ele_list[ipatch] = ele.w

        self.is_dead_ele_list[ipatch] = ele.is_dead

        # posistrons
        pos = self.patches[ipatch].particles[self.positron_ispec]
        self.x_pos_list[ipatch] = pos.x
        if self.dimension >= 2:
            self.y_pos_list[ipatch] = pos.y
        if self.dimension == 3:
            self.z_pos_list[ipatch] = pos.z

        self.ux_pos_list[ipatch] = pos.ux
        self.uy_pos_list[ipatch] = pos.uy
        self.uz_pos_list[ipatch] = pos.uz
        self.inv_gamma_pos_list[ipatch] = pos.inv_gamma
        self.w_pos_list[ipatch] = pos.w

        self.is_dead_pos_list[ipatch] = pos.is_dead
        
    def _update_particle_lists(self) -> None:
        for ipatch, p in enumerate(self.patches):
            for ispec in [self.ispec, self.electron_ispec, self.positron_ispec]:
                if p.particles[ispec].extended:
                    self.update_particle_lists(ipatch)
                    break # if one species is extended, all species are updated

    def event(self, dt: float) -> None:
        from .optical_depth import (
            _integral_pair_prob_along_delta,
            _pair_prob_rate_total_table,
        )
        pairproduction_event_patches(
            self.tau_list, self.chi_list, self.inv_gamma_list,
            self.is_dead_list,
            self.npatches, dt, 
            self.event_list, self.delta_list,
            _integral_pair_prob_along_delta, _pair_prob_rate_total_table
        )


    def create_particles(self, extra_buff=0.25) -> None:
        # extend pairs
        num_ele_extend = get_particle_extension_size_patches(
            self.event_list, self.is_dead_ele_list, self.npatches
        )
        num_pos_extend = get_particle_extension_size_patches(
            self.event_list, self.is_dead_pos_list, self.npatches
        )
        for ipatch in range(self.npatches):
            n1 = num_ele_extend[ipatch]
            if n1 > 0:
                n1 += int(self.patches[ipatch].particles[self.electron_ispec].npart*extra_buff)
                self.patches[ipatch].particles[self.electron_ispec].extend(n1)

            n2 = num_pos_extend[ipatch]
            if n2 > 0:
                n2 += int(self.patches[ipatch].particles[self.positron_ispec].npart*extra_buff)
                self.patches[ipatch].particles[self.positron_ispec].extend(n2)

            if n1 > 0 or n2 > 0:
                self.patches.update_particle_lists(ipatch)
                self.update_particle_lists(ipatch)
        # fillin pairs
        if self.dimension == 2:
            create_pair_patches_2d(
                self.x_list, self.y_list, self.ux_list, self.uy_list, self.uz_list, self.w_list, self.is_dead_list,
                self.x_ele_list, self.y_ele_list, self.ux_ele_list, self.uy_ele_list, self.uz_ele_list, self.inv_gamma_ele_list, self.w_ele_list, self.is_dead_ele_list, 
                self.x_pos_list, self.y_pos_list, self.ux_pos_list, self.uy_pos_list, self.uz_pos_list, self.inv_gamma_pos_list, self.w_pos_list, self.is_dead_pos_list, 
                self.delta_list, self.event_list,
                self.npatches,
            )
        if self.dimension == 3:
            create_pair_patches_3d(
                self.x_list, self.y_list, self.z_list, self.ux_list, self.uy_list, self.uz_list, self.w_list, self.is_dead_list,
                self.x_ele_list, self.y_ele_list, self.z_ele_list, self.ux_ele_list, self.uy_ele_list, self.uz_ele_list, self.inv_gamma_ele_list, self.w_ele_list, self.is_dead_ele_list, 
                self.x_pos_list, self.y_pos_list, self.z_pos_list, self.ux_pos_list, self.uy_pos_list, self.uz_pos_list, self.inv_gamma_pos_list, self.w_pos_list, self.is_dead_pos_list, 
                self.delta_list, self.event_list,
                self.npatches
            )

    def reaction(self) -> None:
        remove_photon_patches(self.event_list, self.is_dead_list, self.npatches)