
from numba import typed

from lambdapic.core.particles import QEDParticles

from ..patch import Patches
from ..species import Electron, Photon, Species
from .cpu import (
    create_photon_patches_2d,
    create_photon_patches_3d,
    get_particle_extension_size_patches,
    photon_recoil_patches,
    radiation_event_patches,
    update_chi_patches,
)


class RadiationBase:
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
        self.photon_ispec = None


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
        assert isinstance(particles, QEDParticles)

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


class NonlinearComptonLCFA(RadiationBase):
    def __init__(self, patches: Patches, ispec: int) -> None:
        super().__init__(patches, ispec)

        radiation_species = patches.species[ispec]
        assert isinstance(radiation_species, Electron), "Only support electron radiation. Please use `Electron` species."
        assert isinstance(radiation_species.photon, Photon), "Please use `Photon` species."
        assert radiation_species.photon.ispec is not None
        self.photon_ispec = radiation_species.photon.ispec

        self.generate_particle_lists()

    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        # electrons
        ispec = self.ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_list = typed.List([p.z for p in particles])
        self.w_list = typed.List([p.w for p in particles])

        # photons
        ispec = self.photon_ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_pho_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_pho_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_pho_list = typed.List([p.z for p in particles])
        self.ux_pho_list = typed.List([p.ux for p in particles])
        self.uy_pho_list = typed.List([p.uy for p in particles])
        self.uz_pho_list = typed.List([p.uz for p in particles])
        self.inv_gamma_pho_list = typed.List([p.inv_gamma for p in particles])
        self.w_pho_list = typed.List([p.w for p in particles])

        self.is_dead_pho_list = typed.List([p.is_dead for p in particles])


    def update_particle_lists(self, ipatch: int) -> None:
        super().update_particle_lists(ipatch)
        # electrons
        electrons = self.patches[ipatch].particles[self.ispec]
        self.x_list[ipatch] = electrons.x
        if self.dimension >= 2:
            self.y_list[ipatch] = electrons.y
        if self.dimension == 3:
            self.z_list[ipatch] = electrons.z
        self.w_list[ipatch] = electrons.w

        # photons
        photons = self.patches[ipatch].particles[self.photon_ispec]
        self.x_pho_list[ipatch] = photons.x
        if self.dimension >= 2:
            self.y_pho_list[ipatch] = photons.y
        if self.dimension == 3:
            self.z_pho_list[ipatch] = photons.z

        self.ux_pho_list[ipatch] = photons.ux
        self.uy_pho_list[ipatch] = photons.uy
        self.uz_pho_list[ipatch] = photons.uz
        self.inv_gamma_pho_list[ipatch] = photons.inv_gamma
        self.w_pho_list[ipatch] = photons.w

        self.is_dead_pho_list[ipatch] = photons.is_dead

    def _update_particle_lists(self) -> None:
        for ipatch, p in enumerate(self.patches):
            for ispec in [self.ispec, self.photon_ispec]:
                if p.particles[ispec].extended:
                    self.update_particle_lists(ipatch)
                    break # if one species is extended, all species are updated

    def event(self, dt: float) -> None:
        from .optical_depth import (
            _integral_photon_prob_along_delta,
            _photon_prob_rate_total_table,
        )
        radiation_event_patches(
            self.tau_list, self.chi_list, self.inv_gamma_list,
            self.is_dead_list,
            self.npatches, dt, 
            self.event_list, self.delta_list,
            _integral_photon_prob_along_delta, _photon_prob_rate_total_table
        )


    def create_particles(self, extra_buff=0.25) -> None:
        # extend photons
        num_photons_extend = get_particle_extension_size_patches(
            self.event_list, self.is_dead_pho_list, self.npatches
        )
        for ipatch in range(self.npatches):
            n = num_photons_extend[ipatch]
            if n > 0:
                n += int(self.patches[ipatch].particles[self.photon_ispec].npart*extra_buff)
                self.patches[ipatch].particles[self.photon_ispec].extend(n)
                self.patches.update_particle_lists(ipatch)
                self.update_particle_lists(ipatch)
        # fillin photons
        if self.dimension == 2:
            create_photon_patches_2d(
                self.x_list, self.y_list, self.ux_list, self.uy_list, self.uz_list, self.w_list, self.is_dead_list,
                self.x_pho_list, self.y_pho_list, self.ux_pho_list, self.uy_pho_list, self.uz_pho_list, self.inv_gamma_pho_list, self.w_pho_list, self.is_dead_pho_list, 
                self.delta_list, self.event_list,
                self.npatches,
            )
        if self.dimension == 3:
            create_photon_patches_3d(
                self.x_list, self.y_list, self.z_list, self.ux_list, self.uy_list, self.uz_list, self.w_list, self.is_dead_list,
                self.x_pho_list, self.y_pho_list, self.z_pho_list, self.ux_pho_list, self.uy_pho_list, self.uz_pho_list, self.inv_gamma_pho_list, self.w_pho_list, self.is_dead_pho_list, 
                self.delta_list, self.event_list,
                self.npatches
            )

    def reaction(self) -> None:
        photon_recoil_patches(
            self.ux_list, self.uy_list, self.uz_list, self.inv_gamma_list,
            self.event_list, self.delta_list, self.is_dead_list,
            self.npatches,
        )

class ContinuousRadiation(RadiationBase):

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
        self.patches = patches
        self.npatches = patches.npatches

        self.ispec = ispec

        radiation_species = patches.species[ispec]
        assert isinstance(radiation_species, Electron)
        assert radiation_species.photon is None

        self.photon_ispec = None
        self.q = radiation_species.q

        self.generate_particle_lists()


    def event(self, dt: float) -> None:
        pass

    def create_particles(self, extra_buff=0.25) -> None:
        pass

    def reaction(self) -> None:
        ...