from time import perf_counter_ns

import numpy as np
from numpy.typing import NDArray
from numba import njit, typed

from ..boundary.cpml import PML, PMLX, PMLY

from ..fields import Fields, Fields2D
from ..particles import ParticlesBase
from ..patch.cpu import (
    fill_particles_2d,
    get_num_macro_particles_2d,
    fill_particles_3d,
    get_num_macro_particles_3d,
)
from ..species import Electron, Photon, Species
from .sync_fields2d import sync_currents_2d, sync_guard_fields_2d
from .sync_fields3d import sync_currents_3d, sync_guard_fields_3d

from . import sync_particles_2d, sync_particles_3d

from enum import IntEnum, auto

class Boundary2D(IntEnum):
    """
    Must be consistent with sync_particles2d.c
    """
    XMIN = 0
    XMAX = auto()
    YMIN = auto()
    YMAX = auto()
    XMINYMIN = auto()
    XMAXYMIN = auto()
    XMINYMAX = auto()
    XMAXYMAX = auto()

class Boundary3D(IntEnum):
    """
    Must be consistent with sync_particles3d.c
    """
    # faces
    XMIN = 0
    XMAX = auto()
    YMIN = auto()
    YMAX = auto()
    ZMIN = auto()
    ZMAX = auto()
    # egdes
    XMINYMIN = auto()
    XMINYMAX = auto()
    XMINZMIN = auto()
    XMINZMAX = auto()
    XMAXYMIN = auto()
    XMAXYMAX = auto()
    XMAXZMIN = auto()
    XMAXZMAX = auto()
    YMINZMIN = auto()
    YMINZMAX = auto()
    YMAXZMIN = auto()
    YMAXZMAX = auto()
    # vertices
    XMINYMINZMIN = auto()
    XMINYMINZMAX = auto()
    XMINYMAXZMIN = auto()
    XMINYMAXZMAX = auto()
    XMAXYMINZMIN = auto()
    XMAXYMINZMAX = auto()
    XMAXYMAXZMIN = auto()
    XMAXYMAXZMAX = auto()
class Patch:
    rank: int|None
    # the global index of the patch
    index: int
    ipatch_x: int
    ipatch_y: int
    ipatch_z: int
    x0: float
    y0: float
    z0: float

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float

    xaxis: NDArray
    yaxis: NDArray
    zaxis: NDArray

    # the global index of the neighbor patch
    neighbor_index: NDArray[np.int64]
    neighbor_rank: NDArray[np.int64]
    # the index of the neighbor patch in the patches of the same rank
    neighbor_ipatch: NDArray[np.int64]

    fields: Fields
    def __init__(self) -> None:
        # PML boundaries
        self.pml_boundary: list[PML] = []

        self.particles : list[ParticlesBase] = []

    @property
    def xmin(self):
        return self.x0
    @property
    def xmax(self):
        return self.x0 + (self.nx-1) * self.dx
    @property
    def ymin(self):
        return self.y0
    @property
    def ymax(self):
        return self.y0 + (self.ny-1) * self.dy
    @property
    def zmin(self):
        return self.z0
    @property
    def zmax(self):
        return self.z0 + (self.nz-1) * self.dz

    def add_particles(self, particles: ParticlesBase) -> None:
        self.particles.append(particles)

    def set_neighbor_index(self, **kwargs):
        raise NotImplementedError
    
    def set_neighbor_rank(self, **kwargs):
        raise NotImplementedError
    
    def set_neighbor_ipatch(self, **kwargs):
        raise NotImplementedError

    def set_fields(self, fields: Fields, **kwargs: dict[str, Fields]) -> None:
        """
        Attach the fields to the patch.

        Parameters
        ----------
        fields : Fields
            Fields of the patch
        kwargs : dict[str, Fields]
            Additional fields to be attached to the patch.

        Returns
        -------
        None

        Examples
        --------
        >>> patch.set_fields(fields, efield=efield, bfield=bfield)
        """
        self.fields = fields
        for fieldname, field in kwargs.items():
            setattr(self, fieldname, field)

    def add_pml_boundary(self, pml: PML) -> None:
        raise NotImplementedError

class Patch2D(Patch):
    def __init__(
        self,
        rank: int|None,
        index: int,
        ipatch_x: int,
        ipatch_y: int,
        x0 : float, 
        y0 : float,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """ 
        Patch2D is a container for the fields and particles of a single patch.
        The patch is a rectangular region of the grid.

        Parameters
        ----------
        rank : int
            rank of the process
        index : int
            index of the patch
        ipatch_x : int
            index of the patch in x direction
        ipatch_y : int
            index of the patch in y direction
        x0 : float
            start x0 of the patch
        y0 : float
            start y0 of the patch
        nx : int
            number of grids in x direction of the patch, n_guard not included
        ny : int
            number of grids in y direction of the patch, n_guard not included
        dx : float
            grid spacing in x direction
        dy : float
            grid spacing in y direction
        """
        super().__init__()
        self.rank = rank
        self.index = index
        self.ipatch_x = ipatch_x
        self.ipatch_y = ipatch_y
        self.x0 = x0
        self.y0 = y0

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        self.xaxis = np.arange(self.nx) * self.dx + x0
        self.yaxis = np.arange(self.ny) * self.dy + y0

        self.neighbor_index  = np.full(len(Boundary2D), -1, dtype=int)
        self.neighbor_rank   = np.full(len(Boundary2D), -1, dtype=int)
        self.neighbor_ipatch = np.full(len(Boundary2D), -1, dtype=int)

    def set_neighbor_index(self, **kwargs):
        for neighbor in kwargs.keys():
            self.neighbor_index[Boundary2D[neighbor.upper()]] = kwargs[neighbor]

    def set_neighbor_rank(self, **kwargs):
        for neighbor in kwargs.keys():
            self.neighbor_rank[Boundary2D[neighbor.upper()]] = kwargs[neighbor]
            
    def set_neighbor_ipatch(self, **kwargs):
        for neighbor in kwargs.keys():
            self.neighbor_ipatch[Boundary2D[neighbor.upper()]] = kwargs[neighbor]

    def add_pml_boundary(self, pml: PML) -> None:
        assert (self.nx >= pml.thickness) and (self.ny >= pml.thickness)
        assert len(self.pml_boundary) < 2, "cannot assign more than 2 PML boundaries to one patch, \
                                             try increasing number of patches."
        if len(self.pml_boundary) == 1:
            assert isinstance(self.pml_boundary[0], PMLX) ^ isinstance(pml, PMLX)
            # assert isinstance(self.pml_boundary[0], PMLY) ^ isinstance(pml, PMLY)
        self.pml_boundary.append(pml)

class Patch3D(Patch):
    def __init__(
        self,
        rank: int,
        index: int,
        ipatch_x: int,
        ipatch_y: int,
        ipatch_z: int,
        x0: float, 
        y0: float,
        z0: float,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None:
        """ 
        Patch3D is a container for the fields and particles of a single patch.
        The patch is a rectangular region of the grid.

        Parameters
        ----------
        rank : int
            rank of the process
        index : int
            index of the patch
        ipatch_x : int
            index of the patch in x direction
        ipatch_y : int
            index of the patch in y direction
        x0 : float
            start x0 of the patch
        y0 : float
            start y0 of the patch
        nx : int
            number of grids in x direction of the patch, n_guard not included
        ny : int
            number of grids in y direction of the patch, n_guard not included
        dx : float
            grid spacing in x direction
        dy : float
            grid spacing in y direction
        """
        super().__init__()
        self.rank = rank
        self.index = index
        self.ipatch_x = ipatch_x
        self.ipatch_y = ipatch_y
        self.ipatch_z = ipatch_z
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.xaxis = np.arange(self.nx) * self.dx + x0
        self.yaxis = np.arange(self.ny) * self.dy + y0
        self.zaxis = np.arange(self.nz) * self.dz + z0

        self.neighbor_index  = np.full(len(Boundary3D), -1, dtype=int)
        self.neighbor_rank   = np.full(len(Boundary3D), -1, dtype=int)
        self.neighbor_ipatch = np.full(len(Boundary3D), -1, dtype=int)

    def set_neighbor_index(self, **kwargs):
        for neighbor in kwargs.keys():
            self.neighbor_index[Boundary3D[neighbor.upper()]] = kwargs[neighbor]

    def set_neighbor_rank(self, **kwargs):
        for neighbor in kwargs.keys():
            self.neighbor_rank[Boundary3D[neighbor.upper()]] = kwargs[neighbor]
    
    def set_neighbor_ipatch(self, **kwargs):
        for neighbor in kwargs.keys():
            self.neighbor_ipatch[Boundary3D[neighbor.upper()]] = kwargs[neighbor]

    def add_pml_boundary(self, pml: PML) -> None:
        assert (self.nx >= pml.thickness) and (self.ny >= pml.thickness)
        assert len(self.pml_boundary) < 3, "cannot assign more than 3 PML boundaries to one patch, \
                                             try increasing number of patches."
        self.pml_boundary.append(pml)

class Patches:
    """ 
    A container for patches of the fields and particles. 

    The class handles synchronization of fields and particles between patches.
    """
    def __init__(self, dimension: int) -> None:
        assert (dimension == 1) or (dimension == 2) or (dimension == 3)
        self.dimension: int = dimension

        self.npatches: int = 0
        self.indices : list[int] = []
        self.patches : list[Patch] = []
        self.species : list[Species|Electron|Photon] = []
    
    def __getitem__(self, i: int) -> Patch:
        return self.patches[i]

    def __len__(self) -> int:
        return self.npatches

    def __iter__(self):
        return iter(self.patches)

    def append(self, patch: Patch):
        # if self.patches:
        #     assert self.patches[-1].index == patch.index - 1
        self.patches.append(patch)
        self.indices.append(patch.index)
        self.npatches += 1

    def prepend(self, patch: Patch):
        # if self.patches:
        #     assert self.patches[0].index == patch.index + 1
        self.patches.insert(0, patch)
        self.indices.insert(0, patch.index)
        self.npatches += 1
    
    def pop(self, i):
        self.indices.pop(i)
        self.npatches -= 1
        p = self.patches.pop(i)

        return p

    def update_lists(self):
        """ 
        Gather all the properties in each patch to the numba list.

        Numba jitted functions cannot accept list of objects as input.
        For parallel execution, it is the only option now.
        
        Not needed for GPU kernels since the kernel calls are asynchronous.

        this update is expensive. 
        """
        particle_lists = []
        for ispec, s in enumerate(self.species):
            particle_lists.append({})
            lists = particle_lists[ispec]
            lists["npart"] = typed.List([p.particles[ispec].npart for p in self.patches])
            lists["is_dead"] = typed.List([p.particles[ispec].is_dead for p in self.patches])

            for attr in self[0].particles[ispec].attrs:
                lists[attr] = typed.List([getattr(p.particles[ispec], attr) for p in self.patches])

        self.particle_lists = particle_lists


    def update_particle_lists(self, ipatch):
        plists = self.particle_lists
        patch = self[ipatch]
        for ispec, s in enumerate(self.species):
            plists[ispec]["npart"][ipatch] = patch.particles[ispec].npart
            plists[ispec]["is_dead"][ipatch] = patch.particles[ispec].is_dead

            for attr in patch.particles[ispec].attrs:
                plists[ispec][attr][ipatch] = getattr(patch.particles[ispec], attr)

    def init_rect_neighbor_index_2d(self, npatch_x: int, npatch_y: int, patch_index_map: dict[tuple, int]={}):
        """ 
        Initialize the neighbor index for a rectangular grid of 2D patches.
        
        Called on **global** patches, or provide a dict of (ipatch_x, ipatch_y) to patch index
        """
        # Define all possible neighbor offsets and their corresponding names
        neighbor_offsets = [
            # faces (4 neighbors)
            ((-1, 0), 'xmin'), ((+1, 0), 'xmax'),
            ((0, -1), 'ymin'), ((0, +1), 'ymax'),
            # edges (corner neighbors) (4)
            ((-1, -1), 'xminymin'), ((+1, -1), 'xmaxymin'),
            ((-1, +1), 'xminymax'), ((+1, +1), 'xmaxymax'),
        ]

        if not patch_index_map:
            patch_index_map = {(p.ipatch_x, p.ipatch_y): p.index for p in self.patches}

        for p in self.patches:
            p.neighbor_index.fill(-1)
            i, j = p.ipatch_x, p.ipatch_y
            
            for (dx, dy), name in neighbor_offsets:
                neighbor_i = i + dx
                neighbor_j = j + dy
                
                # Check if neighbor coordinates are valid
                if 0 <= neighbor_i < npatch_x and 0 <= neighbor_j < npatch_y:
                    neighbor_index = patch_index_map[(neighbor_i, neighbor_j)]
                    p.set_neighbor_index(**{name: neighbor_index})
    
    def init_rect_neighbor_index_3d(self, npatch_x: int, npatch_y: int, npatch_z: int, patch_index_map: dict[tuple, int]={}):
        """ 
        Initialize the neighbor index for a rectangular grid of patches.
        
        Called on **global** patches, or provide a dict of (ipatch_x, ipatch_y, ipatch_z) to patch index
        """
        # Define all possible neighbor offsets and their corresponding names
        neighbor_offsets = [
            # faces (6 neighbors)
            ((-1, 0, 0), 'xmin'), ((+1, 0, 0), 'xmax'),
            ((0, -1, 0), 'ymin'), ((0, +1, 0), 'ymax'),
            ((0, 0, -1), 'zmin'), ((0, 0, +1), 'zmax'),
            
            # edges (12 neighbors)
            ((-1, -1, 0), 'xminymin'), ((+1, -1, 0), 'xmaxymin'),
            ((-1, +1, 0), 'xminymax'), ((+1, +1, 0), 'xmaxymax'),
            ((-1, 0, -1), 'xminzmin'), ((+1, 0, -1), 'xmaxzmin'),
            ((-1, 0, +1), 'xminzmax'), ((+1, 0, +1), 'xmaxzmax'),
            ((0, -1, -1), 'yminzmin'), ((0, +1, -1), 'ymaxzmin'),
            ((0, -1, +1), 'yminzmax'), ((0, +1, +1), 'ymaxzmax'),
            
            # vertices (8 neighbors)
            ((-1, -1, -1), 'xminyminzmin'), ((-1, -1, +1), 'xminyminzmax'),
            ((-1, +1, -1), 'xminymaxzmin'), ((-1, +1, +1), 'xminymaxzmax'),
            ((+1, -1, -1), 'xmaxyminzmin'), ((+1, -1, +1), 'xmaxyminzmax'),
            ((+1, +1, -1), 'xmaxymaxzmin'), ((+1, +1, +1), 'xmaxymaxzmax')
        ]
        
        if not patch_index_map:
            patch_index_map = {(p.ipatch_x, p.ipatch_y, p.ipatch_z): p.index for p in self.patches}

        for p in self.patches:
            p.neighbor_index.fill(-1)
            i, j, k = p.ipatch_x, p.ipatch_y, p.ipatch_z
            
            for (dx, dy, dz), name in neighbor_offsets:
                neighbor_i, neighbor_j, neighbor_k = i + dx, j + dy, k + dz
                
                # Check if neighbor coordinates are valid
                if 0 <= neighbor_i < npatch_x and 0 <= neighbor_j < npatch_y and 0 <= neighbor_k < npatch_z:
                    neighbor_index = patch_index_map[(neighbor_i, neighbor_j, neighbor_k)]
                    p.set_neighbor_index(**{name: neighbor_index})

    def init_neighbor_rank_2d(self, patch_rank_map: dict[int, int]={}):
        """
        Initialize the neighbor rank for a rectangular grid of 2D patches.
        
        Called on **global** patches, or provide a dict of patch index to rank
        """

        if not patch_rank_map:
            patch_rank_map = {}

        for p in self.patches:
            assert p.rank is not None, f"Patch {p.index} rank is None"
            patch_rank_map[p.index] = p.rank
            

        for p in self.patches:
            p.neighbor_rank.fill(-1)
            for bound in Boundary2D:
                neighbor_index = p.neighbor_index[bound]
                if neighbor_index >= 0:
                    neighbor_rank = patch_rank_map[neighbor_index]
                    if neighbor_rank != p.rank:
                        p.set_neighbor_rank(**{bound.name.lower(): neighbor_rank})

    def init_neighbor_rank_3d(self, patch_rank_map: dict[int, int]={}):
        """
        Initialize the neighbor rank for a rectangular grid of 3D patches.
        
        Called on **global** patches, or provide a dict of patch index to rank
        """
        if not patch_rank_map:
            patch_rank_map = {}
            
        for p in self.patches:
            assert p.rank is not None, f"Patch {p.index} rank is None"
            patch_rank_map[p.index] = p.rank

        for p in self.patches:
            p.neighbor_rank.fill(-1)
            for bound in Boundary3D:
                neighbor_index = p.neighbor_index[bound]
                if neighbor_index >= 0:
                    neighbor_rank = self.patches[neighbor_index].rank
                    if neighbor_rank != p.rank:
                        p.set_neighbor_rank(**{bound.name.lower(): neighbor_rank})


    def init_neighbor_ipatch_2d(self):
        """
        Initialize the neighbor ipatch for a rectangular grid of 2D patches.
        
        Called on local patches
        """
        for p in self.patches:
            p.neighbor_ipatch.fill(-1)
            for bound in Boundary2D:
                neighbor_index = p.neighbor_index[bound]
                if neighbor_index >= 0 and neighbor_index in self.indices:
                    neighbor_ipatch = self.indices.index(neighbor_index)
                    p.set_neighbor_ipatch(**{bound.name.lower(): neighbor_ipatch})
    
    def init_neighbor_ipatch_3d(self):
        """
        Initialize the neighbor ipatch for a rectangular grid of 3D patches.
        
        Called on local patches
        """
        for p in self.patches:
            p.neighbor_ipatch.fill(-1)
            for bound in Boundary3D:
                neighbor_index = p.neighbor_index[bound]
                if neighbor_index >= 0 and neighbor_index in self.indices:
                    neighbor_ipatch = self.indices.index(neighbor_index)
                    p.set_neighbor_ipatch(**{bound.name.lower(): neighbor_ipatch})


    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        if self.dimension == 2:
            sync_guard_fields_2d(
                [p.fields for p in self.patches],
                self.patches,
                attrs,
                self.npatches, self.nx, self.ny, self.n_guard
            )
        if self.dimension == 3:
            sync_guard_fields_3d(
                [p.fields for p in self.patches],
                self.patches,
                attrs,
                self.npatches, self.nx, self.ny, self.nz, self.n_guard
            )
        

    def sync_currents(self):
        if self.dimension == 2:
            sync_currents_2d(
                [p.fields for p in self.patches],
                self.patches,
                self.npatches, self.nx, self.ny, self.n_guard,
            )
        if self.dimension == 3:
            sync_currents_3d(
                [p.fields for p in self.patches],
                self.patches,
                self.npatches, self.nx, self.ny, self.nz, self.n_guard,
            )
        
    def sync_particles(self) -> None:
        if self.dimension == 2:
            for ispec, s in enumerate(self.species):

                npart_to_extend, npart_incoming, npart_outgoing = sync_particles_2d.get_npart_to_extend_2d(
                    [p.particles[ispec] for p in self],
                    [p for p in self],
                    self.npatches, self.dx, self.dy,
                )

                # extend the particles in each patch in python mode
                # TODO: extend the particles in each patch in numba mode in parallel
                # typed.List cannot modify the attr of the particle object since
                # the address is modified after being extended.
                for ipatches in range(self.npatches):
                    p = self[ipatches].particles[ispec]
                    if npart_to_extend[ipatches] > 0:
                        p.extend(npart_to_extend[ipatches])
                        p.extended = True
                        self.update_particle_lists(ipatches)
                sync_particles_2d.fill_particles_from_boundary_2d(
                    [p.particles[ispec] for p in self],
                    [p for p in self],
                    npart_incoming, npart_outgoing,
                    self.npatches, self.dx, self.dy,
                    self[0].particles[ispec].attrs
                )
        if self.dimension == 3:
            for ispec, s in enumerate(self.species):
                npart_to_extend, npart_incoming, npart_outgoing = sync_particles_3d.get_npart_to_extend_3d(
                    [p.particles[ispec] for p in self],
                    self.patches,
                    self.npatches, self.dx, self.dy, self.dz,
                )

                for ipatches in range(self.npatches):
                    p = self[ipatches].particles[ispec]
                    if npart_to_extend[ipatches] > 0:
                        p.extend(npart_to_extend[ipatches])
                        p.extended = True
                        self.update_particle_lists(ipatches)
                sync_particles_3d.fill_particles_from_boundary_3d(
                    [p.particles[ispec] for p in self],
                    [p for p in self],
                    npart_incoming, npart_outgoing,
                    self.npatches, self.dx, self.dy, self.dz,
                    self[0].particles[ispec].attrs
                )

        
    @property
    def nx(self) -> int:
        return self[0].fields.nx

    @property
    def ny(self) -> int:
        return self[0].fields.ny

    @property
    def nz(self) -> int:
        return self[0].fields.nz

    @property
    def dx(self) -> float:
        return self[0].fields.dx

    @property
    def dy(self) -> float:
        return self[0].fields.dy

    @property
    def dz(self) -> float:
        return self[0].fields.dz


    @property
    def n_guard(self):
        return self[0].fields.n_guard


    def calculate_npart(self, species: Species) -> NDArray[np.int64]:
        """
        Calculate the number of macro particles in each patch based on the species density function.
        
        Returns:
            num_macro_particles: NDArray[int]
                An array of integers representing the number of macro particles in each patch.
        """
        if self.dimension == 2:
            xaxis = typed.List([p.xaxis for p in self.patches])
            yaxis = typed.List([p.yaxis for p in self.patches])
            
            if species.density is not None:
                num_macro_particles = get_num_macro_particles_2d(
                    species.density_jit,
                    xaxis, 
                    yaxis, 
                    self.npatches, 
                    species.density_min, 
                    species.ppc,
                )
            else:
                num_macro_particles = np.zeros(self.npatches, dtype='int64')
                
        elif self.dimension == 3:
            xaxis = typed.List([p.xaxis for p in self.patches])
            yaxis = typed.List([p.yaxis for p in self.patches])
            zaxis = typed.List([p.zaxis for p in self.patches])
            
            if species.density is not None:
                num_macro_particles = get_num_macro_particles_3d(
                    species.density_jit,
                    xaxis, 
                    yaxis,
                    zaxis,
                    self.npatches, 
                    species.density_min, 
                    species.ppc,
                )
            else:
                num_macro_particles = np.zeros(self.npatches, dtype='int64')
        else:
            raise ValueError('dimension must be 2 or 3')
        return num_macro_particles

        
    def add_species(self, species : Species, aux_attrs: list[str]|None=None) -> int:
        if aux_attrs is None:
            aux_attrs = []

        num_macro_particles = self.calculate_npart(species)

        for ipatch, p in enumerate(self.patches):
            particles : ParticlesBase = species.create_particles(ipatch=ipatch, rank=p.rank)
            particles.attrs += aux_attrs
            particles.initialize(num_macro_particles[ipatch])
            self[ipatch].add_particles(particles)

        self.species.append(species)
        
        num_macro_particles_sum = sum(num_macro_particles)
        return num_macro_particles_sum

    def fill_particles(self):
        for ispec, s in enumerate(self.species):
            if s.density is not None:
                if self.dimension == 2:
                    xaxis = typed.List([p.xaxis for p in self.patches])
                    yaxis = typed.List([p.yaxis for p in self.patches])
                    x_list = typed.List([p.particles[ispec].x for p in self.patches])
                    y_list = typed.List([p.particles[ispec].y for p in self.patches])
                    w_list = typed.List([p.particles[ispec].w for p in self.patches])
                    
                    fill_particles_2d(
                        s.density_jit,
                        xaxis, 
                        yaxis, 
                        self.npatches, 
                        s.density_min, 
                        s.ppc,
                        x_list, y_list, w_list
                    )
                    
                elif self.dimension == 3:
                    xaxis = typed.List([p.xaxis for p in self.patches])
                    yaxis = typed.List([p.yaxis for p in self.patches])
                    zaxis = typed.List([p.zaxis for p in self.patches])
                    x_list = typed.List([p.particles[ispec].x for p in self.patches])
                    y_list = typed.List([p.particles[ispec].y for p in self.patches])
                    z_list = typed.List([p.particles[ispec].z for p in self.patches])
                    w_list = typed.List([p.particles[ispec].w for p in self.patches])
                    
                    fill_particles_3d(
                        s.density_jit,
                        xaxis, 
                        yaxis,
                        zaxis,
                        self.npatches, 
                        s.density_min, 
                        s.ppc,
                        x_list, y_list, z_list, w_list
                    )
    
        self.update_lists()
