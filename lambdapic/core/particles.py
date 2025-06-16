import numpy as np
from scipy.constants import e, m_e
from numpy.typing import NDArray
from numpy import bool_, float64, uint64, bool_
from loguru import logger

class ParticlesBase:
    x: NDArray[float64]
    y: NDArray[float64]
    z: NDArray[float64]
    w: NDArray[float64]
    ux: NDArray[float64]
    uy: NDArray[float64]
    uz: NDArray[float64]
    inv_gamma: NDArray[float64]

    ex_part: NDArray[float64]
    ey_part: NDArray[float64]
    ez_part: NDArray[float64]
    bx_part: NDArray[float64]
    by_part: NDArray[float64]
    bz_part: NDArray[float64]

    is_dead: NDArray[bool_]

    # 64bit float for id. composed of 14bit for rank, 18bit for ipatch, 32bit for local particle
    _id: NDArray[float64]

    npart: int # length of the particles, including dead
    _npart_created: int  # counter for generating sequential local IDs

    def __init__(self, ipatch: int|None=None, rank: int|None=None) -> None:
        """
        Initialize the particle class.

        Parameters
        ----------
        ipatch : int
            The index of the patch the particle class is attached to.
        rank : int, optional
            The rank the particle class is created. default 0
        """
        self.attrs: list[str] = [
            "x", "y", "z", "w", "ux", "uy", "uz", "inv_gamma",
            "ex_part", "ey_part", "ez_part", "bx_part", "by_part", "bz_part",
            "_id"
        ]
        self.extended: bool = False
        self._npart_created = 0

        if rank is None:
            try:
                from mpi4py.MPI import COMM_WORLD
                rank = COMM_WORLD.Get_rank()
            except ImportError:
                rank = 0
            finally:
                rank = 0
        
        if ipatch is None:
            ipatch = 0
            logger.info("ipatch is not specified, set to 0. This may cause ID conflict.")
            
        assert 0 <= rank < 2**14 and 0 <=ipatch < 2**18, "rank and ipatch must be less than 2^12 and 2^18"
        self.rank: int = rank
        self.ipatch: int = ipatch
        self._ipatch_bits = np.uint64(ipatch << 32)
        self._rank_bits = np.uint64(rank << 32+18)

    def _generate_ids(self, start: int, count: int) -> NDArray[float64]:
        """Generate particle IDs with proper bit structure"""
        # Generate local indices (32 bits)
        assert start + count <= 2**32, f"too many particles created in this patch {self.ipatch=} of {self.rank=}, \
                                         local indices must be less than 2^32 = 4294967296"
        local_indices = np.arange(start, start + count, dtype=np.uint32)
        
        # Convert components to uint64 and shift to proper positions
        local_bits = np.uint64(local_indices)
        
        # Combine all bits
        id_int = self._rank_bits | self._ipatch_bits | local_bits
        
        # Convert to float64 while preserving bit pattern
        return id_int.view(np.float64)

    def initialize(
        self,
        npart: int,
    ) -> None:
        assert npart >= 0
        self.npart = npart

        for attr in self.attrs:
            setattr(self, attr, np.zeros(npart))

        self.inv_gamma[:] = 1
        self.is_dead = np.full(npart, False)
        
        # Generate particle IDs
        self._id[:] = self._generate_ids(self._npart_created, npart)
        self._npart_created += npart

    def extend(self, n: int):
        if n <= 0:
            return
        for attr in self.attrs:
            arr: np.ndarray = getattr(self, attr)
            arr.resize(n + self.npart, refcheck=False)
            # new data set to nan
            arr[-n:] = np.nan

        self.w[-n:] = 0

        # Generate new IDs for extended particles
        self._id[-n:] = self._generate_ids(self._npart_created, n)
        self._npart_created += n

        self.is_dead.resize(n + self.npart, refcheck=False)
        self.is_dead[-n:] = True

        self.npart += n
        self.extended = True

    def prune(self, extra_buff=0.1):
        n_alive = self.is_alive.sum()
        npart = int(n_alive * (1 + extra_buff))
        if npart >= self.npart:
            return
        sorted_idx = np.argsort(self.is_dead)
        for attr in self.attrs:
            arr: NDArray[np.float64] = getattr(self, attr)
            arr[:] = arr[sorted_idx]
            arr.resize(npart, refcheck=False)

        self.is_dead[:] = self.is_dead[sorted_idx]
        self.is_dead.resize(npart, refcheck=False)
        self.npart = npart
        self.extended = True
        return sorted_idx

    @property
    def id(self) -> NDArray[uint64]:
        return self._id.view(np.uint64)
    
    @property
    def is_alive(self) -> np.ndarray:
        return np.logical_not(self.is_dead)


class QEDParticles(ParticlesBase):
    chi: NDArray[float64]
    tau: NDArray[float64]
    delta: NDArray[float64]
    event: NDArray[bool_]
    def __init__(self, ipatch: int|None, rank: int|None = 0) -> None:
        super().__init__(ipatch=ipatch, rank=rank)
        self.attrs += ["chi", "tau", "delta"]

    def initialize(self, npart: int) -> None:
        super().initialize(npart)
        self.event = np.full(npart, False)

    def extend(self, n: int):
        self.event.resize(n + self.npart, refcheck=False)
        self.event[-n:] = False
        super().extend(n)

    def prune(self, extra_buff=0.1):
        sorted_idx = super().prune(extra_buff=extra_buff)
        self.event[:] = self.event[sorted_idx]
        self.event.resize(self.npart, refcheck=False)


class SpinParticles(ParticlesBase):
    sx: NDArray[float64]
    sy: NDArray[float64]
    sz: NDArray[float64]
    def __init__(self, ipatch: int|None, rank: int|None = 0) -> None:
        super().__init__(ipatch=ipatch, rank=rank)
        self.attrs += ["sx", "sy", "sz"]


class SpinQEDParticles(SpinParticles, QEDParticles):
    def __init__(self, ipatch: int|None, rank: int|None = 0) -> None:
        super().__init__(ipatch=ipatch, rank=rank)
