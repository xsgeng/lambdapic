from typing import Optional
import numpy as np
from scipy.constants import e, m_e
from numpy.typing import NDArray
from numpy import bool_, float64, uint64, bool_
from loguru import logger

class ParticlesBase:
    """
    The dataclass of particle data.

    This class stores and manages particle attributes including position, momentum,
    electromagnetic fields at particle positions, and particle status (alive/dead).

    Attributes:
        x,y,z (NDArray[float64]): Particle positions in x, y, z coordinates
        w (NDArray[float64]): Particle weights
        ux,uy,uz (NDArray[float64]): Normalized momentum :math:`u_i = \\gamma\\beta_i`
        inv_gamma (NDArray[float64]): Inverse relativistic gamma factor
        ex_part,ey_part,ez_part (NDArray[float64]): Electric fields interpolated at particle positions
        bx_part,by_part,bz_part (NDArray[float64]): Magnetic fields interpolated at particle positions
        is_dead (NDArray[bool]): Boolean array indicating dead particles
        _id (NDArray[float64]): Unique particle IDs stored as float64
        npart (int): Total number of particles (including dead)
        _npart_created (int): Counter for generating sequential local IDs
        attrs (list[str]): List of particle attributes. All attributes will be automatically synced during simulation unless specified.

    Note:
        You can extend/modify the particle attributes manually before calling :any:`ParticlesBase.initialize`.

    """
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

    def __init__(self, ipatch: Optional[int]=None, rank: Optional[int]=None) -> None:
        """
        Args:
            ipatch (Optional[int]): Patch index the particles belong to
            rank (Optional[int]): MPI rank (default: 0)
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
        """Generate particle IDs with proper bit structure.

        Args:
            start (int): Starting index for local particle IDs
            count (int): Number of IDs to generate

        Returns:
            NDArray[float64]: Array of particle IDs encoded as float64

        Raises:
            AssertionError: If start + count exceeds 32-bit limit
        """
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

    def initialize(self, npart: int) -> None:
        """Initialize particle arrays with given size.
        You can extend the particle attributes before calling initialize.

        Args:
            npart (int): Number of particles to initialize

        Raises:
            AssertionError: If npart is negative
        """
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
        """Extend particle arrays by n elements.

        Be careful when using this method, as it changes the address of particle data arrays.
        Update any typed.List storing particle data arrays accordingly.

        Args:
            n (int): Number of elements to add (must be positive)
        """
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

    def prune(self, extra_buff: float = 0.1) -> Optional[np.ndarray]:
        """Remove dead particles and shrink arrays.

        Args:
            extra_buff (float): Buffer size multiplier (default: 0.1)

        Returns:
            Optional[np.ndarray]: Sorting indices used if pruning occurred
        """
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
        """Get particle IDs as uint64 array.

        Returns:
            NDArray[uint64]: Particle IDs
        """
        return self._id.view(np.uint64)
    
    @property
    def is_alive(self) -> np.ndarray:
        """Get boolean mask of alive particles.

        Returns:
            np.ndarray: Boolean array where True indicates alive particles
        """
        return np.logical_not(self.is_dead)

    def __setstate__(self, state):
        for attr in state['attrs']:
            state[attr] = state[attr].copy()
        state['is_dead'] = state['is_dead'].copy()
        self.__dict__.update(state)

class QEDParticles(ParticlesBase):
    """Particle class used for QED processes. With additional attributes below:
    
    Attributes:
        chi (NDArray[float64]): Quantum parameter for radiation
        tau (NDArray[float64]): Optical depth for pair production
        delta (NDArray[float64]): Energy loss fraction
        event (NDArray[bool]): Flags for QED events
    """
    chi: NDArray[float64]
    tau: NDArray[float64]
    delta: NDArray[float64]
    event: NDArray[bool_]
    
    def __init__(self, ipatch: Optional[int], rank: Optional[int] = 0) -> None:
        """Initialize QED particle class.
        
        Args:
            ipatch (Optional[int]): Patch index the particles belong to
            rank (Optional[int]): MPI rank (default: 0)
        """
        super().__init__(ipatch=ipatch, rank=rank)
        self.attrs += ["chi", "tau", "delta"]

    def initialize(self, npart: int) -> None:
        """Initialize QED particle arrays.
        
        Args:
            npart (int): Number of particles to initialize
        """
        super().initialize(npart)
        self.event = np.full(npart, False)

    def extend(self, n: int):
        """Extend QED particle arrays.
        
        Args:
            n (int): Number of elements to add
        """
        self.event.resize(n + self.npart, refcheck=False)
        self.event[-n:] = False
        super().extend(n)

    def prune(self, extra_buff: float = 0.1) -> None:
        """Prune dead QED particles.
        
        Args:
            extra_buff (float): Buffer size multiplier (default: 0.1)
        """
        sorted_idx = super().prune(extra_buff=extra_buff)
        self.event[:] = self.event[sorted_idx]
        self.event.resize(self.npart, refcheck=False)


class SpinParticles(ParticlesBase):
    sx: NDArray[float64]
    sy: NDArray[float64]
    sz: NDArray[float64]
    def __init__(self, ipatch: Optional[int], rank: Optional[int] = 0) -> None:
        super().__init__(ipatch=ipatch, rank=rank)
        self.attrs += ["sx", "sy", "sz"]


class SpinQEDParticles(SpinParticles, QEDParticles):
    def __init__(self, ipatch: Optional[int], rank: Optional[int] = 0) -> None:
        super().__init__(ipatch=ipatch, rank=rank)
