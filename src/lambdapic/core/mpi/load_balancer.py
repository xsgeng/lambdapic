"""Load balancing for MPI-distributed PIC simulations."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ...core.utils.logger import logger, rank_log
from ..mpi.mpi_manager import MPIManager
from ..patch.patch import Patch, Patches


class LoadBalancer:
    """Handles dynamic load balancing of patches across MPI ranks.
    """

    def __init__(
        self,
        patches: Patches,
        mpi: MPIManager,
        load_function: Callable[[Patch], float] | None = None,
    ) -> None:
        """Initialize with patches (called during simulation init).

        Parameters
        ----------
        patches : Patches
            The patches to be load balanced.
        mpi : MPIManager
            The MPI manager for communication.
        load_function : callable | None, optional
            Custom function to calculate load for a single patch. The function
            should accept a `Patch` as its only parameter and return a float
            representing the load. If None, uses the default load calculation.
        """
        self.patches = patches
        self.dimension = patches.dimension
        self.comm = mpi.comm
        self.rank = mpi.rank
        self.comm_size = mpi.size
        self.load_function = load_function

    def __call__(self) -> None:
        """Execute rebalance."""
        if self.comm_size == 1:
            return

        _, _, global_loads = self._gather_loads()

        patches_list, index_to_new_rank = self._compute_distribution(global_loads)

        patches_new: Patches = self.comm.scatter(patches_list, root=0)
        index_to_new_rank = self.comm.bcast(index_to_new_rank, root=0)

        self._exchange(patches_new, index_to_new_rank)
        self._finalize(patches_new)

    @staticmethod
    def _default_load_function(patch: Patch) -> float:
        """Default load calculation for a single patch.

        Load calculation:
        - 2D: load = npart + nx*ny/2
        - 3D: load = npart + nx*ny*nz/2
        """
        load = 0.0

        for ispec in range(len(patch.particles)):
            npart_alive = (~patch.particles[ispec].is_dead).sum()
            load += npart_alive

        if hasattr(patch, 'nz'):
            load += patch.nx * patch.ny * patch.nz / 2
        else:
            load += patch.nx * patch.ny / 2

        return load

    def _gather_loads(self) -> tuple[NDArray, NDArray, NDArray | None]:
        """Gather load information from all MPI ranks."""
        load_func = self.load_function or LoadBalancer._default_load_function
        local_loads = np.array(
            [load_func(p) for p in self.patches], dtype=np.float64
        )
        local_indices = np.array([p.index for p in self.patches], dtype=np.int64)

        all_loads = self.comm.gather(local_loads, root=0)
        all_indices = self.comm.gather(local_indices, root=0)


        global_loads = None
        if self.rank == 0:
            assert all_loads is not None
            assert all_indices is not None
            npatches_total = sum(len(loads) for loads in all_loads)
            global_loads = np.zeros(npatches_total, dtype=np.float64)
            for loads, indices in zip(all_loads, all_indices):
                for load, idx in zip(loads, indices):
                    global_loads[idx] = load

        return local_loads, local_indices, global_loads

    def _compute_distribution(
        self,
        global_loads: NDArray | None,
    ) -> tuple[list[Patches] | None, dict | None]:
        """Compute new patch distribution."""
        from ..patch.metis import compute_rank

        all_patches_skeleton = self.comm.gather(
            [p.copy_skeleton() for p in self.patches], root=0
        )

        patches_list = None
        index_to_new_rank = None

        if self.rank == 0:
            assert all_patches_skeleton is not None
            assert global_loads is not None

            patches_all = Patches(self.dimension)
            for patch in sorted(
                (patch for patches in all_patches_skeleton for patch in patches),
                key=lambda patch: patch.index,
            ):
                patches_all.append(patch)

            weights = global_loads.astype(np.int64)
            new_ranks, _ = compute_rank(
                patches_all,
                nrank=self.comm_size,
                weights=weights,
                rank_prev=np.array([p.rank for p in patches_all]),
            )

            for p, new_rank in zip(patches_all.patches, new_ranks):
                p.rank = new_rank

            if self.dimension == 2:
                patches_all.init_neighbor_rank_2d()
            else:
                patches_all.init_neighbor_rank_3d()

            patches_list = [Patches(self.dimension) for _ in range(self.comm_size)]
            index_to_new_rank = {}
            for p in patches_all.patches:
                assert p.rank is not None
                patches_list[p.rank].append(p)
                index_to_new_rank[p.index] = p.rank

        return patches_list, index_to_new_rank

    def _exchange(
        self,
        patches_new: Patches,
        index_to_new_rank: dict,
    ) -> None:
        """Exchange patches between ranks."""
        import dill as pickle
        from mpi4py import MPI

        patches = self.patches
        comm = self.comm
        rank = self.rank

        old_patch_indices = {p.index: p for p in patches}
        new_patch_indices = {p.index: p for p in patches_new}

        patches_to_send_idx = [
            idx for idx in old_patch_indices if idx not in new_patch_indices
        ]
        patches_to_recv_idx = [
            idx for idx in new_patch_indices if idx not in old_patch_indices
        ]

        logger.debug(f"patches to send: {patches_to_send_idx}", comm)
        logger.debug(f"patches to receive: {patches_to_recv_idx}", comm)

        all_locations = comm.allgather({p.index: rank for p in patches})
        index_to_old_rank = {}
        for loc in all_locations:
            index_to_old_rank.update(loc)

        requests = []
        patches_to_send = []
        for idx in patches_to_send_idx:
            patches_to_send.append(patches.pop(idx))

        for p in patches_to_send:
            target_rank = index_to_new_rank[p.index]
            data = pickle.dumps(p, byref=True, recurse=True)
            logger.debug(f"sending patch {p.index} to rank {target_rank}", comm)
            req = comm.isend(data, dest=target_rank, tag=p.index)
            requests.append(req)

        for idx in patches_to_recv_idx:
            source_rank = index_to_old_rank[idx]
            if source_rank == rank:
                continue
            logger.debug(f"receiving patch {idx} from rank {source_rank}", comm)
            data = comm.recv(source=source_rank, tag=idx)
            p = pickle.loads(data)
            p.rank = rank
            patches.append(p)

        MPI.Request.waitall(requests)

    def _finalize(self, patches_new: Patches) -> None:
        """Finalize rebalance."""
        new_patch_indices = {p.index: p for p in patches_new}

        patches = self.patches
        for p in patches:
            p.neighbor_rank[:] = new_patch_indices[p.index].neighbor_rank[:]

        if self.dimension == 2:
            patches.init_neighbor_ipatch_2d()
        else:
            patches.init_neighbor_ipatch_3d()

