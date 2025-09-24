import signal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import dill as pickle

from ..core.mpi.mpi_manager import MPIManager
from ..core.utils.logger import logger
from ..simulation import Simulation, Simulation3D
from .callback import Callback


class RestartDump(Callback):
    """Callback to dump restart checkpoints per rank.

    Creates a directory per checkpoint with a manifest and one shard per rank:
      - <out_dir>/ckpt_<itime>/rank_<rank>.pkl
    """

    stage = "maxwell second"

    def __init__(self, out_dir: Union[str, Path], interval: Union[int, float, Callable] = 1000, keep: Optional[int] = None, 
                 dump_signals: list[int]|bool=False) -> None:
        self.out_dir = Path(out_dir)
        self.interval = interval
        self.keep = keep
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if dump_signals is False:
            self.dump_signals = []
        elif dump_signals is True:
            self.dump_signals = [signal.SIGINT, signal.SIGTERM]
        else:
            self.dump_signals = dump_signals

        for sig in self.dump_signals:
            signal.signal(sig, self._dump_handler)

        self._dump_requested = False

    def _dump_handler(self, sig, frame):
        self._dump_requested = True

    # ---------------------- save path helpers ----------------------
    def _ckpt_dir(self, itime: int) -> Path:
        return self.out_dir / f"ckpt_{itime:06d}"

    def _rank_shard_path(self, itime: int, rank: int) -> Path:
        return self._ckpt_dir(itime) / f"rank_{rank:06d}.pkl"
    
    # ---------------------- callback entry ----------------------
    def _call(self, sim: Union[Simulation, Simulation3D]):
        comm = sim.mpi.comm
        rank = sim.mpi.rank

        ckpt_dir = self._ckpt_dir(sim.itime)
        if rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        # All ranks write shards
        with open(self._rank_shard_path(sim.itime, rank), "wb") as f:
            pickle.dump(sim, f, byref=True, recurse=True)

        comm.Barrier()

        # Optionally trim old checkpoints (rank 0 only)
        if rank == 0 and self.keep is not None and self.keep > 0:
            self._gc_old_checkpoints(self.keep)

        comm.Barrier()

    def _gc_old_checkpoints(self, keep: int) -> None:
        # Keep most recent N ckpt_* directories
        subdirs = [d for d in self.out_dir.iterdir() if d.is_dir() and d.name.startswith("ckpt_")]
        subdirs.sort(key=lambda p: p.name)
        if len(subdirs) <= keep:
            return
        to_delete = subdirs[: len(subdirs) - keep]
        for d in to_delete:
            try:
                # remove directory recursively
                for path in sorted(d.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if path.is_file():
                        path.unlink(missing_ok=True)
                    elif path.is_dir():
                        path.rmdir()
                d.rmdir()
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {d}: {e}")

    # ---------------------- loader ----------------------
    @staticmethod
    def load(ckpt_dir: Union[str, Path], comm=None) -> Union[Simulation, Simulation3D]:
        """Load a Simulation from a RestartDump checkpoint directory.

        Args:
            ckpt_dir: Path to a single checkpoint directory (ckpt_xxxxxx).
            comm: Optional MPI communicator to use.

        Returns:
            Simulation or Simulation3D instance restored to the checkpoint state.
        """
        ckpt_dir = Path(ckpt_dir)

        if comm is None:
            comm = MPIManager.get_default_comm()

        rank = comm.Get_rank()

        shard_path = ckpt_dir / f"rank_{rank:06d}.pkl"
        with open(shard_path, "rb") as f:
            sim = pickle.load(f)

        sim.update_lists()

        # inc by 1, since restart is called before itime inc
        sim.itime += 1
        comm.Barrier()

        logger.info(f"Rank {rank}: Checkpoint loaded from {ckpt_dir}, itime={sim.itime}")
        return sim

