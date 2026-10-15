# Installation & Usage

## System Requirements

- Python 3.13 (with headers)
- GCC >= 9.3 (for building from source or packages without wheels)
- A working MPI implementation (e.g. MPICH, OpenMPI) and the `mpicc` wrapper
- `uv` for dependency management

## Installation (uv, recommended)

Set up a project-local venv **in the user's working (research) directory** — not a global or conda environment.

1. Install `uv` if missing: see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

2. Create and activate a virtual environment in the working directory:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
```

3. Install `mpi4py` from source (compile-time dependency — must match the system MPI):

```bash
CC=mpicc uv pip install --no-binary=mpi4py --no-cache -U mpi4py
```

4. (Optional) Parallel `h5py` for parallel HDF5 output — the serial wheels are fine for most cases:

```bash
CC=mpicc HDF5_MPI=ON uv pip install --no-binary=h5py h5py
```

5. Install λPIC:

```bash
# From PyPI (recommended)
uv pip install lambdapic

# From source (development or custom builds)
git clone https://github.com/xsgeng/lambdapic.git
cd lambdapic
uv pip install -e ".[test]"
```

**Note**: `mpi4py` (step 3) must be installed before building λPIC from source, because it is required during the build.

6. Verify the installation:

```bash
python -c "import lambdapic"
```

## Alternative: Conda

If the user prefers Conda:

```bash
conda create -n lambdapic python==3.13 mpi4py h5py numpy scipy
conda activate lambdapic
pip install lambdapic
```

## Running Examples

Run an example from the repo, e.g. the 2D laser-target:

```bash
python example/laser-target.py
```

Control the thread count via `OMP_NUM_THREADS` and `NUMBA_NUM_THREADS`:

```bash
export OMP_NUM_THREADS=24
export NUMBA_NUM_THREADS=24
```

For MPI runs, the number of processes should equal the number of NUMA nodes (8 on dual-socket Epyc 9004):

```bash
# Slurm, no need to set num threads
srun -c24 -n8 -u python example/laser-target.py

# MPICH
export OMP_NUM_THREADS=24
export NUMBA_NUM_THREADS=24
mpiexec -np 8 -ppn 8 python -u example/laser-target.py
```

## Auto-reload (CLI)

The `lambdapic` CLI can auto-reload the simulation script on modification — useful in HPC environments to modify and re-run **without re-queuing** the job:

```bash
# in a sbatch script
srun lambdapic autoreload example/laser-target.py

# find something wrong in the output figures
# modify example/laser-target.py ...
# the job automatically restarts with the new script
```

**Note**: put `sim.run` inside the `__name__ == "__main__"` block to avoid running on import, since the CLI reloads and calls run.

## Troubleshooting

- Build errors: ensure GCC >= 9.3 and a working `mpicc` wrapper
- MPI issues: verify `mpi4py` works in the active environment
- Building from source requires development headers for Python, MPI, and (for parallel HDF5) HDF5
