Installation & Usage
====================

System Requirements
-------------------
- Python 3.13 (with headers)
- GCC >= 9.3 (for building from source or packages without wheels)
- A working MPI implementation (e.g. MPICH, OpenMPI) and ``mpicc`` wrapper
- ``uv`` (recommended for dependency management)

Installation
------------

1. Install ``uv``

   If you do not already have ``uv`` installed, see the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

2. Create a virtual environment

   .. code-block:: bash

      uv venv --python 3.13 .venv
      source .venv/bin/activate

3. Install ``mpi4py`` from source

   ``mpi4py`` is a compile-time dependency of λPIC. Build it with the system MPI compiler so it matches your MPI environment.

   .. code-block:: bash

      CC=mpicc uv pip install --no-binary=mpi4py --no-cache -U mpi4py

4. (Optional) Install ``h5py`` with MPI support

   λPIC works with the standard serial ``h5py`` wheels. If you need parallel HDF5 output, build ``h5py`` against an MPI-enabled HDF5 library:

   .. code-block:: bash

      CC=mpicc HDF5_MPI=ON uv pip install --no-binary=h5py h5py

   The serial version is sufficient for most use cases.

5. Install λPIC

   From PyPI (recommended for most users):

   .. code-block:: bash

      uv pip install lambdapic

   From source (for development or custom builds):

   .. code-block:: bash

      git clone https://github.com/xsgeng/lambdapic.git
      cd lambdapic
      uv pip install -e .

   With test dependencies:

   .. code-block:: bash

      uv pip install -e ".[test]"

   .. note::

      ``mpi4py`` must be installed (step 3) before installing λPIC from source, because it is required during the build.

Alternative: Using Conda
------------------------

If you prefer Conda, create an equivalent environment:

.. code-block:: bash

   conda create -n lambdapic python==3.13 mpi4py h5py numpy scipy
   conda activate lambdapic

Then install λPIC with ``pip`` or ``uv pip`` inside the activated environment:

.. code-block:: bash

   pip install lambdapic

Running Examples
----------------

See :doc:`example`, or use examples from the repo. To run the 2D laser-target example:

.. code-block:: bash

   python example/laser-target.py

These examples demonstrate basic functionality and can be modified for your own simulations.

You can set number of threads via :code:`OMP_NUM_THREADS` and :code:`NUMBA_NUM_THREADS`.

.. code-block:: bash

   export OMP_NUM_THREADS=24
   export NUMBA_NUM_THREADS=24

For MPI run, the number of processes should be equal to the number of NUMA nodes. On Epyc 9004, the number of NUMA nodes is 8 on dual socket nodes.

.. code-block:: bash

   # Slurm, no need to set num threads
   srun -c24 -n8 -u python example/laser-target.py

   # MPICH
   export OMP_NUM_THREADS=24
   export NUMBA_NUM_THREADS=24
   mpiexec -np 8 -ppn 8 python -u example/laser-target.py

Auto-reload functionality
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also run from the λPIC commandline interface (CLI).

Currently, the CLI supports autoreload functionality. The simulation script will be automatically reloaded on modification. 
This is useful in HPC environments when you want to modify & re-run the simulation **without re-queuing** the job. 

.. code-block:: bash

   # in a sbatch script
   #SBATCH -c24
   #SBATCH -n8
   #SBATCH -u
   ...
   srun lambdapic autoreload example/laser-target.py

   # find somethig wrong in the output figures.
   # modify example/laser-target.py ...
   # job will automatically restart with new script

.. note::
   You should put the :code:`sim.run` in the :code:`__name__ == "__main__"` block to avoid run on import,
   since the CLI will reload and call run.

Troubleshooting
---------------

- If you encounter build errors, ensure you have GCC 9.3 or newer and a working ``mpicc`` wrapper
- For MPI-related issues, verify ``mpi4py`` is working in your environment
- Building from source requires development headers for Python, MPI, and (if building parallel HDF5) HDF5
