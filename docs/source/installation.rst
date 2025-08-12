Installation & Usage
==================

System Requirements
-------------------
- Python with headers, tested on 3.12
- GCC >= 9.3 (for building from source)
- Conda (recommended for dependency management)

Installation
--------------------

1. Install prebuilt packages from Conda (recommended)

First install prebuilt packages from conda like mpi4py and h5py which can be difficult to build:

.. code-block:: bash

   conda create -n lambdapic python==3.12 mpi4py h5py numpy scipy
   conda activate lambdapic

2. Install λPIC

From PyPI (recommended for most users):

.. code-block:: bash

   pip install lambdapic

From source (for development or custom builds):

.. code-block:: bash

   git clone https://github.com/xsgeng/lambdapic.git
   cd lambdapic
   pip install .

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

- If you encounter build errors, ensure you have GCC 9.3 or newer installed
- For MPI-related issues, verify mpi4py is working in your conda environment
- Building from source requires development headers for Python and MPI
