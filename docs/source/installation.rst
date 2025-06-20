Installation Guide
==================

System Requirements
-------------------
- Python 3.8+
- GCC >= 9.3 (for building from source)
- Conda (recommended for dependency management)

Installation
--------------------

1. Install prebuilt packages from Conda (recommended)

First install prebuilt packages from conda like mpi4py and h5py which can be difficult to build:

.. code-block:: bash

   conda create -n lambdapic mpi4py h5py numpy scipy
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

Troubleshooting
---------------

- If you encounter build errors, ensure you have GCC 9.3 or newer installed
- For MPI-related issues, verify mpi4py is working in your conda environment
- Building from source requires development headers for Python and MPI
