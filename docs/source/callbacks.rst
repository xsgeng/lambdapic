Built-in Callbacks
==================

Plot on-the-fly
---------------

.. autoclass:: lambdapic.callback.plot.PlotFields

.. _hdf5:

HDF5
----

Callbacks for saving simulation data to HDF5 format. These allow saving:

- Electromagnetic fields
- Species densities
- Individual particle data

These callbacks perform parallel writes without need for parallel-hdf5, by initializing chunked dataset on rank 0 then performing parallel writes of patches sequentially on each rank.

Both :any:`SaveFieldsToHDF5` and :any:`SaveSpeciesDensityToHDF5` support an optional ``slice`` parameter for ``np.s_``-style subset selection, e.g. ``slice=np.s_[:, :, 100]`` or ``slice=np.s_[::2, ::2, ::5]``. :any:`SaveFieldsToHDF5` additionally supports an ``mpi`` parameter to control MPI collective I/O behavior.

SaveFieldsToHDF5
~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.hdf5.SaveFieldsToHDF5

SaveSpeciesDensityToHDF5
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.hdf5.SaveSpeciesDensityToHDF5

SaveParticlesToHDF5
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.hdf5.SaveParticlesToHDF5

RestartDump
------------

Callback that dumps simulation checkpoints using dill pickling. Runs at the ``"end"`` stage of each timestep. Checkpoints can be reloaded using :any:`RestartDump.load`.

.. autoclass:: lambdapic.callback.restart.RestartDump
   :members:

MovingWindow
------------

Callback for implementing a moving window that follows the laser pulse.
This maintains high resolution in the region of interest while reducing
computational cost by dropping trailing cells.

.. autoclass:: lambdapic.callback.utils.MovingWindow

.. _laser:

Lasers
------

Laser injection callbacks for PIC simulations.

Simple laser
~~~~~~~~~~~~

Use :any:`SimpleLaser2D` or :any:`SimpleLaser3D`. Supports additional parameters for laser positioning and orientation:

- ``y0``, ``z0``: laser center position (defaults to ``Ly/2``, ``Lz/2``)
- ``angle_y``, ``angle_z``: incident angle in y and z direction (defaults to 0; ``angle_z`` is not implemented and must be 0)
- ``cep``: carrier envelope phase (default: 0)

.. autoclass:: lambdapic.callback.laser.SimpleLaser
.. autoclass:: lambdapic.callback.laser.SimpleLaser2D
.. autoclass:: lambdapic.callback.laser.SimpleLaser3D

Gaussian laser
~~~~~~~~~~~~~~

Use :any:`GaussianLaser2D` or :any:`GaussianLaser3D`.

.. autoclass:: lambdapic.callback.laser.GaussianLaser
.. autoclass:: lambdapic.callback.laser.GaussianLaser2D
.. autoclass:: lambdapic.callback.laser.GaussianLaser3D

Combining lasers
~~~~~~~~~~~~~~~~

Lasers can be combined using the ``+`` operator to inject multiple lasers in a single callback::

    combined = laser1 + laser2

Both lasers must be from the same side and the same dimension (2D or 3D). The resulting combined laser behaves as a single callback that injects all constituent lasers.

Utility callbacks
-----------------

SetTemperature
~~~~~~~~~~~~~~

Set the temperature of a species to a given value in eV.

.. autoclass:: lambdapic.callback.utils.SetTemperature

ExtractSpeciesDensity
~~~~~~~~~~~~~~~~~~~~~

Extract the density of a species to buffer. Supports an optional ``slice`` parameter for ``np.s_``-style subset selection.

.. autoclass:: lambdapic.callback.utils.ExtractSpeciesDensity

LoadParticles
~~~~~~~~~~~~~

Load particles from a hdf5 file.

.. autoclass:: lambdapic.callback.utils.LoadParticles