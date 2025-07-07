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

SaveFieldsToHDF5
~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.hdf5.SaveFieldsToHDF5
    :special-members: __call__

SaveSpeciesDensityToHDF5
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.hdf5.SaveSpeciesDensityToHDF5
    :special-members: __call__

SaveParticlesToHDF5
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.hdf5.SaveParticlesToHDF5
    :special-members: __call__

MovingWindow
------------

Callback for implementing a moving window that follows the laser pulse.
This maintains high resolution in the region of interest while reducing
computational cost by dropping trailing cells.

.. autoclass:: lambdapic.callback.utils.MovingWindow
    :special-members: __call__

.. _laser:

Lasers
------

Laser injection callbacks for PIC simulations.

Simple laser
~~~~~~~~~~~~

Use :any:`SimpleLaser2D` or :any:`SimpleLaser3D`.

.. autoclass:: lambdapic.callback.laser.SimpleLaser
.. autoclass:: lambdapic.callback.laser.SimpleLaser2D
.. autoclass:: lambdapic.callback.laser.SimpleLaser3D

Gaussian laser
~~~~~~~~~~~~~~

Use :any:`GaussianLaser2D` or :any:`GaussianLaser3D`.

.. autoclass:: lambdapic.callback.laser.GaussianLaser
.. autoclass:: lambdapic.callback.laser.GaussianLaser2D
.. autoclass:: lambdapic.callback.laser.GaussianLaser3D

Utility callbacks
~~~~~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.utils.SetTemperature
.. autoclass:: lambdapic.callback.utils.ExtractSpeciesDensity