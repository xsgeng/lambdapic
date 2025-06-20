Callbacks
==========

HDF5
----

Callbacks for saving simulation data to HDF5 format. These allow saving:
- Electromagnetic fields
- Particle species densities
- Individual particle data

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

Lasers
------

Laser injection callbacks for PIC simulations.

Simple laser
~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.laser.SimpleLaser

Gaussian laser
~~~~~~~~~~~~~~

.. autoclass:: lambdapic.callback.laser.GaussianLaser
