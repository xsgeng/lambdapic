Simulation classes
==================

.. autoclass:: lambdapic.simulation.Simulation
   :members: 

.. autoclass:: lambdapic.simulation.Simulation3D
   :members: 
   :show-inheritance:

.. autoclass:: lambdapic.simulation.SimulationCallbacks
   :members: 

.. _simulation-integrated-features:

Integrated Features
-------------------

The :class:`~lambdapic.simulation.Simulation` class includes several features that
are activated automatically during :meth:`~lambdapic.simulation.Simulation.run`:

Progress Bar
~~~~~~~~~~~~

When calling ``run()``, a :class:`~lambdapic.core.utils.progress_bar.ProgressBar` is
created automatically. The progress bar detects whether the output is a terminal or
a log file and adapts its display accordingly: in terminals it uses ``tqdm`` for an
interactive progress bar, and in non-terminal environments it emits structured log
messages at regular intervals.

Dynamic Load Balancing
~~~~~~~~~~~~~~~~~~~~~~

During ``run()``, a :class:`~lambdapic.core.mpi.load_balancer.LoadBalancer` monitors
the computational load across MPI ranks. When the load imbalance exceeds a configurable
threshold, the load balancer triggers automatic patch rebalancing to redistribute work
evenly. After rebalancing, :meth:`~lambdapic.simulation.Simulation.update_patches` is
called automatically to regenerate field and particle lists for all simulation modules.

The load of each patch is calculated as::

    load = npart + (nx * ny [* nz]) / 2

where ``npart`` is the number of alive particles and the grid term accounts for field
computation cost. This calculation is performed internally by
:meth:`~lambdapic.simulation.Simulation._calculate_patch_loads`.