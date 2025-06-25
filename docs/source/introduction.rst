Introduction to λPIC
====================

The λPIC Framework
------------------
`λPIC <https://github.com/xsgeng/lambdapic>`_ is a Particle-In-Cell (PIC) framework with special focus on **laser-plasma interaction**. 
The name reflects the laser wavelength and the callback-centric architecture.

The callback-centric design enables unprecedented runtime customization of physics models, numerical methods, and diagnostics without modifying core logic - a paradigm shift from traditional PIC frameworks. It can be used as a **powerful analysis tool** and enables **rapid physics prototyping** through its flexible callback system.

Features
------------

**Callback system**:

λPIC is a powerful and flexible framework that empowers users perform any kind of diagnostics/modifications to the simulation without the constraints of interface provided by PIC frameworks.

Callback is a function that is called at specific stage during the simulation, with the :any:`Simulation` itself as argument.
It allows reading/writing simulation data during runtime. You can:

- Perform arbitrary diagnostics/outputs
- Adjust simulation data at runtime
- Visualize on the fly
- Drive other physics processeses by simulation data

The package provides built-in callbacks for common diagnostics, such as saving to :ref:`hdf5 <hdf5>`, :ref:`laser <laser>`, plotting, etc.

**Performance & Scaling**:

- Optimized PIC kernels (C/`Numba <https://github.com/numba/numba>`_ accelerated)
- Load balancing via graph partitioning with `METIS <https://github.com/KarypisLab/METIS>`_
- Efficient particle memory management via :code:`is_dead` flag

**Physics**:

- Built-in support for intense laser interactions
- QED processes (photon emission, pair production)
- Efficient user-defined profile evaluation via Numba jit
- [WIP] Collision & Nuclear physics

**Architecture & Extensibility**:

- Protocol-oriented simulation stages
- Zero-core-modification plugin system
- Callback-driven physics process prototyping
- Custom PIC construction using :code:`lambdapic.core`

Code Organization
-----------------
λPIC follows a modular architecture:

- **lambdapic.core**: Contains fundamental PIC algorithms (field solvers, particle pushers, etc.)
- **lambdapic.simulation**: Composes core components into complete simulations
- **lambdapic.callback**: Provides the callback interface and utilities

This separation allows:

- Core algorithms to be optimized independently
- Simulation logic to focus on composition and coordination
- Callbacks to modify behavior without touching core code
