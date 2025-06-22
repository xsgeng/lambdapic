Introduction to λPIC
====================

The λPIC Framework
------------------
λPIC is a Particle-In-Cell (PIC) framework with special focus on **laser-plasma interaction**. 
The name reflects the laser wavelength and the callback-centric architecture.

The callback-centric design enables unprecedented runtime customization of physics models, numerical methods, and diagnostics without modifying core logic - a paradigm shift from traditional PIC frameworks.

Features
------------

**Simulation Control**:

- Modify physics during runtime via callbacks
- Adjust simulation data at runtime
- Runs in notebook, visualization on the fly

**Performance & Scaling**:

- Optimized PIC kernels (C/`Numba <https://github.com/numba/numba>`_ accelerated)
- `METIS <https://github.com/KarypisLab/METIS>`_-based load balancing
- Efficient particle memory management via :code:`is_dead` flag

**Physics**:

- Built-in support for intense laser interactions
- QED processes (photon emission, pair production)
- Efficient profile evaluation via Numba jit

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

