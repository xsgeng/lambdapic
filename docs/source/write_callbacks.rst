Writing your own callbacks
===========================

The :code:`@callback` decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`@callback` decorator adds :code:`stage` and :code:`interval` attributes to the function.

The :code:`stage` specifies when the callback should be called. Leave it empty to run the callback at the end of each loop.

The :code:`interval` specifies how frequent the callback should be called. It can be a number or a function that returns a boolean.
:code:`interval = lambda sim: sim.itime == 42` means the callback will be called at the 42nd iteration.

The decorator also stores both :code:`stage` and :code:`interval` as attributes on the wrapper, so they can be inspected programmatically.

By passing to the :code:`sim.run(1000, callbacks=[your callbacks])`, they will be sequentially called by the :code:`Simulation.run` method.

.. note::
    The :code:`@callback` decorator and :code:`Callback` base class detect whether the process
    is running in a terminal (via :code:`is_terminal()`). In an interactive terminal, a yaspin
    spinner is displayed during callback execution. In non-terminal environments (e.g. batch
    jobs, pipes, or log files), callback execution is logged via the logger instead. Built-in
    callbacks inheriting from :code:`Callback` share this terminal-aware behavior.

.. autoclass:: lambdapic.callback.callback.callback

Class-based callbacks
~~~~~~~~~~~~~~~~~~~~~

Built-in callbacks inherit from the :code:`Callback` base class. For class-based callbacks, define a :code:`DEFAULT_STAGE` class attribute and assign it to :code:`self.stage` in :code:`__init__`. This pattern lets users override the stage at runtime by assigning a different value to :code:`self.stage` after instantiation.

.. code-block:: python

    from lambdapic.callback.callback import Callback

    class MyCallback(Callback):
        DEFAULT_STAGE = "start"
        def __init__(self):
            self.stage = self.DEFAULT_STAGE
            self.interval = 1
        def _call(self, sim):
            print(f"Step {sim.itime}")

    cb = MyCallback()
    # Override the stage at runtime if desired
    cb.stage = "maxwell_1"

Hello world
~~~~~~~~~~~~

.. code-block:: python

    @callback('start', interval=lambda sim: sim.itime == 0)
    def hello(sim: Simulation):
        print("Simulation started!")
    sim = Simulation(...)
    ...

    sim.run(100, callbacks=[hello])

External fields
~~~~~~~~~~~~~~~

Set static external fields by adding fields to particles's local fields.

.. code-block:: python

    @callback('interpolator')
    def set_static_fields(sim: Simulation):
        for p in sim.patches:
            for part in p.particles:
                part.bz_part[:] += 10 # 10T static
                part.ex_part[:] += np.sin(sim.t) # time dependent
                part.ey_part[:] += np.sin(part.x/1e-6) # space dependent

Or faster with numba

.. code-block:: python

    @njit(parallel=True)
    def set_static_fields(x, is_dead, t, ex_part):
        for ipart in prange(ex_part.size):
            if is_dead[ipart]:
                continue
            ex_part[ipart] += 10 # 10T static
            ex_part[ipart] += np.sin(t) # time dependent
            ex_part[ipart] += np.sin(x[ipart]/1e-6) # space dependent

    @callback('interpolator')
    def set_static_fields(sim: Simulation):
        for p in sim.patches:
            part = p.particles[ele.ispec]
            set_static_fields(part.x, part.is_dead, sim.t, part.ex_part)

Reduction/Summation
~~~~~~~~~~~~~~~~~~~

Calculate total EM energy,

.. code-block:: python

    sim = Simulation(...)
    ele = Electron(name='ele', ppc=10, density=...)
    ...

    @callback('start', interval=100)
    def sum_EM_enerty(sim: Simulation):
        Eem = 0.0
        # sum over all patches
        for p in sim.patches:
            f = p.fields
            # NOTE: guard cells are in the [nx_per_patch:, ny_per_patch:] region
            s = np.s_[:sim.nx_per_patch, :sim.ny_per_patch]
            Eem += (0.5*epsilon_0*(f.ex[s]**2+f.ey[s]**2+f.ez[s]**2) + 
                    0.5/mu_0     *(f.bx[s]**2+f.by[s]**2+f.bz[s]**2)).sum()

        # sum over all mpi ranks
        Eem = sim.mpi.comm.reduce(Eem)
        if sim.mpi.rank > 0:
            return
        
        # print, or save to some file
        print(f"{Eem=:g}")

and total electron kinetic energy.

.. code-block:: python

    @callback('start', interval=100)
    def sum_ek(sim: Simulation):
        ek = 0.0

        # sum over all patches
        for p in sim.patches:
            part = p.particles[ele.ispec]
            # select alive particles
            alive = part.is_alive
            ek += ((1/part.inv_gamma[alive] - 1) * ele.m/m_e * part.w[alive]).sum() # mc2

        # sum over all mpi ranks
        ek = sim.mpi.comm.reduce(ek)
        if sim.mpi.rank > 0:
            return
        
        # print, or save to some file
        print(f"{ek=:g} mc2")

Third-party typed.List management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numba kernels can iterate per-patch arrays through ``numba.typed.List``
references. These references must be refreshed after ``extend``/``prune``
replaces particle arrays and rebuilt after load-balance migration or a
moving-window patch rebuild.

:meth:`~lambdapic.simulation.Simulation.register_typed_list_holder` manages
this lifecycle for third-party callbacks:

.. code-block:: python

    class BeamKicker(Callback):
        stage = "current_deposition"

        def __init__(self, species):
            self.interval = 1
            self.species = species

        def _call(self, sim):
            ispec = self.species.ispec
            kick_patches(self.x_list[ispec], self.ux_list[ispec],
                         sim.patches.npatches)

    kicker = BeamKicker(ele)
    sim.register_typed_list_holder(
        kicker,
        particle_lists={"x_list": "x", "ux_list": "ux"},
        species=(ele,),
    )

Particle lists are stored as ``dict[int, typed.List]`` keyed by species
index. Field lists are flat typed lists over patches. Both reference the
live arrays, so writes go directly into simulation state.

Diagnostics
^^^^^^^^^^^

Registration rejects unknown species or source attributes, list-name
collisions, and shadowed holder attributes.
:meth:`~lambdapic.simulation.Simulation.check_typed_list_freshness` returns
structured :class:`~lambdapic.simulation.typed_list_registry.TypedListFreshnessIssue`
entries for stale lists, unbound registrations, missing watched species,
and unregistered callback holders. The run loop emits
unbound/unregistered-holder warnings once after the ``init`` stage.

.. code-block:: python

    issues = sim.check_typed_list_freshness(callbacks=my_callbacks)
    assert not issues

Checkpoint contract
^^^^^^^^^^^^^^^^^^^

Checkpoints preserve registration names, list specifications, and species
watch sets, but drop holder objects. After restoring, re-register each
holder under the same name to rebuild its lists from the restored arrays.
Unbound registrations are diagnosed at run startup. Checkpoints written
before the registry existed restore as an empty registry.

.. autoclass:: lambdapic.simulation.typed_list_registry.TypedListRegistry
   :members:
