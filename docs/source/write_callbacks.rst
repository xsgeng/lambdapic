Writing your own callbacks
===========================

Hello world
~~~~~~~~~~~~

.. code-block:: python

    @callback('start')
    def hello(sim: Simulation):
        if sim.itime == 0:
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

    @callback('start')
    def sum_EM_enerty(sim: Simulation):
        # calculate every 100 time steps
        if it = sim.itime % 100 == 0:
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

    @callback('start')
    def sum_ek(sim: Simulation):
        # calculate every 100 time steps
        if it = sim.itime % 100 == 0:
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