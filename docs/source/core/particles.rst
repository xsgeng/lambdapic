Particle Data
~~~~~~~~~~~~~

ParticlesBase
-------------

The dataclass for particle data. It stores, allocates, extends, prunes data.

.. important::
    The particle data contains dead particles. 

    To access only the live particles, use the :any:`is_alive` property: 

    like :code:`particles.ux[particles.is_alive] += 10` and :code:`np.histogram(particles.ux[particles.is_alive], weights=particles.w[particles.is_alive])`.

.. autoclass:: lambdapic.core.particles.ParticlesBase
    :members:
    :member-order: bysource
    :special-members: 
    
QEDParticles
------------

Inherited from :any:`ParticlesBase`, it adds additional attributes for QED processes.

.. autoclass:: lambdapic.core.particles.QEDParticles
    :members:
    :member-order: bysource
    :show-inheritance:

SpinParticles
-------------

Inherited from :any:`ParticlesBase`, it adds the ``sx``, ``sy``, ``sz`` spin
arrays. It is created automatically when the species has a ``polarization``
set, and the spin is advanced by the TBMT pusher (``pusher='boris+tbmt'``).

.. autoclass:: lambdapic.core.particles.SpinParticles
    :members:
    :member-order: bysource
    :show-inheritance:

SpinQEDParticles
----------------

Combines :any:`QEDParticles` and :any:`SpinParticles`: a polarized species with
radiation enabled carries QED and spin data together.

.. autoclass:: lambdapic.core.particles.SpinQEDParticles
    :members:
    :member-order: bysource
    :show-inheritance: