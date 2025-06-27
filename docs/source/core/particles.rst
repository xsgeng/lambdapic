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
    :special-members: __init__
    
QEDParticles
------------

Inherited from :any:`ParticlesBase`, it adds additional attributes for QED processes.

.. autoclass:: lambdapic.core.particles.QEDParticles
    :members:
    :member-order: bysource
    :special-members: __init__
    :show-inheritance: