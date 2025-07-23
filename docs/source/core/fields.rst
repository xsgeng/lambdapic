Fields Data
~~~~~~~~~~~

Fields
------

Base class for fields data.

.. important::
    Fields data are stored in :code:`[:nx, :ny, :nz]` range, and the guard cells are in the :code:`[nx:, ny:, nz:]` range.
    The guard cells are therefore accessed using :code:`[-n_guard:, -n_guard:, -n_guard:]` and :code:`[nx:nx+n_guard, ny:ny+n_guard, nz:nz+n_guard]`.

.. autoclass:: lambdapic.core.fields.Fields
    :members:
    :member-order: bysource

Fields2D & Fields3D
-------------------

.. autoclass:: lambdapic.core.fields.Fields2D
    :members:
    :member-order: bysource
    :show-inheritance:

.. autoclass:: lambdapic.core.fields.Fields3D
    :members:
    :member-order: bysource
    :show-inheritance: