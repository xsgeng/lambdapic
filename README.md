# λPIC

![λPIC](lambdaPIC.svg)

λPIC is a callback-centric Particle-In-Cell framework.

A Particle-In-Cell (PIC) simulation framework, allowing dynamic behavior modification through callbacks at various simulation stages. It leverages the `libpic` backend, a Python-based PIC simulation framework available at [libpic](https://github.com/xsgeng/libpic).

## Features

- **Flexible Simulation Behavior**: Modify simulation behavior through callbacks without changing core code
- **Multiple Stages**: Well-defined simulation stages for callback injection
- **Flexible Particle Support**: Support various particle types and pushers
- **Extensible**: Easy to add custom behaviors through the callback system

## Installation

```bash
pip install lambdapic
```

## Callback System

Simulation loop stages:

- `start`: Beginning of each timestep
- `maxwell first`: First half of Maxwell solver
- `push position first`: First particle position update
- `interpolator`: Field interpolation
- `push momentum`: Particle momentum update
- `push position second`: Second particle position update
- `current deposition`: Current deposition
- `maxwell second`: Second half of Maxwell solver

Each callback can be attached to any of these stages using the `@callback` decorator.

```python
# laser injector is pre-defined callback
laser = SimpleLaser(
    a0=10,
    w0=5e-6,
    ctau=5e-6
)

# Custom callbacks
@callback()
def energy_diagnostic(sim):
    if sim.itime % 100 != 0:
        return

    total_energy = 0
    for patch in sim.patches:
        # Calculate electromagnetic energy
        ex = patch.fields.ex
        ey = patch.fields.ey
        bz = patch.fields.bz
        energy = 0.5 * (ex**2 + ey**2 + bz**2).sum()
        total_energy += energy
    print(f"Total EM energy: {total_energy}")

@callback(stage="maxwell first")
def mod_ex(sim):
    # Modify fields during the Maxwell solver stage
    for patch in sim.patches:
        patch.fields.ex *= 1.1  # Amplify Ex field by 10%

# Run simulation with callback
sim.run(
    nsteps=1000, 
    callbacks=[
        laser,
        energy_diagnostic,
        mod_ex, 
        callback(stage="start")(lambda sim: print(f"timestep @ {sim.itime} starts")),
        lambda sim: print(f"timestep @ {sim.itime} ends")
    ]
)
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Development Status

λPIC is currently in active development. The API may change without notice.