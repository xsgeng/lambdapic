# λPIC

![λPIC](lambdaPIC.svg)

λPIC represents the callback-centric design of this Particle-In-Cell framework.

A Particle-In-Cell (PIC) simulation framework, allowing dynamic behavior modification through callbacks at various simulation stages.

## Features

- **Flexible Simulation Behavior**: Modify simulation behavior through callbacks without changing core code
- **Multiple Stages**: Well-defined simulation stages for callback injection
- **Flexible Particle Support**: Support various particle types and pushers
- **Extensible**: Easy to add custom behaviors through the callback system

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pydantic

## Installation

```bash
pip install lambdapic
```

## Quick Start

Here's a simple example of using λPIC:

```python
from lambdapic import Simulation, Species
from lambdapic.callback import Callback

# Create simulation
sim = Simulation(nx=100, ny=100, dx=1e-6, dy=1e-6, 
                npatch_x=2, npatch_y=2)

# Add species
electron = Species(name="electron", mass=1, charge=-1)
sim.add_species([electron])

# Define a callback
@Callback(stage="maxwell first")
def custom_field_modification(sim):
    # Modify fields during the Maxwell solver stage
    for patch in sim.patches:
        patch.fields.ex *= 1.1  # Amplify Ex field by 10%

# Run simulation with callback
sim.run(nsteps=1000, callbacks=[custom_field_modification])
```

## Configuration Options

### Simulation Parameters

- `nx`, `ny`: Number of cells in x and y directions
- `dx`, `dy`: Cell sizes in x and y directions
- `npatch_x`, `npatch_y`: Number of patches in x and y directions
- `dt_cfl`: CFL condition factor (default: 0.95)
- `n_guard`: Number of guard cells (default: 3)
- `cpml_thickness`: CPML boundary thickness (default: 6)

### Species Parameters

- `name`: Species name
- `mass`: Particle mass
- `charge`: Particle charge
- `pusher`: Particle pusher type ("boris" or "photon")

## Callback System

λPIC uses a callback system with the following stages:

- `start`: Beginning of each timestep
- `maxwell first`: First half of Maxwell solver
- `push position first`: First particle position update
- `interpolator`: Field interpolation
- `push momentum`: Particle momentum update
- `push position second`: Second particle position update
- `current deposition`: Current deposition
- `maxwell second`: Second half of Maxwell solver

Each callback can be attached to any of these stages using the `@Callback` decorator.

## Advanced Examples

### Custom Diagnostics

```python
@Callback(stage="maxwell second")
def energy_diagnostic(sim):
    total_energy = 0
    for patch in sim.patches:
        # Calculate electromagnetic energy
        ex = patch.fields.ex
        ey = patch.fields.ey
        bz = patch.fields.bz
        energy = 0.5 * (ex**2 + ey**2 + bz**2).sum()
        total_energy += energy
    print(f"Total EM energy: {total_energy}")
```

### Laser Injection

```python
@Callback(stage="maxwell first")
def inject_laser(sim):
    # Inject a Gaussian laser pulse
    for patch in sim.patches:
        if patch.ipatch_x == 0:  # Left boundary
            x = patch.x
            y = patch.y
            t = sim.time
            omega = 2 * pi * c / wavelength
            envelope = exp(-(y - Ly/2)**2 / w0**2)
            patch.fields.ey += envelope * sin(omega * (t - x/c))
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Development Status

λPIC is currently in active development. The API may change without notice until version 1.0.0.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
