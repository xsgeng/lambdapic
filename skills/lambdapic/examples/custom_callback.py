#!/usr/bin/env python3
"""
lambdapic Callback Template

Six practical callback examples covering common use cases:
1. Diagnostic printing (rank‑0 output)
2. Field modification (adding external fields)
3. Particle operations (energy summation)
4. Field plotting (matplotlib visualization)
5. Custom interval (conditional firing)
6. Class‑based callback (alternative style)

Each example includes MPI‑aware code and best practices.
"""

import numpy as np
from pathlib import Path

# Import lambdapic components
from lambdapic import (
    callback,
    e,
    m_e,
    c,
    pi,
    epsilon_0,
    mu_0,
    get_fields,
)
from lambdapic.callback.callback import Callback
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# EXAMPLE 1: DIAGNOSTIC PRINTING (RANK‑0 OUTPUT)
# ============================================================================

@callback(stage='start', interval=100)
def print_simulation_status(sim):
    """
    Print basic simulation information on rank 0.

    This callback runs every 100 steps at the 'start' stage.
    Demonstrates MPI‑gated output and accessing simulation attributes.
    """
    if sim.mpi.rank == 0:
        print(f"Step {sim.itime:6d}, time = {sim.time:.3e} s, "
              f"dt = {sim.dt:.3e} s")
        # Optional: print field energy or other diagnostics
        # (see Example 3 for energy summation)


# ============================================================================
# EXAMPLE 2: FIELD MODIFICATION (ADDING EXTERNAL FIELDS)
# ============================================================================

@callback(stage='_interpolator')
def add_static_magnetic_field(sim):
    """
    Add a static magnetic field in the z‑direction.

    This callback runs during the '_interpolator' stage, which is
    inside the species loop. Fields added here affect particle pushing.

    Note: This modifies fields at particle positions, not grid fields.
    For modifying grid fields, use stage 'maxwell_1' or 'maxwell_2'.
    """
    for p in sim.patches:
        for part in p.particles:
            # Add 10 Tesla static Bz field at particle positions
            part.bz_part[:] += 10.0
            # Optionally add time‑dependent or space‑dependent fields
            # part.ex_part[:] += np.sin(sim.time)  # time‑dependent Ex
            # part.ey_part[:] += np.sin(part.x/1e-6)  # space‑dependent Ey


# ============================================================================
# EXAMPLE 3: PARTICLE OPERATIONS (ENERGY SUMMATION)
# ============================================================================

@callback(stage='start', interval=200)
def sum_electron_kinetic_energy(sim):
    """
    Calculate total electron kinetic energy across all patches and MPI ranks.

    Demonstrates:
    - Looping over patches and particles
    - Using is_alive mask to select active particles
    - MPI reduction to rank 0
    - Physical units (energy in joules)
    """
    # Assuming an Electron species named 'ele' exists
    # In practice, you would need access to the species object
    # For this example, we assume the first species is electrons
    if len(sim.species) == 0:
        return

    # Find electron species (simplified: first species)
    # CUSTOMIZE: Replace with your species reference
    ele = sim.species[0]

    ek_total = 0.0
    for p in sim.patches:
        part = p.particles[ele.ispec]
        alive = part.is_alive  # boolean mask of alive particles

        if np.any(alive):
            # Kinetic energy per particle: (γ‑1) * m * c² * weight
            # inv_gamma = 1/γ, so γ = 1/inv_gamma
            gamma = 1.0 / part.inv_gamma[alive]
            # mc² in joules: ele.m * m_e * c²
            mc2 = ele.m * m_e * c**2
            ek_total += ((gamma - 1.0) * mc2 * part.w[alive]).sum()

    # Sum across all MPI ranks
    ek_total = sim.mpi.comm.reduce(ek_total)

    if sim.mpi.rank == 0:
        print(f"Total electron kinetic energy: {ek_total:.3e} J")


# ============================================================================
# EXAMPLE 4: FIELD PLOTTING (MATPLOTLIB VISUALIZATION)
# ============================================================================

@callback(stage='end', interval=500)
def plot_field_snapshot(sim):
    """
    Create a 2D plot of electric field and charge density.

    Demonstrates:
    - Using get_fields() to assemble global field arrays on rank 0
    - Creating matplotlib figures with custom colormaps
    - Saving plots to disk with unique filenames
    - Proper MPI gating (plotting only on rank 0)
    """
    # Get global field arrays (returns None on worker ranks)
    ex, ey, ez, bx, by, bz, jy, rho = get_fields(
        sim, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jy', 'rho']
    )

    # Only rank 0 has non‑None arrays
    if sim.mpi.rank > 0:
        return

    # Normalize Ey to laser a0 scale for interpretation
    um = 1e-6
    l0 = 0.8 * um
    omega0 = 2 * pi * c / l0
    ey_normalized = ey * e / (m_e * c * omega0)

    # Create custom colormap for overlapping fields
    bwr_alpha = LinearSegmentedColormap(
        'bwr_alpha', dict(
            red=[(0,0,0),(0.5,1,1),(1,1,1)],
            green=[(0,0.5,0),(0.5,1,1),(1,0,0)],
            blue=[(0,1,1),(0.5,1,1),(1,0,0)],
            alpha=[(0,1,1),(0.5,0,0),(1,1,1)]
        )
    )

    fig, ax = plt.subplots(figsize=(5,3), layout='constrained')

    # Plot charge density (grayscale)
    h1 = ax.imshow(rho.T, extent=[0, sim.Lx, 0, sim.Ly],
                   origin='lower', cmap='Greys')

    # Plot normalized Ey (semi‑transparent)
    h2 = ax.imshow(ey_normalized.T, extent=[0, sim.Lx, 0, sim.Ly],
                   origin='lower', cmap=bwr_alpha, alpha=0.7)

    fig.colorbar(h1, label='ρ [C/m³]')
    fig.colorbar(h2, label='Ey [a0]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Step {sim.itime}, t = {sim.time:.2e} s')

    # Save figure
    output_dir = Path('field_plots')
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / f'fields_{sim.itime:06d}.png', dpi=300)
    plt.close(fig)


# ============================================================================
# EXAMPLE 5: CUSTOM INTERVAL (CONDITIONAL FIRING)
# ============================================================================

@callback(
    stage='end',
    interval=lambda sim: 100e-15 <= sim.time <= 200e-15 and sim.itime % 10 == 0
)
def time_window_analysis(sim):
    """
    High‑frequency sampling during interaction phase only.

    Combines physical time window with step‑based interval for efficient
    data collection during critical simulation phases.
    """
    if sim.mpi.rank == 0:
        print(f"Interaction phase: step {sim.itime}, t = {sim.time:.3e} s")


# ============================================================================
# EXAMPLE 6: CLASS‑BASED CALLBACK (ALTERNATIVE STYLE)
# ============================================================================

class SaveEnergyToFile(Callback):
    """
    Class‑based callback that saves electromagnetic energy to a file.

    Demonstrates:
    - Subclassing Callback
    - Setting stage and interval as class attributes
    - File I/O with MPI gating
    - Persistent state across calls (self attributes)
    """
    stage = 'start'
    interval = 100

    def __init__(self, filename='energy.csv'):
        super().__init__()
        self.filename = filename
        # File handle will be opened on rank 0 only
        self.file = None

    def _call(self, sim):
        # Calculate total electromagnetic energy
        E_em = 0.0
        for p in sim.patches:
            f = p.fields
            # Exclude guard cells
            s = np.s_[:sim.nx_per_patch, :sim.ny_per_patch]
            E_em += (
                0.5 * epsilon_0 * (f.ex[s]**2 + f.ey[s]**2 + f.ez[s]**2) +
                0.5 / mu_0 * (f.bx[s]**2 + f.by[s]**2 + f.bz[s]**2)
            ).sum()

        # Sum across MPI ranks
        E_em = sim.mpi.comm.reduce(E_em)

        # Only rank 0 writes to file
        if sim.mpi.rank == 0:
            if self.file is None:
                self.file = open(self.filename, 'w')
                self.file.write('step,time[s],energy[J]\n')

            self.file.write(f'{sim.itime},{sim.time:.6e},{E_em:.6e}\n')
            self.file.flush()

    def __del__(self):
        # Clean up file handle
        if self.file is not None:
            self.file.close()
