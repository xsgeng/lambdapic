import os
from pathlib import Path
from typing import Callable, Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..simulation import Simulation, Simulation3D
from .callback import Callback
from .utils import get_fields


class PlotFields(Callback):
    """Callback to plot and overlay multiple fields with flexible configuration.

    Creates plots with specified fields overlaid using transparency cmaps: `bwr_alpha`, `gold_alpha`, `grey_alpha`, `red_alpha`, `blue_alpha`, `gwb_alpha`.

    Supports both string-named fields (fetched from simulation) and direct array inputs.

    Only rank 0 creates plots in MPI parallel runs.

    Args:
        field_configs (List[Dict]): List of field configurations, each specifying:
            - field: Field name (str) or array (np.ndarray) to plot
            - scale: Scaling factor (default=1.0), multiplied by the field value
            - cmap: Matplotlib colormap (default='viridis')
            - vmin: Minimum value for normalization (optional)
            - vmax: Maximum value for normalization (optional)
        prefix (Union[str, Path]): Output directory for plots
        interval (Union[int, float, Callable] = 100): Save interval
        figsize (Tuple[float, float] = (10, 6)): Figure size
        dpi (int = 300): Image DPI

    Example:
        >>> # Using field names
        >>> field_configs = [
        ...     dict(field='rho', scale=-1/e/nc, cmap='Grays', vmin=-1.0, vmax=1.0),
        ...     dict(field='ey', scale=1/4e12, cmap='bwr_alpha', vmin=-1.0, vmax=1.0),
        ... ]

        >>> # Using direct arrays from other callbacks
        >>> extract_ne = ExtractSpeciesDensity(sim, ele, interval=100)
        >>> field_configs = [
        ...     dict(field=extract_ne.density, scale=1/nc, cmap='viridis'),
        ...     dict(field='ey', scale=1/4e12, cmap='bwr')
        ... ]
        >>> sim.run(callbacks=[
        ...     extract_ne,
        ...     PlotFields(field_configs, prefix='plots', interleval=100)])
    """
    stage = "maxwell second"

    def __init__(self,
                 field_configs: List[Dict],
                 prefix: Union[str, Path],
                 interval: Union[int, float, Callable] = 100,
                 figsize: tuple | None = None,
                 dpi: int = 300):
        self.prefix = Path(prefix)
        self.field_configs = field_configs
        self.interval = interval
        self.figsize = figsize
        self.dpi = dpi

        # Validate field configs
        for cfg in field_configs:
            if 'field' not in cfg:
                raise ValueError("Each field config must have a 'field' key")

        # Create output directory
        self.prefix.mkdir(parents=True, exist_ok=True)


    def _call(self, sim: Union[Simulation, Simulation3D]):
        if self.figsize is None:
            self.figsize = (10, 10*sim.Ly/sim.Lx)

        field_data = []
        field_names = []
        for i, cfg in enumerate(self.field_configs):
            f = cfg['field']
            if isinstance(f, str):
                field_data.append(get_fields(sim, [f])[0])
                field_names.append(f)
            elif isinstance(f, np.ndarray):
                field_data.append(f)
                field_names.append('data')
            else:
                raise TypeError(f"Invalid field type: {type(f)}")

        # Skip non-root ranks in MPI
        if sim.mpi.comm.Get_rank() > 0:
            return

        register_cmaps()

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each field with its configuration
        for i, cfg in enumerate(self.field_configs):
            scale = cfg.get('scale', 1.0)

            cmap = cfg.get('cmap')
            vmin = cfg.get('vmin')
            vmax = cfg.get('vmax')

            im = ax.imshow(
                field_data[i].T * scale,
                extent=[0, sim.Lx, 0, sim.Ly],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            # Add colorbar with field name
            cbar = fig.colorbar(im, ax=ax)
            field_names = cfg['field'] if isinstance(
                cfg['field'], str) else 'field'
            cbar.set_label(f"{field_names} x{scale:.2e}")

        ax.set_title(f"Fields at t = {sim.time:.2e}, step = {sim.itime}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # Save figure
        fig.savefig(
            self.prefix/f"fields_{sim.itime:06d}.png",
            dpi=self.dpi,
        )
        plt.close(fig)

        unregister_cmaps()


# custom colormaps
_cmaps = [
    LinearSegmentedColormap(
        'bwr_alpha',
        dict(red=[(0, 0, 0), (0.5, 1, 1), (1, 1, 1)],
             green=[(0, 0.5, 0), (0.5, 1, 1), (1, 0, 0)],
             blue=[(0, 1, 1), (0.5, 1, 1), (1, 0, 0)],
             alpha=[(0, 1, 1), (0.5, 0, 0), (1, 1, 1)])
    ),
    LinearSegmentedColormap(
        'gold_alpha',
        dict(red=[(0, 1, 1), (1, 1, 1)],
             green=[(0, 1, 1), (1, 0.9, 1)],
             blue=[(0, 1, 1), (1, 0, 1)],
             alpha=[(0, 0, 0), (1, 1, 1)])
    ),
    LinearSegmentedColormap(
        'grey_alpha',
        dict(red=[(0, 0, 1), (1, 0, 1)],
             green=[(0, 0, 1), (1, 0, 1)],
             blue=[(0, 0, 1), (1, 0, 1)],
             alpha=[(0, 0, 0), (1, 1, 1)])
    ),
    LinearSegmentedColormap(
        'red_alpha',
        dict(red=[(0, 0, 1), (1, 1, 1)],
             green=[(0, 0, 1), (1, 0, 0)],
             blue=[(0, 0, 1), (1, 0, 0)],
             alpha=[(0, 0, 0), (1, 1, 1)]),
    ),
    LinearSegmentedColormap(
        'blue_alpha',
        dict(red=[(0, 0, 1), (1, 0, 1)],
             green=[(0, 0, 1), (1, 0, 0)],
             blue=[(0, 0, 1), (1, 1, 0)],
             alpha=[(0, 0, 0), (1, 1, 1)]),
    ),
    LinearSegmentedColormap(
        'gwb_alpha',
        dict(red=[(0,   0, 0.36), (0.5, 1, 1), (1,   0.96, 1)],
             green=[(0, 0, 0.75), (0.5, 1, 1), (1, 0.79, 0)],
             blue=[(0, 0, 0.75), (0.5, 1, 1), (1, 0.29, 0)],
             alpha=[(0, 1, 1), (0.5, 0, 0), (1, 1, 1)]),
    )
]

def register_cmaps():
    """Register custom colormaps."""
    for cmap in _cmaps:
        matplotlib.colormaps.register(cmap)

def unregister_cmaps():
    """Unregister custom colormaps."""
    for cmap in _cmaps:
        matplotlib.colormaps.unregister(cmap.name)
