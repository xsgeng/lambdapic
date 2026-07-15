"""Internal mask-driven simulation class for irregular domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import Simulation
from ..core.patch.patch import Patch2D, Patches, Boundary2D
from ..core.boundary.cpml import PMLXmin, PMLXmax, PMLYmin, PMLYmax


@dataclass
class _MaskSimulation(Simulation):
    """2D simulation with mask-driven irregular domain.

    Only patches whose centers satisfy `mask(xc, yc) == True` are included.
    Patches with no neighbor on a face automatically get PML on that face.
    Periodic boundaries are not supported — all open faces get PML.

    Parameters
    ----------
    mask : Callable[[float, float], bool]
        Function `mask(x, y) -> bool` evaluated at each patch center.
        Only patches where mask returns True are created.

    Notes
    -----
    - For MPI: `npatches_included` must be >= `comm_size` to avoid empty ranks.
    - The `boundary_conditions` parameter is inherited but ignored;
      all open faces get PML regardless.
    """
    mask: Callable[[float, float], bool] | None = field(default=None)

    def create_patches(self) -> Patches:
        """Create patches filtered by the mask function."""
        assert self.mask is not None, "mask must be provided"
        patches = Patches(dimension=2)
        index = 0
        for j in range(self.npatch_y):
            for i in range(self.npatch_x):
                xc = (i + 0.5) * self.Lx / self.npatch_x
                yc = (j + 0.5) * self.Ly / self.npatch_y
                if not self.mask(xc, yc):
                    continue
                p = Patch2D(
                    rank=None,
                    index=index,
                    ipatch_x=i,
                    ipatch_y=j,
                    x0=i * self.Lx / self.npatch_x,
                    y0=j * self.Ly / self.npatch_y,
                    nx=self.nx_per_patch,
                    ny=self.ny_per_patch,
                    dx=self.dx,
                    dy=self.dy,
                )
                patches.append(p)
                index += 1
        assert patches.npatches > 0, "mask produced no patches"
        patches.init_rect_neighbor_index_2d(
            npatch_x=self.npatch_x,
            npatch_y=self.npatch_y,
            boundary_conditions={
                "xmin": "pml", "xmax": "pml",
                "ymin": "pml", "ymax": "pml",
            },
        )
        patches.update_lists()

        # Build domain mask: True where a patch exists
        import numpy as np

        self.domain_mask = np.zeros((self.nx, self.ny), dtype=bool)
        for p in patches.patches:
            s = np.s_[
                p.ipatch_x * self.nx_per_patch:(p.ipatch_x + 1) * self.nx_per_patch,
                p.ipatch_y * self.ny_per_patch:(p.ipatch_y + 1) * self.ny_per_patch,
            ]
            self.domain_mask[s] = True

        return patches

    def _init_pml(self) -> None:
        """Attach PML to any face with no neighbor."""
        for p in self.patches:
            if p.neighbor_index[Boundary2D.XMIN] < 0:
                p.add_pml_boundary(
                    PMLXmin(p.fields, thickness=self.cpml_thickness)
                )
            if p.neighbor_index[Boundary2D.XMAX] < 0:
                p.add_pml_boundary(
                    PMLXmax(p.fields, thickness=self.cpml_thickness)
                )
            if p.neighbor_index[Boundary2D.YMIN] < 0:
                p.add_pml_boundary(
                    PMLYmin(p.fields, thickness=self.cpml_thickness)
                )
            if p.neighbor_index[Boundary2D.YMAX] < 0:
                p.add_pml_boundary(
                    PMLYmax(p.fields, thickness=self.cpml_thickness)
                )
