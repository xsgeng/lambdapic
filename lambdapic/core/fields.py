import numpy as np
from numpy.typing import NDArray


class Fields:
    nx: int
    ny: int
    nz: int
    n_guard: int
    dx: float
    dy: float
    dz: float
    shape: tuple

    x0: float
    y0: float
    z0: float

    ex: NDArray[np.float64]
    ey: NDArray[np.float64]
    ez: NDArray[np.float64]
    bx: NDArray[np.float64]
    by: NDArray[np.float64]
    bz: NDArray[np.float64]
    jx: NDArray[np.float64]
    jy: NDArray[np.float64]
    jz: NDArray[np.float64]
    rho: NDArray[np.float64]

    xaxis: NDArray[np.float64]
    yaxis: NDArray[np.float64]
    zaxis: NDArray[np.float64]

    attrs = ["ex", "ey", "ez", "bx", "by", "bz", "jx", "jy", "jz", "rho"]

    def _init_fields(self, attrs: list[str]|None):
        if attrs is not None:
            self.attrs = attrs
        for attr in self.attrs:
            setattr(self, attr, np.zeros(self.shape))


class Fields2D(Fields):

    def __init__(self, nx, ny, dx, dy, x0, y0, n_guard, attrs: list[str]|None=None) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.n_guard = n_guard

        self.shape = (nx+2*n_guard, ny+2*n_guard)
        self._init_fields(attrs)

        xaxis = np.arange(nx+n_guard*2, dtype=float)
        xaxis[-n_guard:] = np.arange(-n_guard, 0)
        xaxis *= dx
        self.x0 = x0
        self.xaxis = xaxis[:, None] + x0

        yaxis = np.arange(ny+2*n_guard, dtype=float)
        yaxis[-n_guard:] = np.arange(-n_guard, 0)
        yaxis *= dy
        self.y0 = y0
        self.yaxis = yaxis[None, :] + y0


class Fields3D(Fields):

    def __init__(self, nx, ny, nz, dx, dy, dz, x0, y0, z0, n_guard, attrs: list[str]|None=None) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.n_guard = n_guard

        self.shape = (nx+2*n_guard, ny+2*n_guard, nz+2*n_guard)
        self._init_fields(attrs)

        # x axis
        xaxis = np.arange(nx+n_guard*2, dtype=float)
        xaxis[-n_guard:] = np.arange(-n_guard, 0)
        xaxis *= dx
        self.x0 = x0
        self.xaxis = xaxis[:, None, None] + x0

        # y axis
        yaxis = np.arange(ny+2*n_guard, dtype=float)
        yaxis[-n_guard:] = np.arange(-n_guard, 0)
        yaxis *= dy
        self.y0 = y0
        self.yaxis = yaxis[None, :, None] + y0

        # z axis
        zaxis = np.arange(nz+2*n_guard, dtype=float)
        zaxis[-n_guard:] = np.arange(-n_guard, 0)
        zaxis *= dz
        self.z0 = z0
        self.zaxis = zaxis[None, None, :] + z0
