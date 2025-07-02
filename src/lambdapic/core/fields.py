from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


class Fields:
    """Base class for electromagnetic field data in particle-in-cell simulations.

    Attributes:
        nx,ny,nz (int): Grid dimensions in x, y, z directions
        n_guard (int): Number of guard cells
        dx,dy,dz (float): Grid spacings
        shape (tuple): Full array shape including guard cells
        x0,y0,z0 (float): Grid origin coordinates
        
        ex,ey,ez (NDArray[float64]): Electric field components
        bx,by,bz (NDArray[float64]): Magnetic field components  
        jx,jy,jz (NDArray[float64]): Current density components
        rho (NDArray[float64]): Charge density
        
        xaxis,yaxis,zaxis (NDArray[float64]): Coordinate axes including guard cells
        attrs (list[str]): List of field attribute names

    Note:
        Fields data are stored in [:nx, :ny, :nz] range, and the guard cells are in the [nx:, ny:, nz:] range.
        The guard cells are therefore accessed using [-n_guard:, -n_guard:, -n_guard:] and [nx:nx+n_guard, ny:ny+n_guard, nz:nz+n_guard].
    """
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

    def _init_fields(self, attrs: Optional[List[str]]):
        """Initialize field arrays with zeros.
        
        Args:
            attrs (Optional[List[str]]): Optional list of field attributes to initialize.
                  If None, uses default attrs list.
        """
        if attrs is not None:
            self.attrs = attrs
        for attr in self.attrs:
            setattr(self, attr, np.zeros(self.shape))


class Fields2D(Fields):
    """2D electromagnetic field data for particle-in-cell simulations.

    Attributes:
        Inherits all attributes from Fields class.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, 
                 x0: float, y0: float, n_guard: int, 
                 attrs: Optional[List[str]]=None) -> None:
        """Initialize 2D field data.
        
        Args:
            nx (int): Number of grid cells in x direction
            ny (int): Number of grid cells in y direction
            dx (float): Grid spacing in x direction
            dy (float): Grid spacing in y direction
            x0 (float): x-coordinate origin
            y0 (float): y-coordinate origin
            n_guard (int): Number of guard cells
            attrs (Optional[List[str]]): Optional list of field attributes to initialize
        """
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
    """3D electromagnetic field data for particle-in-cell simulations.

    Attributes:
        Inherits all attributes from Fields class.
    """

    def __init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float,
                 x0: float, y0: float, z0: float, n_guard: int,
                 attrs: Optional[List[str]]=None) -> None:
        """Initialize 3D field data.
        
        Args:
            nx (int): Number of grid cells in x direction
            ny (int): Number of grid cells in y direction
            nz (int): Number of grid cells in z direction
            dx (float): Grid spacing in x direction
            dy (float): Grid spacing in y direction
            dz (float): Grid spacing in z direction
            x0 (float): x-coordinate origin
            y0 (float): y-coordinate origin
            z0 (float): z-coordinate origin
            n_guard (int): Number of guard cells
            attrs (Optional[List[str]]): Optional list of field attributes to initialize
        """
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
