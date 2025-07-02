from scipy.constants import c, epsilon_0

def clamp_zero_bfield(bx, by, bz, n_guard):
    clamp_zero_xmin(bx, n_guard, halfgrid=False)
    clamp_zero_xmax(bx, n_guard, halfgrid=False)
    clamp_zero_xmin(by, n_guard, halfgrid=True)
    clamp_zero_xmax(by, n_guard, halfgrid=True)
    clamp_zero_xmin(bz, n_guard, halfgrid=True)
    clamp_zero_xmax(bz, n_guard, halfgrid=True)

    clamp_zero_ymin(bx, n_guard, halfgrid=True)
    clamp_zero_ymax(bx, n_guard, halfgrid=True)
    clamp_zero_ymin(by, n_guard, halfgrid=False)
    clamp_zero_ymax(by, n_guard, halfgrid=False)
    clamp_zero_ymin(bz, n_guard, halfgrid=True)
    clamp_zero_ymax(bz, n_guard, halfgrid=True)

def clamp_zero_efield(ex, ey, ez, n_guard):
    clamp_zero_xmin(ex, n_guard, halfgrid=True)
    clamp_zero_xmax(ex, n_guard, halfgrid=True)
    clamp_zero_xmin(ey, n_guard, halfgrid=False)
    clamp_zero_xmax(ey, n_guard, halfgrid=False)
    clamp_zero_xmin(ez, n_guard, halfgrid=False)
    clamp_zero_xmax(ez, n_guard, halfgrid=False)
    
    clamp_zero_ymin(ex, n_guard, halfgrid=False)
    clamp_zero_ymax(ex, n_guard, halfgrid=False)
    clamp_zero_ymin(ey, n_guard, halfgrid=True)
    clamp_zero_ymax(ey, n_guard, halfgrid=True)
    clamp_zero_ymin(ez, n_guard, halfgrid=False)
    clamp_zero_ymax(ez, n_guard, halfgrid=False)


def outflow_x_min(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, nx, ny, dt, n_guard):

    sum = 1 / (dt*c**2/dx + c)
    diff = dt * c**2 / dx - c
    bx[-1, :ny] = bx[0, :ny]
    by[-1, :ny] = sum*(
        - 2 * (ez[0, :ny] + c*by[0, :ny]) \
        + 2 * ez[0, :ny] \
        - dt*c**2 / dy * (bx[0, :ny] - bx[:, :ny]) \
        - dt/epsilon_0 * jz[0, :ny] \
        + diff * by[0, :ny]
    )
    bz[-1, :ny] = sum*(
        + 2  * (ey[0, :ny] + c*bz[0, :ny]) \
        - 2 * ey[0, :ny] \
        + dt/epsilon_0 * jy[0, :ny] \
        + diff * bz[0, :ny]
    )


def clamp_zero_xmin(field, n_guard, halfgrid):
    if halfgrid:
        field[-n_guard:-1, :] = -field[n_guard-2::-1,:]
        field[-1, :] = 0
    else:
        field[-n_guard:, :] = -field[n_guard-1::-1,:]


def clamp_zero_xmax(field, n_guard, halfgrid):
    nx = field.shape[0] - 2*n_guard
    if halfgrid:
        field[nx, :] = 0
        field[nx+1:nx+n_guard, :] = -field[nx-1:nx-n_guard:-1, :]
    else:
        field[nx:nx+n_guard, :] = -field[nx-1:nx-1-n_guard:-1, :]


def clamp_zero_ymin(field, n_guard, halfgrid):
    if halfgrid:
        field[:, -n_guard:-1] = -field[:, n_guard-2::-1]
        field[:, -1] = 0
    else:
        field[:, -n_guard:] = -field[:, n_guard-1::-1]


def clamp_zero_ymax(field, n_guard, halfgrid):
    ny = field.shape[1] - 2*n_guard
    if halfgrid:
        field[:, ny] = 0
        field[:, ny+1:ny+n_guard] = -field[:, ny-1:ny-n_guard:-1]
    else:
        field[:, ny:ny+n_guard] = -field[:, ny-1:ny-1-n_guard:-1]



def zero_gradient_xmin(field, n_guard, halfgrid):
    if halfgrid:
        field[-n_guard:-1, :] = field[n_guard-2::-1,:]
        field[-1, :] = 0
    else:
        field[-n_guard:, :] = field[n_guard-1::-1,:]


def zero_gradient_xmax(field, n_guard, halfgrid):
    nx = field.shape[0] - 2*n_guard
    if halfgrid:
        field[nx, :] = 0
        field[nx+1:nx+n_guard, :] = field[nx-1:nx-n_guard:-1, :]
    else:
        field[nx:nx+n_guard, :] = field[nx-1:nx-1-n_guard:-1, :]


def zero_gradient_ymin(field, n_guard, halfgrid):
    if halfgrid:
        field[:, -n_guard:-1] = field[:, n_guard-2::-1]
        field[:, -1] = 0
    else:
        field[:, -n_guard:] = field[:, n_guard-1::-1]


def zero_gradient_ymax(field, n_guard, halfgrid):
    ny = field.shape[1] - 2*n_guard
    if halfgrid:
        field[:, ny] = 0
        field[:, ny+1:ny+n_guard] = field[:, ny-1:ny-n_guard:-1]
    else:
        field[:, ny:ny+n_guard] = field[:, ny-1:ny-1-n_guard:-1]

