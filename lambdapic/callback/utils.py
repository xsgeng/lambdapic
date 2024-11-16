from ..simulation import Simulation
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
import numpy as np

from libpic.species import Species

from pathlib import Path


def get_fields(sim: Simulation, fields=[]) -> list[np.ndarray]:
    ret = []
    patches = sim.patches
    nx_per_patch = sim.nx_per_patch
    ny_per_patch = sim.ny_per_patch
    n_guard = sim.n_guard
    for field in fields:
        field_ = np.zeros((sim.nx, sim.ny))
        for ipatch, p in enumerate(patches):
            s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                        p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
            field_[s] = getattr(p.fields, field)[:-2*n_guard, :-2*n_guard]
        ret.append(field_)

    return ret

def save_fields_to_hdf5(sim: Simulation, fields: list, every: int, prefix: [Path|str]='.'):
    import h5py

    prefix = Path(prefix)
    def save_fields(it):
        if it % every == 0:
            field_data = get_fields(sim, fields)
            
            # 创建HDF5文件
            with h5py.File(prefix/f'fields_{it:04d}.h5', 'w') as f:
                # 存储电磁场数据
                for field, data in zip(fields, field_data):
                    f.create_dataset(field, data=data)
                
                # 存储其他相关参数
                f.attrs['nx'] = sim.nx
                f.attrs['ny'] = sim.ny
                f.attrs['dx'] = sim.dx
                f.attrs['dy'] = sim.dy
                f.attrs['Lx'] = sim.Lx
                f.attrs['Ly'] = sim.Ly
                f.attrs['it'] = it

    return save_fields


def create_store_density_cb(sim: Simulation, species: Species, every: int) -> tuple[np.ndarray, callable]:
    ispec_target = sim.species.index(species)
    buf = np.zeros((sim.nx, sim.ny))

    patches = sim.patches
    nx_per_patch = sim.nx_per_patch
    ny_per_patch = sim.ny_per_patch
    n_guard = sim.n_guard

    if ispec_target == 0:
        def store_density(it, ispec):
            if it % every == 0:
                if ispec == 0:
                    for ipatch, p in enumerate(patches):
                        s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                                p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
                        buf[s] = p.fields.rho[:-2*n_guard, :-2*n_guard] / sim.species[ispec].q

    elif ispec_target > 0:
        def store_density(it, ispec):
            if it % every == 0:
                if ispec == ispec_target - 1:
                    for ipatch, p in enumerate(patches):
                        s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                                p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
                        # store previous rho
                        buf[s] = p.fields.rho[:-2*n_guard, :-2*n_guard]
                if ispec == ispec_target:
                    for ipatch, p in enumerate(patches):
                        s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                                p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
                        # subtract previous rho
                        buf[s] = p.fields.rho[:-2*n_guard, :-2*n_guard] - buf[s]
                        buf[s] /= sim.species[ispec].q

    return buf, store_density


def species_transfer(s1, s2):
    # if ..., s1 particles become s2
    pass