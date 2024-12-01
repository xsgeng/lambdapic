import pytest
from pathlib import Path
from time import perf_counter_ns

import matplotlib.pyplot as plt
import mytools
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from mpi4py.MPI import COMM_WORLD as comm
from objprint import op
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic.core.particles import ParticlesBase
from lambdapic import Electron
from lambdapic.core.patch.patch import Patch2D, Patches

from lambdapic.core.mpi.sync_particles_2d import (
    fill_particles_from_boundary_2d,
    get_npart_to_extend_2d,
)

from lambdapic.core.patch.patch import Boundary2D

def test_syncparticles():
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    if comm_size != 9:
        pytest.skip('must have 9 ranks')    

    patches = Patches(2)

    npatchx = 9
    npatchy = 9

    nrankx = 3
    nranky = 3
    
    npatchx_per_rank = npatchx // nrankx
    npatchy_per_rank = npatchy // nranky

    patches_list = [Patches(2) for _ in range(comm_size)]
    if rank == 0:
        for ipatch in range(npatchx*npatchy):
            ipatch_x = ipatch % npatchx
            ipatch_y = ipatch // npatchx
            irank_x = ipatch_x // npatchx_per_rank
            irank_y = ipatch_y // npatchy_per_rank
            
            # print(f"rank {rank}: ipatch {ipatch} ipatch_x {ipatch_x} ipatch_y {ipatch_y} irank_x {irank_x} irank_y {irank_y}")
            patch = Patch2D(
                rank=irank_y*nrankx+irank_x, index=ipatch, ipatch_x=ipatch_x, ipatch_y=ipatch_y,
                nx=5, ny=5, x0=9*ipatch_x, y0=9*ipatch_y, dx=1.0, dy=1.0
            )
            
            # xmin
            if irank_x > 0 and ipatch_x % nrankx == 0:
                patch.set_neighbor_rank(xmin=irank_y*nrankx+irank_x-1)
            # xmax
            if irank_x < nrankx-1 and ipatch_x % nrankx == nrankx-1:
                patch.set_neighbor_rank(xmax=irank_y*nrankx+irank_x+1)
            # ymin
            if irank_y > 0 and ipatch_y % nranky == 0:
                patch.set_neighbor_rank(ymin=(irank_y-1)*nrankx+irank_x)
            # ymax
            if irank_y < nranky-1 and ipatch_y % nranky == nranky-1:
                patch.set_neighbor_rank(ymax=(irank_y+1)*nrankx+irank_x)
            
            # xminymin
            if irank_x > 0 and irank_y > 0 and ipatch_x % nrankx == 0 and ipatch_y % nranky == 0:
                patch.set_neighbor_rank(xminymin=(irank_y-1)*nrankx+irank_x-1)
            # xminymax
            if irank_x > 0 and irank_y < nranky-1 and ipatch_x % nrankx == 0 and ipatch_y % nranky == nranky-1:
                patch.set_neighbor_rank(xminymax=(irank_y+1)*nrankx+irank_x-1)
            # xmaxymin
            if irank_x < nrankx-1 and irank_y > 0 and ipatch_x % nrankx == nrankx-1 and ipatch_y % nranky == 0:
                patch.set_neighbor_rank(xmaxymin=(irank_y-1)*nrankx+irank_x+1)
            # xmaxymax
            if irank_x < nrankx-1 and irank_y < nranky-1 and ipatch_x % nrankx == nrankx-1 and ipatch_y % nranky == nranky-1:
                patch.set_neighbor_rank(xmaxymax=(irank_y+1)*nrankx+irank_x+1)
            
            patches_list[irank_y*nrankx+irank_x].append(patch)
        
        for p in patches_list:
            p.init_rect_neighbor_index_2d(npatch_x=npatchx, npatch_y=npatchy)
            
    patches: Patches = comm.scatter(patches_list, root=0)

    s = Electron(density=lambda x, y: 1.0, ppc=10)
    patches.add_species(s)
    patches.fill_particles()

    for p in patches:
        p.particles[0].x[:] += 1
        
    npart_to_extend, npart_incoming, npart_outgoing = get_npart_to_extend_2d([p.particles[0] for p in patches], patches.patches, comm, 9, 1.0, 1.0)
    for ipatch, p in enumerate(patches):
        assert npart_to_extend[ipatch] >= npart_incoming[ipatch].sum()
        if p.rank % nrankx == 0:
            if p.ipatch_x % npatchx_per_rank == 0:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
            elif p.ipatch_x % npatchx_per_rank == npatchx_per_rank - 1:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == p.ny*s.ppc
            else:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
        elif p.rank % nrankx == nrankx-1:
            if p.ipatch_x % npatchx_per_rank == 0:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == p.ny*s.ppc
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
            elif p.ipatch_x % npatchx_per_rank == npatchx_per_rank - 1:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
            else:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
        else:
            if p.ipatch_x % npatchx_per_rank == 0:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == p.ny*s.ppc
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
            elif p.ipatch_x % npatchx_per_rank == npatchx_per_rank - 1:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == p.ny*s.ppc
            else:
                assert npart_incoming[ipatch][Boundary2D.XMIN] == 0
                assert npart_outgoing[ipatch][Boundary2D.XMAX] == 0
        
    for ipatch, p in enumerate(patches):
        p.particles[0].extend(npart_to_extend[ipatch])
        
    fill_particles_from_boundary_2d(
        [p.particles[0] for p in patches], 
        patches.patches, 
        npart_incoming, 
        npart_outgoing, 
        comm, 
        9, 1.0, 1.0, 
        patches[0].particles[0].attrs
    )
    
    # check num particles
    for ipatch, p in enumerate(patches):
        assert p.particles[0].is_alive.sum() == p.nx*p.ny*s.ppc + npart_incoming[ipatch].sum() - npart_outgoing[ipatch].sum()
        
if __name__ == '__main__':
    test_syncparticles()