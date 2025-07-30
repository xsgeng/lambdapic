import unittest

import numpy as np

from lambdapic.simulation import Simulation3D


class Test2D(unittest.TestCase):
    def setUp(self) -> None:
        from lambdapic import Simulation
        from lambdapic.core.species import Electron
        
        dx = 1e-6
        dy = 1e-6

        nx = 16
        ny = 16

        npatch_x = 2
        npatch_y = 2

        sim = Simulation(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            npatch_x=npatch_x,
            npatch_y=npatch_y,
        )
        
        ele = Electron(density=lambda x, y: 1, ppc=8)

        sim.add_species([ele])

        self.sim = sim
        
    def test_init_sort(self):
        self.sim.initialize()
        sorter = self.sim.sorter[0]

        np.random.shuffle(self.sim.patches[0].particles[0].x)
        np.random.shuffle(self.sim.patches[0].particles[0].y)

        x = self.sim.patches[0].particles[0].x.copy()
        y = self.sim.patches[0].particles[0].y.copy()
        sorter()
        
        # Check bucket count
        bucket_count, _, _ = np.histogram2d(x, y, bins=[
            np.arange(sorter.nx_buckets+1)*sorter.dx_buckets+sorter.x0s[0], 
            np.arange(sorter.ny_buckets+1)*sorter.dy_buckets+sorter.y0s[0]])
        np.testing.assert_array_equal(bucket_count, sorter.bucket_count_list[0])
        
        # Check bucket bounds
        bucket_bound_min = np.cumsum(bucket_count, axis=0) - bucket_count[0]
        bucket_bound_max = bucket_bound_min + bucket_count
        np.testing.assert_array_equal(bucket_bound_max, sorter.bucket_bound_max_list[0])
        np.testing.assert_array_equal(bucket_bound_min, sorter.bucket_bound_min_list[0])
        
        # Check particle sorting
        x_sort = self.sim.patches[0].particles[0].x.copy()
        y_sort = self.sim.patches[0].particles[0].y.copy()
        
        ix = np.floor((x - sorter.x0s[0]) / sorter.dx_buckets).astype(int)
        iy = np.floor((y - sorter.y0s[0]) / sorter.dy_buckets).astype(int)
        ibin = iy + ix * sorter.ny_buckets

        ix_sort = np.floor((x_sort - sorter.x0s[0]) / sorter.dx_buckets).astype(int)
        iy_sort = np.floor((y_sort - sorter.y0s[0]) / sorter.dy_buckets).astype(int)
        ibin_sort = iy_sort + ix_sort * sorter.ny_buckets

        self.assertTrue((sorter.particle_index_list[0] == ibin).all())
        self.assertTrue((np.sort(ibin) == ibin_sort).all())

class Test3D(unittest.TestCase):
    def setUp(self) -> None:
        from lambdapic import Simulation
        from lambdapic.core.species import Electron

        dx = 1e-6
        dy = 1e-6
        dz = 1e-6

        nx = 16
        ny = 16
        nz = 16

        npatch_x = 2
        npatch_y = 2
        npatch_z = 2

        sim = Simulation3D(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            npatch_x=npatch_x,
            npatch_y=npatch_y,
            npatch_z=npatch_z,
        )
        
        ele = Electron(density=lambda x, y, z: 1, ppc=8)

        sim.add_species([ele])

        self.sim = sim
        
    def test_init_sort(self):
        self.sim.initialize()
        sorter = self.sim.sorter[0]

        np.random.shuffle(self.sim.patches[0].particles[0].x)
        np.random.shuffle(self.sim.patches[0].particles[0].y)
        np.random.shuffle(self.sim.patches[0].particles[0].z)

        x = self.sim.patches[0].particles[0].x.copy()
        y = self.sim.patches[0].particles[0].y.copy()
        z = self.sim.patches[0].particles[0].z.copy()
        sorter()
        
        # Check bucket count
        bucket_count, _ = np.histogramdd([x, y, z], bins=[
            np.arange(sorter.nx_buckets+1)*sorter.dx_buckets+sorter.x0s[0], 
            np.arange(sorter.ny_buckets+1)*sorter.dy_buckets+sorter.y0s[0],
            np.arange(sorter.nz_buckets+1)*sorter.dz_buckets+sorter.z0s[0]])
        np.testing.assert_array_equal(bucket_count, sorter.bucket_count_list[0])
        
        # Check bucket bounds
        bucket_bound_min = np.cumsum(bucket_count, axis=0) - bucket_count[0]
        bucket_bound_max = bucket_bound_min + bucket_count
        np.testing.assert_array_equal(bucket_bound_max, sorter.bucket_bound_max_list[0])
        np.testing.assert_array_equal(bucket_bound_min, sorter.bucket_bound_min_list[0])
        
        # Check particle sorting
        x_sort = self.sim.patches[0].particles[0].x.copy()
        y_sort = self.sim.patches[0].particles[0].y.copy()
        z_sort = self.sim.patches[0].particles[0].z.copy()
        
        ix = np.floor((x - sorter.x0s[0]) / sorter.dx_buckets).astype(int)
        iy = np.floor((y - sorter.y0s[0]) / sorter.dy_buckets).astype(int)
        iz = np.floor((z - sorter.z0s[0]) / sorter.dz_buckets).astype(int)
        ibin = iz + iy * sorter.nz_buckets + ix * sorter.ny_buckets * sorter.nz_buckets

        ix_sort = np.floor((x_sort - sorter.x0s[0]) / sorter.dx_buckets).astype(int)
        iy_sort = np.floor((y_sort - sorter.y0s[0]) / sorter.dy_buckets).astype(int)
        iz_sort = np.floor((z_sort - sorter.z0s[0]) / sorter.dz_buckets).astype(int)
        ibin_sort = iz_sort + iy_sort * sorter.nz_buckets + ix_sort * sorter.ny_buckets * sorter.nz_buckets

        self.assertTrue((sorter.particle_index_list[0] == ibin).all())
        self.assertTrue((np.sort(ibin) == ibin_sort).all())