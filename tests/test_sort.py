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
            boundary_conditions={'xmin': 'periodic', 'xmax': 'periodic', 'ymin': 'periodic', 'ymax': 'periodic'}
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

    def test_reversed_bucket_order(self):
        """Species drifting towards -x get a mirrored x bucket numbering.

        Bucket b then corresponds to physical cell nx-1-b, with out-of-bound
        coordinates clamped into the domain. Sorting must remain a valid
        bucketed layout.
        """
        self.sim.initialize()
        sorter = self.sim.sorter[0]
        self.assertEqual(sorter.ny_buckets, 1)  # 1D bucket path (no collisions)

        # drift decides the ordering on the first call
        for p in self.sim.patches:
            p.particles[0].ux[:] = -1.0

        np.random.shuffle(self.sim.patches[0].particles[0].x)
        x = self.sim.patches[0].particles[0].x.copy()
        sorter()

        self.assertTrue(sorter.reverse_x)

        nxb = sorter.nx_buckets
        ix = np.clip(np.floor((x - sorter.x0s[0]) / sorter.dx_buckets).astype(int), 0, nxb-1)
        ibin = nxb - 1 - ix  # mirrored

        # Check bucket count against mirrored physical cells
        bucket_count = np.bincount(ibin, minlength=nxb).reshape(nxb, 1)
        np.testing.assert_array_equal(bucket_count, sorter.bucket_count_list[0])

        # Check bucket bounds
        cnt = sorter.bucket_count_list[0].ravel()
        bound_min = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        np.testing.assert_array_equal(bound_min.reshape(nxb, 1), sorter.bucket_bound_min_list[0])
        np.testing.assert_array_equal((bound_min+cnt).reshape(nxb, 1), sorter.bucket_bound_max_list[0])

        # Check particle sorting: per-particle bucket and ascending post-sort layout
        x_sort = self.sim.patches[0].particles[0].x.copy()
        ix_sort = np.clip(np.floor((x_sort - sorter.x0s[0]) / sorter.dx_buckets).astype(int), 0, nxb-1)
        ibin_sort = nxb - 1 - ix_sort

        self.assertTrue((sorter.particle_index_list[0] == ibin).all())
        self.assertTrue((np.sort(ibin_sort) == ibin_sort).all())

    def test_reverse_override(self):
        """Explicit reverse_x override wins over auto-detection; full 2D bucket
        sorts (collisions/fusion) always keep the default ordering."""
        from lambdapic.core.sort.particle_sort import ParticleSort2D

        self.sim.initialize()
        sorter = self.sim.sorter[0]

        # +x drift would auto-select False; force True instead
        for p in self.sim.patches:
            p.particles[0].ux[:] = 1.0
        sorter._reverse_x_override = True
        sorter()
        self.assertTrue(sorter.reverse_x)

        # full 2D buckets: override is ignored
        sorter_2d = ParticleSort2D(self.sim.patches, self.sim.patches.species[0], reverse_x=True)
        sorter_2d()
        self.assertGreater(sorter_2d.ny_buckets, 1)
        self.assertFalse(sorter_2d.reverse_x)

    def test_nbuf_zero_when_already_sorted(self):
        """A second sort without any motion/sync must not move any particle."""
        self.sim.initialize()
        sorter = self.sim.sorter[0]

        sorter()
        nbuf = sorter()
        self.assertEqual(nbuf, 0)
        self.assertEqual(sorter.nbuf_last, 0)


class TestDriftSymmetry(unittest.TestCase):
    """Regression test: sort cost must be symmetric w.r.t. drift direction.

    The bucket layout advects with the flow only when particles enter from the
    bucket-0 side; before mirroring, a -x drifting species misplaced ~4x more
    particles per step than its +x counterpart.
    """
    def setUp(self) -> None:
        from lambdapic import Simulation
        from lambdapic.core.species import Electron

        sim = Simulation(
            nx=64, ny=64, dx=1e-6, dy=1e-6,
            npatch_x=2, npatch_y=2,
            boundary_conditions={'xmin': 'periodic', 'xmax': 'periodic', 'ymin': 'periodic', 'ymax': 'periodic'}
        )
        beam_pos = Electron(name="beam_pos", density=lambda x, y: 1, ppc=8)
        beam_neg = Electron(name="beam_neg", density=lambda x, y: 1, ppc=8)
        sim.add_species([beam_pos, beam_neg])
        sim.initialize()

        for p in sim.patches:
            p.particles[beam_pos.ispec].ux[:] = 1.0
            p.particles[beam_neg.ispec].ux[:] = -1.0

        self.sim = sim
        self.ispec_pos = beam_pos.ispec
        self.ispec_neg = beam_neg.ispec

    def test_drift_symmetry_nbuf(self):
        sim = self.sim
        nbuf_pos, nbuf_neg = [], []

        for _ in range(30):
            # stream both species by ~0.58 cells/step like a relativistic beam
            for p in sim.patches:
                part_pos = p.particles[self.ispec_pos]
                part_neg = p.particles[self.ispec_neg]
                part_pos.x[~part_pos.is_dead] += 0.58 * sim.dx
                part_neg.x[~part_neg.is_dead] -= 0.58 * sim.dx
            sim.patches.sync_particles()
            nbuf_pos.append(sim.sorter[self.ispec_pos]())
            nbuf_neg.append(sim.sorter[self.ispec_neg]())

        # ordering auto-detection
        self.assertFalse(sim.sorter[self.ispec_pos].reverse_x)
        self.assertTrue(sim.sorter[self.ispec_neg].reverse_x)

        npart_pos = sum(p.particles[self.ispec_pos].x.size for p in sim.patches)
        npart_neg = sum(p.particles[self.ispec_neg].x.size for p in sim.patches)
        mean_pos = np.mean(nbuf_pos[10:]) / npart_pos
        mean_neg = np.mean(nbuf_neg[10:]) / npart_neg

        # pre-fix ratio was ~0.58/0.15 ~ 4; post-fix both sit at ~0.1-0.2
        self.assertLess(mean_neg, 1.5 * max(mean_pos, 1e-12))

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
            boundary_conditions={'xmin': 'periodic', 'xmax': 'periodic', 'ymin': 'periodic', 'ymax': 'periodic', 'zmin': 'periodic', 'zmax': 'periodic'}
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

    def test_reversed_bucket_order(self):
        """Mirrored x bucket numbering for -x drifting species (1D buckets)."""
        self.sim.initialize()
        sorter = self.sim.sorter[0]
        self.assertEqual(sorter.ny_buckets, 1)
        self.assertEqual(sorter.nz_buckets, 1)

        for p in self.sim.patches:
            p.particles[0].ux[:] = -1.0

        np.random.shuffle(self.sim.patches[0].particles[0].x)
        x = self.sim.patches[0].particles[0].x.copy()
        sorter()

        self.assertTrue(sorter.reverse_x)

        nxb = sorter.nx_buckets
        ix = np.clip(np.floor((x - sorter.x0s[0]) / sorter.dx_buckets).astype(int), 0, nxb-1)
        ibin = nxb - 1 - ix  # mirrored

        bucket_count = np.bincount(ibin, minlength=nxb).reshape(nxb, 1, 1)
        np.testing.assert_array_equal(bucket_count, sorter.bucket_count_list[0])

        x_sort = self.sim.patches[0].particles[0].x.copy()
        ix_sort = np.clip(np.floor((x_sort - sorter.x0s[0]) / sorter.dx_buckets).astype(int), 0, nxb-1)
        ibin_sort = nxb - 1 - ix_sort

        self.assertTrue((sorter.particle_index_list[0] == ibin).all())
        self.assertTrue((np.sort(ibin_sort) == ibin_sort).all())