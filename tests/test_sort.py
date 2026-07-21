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

    def test_extend_resort(self):
        """Sorting must stay correct after the particle arrays are extended.

        Extension notifies sorter.update_particle_lists via the extended flag
        and Simulation.update_lists; the per-patch index buffers are recreated
        and the layout must remain a valid bucketed one.
        """
        self.sim.initialize()
        sorter = self.sim.sorter[0]
        sorter()

        rng = np.random.default_rng(42)
        n_add = 16
        for ipatch, p in enumerate(self.sim.patches):
            part = p.particles[0]
            part.extend(n_add)
            # revive the new (dead-by-default) slots inside this patch
            part.x[-n_add:] = rng.uniform(
                sorter.x0s[ipatch], sorter.x0s[ipatch] + sorter.nx_buckets*sorter.dx_buckets, n_add)
            part.y[-n_add:] = rng.uniform(
                sorter.y0s[ipatch], sorter.y0s[ipatch] + p.ny*p.dy, n_add)
            part.ux[-n_add:] = 0.0
            part.uy[-n_add:] = 0.0
            part.uz[-n_add:] = 0.0
            part.is_dead[-n_add:] = False

        # the extended flag routes through Simulation.update_lists
        self.sim.update_lists()

        for ipatch, p in enumerate(self.sim.patches):
            part = p.particles[0]
            self.assertEqual(sorter.particle_index_list[ipatch].size, part.npart)

        x_before = [p.particles[0].x.copy() for p in self.sim.patches]
        sorter()

        for ipatch, p in enumerate(self.sim.patches):
            part = p.particles[0]
            x, y = x_before[ipatch], part.y  # y unused below, positions unchanged by sort
            ix = np.floor((x - sorter.x0s[ipatch]) / sorter.dx_buckets).astype(int)
            ibin = np.clip(ix, 0, sorter.nx_buckets-1)  # 1D buckets, default order

            np.testing.assert_array_equal(sorter.particle_index_list[ipatch], ibin)

            cnt = np.bincount(ibin, minlength=sorter.nx_buckets).reshape(sorter.nx_buckets, 1)
            np.testing.assert_array_equal(cnt, sorter.bucket_count_list[ipatch])

            ix_sort = np.floor((part.x - sorter.x0s[ipatch]) / sorter.dx_buckets).astype(int)
            ibin_sort = np.clip(ix_sort, 0, sorter.nx_buckets-1)
            self.assertTrue((np.sort(ibin) == ibin_sort).all())

    def test_sort_with_dead_particles(self):
        """Dead particles are counted into the bucket of the previous slot
        (inherit chain) and relocated consistently with the float attrs."""
        self.sim.initialize()
        sorter = self.sim.sorter[0]
        self.assertEqual(sorter.ny_buckets, 1)  # 1D bucket path

        rng = np.random.default_rng(123)
        snaps = []
        for ipatch, p in enumerate(self.sim.patches):
            part = p.particles[0]
            npart = part.npart
            part.x[:] = rng.uniform(
                sorter.x0s[ipatch], sorter.x0s[ipatch] + sorter.nx_buckets*sorter.dx_buckets, npart)
            part.ux[:] = rng.normal(size=npart)  # per-particle tag
            part.is_dead[:] = rng.random(npart) < 0.2
            snaps.append((part.x.copy(), part.ux.copy(), part.is_dead.copy()))

        sorter()

        for ipatch, p in enumerate(self.sim.patches):
            part = p.particles[0]
            x_b, ux_b, dead_b = snaps[ipatch]

            # multisets of (x, ux) are preserved exactly (bitwise relocation),
            # separately among alive and dead particles
            for dead_mask_b, dead_mask_a in ((~dead_b, ~part.is_dead), (dead_b, part.is_dead)):
                self.assertEqual(dead_mask_a.sum(), dead_mask_b.sum())
                key_b = np.lexsort((ux_b[dead_mask_b], x_b[dead_mask_b]))
                key_a = np.lexsort((part.ux[dead_mask_a], part.x[dead_mask_a]))
                np.testing.assert_array_equal(
                    x_b[dead_mask_b][key_b], part.x[dead_mask_a][key_a])
                np.testing.assert_array_equal(
                    ux_b[dead_mask_b][key_b], part.ux[dead_mask_a][key_a])

            # every alive particle sits in a slot range belonging to its bucket
            alive = ~part.is_dead
            ix = np.floor((part.x[alive] - sorter.x0s[ipatch]) / sorter.dx_buckets).astype(int)
            ibin = np.clip(ix, 0, sorter.nx_buckets-1)
            bmin = sorter.bucket_bound_min_list[ipatch].ravel()
            bmax = sorter.bucket_bound_max_list[ipatch].ravel()
            slots = np.nonzero(alive)[0]
            self.assertTrue((bmin[ibin] <= slots).all())
            self.assertTrue((slots < bmax[ibin]).all())

            # bucket_count = alive histogram + dead counts per slot range
            hist = np.bincount(ibin, minlength=sorter.nx_buckets)
            dead_cumsum = np.concatenate([[0], np.cumsum(part.is_dead.astype(int))])
            dead_per_bucket = dead_cumsum[bmax] - dead_cumsum[bmin]
            np.testing.assert_array_equal(
                sorter.bucket_count_list[ipatch].ravel(), hist + dead_per_bucket)

    def test_all_dead_species(self):
        """A species with no alive particle keeps the default ordering and
        sorts to a no-op without crashing."""
        self.sim.initialize()
        sorter = self.sim.sorter[0]

        for p in self.sim.patches:
            p.particles[0].is_dead[:] = True

        nbuf = sorter()
        self.assertFalse(sorter.reverse_x)  # no live weight: default ordering
        self.assertEqual(nbuf, 0)  # all dead inherit one chain: already sorted

    def test_reverse_override_false(self):
        """Explicit reverse_x=False wins over the -x drift auto-detection."""
        self.sim.initialize()
        sorter = self.sim.sorter[0]

        for p in self.sim.patches:
            p.particles[0].ux[:] = -1.0
        sorter._reverse_x_override = False
        sorter()
        self.assertFalse(sorter.reverse_x)


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