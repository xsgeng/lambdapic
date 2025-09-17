import numpy as np
import pytest
from scipy.constants import c, epsilon_0, e, m_e, k
from lambdapic.core.collision.cpu import debye_length_patch, debye_length_cell
from lambdapic.core.collision.utils import ParticleData
from lambdapic.core.particles import ParticlesBase
from numba import typed


class TestDebyeLengthPatch:
    """测试debye_length_patch函数的pytest测试类"""
    
    def create_test_particles(self, n_particles: int, 
                            inv_gamma_value: float = 0.9,
                            weight_value: float = 1.0) -> ParticleData:
        """创建测试用的ParticleData对象"""
        particles = ParticlesBase()
        particles.initialize(n_particles)
        
        # 设置粒子属性
        particles.inv_gamma[:] = inv_gamma_value
        particles.w[:] = weight_value
        particles.is_dead[:] = False
        
        # 创建ParticleData对象
        return ParticleData(
            x=particles.x, y=particles.y, z=particles.z,
            ux=particles.ux, uy=particles.uy, uz=particles.uz,
            inv_gamma=particles.inv_gamma, w=particles.w,
            is_dead=particles.is_dead, m=m_e, q=-e
        )
    
    def test_basic_functionality(self):
        """测试基本功能：验证debye_length_patch能正确计算德拜长度"""
        # 设置测试参数
        nx, ny = 4, 4
        n_particles = 100
        
        # 创建测试粒子数据
        particle_data = self.create_test_particles(n_particles, inv_gamma_value=0.9)
        
        # 创建bucket边界
        bucket_bound_min = np.zeros(nx * ny, dtype=np.int64)
        bucket_bound_max = np.full(nx * ny, n_particles, dtype=np.int64)
        
        # 将粒子分配到第一个cell
        bucket_bound_min[0] = 0
        bucket_bound_max[0] = n_particles
        for i in range(1, nx * ny):
            bucket_bound_min[i] = 0
            bucket_bound_max[i] = 0
        
        # 设置物理参数
        dx = dy = dz = 1e-6  # 1微米
        cell_vol = dx * dy * dz
        
        # 创建输出数组
        debye_length_inv_square = np.zeros((nx, ny), dtype=np.float64)
        
        # 调用函数
        debye_length_patch(
            particle_data,
            bucket_bound_min, bucket_bound_max,
            cell_vol,
            debye_length_inv_square
        )
        
        # 验证结果
        assert not np.allclose(debye_length_inv_square, 0.0)
        assert debye_length_inv_square[0, 0] > 0  # 第一个cell应该有值
        assert np.all(debye_length_inv_square[1:, :] == 0)  # 其他cell应该为0
        
    def test_empty_cells(self):
        """测试空cell的情况"""
        nx, ny = 2, 2
        
        # 创建空粒子数据
        empty_data = self.create_test_particles(0)
        
        # 所有cell都没有粒子
        bucket_bound_min = np.zeros(nx * ny, dtype=np.int64)
        bucket_bound_max = np.zeros(nx * ny, dtype=np.int64)
        
        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz
        
        debye_length_inv_square = np.zeros((nx, ny), dtype=np.float64)
        
        # 调用函数
        debye_length_patch(
            empty_data,
            bucket_bound_min, bucket_bound_max,
            cell_vol,
            debye_length_inv_square
        )
        
        # 验证所有值保持为0（因为没有粒子，函数不会更新数组）
        expected = np.zeros((nx, ny))
        np.testing.assert_array_equal(debye_length_inv_square, expected)
        
    def test_dead_particles(self):
        """测试所有粒子都是dead的情况"""
        nx, ny = 2, 2
        n_particles = 10
        
        # 创建所有粒子都是dead的测试数据
        dead_data = self.create_test_particles(n_particles, inv_gamma_value=1.0, weight_value=1.0)
        
        # 设置所有粒子为dead
        dead_data.is_dead[:] = True
        
        bucket_bound_min = np.array([0, 0, 0, 0], dtype=np.int64)
        bucket_bound_max = np.array([n_particles, 0, 0, 0], dtype=np.int64)
        
        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz
        
        debye_length_inv_square = np.zeros((nx, ny), dtype=np.float64)
        
        debye_length_patch(
            dead_data,
            bucket_bound_min, bucket_bound_max,
            cell_vol,
            debye_length_inv_square
        )
        
        # 验证结果保持为0（因为没有有效粒子，函数不会更新数组）
        assert debye_length_inv_square[0, 0] == 0.0
        
    def test_uniform_distribution(self):
        """测试粒子均匀分布的情况"""
        nx, ny = 4, 4
        n_particles_per_cell = 10
        n_total = nx * ny * n_particles_per_cell
        
        # 创建均匀分布的粒子
        uniform_data = self.create_test_particles(n_total, inv_gamma_value=0.95, weight_value=1.0)
        
        # 均匀分配到所有cell
        bucket_bound_min = np.zeros(nx * ny, dtype=np.int64)
        bucket_bound_max = np.zeros(nx * ny, dtype=np.int64)
        
        for i in range(nx * ny):
            bucket_bound_min[i] = i * n_particles_per_cell
            bucket_bound_max[i] = (i + 1) * n_particles_per_cell
        
        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz
        
        debye_length_inv_square = np.zeros((nx, ny), dtype=np.float64)
        
        debye_length_patch(
            uniform_data,
            bucket_bound_min, bucket_bound_max,
            cell_vol,
            debye_length_inv_square
        )
        
        # 验证所有cell都有相同的德拜长度倒数平方
        expected_value = debye_length_inv_square[0, 0]
        assert expected_value > 0
        np.testing.assert_allclose(debye_length_inv_square, expected_value)
        
    def test_physical_correctness(self):
        """测试物理正确性：验证计算结果与理论值的一致性"""
        # 创建简单的测试场景
        n_particles = 1000

        from lambdapic.callback.utils import SetTemperature

        # 设置非相对论情况
        kT_mc2 = 0.01
        ux, uy, uz = SetTemperature.sample_maxwell_juttner(n_particles, kT_mc2)
        
        particles = ParticlesBase()
        particles.initialize(n_particles)
        particles.ux[:] = ux
        particles.uy[:] = uy
        particles.uz[:] = uz
        particles.inv_gamma[:] = 1.0 / np.sqrt(1 + ux**2 + uy**2 + uz**2)
        particles.w[:] = 1.0
        particles.is_dead[:] = False
        
        physical_data = ParticleData(
            x=particles.x, y=particles.y, z=particles.z,
            ux=particles.ux, uy=particles.uy, uz=particles.uz,
            inv_gamma=particles.inv_gamma, w=particles.w,
            is_dead=particles.is_dead, m=m_e, q=-e
        )

        bucket_bound_min = np.array([0], dtype=np.int64)
        bucket_bound_max = np.array([n_particles], dtype=np.int64)

        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz

        # 计算理论德拜长度
        density = n_particles / cell_vol  # 粒子数密度

        # 计算温度：对于gamma = 1.01，动能 = (gamma-1)mc^2
        kT = kT_mc2 * m_e * c**2

        # 理论德拜长度倒数平方
        lambda_D_inv_sq_theory = density * e**2 / (epsilon_0 * kT)

        debye_length_inv_square = np.zeros((1, 1), dtype=np.float64)

        debye_length_patch(
            physical_data,
            bucket_bound_min, bucket_bound_max,
            cell_vol,
            debye_length_inv_square
        )

        # 验证计算值与理论值数量级一致
        calculated = debye_length_inv_square[0, 0]
        assert calculated > 0
        ratio = abs(calculated - lambda_D_inv_sq_theory) / lambda_D_inv_sq_theory
        assert ratio < 0.05
        
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试单个粒子
        edge_data = self.create_test_particles(1, inv_gamma_value=1.0, weight_value=1.0)
        
        bucket_bound_min = np.array([0], dtype=np.int64)
        bucket_bound_max = np.array([1], dtype=np.int64)
        
        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz
        
        debye_length_inv_square = np.zeros((1, 1), dtype=np.float64)
        
        debye_length_patch(
            edge_data,
            bucket_bound_min, bucket_bound_max,
            cell_vol,
            debye_length_inv_square
        )
        
        # 单个粒子也应该有合理的德拜长度
        assert debye_length_inv_square[0, 0] > 0
        
    def test_debye_length_cell_directly(self):
        """直接测试debye_length_cell函数"""
        n_particles = 10
        
        cell_data = self.create_test_particles(n_particles, inv_gamma_value=0.9, weight_value=1.0)
        
        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz
        
        # 调用底层函数
        result = debye_length_cell(
            cell_data,
            np.int64(0), np.int64(n_particles), 
            cell_vol
        )
        
        # 验证结果合理
        assert result[0] > 0
        
        # 测试空范围
        empty_result = debye_length_cell(
            cell_data,
            np.int64(0), np.int64(0), 
            cell_vol
        )
        assert empty_result[0] == -1.0
        
        # 测试全dead粒子
        dead_cell_data = self.create_test_particles(n_particles, inv_gamma_value=0.9, weight_value=1.0)
        dead_cell_data.is_dead[:] = True
        
        dead_result = debye_length_cell(
            dead_cell_data,
            np.int64(0), np.int64(n_particles), 
            cell_vol
        )
        assert dead_result[0] == -1.0

    def test_debye_length_patches(self):
        """测试debye_length_patches函数（多patch版本）"""
        from lambdapic.core.collision.cpu import debye_length_patches
        
        npatches = 3
        nx, ny = 2, 2
        n_particles_per_patch = 20
        
        # 创建多个patch的数据，使用Numba的typed.List
        bucket_bound_min_list = typed.List() # type: ignore
        bucket_bound_max_list = typed.List() # type: ignore
        particle_data_list = typed.List() # type: ignore
        debye_length_inv_square_list = typed.List() # type: ignore
        total_density_list = typed.List() # type: ignore
        
        for ipatch in range(npatches):
            # 使用create_test_particles创建粒子数据
            particle_data = self.create_test_particles(
                n_particles_per_patch, 
                inv_gamma_value=0.95, 
                weight_value=1.0
            )
            
            # 均匀分配到所有cell
            bucket_bound_min = np.zeros(nx * ny, dtype=np.int64)
            bucket_bound_max = np.zeros(nx * ny, dtype=np.int64)
            
            particles_per_cell = n_particles_per_patch // (nx * ny)
            for i in range(nx * ny):
                bucket_bound_min[i] = i * particles_per_cell
                bucket_bound_max[i] = (i + 1) * particles_per_cell
            
            # 添加到列表中
            particle_data_list.append(particle_data)
            bucket_bound_min_list.append(bucket_bound_min)
            bucket_bound_max_list.append(bucket_bound_max)
            debye_length_inv_square_list.append(np.zeros((nx, ny), dtype=np.float64))
            total_density_list.append(np.zeros((nx, ny), dtype=np.float64))
        
        dx = dy = dz = 1e-6
        cell_vol = dx * dy * dz
        
        # 调用多patch函数
        debye_length_patches(
            particle_data_list,
            bucket_bound_min_list, bucket_bound_max_list,
            cell_vol,
            debye_length_inv_square_list,
            total_density_list
        )
        
        # 验证所有patch都有合理的结果
        for ipatch in range(npatches):
            result = debye_length_inv_square_list[ipatch]
            assert np.all(result >= 0)  # 所有值应该非负
            assert np.any(result > 0)   # 至少有一些cell有值
        
        # 验证所有patch的结果应该相同（因为输入相同）
        for ipatch in range(1, npatches):
            np.testing.assert_allclose(
                debye_length_inv_square_list[ipatch],
                debye_length_inv_square_list[0]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])