import numpy as np
import pytest
from lambdapic import Simulation3D, Electron
from lambdapic.callback.utils import get_fields_3d


class TestGetFields3D:
    """测试 get_fields_3d 函数的功能"""
    
    def setup_method(self):
        """在每个测试方法前设置测试环境"""
        # 创建一个简单的 3D 模拟
        self.sim = Simulation3D(
            nx=32, ny=32, nz=32,
            dx=0.1, dy=0.1, dz=0.1,
            npatch_x=4, npatch_y=4, npatch_z=4,
        )
        
        self.sim.initialize()
        
        # 在每个 patch 中设置一些简单的场数据
        for p in self.sim.patches:
            # 设置电场
            p.fields.ex[:, :, :] = 1.0
            p.fields.ey[:, :, :] = 2.0
            p.fields.ez[:, :, :] = 3.0
            
            # 设置磁场
            p.fields.bx[:, :, :] = 4.0
            p.fields.by[:, :, :] = 5.0
            p.fields.bz[:, :, :] = 6.0

    def test_get_fields_3d_basic(self):
        """测试基本的 get_fields_3d 功能"""
        # 测试获取所有场
        fields = ['ex', 'ey', 'ez', 'bx', 'by', 'bz']
        result = get_fields_3d(self.sim, fields, slice_at=self.sim.Lz/2)
        
        # 验证结果
        assert len(result) == len(fields)
        
        # 只在 rank 0 上验证结果
        if self.sim.mpi.rank == 0:
            for i, field in enumerate(fields):
                assert result[i] is not None
                assert result[i].shape == (self.sim.nx, self.sim.ny)
                
                # 验证场值
                if field == 'ex':
                    np.testing.assert_array_equal(result[i], 1.0)
                elif field == 'ey':
                    np.testing.assert_array_equal(result[i], 2.0)
                elif field == 'ez':
                    np.testing.assert_array_equal(result[i], 3.0)
                elif field == 'bx':
                    np.testing.assert_array_equal(result[i], 4.0)
                elif field == 'by':
                    np.testing.assert_array_equal(result[i], 5.0)
                elif field == 'bz':
                    np.testing.assert_array_equal(result[i], 6.0)
        else:
            # 其他 rank 应该返回 None
            for res in result:
                assert res is None

    def test_get_fields_3d_different_slice(self):
        """测试在不同 z 位置切片"""
        # 创建一个简单的 3D 模拟
        sim = self.sim
        
        # 设置随 z 变化的电场
        for p in sim.patches:
            z_values = p.fields.zaxis[0, 0, :]
            for iz in range(p.nz):
                p.fields.ex[:, :, iz] = z_values[iz]
        
        # 测试在不同 z 位置获取场
        slice_positions = [sim.Lz/4, sim.Lz/2, 3*sim.Lz/4]
        
        for slice_pos in slice_positions:
            result = get_fields_3d(sim, ['ex'], slice_at=slice_pos)[0]
            
            # 只在 rank 0 上验证结果
            if sim.mpi.rank == 0:
                assert result is not None
                assert result.shape == (sim.nx, sim.ny)
                
                # 验证场值是否与切片位置匹配
                expected_value = slice_pos
                # 由于场值设置方式，我们需要检查平均值
                assert result[0, 0] == expected_value
            else:
                assert result is None


    def test_get_fields_3d_invalid_slice(self):
        """测试无效的切片位置"""
        # 测试无效的切片位置（负值）
        with pytest.raises(ValueError, match="Slice position .* is outside the simulation domain"):
            get_fields_3d(self.sim, ['ex'], slice_at=-1.0)
        
        # 测试无效的切片位置（超过 Lz）
        with pytest.raises(ValueError, match="Slice position .* is outside the simulation domain"):
            get_fields_3d(self.sim, ['ex'], slice_at=self.sim.Lz + 1.0)
