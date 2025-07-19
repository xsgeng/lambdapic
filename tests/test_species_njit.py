"""Tests for njit functionality in species.py"""

import pytest
import numpy as np
from numba import njit
from numba.core.dispatcher import Dispatcher

from lambdapic.core.species import Species


class TestSpeciesNjit:
    """Test class for Species njit compilation functionality."""

    def test_compile_jit_with_already_jitted_function(self):
        """Test compile_jit with already jitted function."""
        @njit
        def already_jitted_2d(x, y):
            return x + y
            
        result = Species.compile_jit(already_jitted_2d, 2)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0) == 3.0
        
    def test_compile_jit_with_2d_function(self):
        """Test compile_jit with 2D function."""
        def func_2d(x, y):
            return x**2 + y**2
            
        result = Species.compile_jit(func_2d, 2)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0) == 5.0
        assert result(3.0, 4.0) == 25.0
        
    def test_compile_jit_with_3d_function(self):
        """Test compile_jit with 3D function."""
        def func_3d(x, y, z):
            return x**2 + y**2 + z**2
            
        result = Species.compile_jit(func_3d, 3)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0, 3.0) == 14.0
        assert result(2.0, 3.0, 4.0) == 29.0
        
    def test_compile_jit_with_constant_2d(self):
        """Test compile_jit with constant value for 2D."""
        constant = 42.0
        result = Species.compile_jit(constant, 2)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0) == constant
        assert result(10.0, 20.0) == constant
        assert result(-5.0, 3.14) == constant
        
    def test_compile_jit_with_constant_3d(self):
        """Test compile_jit with constant value for 3D."""
        constant = 3.14
        result = Species.compile_jit(constant, 3)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0, 3.0) == constant
        assert result(10.0, 20.0, 30.0) == constant
        assert result(-1.0, -2.0, -3.0) == constant
        
    def test_compile_jit_argument_mismatch_error_2d(self):
        """Test compile_jit raises error for argument mismatch in 2D."""
        def func_3_args(x, y, z):
            return x + y + z
            
        with pytest.raises(ValueError, match="function.*must have 2 arguments"):
            Species.compile_jit(func_3_args, 2)
            
    def test_compile_jit_argument_mismatch_error_3d(self):
        """Test compile_jit raises error for argument mismatch in 3D."""
        def func_2_args(x, y):
            return x + y
            
        with pytest.raises(ValueError, match="function.*must have 3 arguments"):
            Species.compile_jit(func_2_args, 3)
            
    def test_compile_jit_invalid_dimension_error(self):
        """Test compile_jit raises error for invalid dimension."""
        constant = 42.0
        
        with pytest.raises(ValueError, match="dimension must be 2 or 3"):
            Species.compile_jit(constant, 4)
            
    def test_compile_jit_invalid_input_type_error(self):
        """Test compile_jit raises error for invalid input type."""
        invalid_input = "not_a_function_or_number"
        
        with pytest.raises(ValueError, match="Invalid profile"):
            Species.compile_jit(invalid_input, 2)
            
    def test_compile_jit_with_int_constant_2d(self):
        """Test compile_jit with integer constant for 2D."""
        constant = 100
        result = Species.compile_jit(constant, 2)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0) == float(constant)
        
    def test_compile_jit_with_int_constant_3d(self):
        """Test compile_jit with integer constant for 3D."""
        constant = 200
        result = Species.compile_jit(constant, 3)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0, 3.0) == float(constant)
        
    def test_jitted_function_performance_2d(self):
        """Test that jitted 2D function performs correctly."""
        def complex_2d_func(x, y):
            return np.sin(x) * np.cos(y) + x * y
            
        jitted_func = Species.compile_jit(complex_2d_func, 2)
        
        # Test with arrays
        x_vals = np.array([1.0, 2.0, 3.0])
        y_vals = np.array([4.0, 5.0, 6.0])
        
        expected = [complex_2d_func(x, y) for x, y in zip(x_vals, y_vals)]
        actual = [jitted_func(x, y) for x, y in zip(x_vals, y_vals)]
        
        np.testing.assert_array_almost_equal(actual, expected)
        
    def test_jitted_function_performance_3d(self):
        """Test that jitted 3D function performs correctly."""
        def complex_3d_func(x, y, z):
            return np.sin(x) * np.cos(y) * np.exp(z/10) + x * y * z
            
        jitted_func = Species.compile_jit(complex_3d_func, 3)
        
        # Test with arrays
        x_vals = np.array([1.0, 2.0])
        y_vals = np.array([3.0, 4.0])
        z_vals = np.array([5.0, 6.0])
        
        expected = [complex_3d_func(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)]
        actual = [jitted_func(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)]
        
        np.testing.assert_array_almost_equal(actual, expected)
        
    def test_compile_jit_with_zero_constant(self):
        """Test compile_jit with zero constant."""
        result = Species.compile_jit(0.0, 2)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0) == 0.0
        assert result(-10.0, 5.0) == 0.0
        
    def test_compile_jit_with_negative_constant(self):
        """Test compile_jit with negative constant."""
        constant = -42.5
        result = Species.compile_jit(constant, 3)
        
        assert isinstance(result, Dispatcher)
        assert result(1.0, 2.0, 3.0) == constant