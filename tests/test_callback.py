import pytest
from unittest.mock import Mock, patch

from lambdapic.callback.callback import callback, SimulationStage
from lambdapic.simulation import Simulation, SimulationCallbacks

@pytest.fixture
def mock_sim():
    """Fixture providing a mock simulation."""
    sim = Mock()
    sim.patches = []
    sim.itime = 0
    sim.mpi.rank = 0
    return sim

@pytest.mark.unit
class TestCallback:
        
    def test_callback_decorator(self, mock_sim):
        """Test that callback works as a decorator."""
        @callback(stage="maxwell first")
        def test_func(sim):
            return "test_result"
            
        # Test function metadata is preserved
        assert test_func.__name__ == "test_func"
        
        # Test stage is properly set
        assert test_func.stage == "maxwell first"
        
        # Test function is still callable
        result = test_func(mock_sim)
        assert result == "test_result"
        
    def test_callback_default_stage(self, mock_sim):
        """Test that callback uses default stage when none specified."""
        @callback()
        def test_func(sim):
            pass
            
        assert test_func.stage == "maxwell second"
        
    def test_invalid_stage(self, mock_sim):
        """Test that callback raises error for invalid stage."""
        with pytest.raises(ValueError):
            @callback(stage="invalid_stage")
            def test_func(sim):
                pass
                
    def test_callback_execution(self, mock_sim):
        """Test that callback can be executed both directly and via execute()."""
        test_value = []
        
        @callback(stage="start")
        def test_func(sim):
            test_value.append(1)
            
        # Test direct execution
        test_func(mock_sim)
        assert len(test_value) == 1

    def test_simulation_integration(self, mock_sim):
        """Test that callbacks integrate properly with SimulationCallbacks."""
        executed_stages = []
        
        @callback(stage="maxwell first")
        def callback1(sim):
            executed_stages.append("maxwell first")
            
        @callback(stage="start")
        def callback2(sim):
            executed_stages.append("start")
            
        # Create SimulationCallbacks instance
        callbacks = SimulationCallbacks([callback1, callback2], mock_sim)
        
        # Test callbacks are executed at correct stages
        callbacks.run("start")
        assert executed_stages == ["start"]
        
        callbacks.run("maxwell first")
        assert executed_stages == ["start", "maxwell first"]
        
        # Test stage with no callbacks doesn't affect execution
        callbacks.run("interpolator")
        assert executed_stages == ["start", "maxwell first"]
        
    def test_multiple_callbacks_same_stage(self, mock_sim):
        """Test that multiple callbacks can be registered for the same stage."""
        executed = []
        
        @callback(stage="start")
        def callback1(sim):
            executed.append(1)
            
        @callback(stage="start")
        def callback2(sim):
            executed.append(2)
            
        callbacks = SimulationCallbacks([callback1, callback2], mock_sim)
        callbacks.run("start")
        
        assert executed == [1, 2]
        
    def test_plain_function_wrapping(self, mock_sim):
        """Test that plain functions are automatically wrapped with default stage."""
        def plain_func(sim):
            return "plain"
            
        callbacks = SimulationCallbacks([plain_func], mock_sim)
        
        # Plain function should be wrapped and assigned to default stage
        assert any(cb.stage == "maxwell second" 
                 for cb in callbacks.stage_callbacks["maxwell second"])

    def test_class_method_callback(self, mock_sim):
        """Test that callback can handle class methods."""
        class TestClass:
            @callback(stage="maxwell first")
            def class_method(self, sim):
                return "class_method_result"

        testclass = TestClass()
        
        # Test class method metadata is preserved
        assert testclass.class_method.__name__ == "class_method"
        
        # Test stage is properly set
        assert testclass.class_method.stage == "maxwell first"
        
        # Test class method is still callable
        result = testclass.class_method(mock_sim)
        assert result == "class_method_result"
        

    def test_lambda_callback(self, mock_sim):
        """Test that callback can handle lambda functions."""
        lambda_func = callback(stage="maxwell first")(lambda sim: "lambda_result")
        
        # Test lambda function metadata is preserved
        assert lambda_func.__name__ == "<lambda>"
        
        # Test stage is properly set
        assert lambda_func.stage == "maxwell first"
        
        # Test lambda function is still callable
        result = lambda_func(mock_sim)
        assert result == "lambda_result"
