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

@pytest.fixture
def mock_sim_with_time():
    """Fixture providing a mock simulation with time and dt properties."""
    sim = Mock()
    sim.patches = []
    sim.itime = 0
    sim.time = 0.0
    sim.dt = 0.1  # Default time step
    sim.mpi.rank = 0
    sim.mpi.comm.Barrier = Mock()
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

    def test_float_interval_validation_valid(self, mock_sim):
        """Test that valid float intervals are accepted."""
        # Test valid float intervals (0 < interval < 1)
        valid_intervals = [0.1, 0.5, 0.9, 0.001, 0.999]
        
        for interval in valid_intervals:
            @callback(stage="start", interval=interval)
            def test_func(sim):
                return f"result_{interval}"
            
            # Should not raise any error
            assert test_func.stage == "start"

    def test_float_interval_validation_invalid(self, mock_sim):
        """Test that invalid float intervals raise ValueError."""
        # Test invalid float intervals
        invalid_intervals = [0.0, 1.0, -0.1, 1.1, -1.0, 2.0]
        
        for interval in invalid_intervals:
            with pytest.raises(ValueError, match=f"Invalid interval: {interval}"):
                @callback(stage="start", interval=interval)
                def test_func(sim):
                    pass

    def test_float_interval_execution_logic(self, mock_sim_with_time):
        """Test the execution logic for float intervals."""
        executed = []
        
        @callback(stage="start", interval=0.5)
        def test_func(sim):
            executed.append(sim.time)
            return "executed"
        
        # Test at different time points
        test_cases = [
            (0.0, True),     # time=0.0, should execute (0.0 % 0.5 = 0.0 > 0.1? False → execute)
            (0.4, False),    # time=0.4, should not execute (0.4 % 0.5 = 0.4 > 0.1? True → not execute)
            (0.49, False),   # time=0.49, should not execute (0.49 % 0.5 = 0.49 > 0.1? True → not execute)
            (0.5, True),     # time=0.5, should execute (0.5 % 0.5 = 0.0 > 0.1? False → execute)
            (1.0, True),     # time=1.0, should execute (1.0 % 0.5 = 0.0 > 0.1? False → execute)
            (1.4, False),    # time=1.4, should not execute (1.4 % 0.5 = 0.4 > 0.1? True → not execute)
        ]
        
        for time, should_execute in test_cases:
            mock_sim_with_time.time = time
            executed.clear()
            result = test_func(mock_sim_with_time)
            
            if should_execute:
                assert len(executed) == 1
                assert executed[0] == time
                assert result == "executed"
            else:
                assert len(executed) == 0
                assert result is None

    def test_float_interval_boundary_cases(self, mock_sim):
        """Test boundary cases for float intervals."""
        # Test very small interval (close to 0) - should be valid
        @callback(stage="start", interval=0.0001)
        def test_func1(sim):
            pass
        assert test_func1.stage == "start"
        
        # Test interval very close to 1 - should be valid
        @callback(stage="start", interval=0.9999)
        def test_func2(sim):
            pass
        assert test_func2.stage == "start"
        
        # Test exactly 0 - should raise error
        with pytest.raises(ValueError, match="Invalid interval: 0.0"):
            @callback(stage="start", interval=0.0)
            def test_func3(sim):
                pass
        
        # Test exactly 1 - should raise error
        with pytest.raises(ValueError, match="Invalid interval: 1.0"):
            @callback(stage="start", interval=1.0)
            def test_func4(sim):
                pass

    def test_callback_class_float_interval(self, mock_sim_with_time):
        """Test that Callback class handles float intervals correctly."""
        from lambdapic.callback.callback import Callback
        
        class TestCallback(Callback):
            def __init__(self, interval):
                self.interval = interval
                self.stage = "start"
            
            def _call(self, sim):
                return None
        
        # Test valid float interval
        callback = TestCallback(0.5)
        mock_sim_with_time.time = 0.5  # Should execute at this time
        result = callback(mock_sim_with_time)
        assert result is None  # Callback.__call__ returns None
        
        # Test invalid float interval should raise ValueError when called
        invalid_callback = TestCallback(1.0)
        with pytest.raises(ValueError, match="Invalid interval: 1.0"):
            invalid_callback(mock_sim_with_time)
