import unittest
from unittest.mock import Mock, patch

from lambdapic.callback.callback import callback, SimulationStage
from lambdapic.simulation import Simulation, SimulationCallbacks

class TestCallback(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock simulation for testing
        self.sim = Mock()
        self.sim.patches = []
        
    def test_callback_decorator(self):
        """Test that callback works as a decorator."""
        @callback(stage="maxwell first")
        def test_func(sim):
            return "test_result"
            
        # Test function metadata is preserved
        self.assertEqual(test_func.__name__, "test_func")
        
        # Test stage is properly set
        self.assertEqual(test_func.stage, "maxwell first")
        
        # Test function is still callable
        result = test_func(self.sim)
        self.assertEqual(result, "test_result")
        
    def test_callback_default_stage(self):
        """Test that callback uses default stage when none specified."""
        @callback()
        def test_func(sim):
            pass
            
        self.assertEqual(test_func.stage, "maxwell second")
        
    def test_invalid_stage(self):
        """Test that callback raises error for invalid stage."""
        with self.assertRaises(ValueError):
            @callback(stage="invalid_stage")
            def test_func(sim):
                pass
                
    def test_callback_execution(self):
        """Test that callback can be executed both directly and via execute()."""
        test_value = []
        
        @callback(stage="start")
        def test_func(sim):
            test_value.append(1)
            
        # Test direct execution
        test_func(self.sim)
        self.assertEqual(len(test_value), 1)
        
        # Test execution via execute()
        test_func.execute(self.sim)
        self.assertEqual(len(test_value), 2)
        
    def test_simulation_integration(self):
        """Test that callbacks integrate properly with SimulationCallbacks."""
        executed_stages = []
        
        @callback(stage="maxwell first")
        def callback1(sim):
            executed_stages.append("maxwell first")
            
        @callback(stage="start")
        def callback2(sim):
            executed_stages.append("start")
            
        # Create SimulationCallbacks instance
        callbacks = SimulationCallbacks([callback1, callback2], self.sim)
        
        # Test callbacks are executed at correct stages
        callbacks.run("start")
        self.assertEqual(executed_stages, ["start"])
        
        callbacks.run("maxwell first")
        self.assertEqual(executed_stages, ["start", "maxwell first"])
        
        # Test stage with no callbacks doesn't affect execution
        callbacks.run("interpolator")
        self.assertEqual(executed_stages, ["start", "maxwell first"])
        
    def test_multiple_callbacks_same_stage(self):
        """Test that multiple callbacks can be registered for the same stage."""
        executed = []
        
        @callback(stage="start")
        def callback1(sim):
            executed.append(1)
            
        @callback(stage="start")
        def callback2(sim):
            executed.append(2)
            
        callbacks = SimulationCallbacks([callback1, callback2], self.sim)
        callbacks.run("start")
        
        self.assertEqual(executed, [1, 2])
        
    def test_plain_function_wrapping(self):
        """Test that plain functions are automatically wrapped with default stage."""
        def plain_func(sim):
            return "plain"
            
        callbacks = SimulationCallbacks([plain_func], self.sim)
        
        # Plain function should be wrapped and assigned to default stage
        self.assertTrue(any(cb.stage == "maxwell second" 
                          for cb in callbacks.stage_callbacks["maxwell second"]))

    def test_class_method_callback(self):
        """Test that callback can handle class methods."""
        class TestClass:
            @callback(stage="maxwell first")
            def class_method(self, sim):
                return "class_method_result"

        testclass = TestClass()
        
        # Test class method metadata is preserved
        self.assertEqual(testclass.class_method.__name__, "class_method")
        
        # Test stage is properly set
        self.assertEqual(testclass.class_method.stage, "maxwell first")
        
        # Test class method is still callable
        result = testclass.class_method(self.sim)
        self.assertEqual(result, "class_method_result")
        

    def test_lambda_callback(self):
        """Test that callback can handle lambda functions."""
        lambda_func = callback(stage="maxwell first")(lambda sim: "lambda_result")
        
        # Test lambda function metadata is preserved
        self.assertEqual(lambda_func.__name__, "<lambda>")
        
        # Test stage is properly set
        self.assertEqual(lambda_func.stage, "maxwell first")
        
        # Test lambda function is still callable
        result = lambda_func(self.sim)
        self.assertEqual(result, "lambda_result")

if __name__ == '__main__':
    unittest.main()
