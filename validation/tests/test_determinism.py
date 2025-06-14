"""
Determinism validation tests for checkpoint-based neural network inference.

This module verifies that checkpoint-based inference produces identical results
to standard inference, validating the core theoretical claim.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import (
    StandardNeuralNetwork,
    CheckpointNeuralNetwork,
    create_test_network,
    optimal_checkpoint_placement
)


class DeterminismValidator:
    """Validates determinism properties of checkpoint-based inference."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = []
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[DETERMINISM] {message}")
    
    def test_identical_outputs(self, 
                             layer_configs: List,
                             checkpoint_indices: List[int],
                             num_tests: int = 10,
                             input_shape: Tuple[int, ...] = (100,)) -> Dict:
        """
        Test that standard and checkpoint networks produce identical outputs.
        
        Args:
            layer_configs: Network layer configurations
            checkpoint_indices: Checkpoint placement strategy
            num_tests: Number of random inputs to test
            input_shape: Shape of input tensors
            
        Returns:
            Dictionary with test results
        """
        self.log(f"Testing identical outputs with {len(checkpoint_indices)} checkpoints")
        
        # Create networks with same seed for identical weights
        seed = 42
        standard_net = StandardNeuralNetwork(layer_configs, seed=seed)
        checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=seed)
        
        max_diff = 0.0
        differences = []
        
        for test_idx in range(num_tests):
            # Generate random input
            np.random.seed(test_idx + 1000)  # Different seed for inputs
            x = np.random.randn(*input_shape)
            
            # Forward pass through both networks
            standard_output, _ = standard_net.forward(x)
            checkpoint_output, _ = checkpoint_net.forward(x)
            
            # Calculate absolute difference
            diff = np.abs(standard_output - checkpoint_output)
            max_test_diff = np.max(diff)
            mean_test_diff = np.mean(diff)
            
            differences.append({
                'test_idx': test_idx,
                'max_diff': max_test_diff,
                'mean_diff': mean_test_diff,
                'output_norm': np.linalg.norm(standard_output)
            })
            
            max_diff = max(max_diff, max_test_diff)
            
            if self.verbose and test_idx < 3:  # Show first few tests
                self.log(f"Test {test_idx}: max_diff = {max_test_diff:.2e}, "
                        f"mean_diff = {mean_test_diff:.2e}")
        
        # Test results
        result = {
            'test_type': 'identical_outputs',
            'num_tests': num_tests,
            'max_difference': max_diff,
            'mean_difference': np.mean([d['mean_diff'] for d in differences]),
            'passed': max_diff < 1e-14,  # Machine precision threshold
            'differences': differences,
            'checkpoint_indices': checkpoint_indices,
            'network_params': standard_net.total_params
        }
        
        self.log(f"Maximum difference across all tests: {max_diff:.2e}")
        self.log(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
        
        return result
    
    def test_multiple_runs_same_input(self,
                                    layer_configs: List,
                                    checkpoint_indices: List[int],
                                    num_runs: int = 5) -> Dict:
        """
        Test that multiple runs with same input produce identical results.
        
        Args:
            layer_configs: Network layer configurations
            checkpoint_indices: Checkpoint placement strategy
            num_runs: Number of runs with same input
            
        Returns:
            Dictionary with test results
        """
        self.log(f"Testing multiple runs with same input ({num_runs} runs)")
        
        # Create network
        checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=42)
        
        # Fixed input - use appropriate input dimension
        np.random.seed(123)
        input_dim = layer_configs[0].input_dim
        x = np.random.randn(input_dim)
        
        outputs = []
        for run_idx in range(num_runs):
            output, _ = checkpoint_net.forward(x)
            outputs.append(output.copy())
        
        # Check all outputs are identical
        max_diff = 0.0
        for i in range(1, num_runs):
            diff = np.max(np.abs(outputs[0] - outputs[i]))
            max_diff = max(max_diff, diff)
        
        result = {
            'test_type': 'multiple_runs_same_input',
            'num_runs': num_runs,
            'max_difference': max_diff,
            'passed': max_diff < 1e-15,
            'checkpoint_indices': checkpoint_indices
        }
        
        self.log(f"Maximum difference between runs: {max_diff:.2e}")
        self.log(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
        
        return result
    
    def test_different_checkpoint_strategies(self,
                                           layer_configs: List,
                                           strategies: List[List[int]]) -> Dict:
        """
        Test that different checkpoint strategies produce identical results.
        
        Args:
            layer_configs: Network layer configurations
            strategies: List of different checkpoint placement strategies
            
        Returns:
            Dictionary with test results
        """
        self.log(f"Testing {len(strategies)} different checkpoint strategies")
        
        # Create standard network as reference
        standard_net = StandardNeuralNetwork(layer_configs, seed=42)
        
        # Fixed input - use appropriate input dimension  
        np.random.seed(456)
        input_dim = layer_configs[0].input_dim
        x = np.random.randn(input_dim)
        standard_output, _ = standard_net.forward(x)
        
        results = []
        max_overall_diff = 0.0
        
        for strategy_idx, checkpoint_indices in enumerate(strategies):
            checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=42)
            checkpoint_output, _ = checkpoint_net.forward(x)
            
            diff = np.max(np.abs(standard_output - checkpoint_output))
            max_overall_diff = max(max_overall_diff, diff)
            
            strategy_result = {
                'strategy_idx': strategy_idx,
                'checkpoint_indices': checkpoint_indices,
                'max_diff': diff,
                'passed': diff < 1e-14
            }
            results.append(strategy_result)
            
            self.log(f"Strategy {strategy_idx} (checkpoints {checkpoint_indices}): "
                    f"diff = {diff:.2e}")
        
        overall_result = {
            'test_type': 'different_checkpoint_strategies',
            'num_strategies': len(strategies),
            'max_difference': max_overall_diff,
            'passed': max_overall_diff < 1e-14,
            'strategy_results': results
        }
        
        self.log(f"Overall test {'PASSED' if overall_result['passed'] else 'FAILED'}")
        
        return overall_result
    
    def run_comprehensive_determinism_tests(self) -> Dict:
        """Run all determinism tests and return comprehensive results."""
        self.log("=" * 60)
        self.log("COMPREHENSIVE DETERMINISM VALIDATION")
        self.log("=" * 60)
        
        all_results = []
        
        # Test configurations
        test_configs = [
            {
                'name': 'Small Network',
                'layers': create_test_network(num_layers=4, layer_size=50, input_dim=50),
                'checkpoints': [0, 2]
            },
            {
                'name': 'Medium Network', 
                'layers': create_test_network(num_layers=8, layer_size=100, input_dim=100),
                'checkpoints': optimal_checkpoint_placement(8)
            },
            {
                'name': 'Large Network',
                'layers': create_test_network(num_layers=16, layer_size=100, input_dim=100), 
                'checkpoints': optimal_checkpoint_placement(16)
            }
        ]
        
        for config in test_configs:
            self.log(f"\nTesting {config['name']}...")
            self.log(f"Layers: {len(config['layers'])}, Checkpoints: {config['checkpoints']}")
            
            # Test 1: Identical outputs  
            input_dim = config['layers'][0].input_dim
            result1 = self.test_identical_outputs(
                config['layers'], 
                config['checkpoints'],
                num_tests=20,
                input_shape=(input_dim,)
            )
            result1['network_name'] = config['name']
            all_results.append(result1)
            
            # Test 2: Multiple runs same input
            result2 = self.test_multiple_runs_same_input(
                config['layers'],
                config['checkpoints'],
                num_runs=10
            )
            result2['network_name'] = config['name']
            all_results.append(result2)
            
            # Test 3: Different checkpoint strategies (for medium network)
            if config['name'] == 'Medium Network':
                num_layers = len(config['layers'])
                strategies = [
                    [0, 3, 6],      # Uniform spacing (valid for 8-layer network)
                    [0, 2, 4, 6],   # Dense early (exclude final layer 7)
                    [0, 6],         # Sparse (exclude final layer 7)
                    optimal_checkpoint_placement(num_layers)  # Theoretical optimal
                ]
                result3 = self.test_different_checkpoint_strategies(
                    config['layers'],
                    strategies
                )
                result3['network_name'] = config['name']
                all_results.append(result3)
        
        # Summary
        passed_tests = sum(1 for r in all_results if r['passed'])
        total_tests = len(all_results)
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests,
            'all_passed': passed_tests == total_tests,
            'detailed_results': all_results
        }
        
        self.log("\n" + "=" * 60)
        self.log("DETERMINISM TEST SUMMARY")
        self.log("=" * 60)
        self.log(f"Tests passed: {passed_tests}/{total_tests}")
        self.log(f"Success rate: {summary['success_rate']:.1%}")
        self.log(f"Overall result: {'PASS' if summary['all_passed'] else 'FAIL'}")
        
        return summary


def main():
    """Run determinism validation tests."""
    validator = DeterminismValidator(verbose=True)
    results = validator.run_comprehensive_determinism_tests()
    
    # Save results
    import json
    results_file = '../results/determinism_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    # Recursively convert numpy types
    def deep_convert(data):
        if isinstance(data, dict):
            return {k: deep_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [deep_convert(item) for item in data]
        else:
            return convert_numpy(data)
    
    json_results = deep_convert(results)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results['all_passed']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)