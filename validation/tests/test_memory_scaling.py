"""
Memory scaling validation tests for checkpoint-based neural network inference.

This module validates the O(√n) memory scaling claim by measuring actual
memory usage across different network sizes and checkpoint strategies.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import time
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import (
    StandardNeuralNetwork,
    CheckpointNeuralNetwork,
    create_test_network,
    optimal_checkpoint_placement,
    MemoryTracker
)


class MemoryScalingValidator:
    """Validates memory scaling properties of checkpoint-based inference."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = []
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[MEMORY] {message}")
    
    def measure_memory_usage(self,
                           network_sizes: List[int],
                           layer_size: int = 100) -> Dict:
        """
        Measure memory usage across different network sizes.
        
        Args:
            network_sizes: List of number of layers to test
            layer_size: Size of each layer
            
        Returns:
            Dictionary with memory measurements
        """
        self.log(f"Measuring memory usage for network sizes: {network_sizes}")
        
        results = {
            'network_sizes': network_sizes,
            'standard_memory': [],
            'checkpoint_memory': [],
            'theoretical_params': [],
            'checkpoint_params': [],
            'memory_reduction': []
        }
        
        for num_layers in network_sizes:
            self.log(f"\nTesting {num_layers} layers...")
            
            # Create network configuration
            layer_configs = create_test_network(
                num_layers=num_layers,
                layer_size=layer_size,
                input_dim=layer_size
            )
            
            # Calculate optimal checkpoints
            checkpoint_indices = optimal_checkpoint_placement(num_layers)
            
            # Create networks
            standard_net = StandardNeuralNetwork(layer_configs, seed=42)
            checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=42)
            
            # Create test input
            x = np.random.randn(layer_size)
            
            # Measure standard network memory
            _, standard_memory_stats = standard_net.forward(x, track_memory=True)
            
            # Measure checkpoint network memory  
            _, checkpoint_memory_stats = checkpoint_net.forward(x, track_memory=True)
            
            # Get theoretical memory usage
            checkpoint_usage = checkpoint_net.get_memory_usage()
            
            # Store results
            results['standard_memory'].append(standard_memory_stats['peak'])
            results['checkpoint_memory'].append(checkpoint_memory_stats['peak'])
            results['theoretical_params'].append(standard_net.total_params)
            results['checkpoint_params'].append(checkpoint_usage['checkpoint_params'])
            results['memory_reduction'].append(checkpoint_usage['memory_reduction_ratio'])
            
            self.log(f"  Total params: {standard_net.total_params:,}")
            self.log(f"  Checkpoint params: {checkpoint_usage['checkpoint_params']:,}")
            self.log(f"  Memory reduction: {checkpoint_usage['memory_reduction_ratio']:.1f}x")
            self.log(f"  Standard memory: {standard_memory_stats['peak']:,} bytes")
            self.log(f"  Checkpoint memory: {checkpoint_memory_stats['peak']:,} bytes")
        
        return results
    
    def validate_sqrt_scaling(self, memory_data: Dict) -> Dict:
        """
        Validate that checkpoint memory usage scales as O(√n).
        
        Args:
            memory_data: Memory measurement data
            
        Returns:
            Dictionary with scaling analysis results
        """
        self.log("\nValidating O(√n) scaling...")
        
        params = np.array(memory_data['theoretical_params'])
        checkpoint_params = np.array(memory_data['checkpoint_params'])
        
        # Theoretical sqrt(n) scaling
        sqrt_params = np.sqrt(params)
        
        # Fit linear model: checkpoint_params = a * sqrt(params) + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            sqrt_params, checkpoint_params
        )
        
        # Calculate correlation with sqrt scaling
        correlation = np.corrcoef(sqrt_params, checkpoint_params)[0, 1]
        
        # Test linear scaling for comparison (should have lower correlation)
        linear_correlation = np.corrcoef(params, checkpoint_params)[0, 1]
        
        # Test quadratic scaling for comparison (should have lower correlation)  
        quad_correlation = np.corrcoef(params**2, checkpoint_params)[0, 1]
        
        result = {
            'sqrt_correlation': correlation,
            'linear_correlation': linear_correlation,
            'quadratic_correlation': quad_correlation,
            'r_squared': r_value**2,
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'sqrt_scaling_confirmed': correlation > 0.95 and correlation > linear_correlation,
            'params': params.tolist(),
            'checkpoint_params': checkpoint_params.tolist(),
            'sqrt_params': sqrt_params.tolist()
        }
        
        self.log(f"  √n correlation: {correlation:.4f}")
        self.log(f"  Linear correlation: {linear_correlation:.4f}") 
        self.log(f"  Quadratic correlation: {quad_correlation:.4f}")
        self.log(f"  R²: {r_value**2:.4f}")
        self.log(f"  √n scaling confirmed: {result['sqrt_scaling_confirmed']}")
        
        return result
    
    def test_checkpoint_strategies(self,
                                 num_layers: int = 10,
                                 layer_size: int = 100) -> Dict:
        """
        Test different checkpoint strategies and their memory usage.
        
        Args:
            num_layers: Number of layers in test network
            layer_size: Size of each layer
            
        Returns:
            Dictionary with strategy comparison results
        """
        self.log(f"\nTesting checkpoint strategies for {num_layers}-layer network...")
        
        # Create network configuration
        layer_configs = create_test_network(
            num_layers=num_layers,
            layer_size=layer_size,
            input_dim=layer_size
        )
        
        # Different checkpoint strategies
        strategies = {
            'no_checkpoints': [],
            'all_checkpoints': list(range(num_layers)),
            'uniform_2': [i for i in range(0, num_layers, num_layers//2)],
            'uniform_3': [i for i in range(0, num_layers, num_layers//3)],
            'optimal_sqrt': optimal_checkpoint_placement(num_layers),
            'first_half': list(range(num_layers//2)),
            'second_half': list(range(num_layers//2, num_layers))
        }
        
        results = {}
        x = np.random.randn(layer_size)
        
        for strategy_name, checkpoint_indices in strategies.items():
            if not checkpoint_indices:  # Handle no checkpoints case
                checkpoint_indices = [0]  # Need at least one checkpoint
            
            # Create checkpoint network
            checkpoint_net = CheckpointNeuralNetwork(
                layer_configs, checkpoint_indices, seed=42
            )
            
            # Measure memory usage
            _, memory_stats = checkpoint_net.forward(x, track_memory=True)
            usage_stats = checkpoint_net.get_memory_usage()
            
            results[strategy_name] = {
                'checkpoint_indices': checkpoint_indices,
                'num_checkpoints': len(checkpoint_indices),
                'checkpoint_params': usage_stats['checkpoint_params'],
                'memory_reduction': usage_stats['memory_reduction_ratio'],
                'measured_memory': memory_stats['peak'],
                'theoretical_ratio': usage_stats['total_params'] / usage_stats['checkpoint_params']
            }
            
            self.log(f"  {strategy_name}: {len(checkpoint_indices)} checkpoints, "
                    f"{usage_stats['memory_reduction_ratio']:.1f}x reduction")
        
        return results
    
    def create_scaling_plots(self, memory_data: Dict, scaling_data: Dict, save_path: str = None):
        """
        Create visualization plots for memory scaling analysis.
        
        Args:
            memory_data: Memory measurement data
            scaling_data: Scaling analysis results
            save_path: Path to save plots (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        params = np.array(memory_data['theoretical_params'])
        checkpoint_params = np.array(memory_data['checkpoint_params'])
        sqrt_params = np.sqrt(params)
        
        # Plot 1: Memory reduction over network size
        ax1.plot(memory_data['network_sizes'], memory_data['memory_reduction'], 'bo-')
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('Memory Reduction Factor')
        ax1.set_title('Memory Reduction vs Network Size')
        ax1.grid(True)
        
        # Plot 2: Checkpoint params vs √n (main scaling validation)
        ax2.scatter(sqrt_params, checkpoint_params, alpha=0.7, s=50)
        
        # Add fitted line
        slope = scaling_data['slope']
        intercept = scaling_data['intercept']
        fit_line = slope * sqrt_params + intercept
        ax2.plot(sqrt_params, fit_line, 'r--', 
                label=f'Fit: y = {slope:.1f}√n + {intercept:.0f}\nR² = {scaling_data["r_squared"]:.3f}')
        
        ax2.set_xlabel('√(Total Parameters)')
        ax2.set_ylabel('Checkpoint Parameters')
        ax2.set_title('Checkpoint Parameters vs √n (Scaling Validation)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Comparison of different scalings
        correlations = [
            scaling_data['sqrt_correlation'],
            scaling_data['linear_correlation'], 
            scaling_data['quadratic_correlation']
        ]
        scaling_types = ['√n', 'n', 'n²']
        
        bars = ax3.bar(scaling_types, correlations, color=['green', 'blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Correlation Coefficient')
        ax3.set_title('Correlation with Different Scaling Laws')
        ax3.set_ylim(0, 1)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # Plot 4: Actual vs Theoretical memory usage
        standard_memory = np.array(memory_data['standard_memory'])
        checkpoint_memory = np.array(memory_data['checkpoint_memory'])
        
        ax4.loglog(params, standard_memory, 'ro-', label='Standard', alpha=0.7)
        ax4.loglog(params, checkpoint_memory, 'go-', label='Checkpoint', alpha=0.7)
        ax4.set_xlabel('Total Parameters')
        ax4.set_ylabel('Peak Memory Usage (bytes)')
        ax4.set_title('Memory Usage Scaling (Log-Log)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log(f"Plots saved to: {save_path}")
        
        return fig
    
    def run_comprehensive_memory_tests(self) -> Dict:
        """Run all memory scaling tests and return comprehensive results."""
        self.log("=" * 60)
        self.log("COMPREHENSIVE MEMORY SCALING VALIDATION")
        self.log("=" * 60)
        
        # Test different network sizes
        network_sizes = [3, 5, 8, 12, 16, 24, 32]
        layer_size = 64  # Smaller to fit in memory
        
        # Measure memory usage
        memory_data = self.measure_memory_usage(network_sizes, layer_size)
        
        # Validate scaling
        scaling_data = self.validate_sqrt_scaling(memory_data)
        
        # Test checkpoint strategies
        strategy_data = self.test_checkpoint_strategies(num_layers=16, layer_size=64)
        
        # Create plots
        plot_path = '../results/memory_scaling_plots.png'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig = self.create_scaling_plots(memory_data, scaling_data, plot_path)
        plt.close(fig)
        
        # Summary
        summary = {
            'sqrt_scaling_confirmed': scaling_data['sqrt_scaling_confirmed'],
            'sqrt_correlation': scaling_data['sqrt_correlation'],
            'r_squared': scaling_data['r_squared'],
            'average_memory_reduction': np.mean(memory_data['memory_reduction']),
            'max_memory_reduction': np.max(memory_data['memory_reduction']),
            'memory_data': memory_data,
            'scaling_analysis': scaling_data,
            'checkpoint_strategies': strategy_data
        }
        
        self.log("\n" + "=" * 60)
        self.log("MEMORY SCALING TEST SUMMARY")
        self.log("=" * 60)
        self.log(f"√n scaling confirmed: {summary['sqrt_scaling_confirmed']}")
        self.log(f"√n correlation: {summary['sqrt_correlation']:.4f}")
        self.log(f"R²: {summary['r_squared']:.4f}")
        self.log(f"Average memory reduction: {summary['average_memory_reduction']:.1f}x")
        self.log(f"Maximum memory reduction: {summary['max_memory_reduction']:.1f}x")
        
        return summary


def main():
    """Run memory scaling validation tests."""
    validator = MemoryScalingValidator(verbose=True)
    results = validator.run_comprehensive_memory_tests()
    
    # Save results
    import json
    results_file = '../results/memory_scaling_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
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
    return results['sqrt_scaling_confirmed']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)