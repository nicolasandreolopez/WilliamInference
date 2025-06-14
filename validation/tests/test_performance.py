"""
Performance benchmarking tests for checkpoint-based neural network inference.

This module measures the computational overhead of checkpoint-based inference
and analyzes the space-time tradeoffs.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import time
import matplotlib.pyplot as plt
from statistics import mean, stdev

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import (
    StandardNeuralNetwork,
    CheckpointNeuralNetwork,
    create_test_network,
    optimal_checkpoint_placement
)


class PerformanceBenchmark:
    """Benchmarks performance characteristics of checkpoint-based inference."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.benchmark_results = []
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[PERF] {message}")
    
    def measure_inference_time(self,
                             network,
                             input_data: np.ndarray,
                             num_runs: int = 100,
                             warmup_runs: int = 10) -> Dict:
        """
        Measure inference time with statistical analysis.
        
        Args:
            network: Neural network to benchmark
            input_data: Input tensor for inference
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs (excluded from timing)
            
        Returns:
            Dictionary with timing statistics
        """
        # Warmup runs
        for _ in range(warmup_runs):
            network.forward(input_data)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            network.forward(input_data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times),
            'median_time': sorted(times)[len(times)//2],
            'num_runs': num_runs,
            'all_times': times
        }
    
    def benchmark_overhead(self,
                         layer_configs: List,
                         checkpoint_indices: List[int],
                         input_shape: Tuple[int, ...] = (100,),
                         num_runs: int = 50) -> Dict:
        """
        Benchmark computational overhead of checkpoint vs standard inference.
        
        Args:
            layer_configs: Network layer configurations
            checkpoint_indices: Checkpoint placement strategy
            input_shape: Shape of input tensors
            num_runs: Number of timing runs
            
        Returns:
            Dictionary with benchmark results
        """
        self.log(f"Benchmarking overhead for {len(checkpoint_indices)} checkpoints...")
        
        # Create networks
        standard_net = StandardNeuralNetwork(layer_configs, seed=42)
        checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=42)
        
        # Create test input
        x = np.random.randn(*input_shape)
        
        # Benchmark standard inference
        standard_times = self.measure_inference_time(standard_net, x, num_runs)
        
        # Benchmark checkpoint inference
        checkpoint_times = self.measure_inference_time(checkpoint_net, x, num_runs)
        
        # Calculate overhead
        overhead_ratio = checkpoint_times['mean_time'] / standard_times['mean_time']
        overhead_absolute = checkpoint_times['mean_time'] - standard_times['mean_time']
        
        result = {
            'standard_timing': standard_times,
            'checkpoint_timing': checkpoint_times,
            'overhead_ratio': overhead_ratio,
            'overhead_absolute_ms': overhead_absolute * 1000,
            'checkpoint_indices': checkpoint_indices,
            'num_checkpoints': len(checkpoint_indices),
            'network_layers': len(layer_configs),
            'total_params': standard_net.total_params
        }
        
        self.log(f"  Standard: {standard_times['mean_time']*1000:.2f}±{standard_times['std_time']*1000:.2f}ms")
        self.log(f"  Checkpoint: {checkpoint_times['mean_time']*1000:.2f}±{checkpoint_times['std_time']*1000:.2f}ms")
        self.log(f"  Overhead: {overhead_ratio:.2f}x ({overhead_absolute*1000:.2f}ms)")
        
        return result
    
    def analyze_checkpoint_density_tradeoff(self,
                                          num_layers: int = 16,
                                          layer_size: int = 100) -> Dict:
        """
        Analyze space-time tradeoff for different checkpoint densities.
        
        Args:
            num_layers: Number of layers in test network
            layer_size: Size of each layer
            
        Returns:
            Dictionary with tradeoff analysis results
        """
        self.log(f"\nAnalyzing checkpoint density tradeoffs for {num_layers}-layer network...")
        
        # Create network configuration
        layer_configs = create_test_network(
            num_layers=num_layers,
            layer_size=layer_size,
            input_dim=layer_size
        )
        
        # Test different checkpoint densities
        checkpoint_strategies = []
        
        # No checkpoints (baseline, just store input)
        checkpoint_strategies.append(('None', [0]))
        
        # Sparse checkpoints
        for interval in [8, 4, 2]:
            if interval < num_layers:
                indices = list(range(0, num_layers, interval))
                if indices[-1] != num_layers - 1:
                    indices.append(num_layers - 1)
                checkpoint_strategies.append((f'Every {interval}', indices))
        
        # Optimal sqrt strategy
        optimal_indices = optimal_checkpoint_placement(num_layers)
        checkpoint_strategies.append(('Optimal √n', optimal_indices))
        
        # Dense checkpoints
        checkpoint_strategies.append(('Dense', list(range(0, num_layers, 1))))
        
        results = []
        x = np.random.randn(layer_size)
        
        for strategy_name, checkpoint_indices in checkpoint_strategies:
            # Benchmark performance
            perf_result = self.benchmark_overhead(
                layer_configs, checkpoint_indices, (layer_size,), num_runs=30
            )
            
            # Calculate memory usage
            checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=42)
            memory_usage = checkpoint_net.get_memory_usage()
            
            strategy_result = {
                'strategy_name': strategy_name,
                'checkpoint_indices': checkpoint_indices,
                'num_checkpoints': len(checkpoint_indices),
                'checkpoint_density': len(checkpoint_indices) / num_layers,
                'mean_time_ms': perf_result['checkpoint_timing']['mean_time'] * 1000,
                'overhead_ratio': perf_result['overhead_ratio'],
                'memory_reduction': memory_usage['memory_reduction_ratio'],
                'checkpoint_params': memory_usage['checkpoint_params'],
                'total_params': memory_usage['total_params']
            }
            
            results.append(strategy_result)
            
            self.log(f"  {strategy_name}: {len(checkpoint_indices)} checkpoints, "
                    f"{strategy_result['overhead_ratio']:.2f}x time, "
                    f"{strategy_result['memory_reduction']:.1f}x memory reduction")
        
        return {
            'network_config': {
                'num_layers': num_layers,
                'layer_size': layer_size,
                'total_params': results[0]['total_params']
            },
            'strategy_results': results
        }
    
    def benchmark_scaling_performance(self,
                                    network_sizes: List[int],
                                    layer_size: int = 64) -> Dict:
        """
        Benchmark performance scaling across different network sizes.
        
        Args:
            network_sizes: List of number of layers to test
            layer_size: Size of each layer
            
        Returns:
            Dictionary with scaling performance results
        """
        self.log(f"\nBenchmarking performance scaling for sizes: {network_sizes}")
        
        results = []
        
        for num_layers in network_sizes:
            self.log(f"\nTesting {num_layers} layers...")
            
            # Create network configuration
            layer_configs = create_test_network(
                num_layers=num_layers,
                layer_size=layer_size,
                input_dim=layer_size
            )
            
            # Optimal checkpoint strategy
            checkpoint_indices = optimal_checkpoint_placement(num_layers)
            
            # Benchmark
            perf_result = self.benchmark_overhead(
                layer_configs, checkpoint_indices, (layer_size,), num_runs=20
            )
            
            # Memory usage
            checkpoint_net = CheckpointNeuralNetwork(layer_configs, checkpoint_indices, seed=42)
            memory_usage = checkpoint_net.get_memory_usage()
            
            scaling_result = {
                'num_layers': num_layers,
                'total_params': perf_result['total_params'],
                'standard_time_ms': perf_result['standard_timing']['mean_time'] * 1000,
                'checkpoint_time_ms': perf_result['checkpoint_timing']['mean_time'] * 1000,
                'overhead_ratio': perf_result['overhead_ratio'],
                'memory_reduction': memory_usage['memory_reduction_ratio'],
                'num_checkpoints': len(checkpoint_indices),
                'checkpoint_density': len(checkpoint_indices) / num_layers
            }
            
            results.append(scaling_result)
        
        return {
            'layer_size': layer_size,
            'scaling_results': results
        }
    
    def create_performance_plots(self, 
                               tradeoff_data: Dict,
                               scaling_data: Dict,
                               save_path: str = None):
        """
        Create visualization plots for performance analysis.
        
        Args:
            tradeoff_data: Checkpoint density tradeoff data
            scaling_data: Performance scaling data
            save_path: Path to save plots (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Space-Time Tradeoff
        strategies = tradeoff_data['strategy_results']
        memory_reductions = [s['memory_reduction'] for s in strategies]
        overhead_ratios = [s['overhead_ratio'] for s in strategies]
        strategy_names = [s['strategy_name'] for s in strategies]
        
        scatter = ax1.scatter(memory_reductions, overhead_ratios, 
                            s=100, alpha=0.7, c=range(len(strategies)), cmap='viridis')
        
        for i, name in enumerate(strategy_names):
            ax1.annotate(name, (memory_reductions[i], overhead_ratios[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Memory Reduction Factor')
        ax1.set_ylabel('Time Overhead Ratio')
        ax1.set_title('Space-Time Tradeoff')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Checkpoint Density vs Performance
        densities = [s['checkpoint_density'] for s in strategies]
        times = [s['mean_time_ms'] for s in strategies]
        
        ax2.plot(densities, times, 'bo-', alpha=0.7)
        ax2.set_xlabel('Checkpoint Density (checkpoints/layers)')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Performance vs Checkpoint Density')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance Scaling
        scaling_results = scaling_data['scaling_results']
        layer_counts = [r['num_layers'] for r in scaling_results]
        standard_times = [r['standard_time_ms'] for r in scaling_results]
        checkpoint_times = [r['checkpoint_time_ms'] for r in scaling_results]
        
        ax3.plot(layer_counts, standard_times, 'ro-', label='Standard', alpha=0.7)
        ax3.plot(layer_counts, checkpoint_times, 'go-', label='Checkpoint', alpha=0.7)
        ax3.set_xlabel('Number of Layers')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.set_title('Performance Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overhead vs Memory Reduction Scaling
        param_counts = [r['total_params'] for r in scaling_results]
        overhead_ratios_scaling = [r['overhead_ratio'] for r in scaling_results]
        memory_reductions_scaling = [r['memory_reduction'] for r in scaling_results]
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(param_counts, overhead_ratios_scaling, 'r.-', label='Time Overhead', alpha=0.7)
        line2 = ax4_twin.plot(param_counts, memory_reductions_scaling, 'b.-', label='Memory Reduction', alpha=0.7)
        
        ax4.set_xlabel('Total Parameters')
        ax4.set_ylabel('Time Overhead Ratio', color='red')
        ax4_twin.set_ylabel('Memory Reduction Factor', color='blue')
        ax4.set_title('Overhead and Memory Reduction vs Network Size')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log(f"Performance plots saved to: {save_path}")
        
        return fig
    
    def run_comprehensive_performance_tests(self) -> Dict:
        """Run all performance tests and return comprehensive results."""
        self.log("=" * 60)
        self.log("COMPREHENSIVE PERFORMANCE BENCHMARKING")
        self.log("=" * 60)
        
        # Test checkpoint density tradeoffs
        tradeoff_data = self.analyze_checkpoint_density_tradeoff(
            num_layers=16, layer_size=64
        )
        
        # Test performance scaling
        scaling_data = self.benchmark_scaling_performance(
            network_sizes=[4, 8, 12, 16, 20], layer_size=64
        )
        
        # Create plots
        plot_path = '../results/performance_plots.png'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig = self.create_performance_plots(tradeoff_data, scaling_data, plot_path)
        plt.close(fig)
        
        # Calculate summary statistics
        optimal_strategy = None
        min_overhead = float('inf')
        
        for strategy in tradeoff_data['strategy_results']:
            if strategy['memory_reduction'] > 2.0 and strategy['overhead_ratio'] < min_overhead:
                min_overhead = strategy['overhead_ratio']
                optimal_strategy = strategy
        
        avg_overhead = mean([r['overhead_ratio'] for r in scaling_data['scaling_results']])
        avg_memory_reduction = mean([r['memory_reduction'] for r in scaling_data['scaling_results']])
        
        summary = {
            'optimal_strategy': optimal_strategy,
            'average_overhead_ratio': avg_overhead,
            'average_memory_reduction': avg_memory_reduction,
            'performance_acceptable': avg_overhead < 3.0,  # Less than 3x overhead
            'tradeoff_analysis': tradeoff_data,
            'scaling_analysis': scaling_data
        }
        
        self.log("\n" + "=" * 60)
        self.log("PERFORMANCE BENCHMARK SUMMARY")
        self.log("=" * 60)
        self.log(f"Average time overhead: {avg_overhead:.2f}x")
        self.log(f"Average memory reduction: {avg_memory_reduction:.1f}x")
        self.log(f"Performance acceptable: {summary['performance_acceptable']}")
        
        if optimal_strategy:
            self.log(f"Optimal strategy: {optimal_strategy['strategy_name']}")
            self.log(f"  Overhead: {optimal_strategy['overhead_ratio']:.2f}x")
            self.log(f"  Memory reduction: {optimal_strategy['memory_reduction']:.1f}x")
        
        return summary


def main():
    """Run performance benchmarking tests."""
    benchmark = PerformanceBenchmark(verbose=True)
    results = benchmark.run_comprehensive_performance_tests()
    
    # Save results
    import json
    results_file = '../results/performance_results.json'
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
    return results['performance_acceptable']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)