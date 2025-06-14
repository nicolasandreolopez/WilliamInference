"""
Comprehensive validation suite for space-efficient neural network inference.

Runs all validation tests and generates a summary report.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from tests.test_determinism import DeterminismValidator
from tests.test_memory_scaling import MemoryScalingValidator
from tests.test_performance import PerformanceBenchmark


class ValidationSuite:
    """Main validation suite runner."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SUITE] {message}")
    
    def run_all_validations(self) -> Dict:
        """
        Run all validation tests and return comprehensive results.
        
        Returns:
            Dictionary with all validation results
        """
        self.start_time = time.time()
        
        self.log("=" * 80)
        self.log("SPACE-EFFICIENT NEURAL NETWORK INFERENCE VALIDATION SUITE")
        self.log("=" * 80)
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        all_tests_passed = True
        
        # Test 1: Determinism Validation
        self.log("🔍 Running Determinism Validation Tests...")
        try:
            determinism_validator = DeterminismValidator(verbose=self.verbose)
            determinism_results = determinism_validator.run_comprehensive_determinism_tests()
            self.results['determinism'] = determinism_results
            
            if not determinism_results['all_passed']:
                all_tests_passed = False
                self.log("❌ Determinism tests FAILED")
            else:
                self.log("✅ Determinism tests PASSED")
                
        except Exception as e:
            self.log(f"❌ Determinism tests ERROR: {str(e)}")
            self.results['determinism'] = {'error': str(e), 'all_passed': False}
            all_tests_passed = False
        
        self.log("")
        
        # Test 2: Memory Scaling Validation
        self.log("📊 Running Memory Scaling Validation Tests...")
        try:
            memory_validator = MemoryScalingValidator(verbose=self.verbose)
            memory_results = memory_validator.run_comprehensive_memory_tests()
            self.results['memory_scaling'] = memory_results
            
            if not memory_results['sqrt_scaling_confirmed']:
                all_tests_passed = False
                self.log("❌ Memory scaling tests FAILED")
            else:
                self.log("✅ Memory scaling tests PASSED")
                
        except Exception as e:
            self.log(f"❌ Memory scaling tests ERROR: {str(e)}")
            self.results['memory_scaling'] = {'error': str(e), 'sqrt_scaling_confirmed': False}
            all_tests_passed = False
        
        self.log("")
        
        # Test 3: Performance Benchmarking
        self.log("⚡ Running Performance Benchmarking Tests...")
        try:
            performance_benchmark = PerformanceBenchmark(verbose=self.verbose)
            performance_results = performance_benchmark.run_comprehensive_performance_tests()
            self.results['performance'] = performance_results
            
            if not performance_results['performance_acceptable']:
                self.log("⚠️  Performance tests show high overhead (but still valid)")
            else:
                self.log("✅ Performance tests PASSED")
                
        except Exception as e:
            self.log(f"❌ Performance tests ERROR: {str(e)}")
            self.results['performance'] = {'error': str(e), 'performance_acceptable': False}
            # Performance is not a failure condition for the core theory
        
        self.end_time = time.time()
        
        # Generate summary
        summary = self.generate_summary_report(all_tests_passed)
        self.results['summary'] = summary
        
        return self.results
    
    def generate_summary_report(self, all_tests_passed: bool) -> Dict:
        """
        Generate a comprehensive summary report.
        
        Args:
            all_tests_passed: Whether all critical tests passed
            
        Returns:
            Dictionary with summary information
        """
        self.log("")
        self.log("=" * 80)
        self.log("VALIDATION SUMMARY REPORT")
        self.log("=" * 80)
        
        # Calculate runtime
        runtime_seconds = self.end_time - self.start_time if self.end_time and self.start_time else 0
        runtime_minutes = runtime_seconds / 60
        
        # Extract key metrics
        determinism_passed = self.results.get('determinism', {}).get('all_passed', False)
        memory_scaling_confirmed = self.results.get('memory_scaling', {}).get('sqrt_scaling_confirmed', False)
        performance_acceptable = self.results.get('performance', {}).get('performance_acceptable', False)
        
        # Key findings
        key_findings = []
        
        if determinism_passed:
            key_findings.append("✅ Checkpoint-based inference produces identical outputs to standard inference")
            max_diff = self.results.get('determinism', {}).get('detailed_results', [{}])[0].get('max_difference', 0)
            key_findings.append(f"   Maximum output difference: {max_diff:.2e} (machine precision)")
        
        if memory_scaling_confirmed:
            sqrt_corr = self.results.get('memory_scaling', {}).get('sqrt_correlation', 0)
            r_squared = self.results.get('memory_scaling', {}).get('r_squared', 0)
            avg_reduction = self.results.get('memory_scaling', {}).get('average_memory_reduction', 0)
            key_findings.append(f"✅ Memory usage scales as O(√n) with {sqrt_corr:.3f} correlation (R² = {r_squared:.3f})")
            key_findings.append(f"   Average memory reduction: {avg_reduction:.1f}x")
        
        if 'performance' in self.results and 'error' not in self.results['performance']:
            avg_overhead = self.results['performance'].get('average_overhead_ratio', 0)
            key_findings.append(f"⚡ Average computational overhead: {avg_overhead:.2f}x")
            
            optimal_strategy = self.results['performance'].get('optimal_strategy')
            if optimal_strategy:
                key_findings.append(f"   Optimal strategy: {optimal_strategy['strategy_name']} "
                                  f"({optimal_strategy['overhead_ratio']:.2f}x time, "
                                  f"{optimal_strategy['memory_reduction']:.1f}x memory)")
        
        # Theoretical validation
        theory_validated = determinism_passed and memory_scaling_confirmed
        
        self.log(f"Runtime: {runtime_minutes:.1f} minutes ({runtime_seconds:.1f} seconds)")
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        self.log("KEY FINDINGS:")
        for finding in key_findings:
            self.log(f"  {finding}")
        self.log("")
        self.log(f"CORE THEORY VALIDATION: {'✅ CONFIRMED' if theory_validated else '❌ FAILED'}")
        self.log(f"OVERALL RESULT: {'✅ SUCCESS' if all_tests_passed else '❌ FAILURE'}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': runtime_seconds,
            'runtime_minutes': runtime_minutes,
            'all_tests_passed': all_tests_passed,
            'theory_validated': theory_validated,
            'determinism_passed': determinism_passed,
            'memory_scaling_confirmed': memory_scaling_confirmed,
            'performance_acceptable': performance_acceptable,
            'key_findings': key_findings
        }
        
        return summary
    
    def save_results(self, filename: str = None):
        """
        Save all results to JSON file.
        
        Args:
            filename: Optional filename, defaults to timestamped name
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'validation_results_{timestamp}.json'
        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        json_results = deep_convert(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.log(f"📁 Complete results saved to: {filepath}")
        return filepath


def main():
    """Run the complete validation suite."""
    print("Installing required packages...")
    
    # Install required packages
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not install requirements: {e}")
        print("Some tests may fail due to missing dependencies.")
    
    # Run validation suite
    suite = ValidationSuite(verbose=True)
    results = suite.run_all_validations()
    
    # Save results
    results_file = suite.save_results()
    
    # Return appropriate exit code
    success = results['summary']['all_tests_passed']
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)