"""
Neural Network implementations for validation testing.
Includes both standard and checkpoint-based inference methods.
"""

import numpy as np
import tracemalloc
import psutil
import os
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    input_dim: int
    output_dim: int
    activation: str = 'relu'


class MemoryTracker:
    """Utility class for tracking memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.current_memory = 0
    
    def start_tracking(self):
        """Start memory tracking."""
        tracemalloc.start()
        self.peak_memory = 0
    
    def get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        current, peak = tracemalloc.get_traced_memory()
        self.current_memory = current
        self.peak_memory = max(self.peak_memory, peak)
        return current
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        _, peak = tracemalloc.get_traced_memory()
        self.peak_memory = max(self.peak_memory, peak)
        return self.peak_memory
    
    def stop_tracking(self) -> Dict[str, int]:
        """Stop tracking and return memory statistics."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'current': current,
            'peak': max(self.peak_memory, peak),
            'rss': self.process.memory_info().rss
        }


class StandardNeuralNetwork:
    """Standard neural network with full activation storage."""
    
    def __init__(self, layer_configs: List[LayerConfig], seed: int = 42):
        """
        Initialize network with given layer configurations.
        
        Args:
            layer_configs: List of layer configurations
            seed: Random seed for reproducible weights
        """
        # Validate inputs
        if not layer_configs:
            raise ValueError("layer_configs cannot be empty")
        
        # Validate layer configurations
        for i, config in enumerate(layer_configs):
            if config.input_dim <= 0 or config.output_dim <= 0:
                raise ValueError(f"Layer {i} has invalid dimensions: {config.input_dim} -> {config.output_dim}")
        
        np.random.seed(seed)
        self.layer_configs = layer_configs
        self.weights = []
        self.biases = []
        self.total_params = 0
        
        # Initialize weights and biases
        for config in layer_configs:
            # Xavier initialization
            fan_in, fan_out = config.input_dim, config.output_dim
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            W = np.random.uniform(-limit, limit, (fan_out, fan_in))
            b = np.zeros(fan_out)
            
            self.weights.append(W)
            self.biases.append(b)
            self.total_params += W.size + b.size
    
    def _activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: np.ndarray, track_memory: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Standard forward pass storing all activations.
        
        Args:
            x: Input tensor
            track_memory: Whether to track memory usage
            
        Returns:
            Tuple of (output, memory_stats)
        """
        memory_tracker = MemoryTracker() if track_memory else None
        if track_memory:
            memory_tracker.start_tracking()
        
        # Store all activations (this is what we want to reduce)
        activations = [x]
        
        current = x
        for i, (W, b, config) in enumerate(zip(self.weights, self.biases, self.layer_configs)):
            # Linear transformation
            current = W @ current + b
            
            # Apply activation
            current = self._activation(current, config.activation)
            
            # Store activation for potential backprop (standard approach)
            activations.append(current.copy())
            
            if track_memory:
                memory_tracker.get_current_memory()
        
        memory_stats = memory_tracker.stop_tracking() if track_memory else None
        return current, memory_stats


class CheckpointNeuralNetwork:
    """Neural network with checkpoint-based inference to save memory."""
    
    def __init__(self, layer_configs: List[LayerConfig], checkpoint_indices: List[int], seed: int = 42):
        """
        Initialize checkpoint-based network.
        
        Args:
            layer_configs: List of layer configurations
            checkpoint_indices: Indices of layers to checkpoint (0-indexed)
            seed: Random seed for reproducible weights (must match StandardNeuralNetwork)
        """
        # Validate inputs
        if not layer_configs:
            raise ValueError("layer_configs cannot be empty")
        
        if not checkpoint_indices:
            raise ValueError("checkpoint_indices cannot be empty")
        
        # Validate checkpoint indices
        for idx in checkpoint_indices:
            if idx < 0:
                raise ValueError(f"Checkpoint index {idx} cannot be negative")
            if idx >= len(layer_configs):
                raise ValueError(f"Checkpoint index {idx} is out of range for {len(layer_configs)} layers")
        
        # Use same initialization as standard network for fair comparison
        np.random.seed(seed)
        self.layer_configs = layer_configs
        self.checkpoint_indices = set(checkpoint_indices)
        self.weights = []
        self.biases = []
        self.total_params = 0
        
        # Initialize with identical weights to standard network
        for config in layer_configs:
            fan_in, fan_out = config.input_dim, config.output_dim
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            W = np.random.uniform(-limit, limit, (fan_out, fan_in))
            b = np.zeros(fan_out)
            
            self.weights.append(W)
            self.biases.append(b)
            self.total_params += W.size + b.size
        
        # Calculate memory usage for checkpointed layers only
        self.checkpoint_params = 0
        for i in self.checkpoint_indices:
            if i < len(self.weights):
                self.checkpoint_params += self.weights[i].size + self.biases[i].size
    
    def _activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function (identical to standard network)."""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _forward_segment(self, x: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Forward pass through a segment of layers.
        
        Args:
            x: Input to the segment
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (exclusive)
            
        Returns:
            Output of the segment
        """
        current = x
        for i in range(start_idx, end_idx):
            W, b, config = self.weights[i], self.biases[i], self.layer_configs[i]
            current = W @ current + b
            current = self._activation(current, config.activation)
        return current
    
    def forward(self, x: np.ndarray, track_memory: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Checkpoint-based forward pass with reduced memory usage.
        
        This implementation demonstrates the theoretical approach but for validation purposes,
        we compute the full forward pass while only storing checkpointed activations.
        In practice, this would recompute intermediate activations from checkpoints as needed.
        
        Args:
            x: Input tensor
            track_memory: Whether to track memory usage
            
        Returns:
            Tuple of (output, memory_stats)
        """
        memory_tracker = MemoryTracker() if track_memory else None
        if track_memory:
            memory_tracker.start_tracking()
        
        # Validate checkpoint indices
        for idx in self.checkpoint_indices:
            if idx >= len(self.layer_configs):
                raise ValueError(f"Checkpoint index {idx} is out of range for {len(self.layer_configs)} layers")
        
        # Store only checkpointed activations to simulate memory reduction
        checkpoints = {}
        
        # For determinism validation, we'll compute the full forward pass
        # but only "store" the checkpointed activations in memory
        current = x
        
        for layer_idx in range(len(self.layer_configs)):
            # Apply current layer transformation
            W, b, config = self.weights[layer_idx], self.biases[layer_idx], self.layer_configs[layer_idx]
            
            # Validate dimensions before matrix multiplication
            if W.shape[1] != current.shape[0]:
                raise ValueError(
                    f"Dimension mismatch at layer {layer_idx}: "
                    f"Weight matrix expects input of size {W.shape[1]} "
                    f"but got activation of size {current.shape[0]}"
                )
            
            current = W @ current + b
            current = self._activation(current, config.activation)
            
            # Store checkpoint if this layer is marked for checkpointing
            if layer_idx in self.checkpoint_indices:
                checkpoints[layer_idx] = current.copy()
                
                if track_memory:
                    memory_tracker.get_current_memory()
        
        # For memory calculation purposes, we only count checkpointed parameters
        # The actual memory savings come from not storing all intermediate activations
        
        memory_stats = memory_tracker.stop_tracking() if track_memory else None
        return current, memory_stats
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get theoretical memory usage statistics."""
        return {
            'total_params': self.total_params,
            'checkpoint_params': self.checkpoint_params,
            'memory_reduction_ratio': self.total_params / max(self.checkpoint_params, 1)
        }


def create_test_network(num_layers: int, layer_size: int, input_dim: int = 100) -> List[LayerConfig]:
    """Create a test network configuration."""
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    
    configs = []
    
    if num_layers == 1:
        # Single layer network
        configs.append(LayerConfig(input_dim, 10, 'linear'))
    else:
        # First layer
        configs.append(LayerConfig(input_dim, layer_size, 'relu'))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            configs.append(LayerConfig(layer_size, layer_size, 'relu'))
        
        # Output layer
        configs.append(LayerConfig(layer_size, 10, 'linear'))
    
    return configs


def optimal_checkpoint_placement(num_layers: int) -> List[int]:
    """
    Calculate optimal checkpoint placement based on sqrt(n) theory.
    
    Checkpoints are placed at intermediate layers only (not the final output layer).
    The final layer produces the output and doesn't need checkpointing for recomputation.
    
    Args:
        num_layers: Total number of layers
        
    Returns:
        List of checkpoint indices (0-indexed, excluding final layer)
    """
    if num_layers <= 1:
        return [0] if num_layers == 1 else []
    
    if num_layers == 2:
        return [0]  # Only checkpoint the first layer
    
    # Place checkpoints every sqrt(L) layers approximately
    checkpoint_interval = max(1, int(np.sqrt(num_layers)))
    checkpoints = []
    
    # Checkpoint intermediate layers only (exclude final layer)
    for i in range(0, num_layers - 1, checkpoint_interval):
        checkpoints.append(i)
    
    # Ensure we have at least the input layer checkpointed
    if not checkpoints or checkpoints[0] != 0:
        checkpoints.insert(0, 0)
    
    # Remove duplicates and sort
    checkpoints = sorted(list(set(checkpoints)))
    
    # Validate that all checkpoints are valid intermediate layers
    checkpoints = [i for i in checkpoints if 0 <= i < num_layers - 1]
    
    # Ensure we always have at least one checkpoint (input layer)
    if not checkpoints:
        checkpoints = [0]
    
    return checkpoints