#!/usr/bin/env python3
import numpy as np
from ml_dtypes import bfloat16
import struct

EMBEDDING_DIM = 2048
HIDDEN_DIM = 8192
NUM_TEST_CASES = 2

def silu(x):
    return x / (1.0 + np.exp(-x))

def swiglu_reference(input_data, weights_1, weights_2, weights_3):
    left = np.dot(input_data, weights_1.T)
    right = np.dot(input_data, weights_2.T)
    
    left_swished = silu(left)
    
    intermediate = left_swished * right
    
    output = np.dot(intermediate, weights_3.T)
    
    # Return all intermediate results
    return {
        'left': left.astype(bfloat16),
        'right': right.astype(bfloat16),
        'left_swished': left_swished.astype(bfloat16),
        'intermediate': intermediate.astype(bfloat16),
        'output': output.astype(bfloat16)
    }

def generate_random_inputs():
    """Generate random inputs for testing"""
    np.random.seed(42)  # Fixed seed for reproducibility
    
    inputs = []
    weights = []
    
    for i in range(NUM_TEST_CASES):
        # Generate random input and weights with reasonable ranges
        input_data = (np.random.rand(EMBEDDING_DIM).astype(np.float32)).astype(bfloat16)
        w1 = (np.random.randn(HIDDEN_DIM, EMBEDDING_DIM).astype(np.float32) * 0.2).astype(bfloat16)
        w2 = (np.random.randn(HIDDEN_DIM, EMBEDDING_DIM).astype(np.float32) * 0.2).astype(bfloat16)
        w3 = (np.random.randn(EMBEDDING_DIM, HIDDEN_DIM).astype(np.float32) * 0.2).astype(bfloat16)
        
        inputs.append(input_data)
        weights.append((w1, w2, w3))
    
    return inputs, weights

def save_reference():
    inputs, weights = generate_random_inputs()
    
    for test_idx in range(NUM_TEST_CASES):
        input_data = inputs[test_idx]
        w1, w2, w3 = weights[test_idx]
        
        # Save inputs
        with open(f'input_{test_idx}.bin', 'wb') as f:
            input_data.tofile(f)
        
        with open(f'weights_1_{test_idx}.bin', 'wb') as f:
            w1.tofile(f)
        
        with open(f'weights_2_{test_idx}.bin', 'wb') as f:
            w2.tofile(f)
        
        with open(f'weights_3_{test_idx}.bin', 'wb') as f:
            w3.tofile(f)
        
        # Compute and save all reference outputs (including intermediates)
        results = swiglu_reference(input_data, w1, w2, w3)
        
        with open(f'reference_left_{test_idx}.bin', 'wb') as f:
            results['left'].tofile(f)
        
        with open(f'reference_right_{test_idx}.bin', 'wb') as f:
            results['right'].tofile(f)
        
        with open(f'reference_left_swished_{test_idx}.bin', 'wb') as f:
            results['left_swished'].tofile(f)
        
        with open(f'reference_intermediate_{test_idx}.bin', 'wb') as f:
            results['intermediate'].tofile(f)
        
        with open(f'reference_output_{test_idx}.bin', 'wb') as f:
            results['output'].tofile(f)
        
        print(f"\nTest case {test_idx}:")
        print(f"  Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        print(f"  Weights_1 shape: {w1.shape}, dtype: {w1.dtype}")
        print(f"  Weights_2 shape: {w2.shape}, dtype: {w2.dtype}")
        print(f"  Weights_3 shape: {w3.shape}, dtype: {w3.dtype}")
        print(f"  Left shape: {results['left'].shape}, dtype: {results['left'].dtype}")
        print(f"  Right shape: {results['right'].shape}, dtype: {results['right'].dtype}")
        print(f"  Left_swished shape: {results['left_swished'].shape}, dtype: {results['left_swished'].dtype}")
        print(f"  Intermediate shape: {results['intermediate'].shape}, dtype: {results['intermediate'].dtype}")
        print(f"  Output shape: {results['output'].shape}, dtype: {results['output'].dtype}")
        print(f"  Output first 5 values: {results['output'][:5]}")
        print(f"  Output last 5 values: {results['output'][-5:]}")

if __name__ == "__main__":
    save_reference()
