#!/usr/bin/env python3
import numpy as np
from ml_dtypes import bfloat16
import struct

EMBEDDING_DIM = 2048
HIDDEN_DIM = 8192

def silu(x):
    return x / (1.0 + np.exp(-x))

def swiglu_reference():
    input_data = np.ones(EMBEDDING_DIM, dtype=bfloat16)
    weights_1 = np.full((EMBEDDING_DIM, HIDDEN_DIM), 0.5, dtype=bfloat16)
    weights_2 = np.full((EMBEDDING_DIM, HIDDEN_DIM), 0.5, dtype=bfloat16)
    weights_3 = np.full((HIDDEN_DIM, EMBEDDING_DIM), 0.5, dtype=bfloat16)
    
    left = np.dot(input_data, weights_1)
    right = np.dot(input_data, weights_2)
    
    left_swished = silu(left)
    
    intermediate = left_swished * right
    
    output = np.dot(intermediate, weights_3)
    
    return output.astype(bfloat16)

def save_reference():
    output = swiglu_reference()
    
    with open('reference_output.bin', 'wb') as f:
        output.tofile(f)
    
    print(f"Reference output saved: {len(output)} elements")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"First 10 values: {output[:10]}")
    print(f"Last 10 values: {output[-10:]}")

if __name__ == "__main__":
    save_reference()
