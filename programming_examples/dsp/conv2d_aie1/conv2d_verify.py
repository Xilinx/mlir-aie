#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import numpy as np
from scipy.signal import correlate

# Reduced parameters for demonstration
M = 13  # Input width
N = 13  # Input height
Cin = 8  # Input channels
Cout = 8  # Output channels
F = 3  # Kernel size
S = 1  # Stride
P = 0  # Padding

# Derived parameters
outHeight = (N - F + 2 * P) // S + 1
outWidthR = (M - F + 2 * P) // S + 1
CinUp = ((Cin + 7) // 8) * 8  # Align to multiple of 8
CoutUp = ((Cout + 7) // 8) * 8  # Align to multiple of 8

# Set random seed for reproducibility
np.random.seed(42)

# Generate random input data with shape (M, N, CinUp)
inputs = np.random.randint(0, 256, size=(M, N, CinUp), dtype=np.uint8)

# Generate random weights with shape (F, F, CinUp, CoutUp)
weights = np.random.randint(-128, 128, size=(F, F, CinUp, CoutUp), dtype=np.int8)

# Function to perform 2D convolution using given weights and inputs
def conv2d(inputs, weights, stride=1, padding=0):
    M, N, Cin = inputs.shape
    F, _, Cin, Cout = weights.shape

    # Calculate output dimensions
    outHeight = (N - F + 2 * padding) // stride + 1
    outWidth = (M - F + 2 * padding) // stride + 1
    
    # Initialize output tensor
    outputs = np.zeros((outHeight, outWidth, Cout), dtype=np.int32)
    
    # Apply the convolution
    for cout in range(Cout):
        for cin in range(Cin):
            # Extract the cin-th input channel and the corresponding filter
            input_channel = inputs[:, :, cin]
            weight_filter = weights[:, :, cin, cout]
            # Convolve the input with the filter and add to the output
            outputs[:, :, cout] += correlate(input_channel, weight_filter, mode='valid')

    return outputs

# Perform the convolution
outputs = conv2d(inputs, weights, stride=S, padding=P)

# Save inputs, weights, and outputs to text files

input_filename = 'data/python/inputs.txt'
weights_filename = 'data/python/weights.txt'
outputs_filename = 'data/python/outputs.txt'

np.set_printoptions(threshold=np.inf)
# Save inputs
np.savetxt(input_filename, inputs.flatten(), fmt='%d')

# Save weights
# reshaped_weights = weights.transpose(3, 2, 0, 1).reshape(Cout, Cin * F * F)
reshaped_weights = weights
print(reshaped_weights)
np.savetxt(weights_filename, reshaped_weights.flatten(), fmt='%d')

# Save outputs
np.savetxt(outputs_filename, outputs.flatten(), fmt='%d')

input_filename, weights_filename, outputs_filename
