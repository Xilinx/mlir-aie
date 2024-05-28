<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>ResNet with Offloaded Conv2_x Layers and Post-Training Quantization</ins>

Quantization involves reducing the precision of the weights and activations of a neural network from floating-point (e.g., 32-bit float) to lower bit-width formats (e.g., 8-bit integers). Quantization reduces model size and speeds up inference, making it more suitable for deployment on resource-constrained devices. In AI Engine (AIE), we use a power-of-two scale factor to set up the SRS to shift and scale the values to the integer range. A power of two is a number of the form 2^n, where n is an integer. Power-of-two scale factors can lead to more efficient hardware implementations, as multiplication by a power of two can be performed using bit shifts rather than more complex multiplication operations.

[Brevitas](https://github.com/Xilinx/brevitas) is a PyTorch-based library designed for quantization of neural networks. It enables users to train models with reduced numerical precision, typically using lower bit widths for weights and activations, which can lead to significant improvements in computational efficiency and memory usage. Brevitas supports various quantization schemes, including uniform and non-uniform quantization, and can be used to target a wide range of hardware platforms, including FPGAs, ASICs, and CPUs. We use Brevitas to:
1. Quantize weights and activations of a model to lower bit format for AIE deployment, and
2. Extract proper power-of-two scale factors to set up the SRS unit.

## Source Files Overview

```
.
+-- ptq_conv2x                      # Implementation of ResNet conv2_x layers on NPU with PTQ
+-- +--  data                       # Labels for CIFAR dataset.
|   +-- aie2.py                     # A Python script that defines the AIE array structural design using MLIR-AIE operations.
|   +-- Makefile                    # Contains instructions for building and compiling software projects.
|   +-- model.py                    # Python code for ResNet Model where we apply PTQ.
|   +-- README.md                   # This file.
|   +-- requirements.txt            # pip requirements to perform PTQ.
|   +-- run_makefile.lit            # For LLVM Integrated Tester (LIT) of the design.
|   +-- test.py                     # Python code testbench for the design example.
|   +-- utils.py                    # Python code for miscellaneous functions needed for inference.


```

# Post-Training Quantization Using Brevitas
To enhance the efficiency of our implementation, we perform post-training quantization on the model using the Brevitas library. This step converts the model to use 8-bit weights and power-of-two scale factors, optimizing it for deployment on hardware with limited precision requirements.


##  Step-by-Step Process
We use test.py to:

**1. Loading the Pre-trained ResNet Model**: The script begins by loading a pre-trained ResNet model, which serves as the baseline for quantization and inference.

**2. Applying Post-Training Quantization (PTQ)**: Using the Brevitas library, the script applies PTQ to the conv2_x layers of the ResNet model. This involves converting the weights and activations to 8-bit precision.

**3. Extracting Power-of-Two Scale Factors**: After quantizing the weights and activations, the script extracts the power-of-two scale factors. These factors are crucial for efficient hardware implementation, as they simplify multiplication operations to bit shifts.

**4. Calculating Combined Scales**: The combined scale factors are calculated by multiplying the extracted weight and activation scales for each layer. These combined scales are then used to set up the SRS unit.

**5. Setting Up the SRS Unit**:
The SRS unit uses the calculated combined scales to efficiently shift and scale the values to the integer range required for the NPU.

**6. Running Inference**: Finally, the script runs inference on the quantized model. The conv2_x layers are offloaded to the NPU, utilizing the SRS unit to scale the quantized weights and activations to the int8 range properly.

# Compilation and Execution

## Prerequisites
Ensure you have the necessary dependencies installed. You can install the required packages using:

```
pip install -r requirements.txt
```
## Compilation
To compile the design:
```
make
```

## Running the Design

To run the design:
```
make run_py
```
