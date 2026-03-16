<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# GeLU

GeLU (Gaussian Error Linear Unit) is an activation function widely used in transformer-based models such as BERT and GPT. It is defined as:

$$\text{GeLU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. In practice a fast approximation is commonly used:

$$\text{GeLU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right)$$

This design implements a `bfloat16` based GeLU on a vector, distributed in parallel across multiple AIE cores and NPU columns. Like other element-wise activation functions, GeLU is I/O bound due to its low compute intensity relative to data movement.

## Source Files Overview

1. `gelu.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.bin for the NPU in Ryzen™ AI).

1. `gelu_bf16` kernel: A vectorized C++ implementation of GeLU for AIE cores, compiled into `kernels.a`. The kernel operates on 1024-element `bfloat16` chunks per invocation.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the testbench verifies the GeLU results against a CPU reference.

## Usage

### C++ Testbench

To compile the design and C++ testbench:
```shell
make
```

To run the design:
```shell
make run
```
