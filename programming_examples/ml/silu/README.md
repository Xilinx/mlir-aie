<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# SiLU

SiLU (Sigmoid Linear Unit), also known as the Swish activation function, is defined as:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

where $\sigma(x)$ is the sigmoid function. SiLU is used as an activation function in models such as EfficientNet and various vision transformers. It is smooth and non-monotonic, which can improve training dynamics compared to ReLU.

This design implements a `bfloat16` based SiLU on a vector, distributed in parallel across multiple AIE cores and NPU columns. Like other element-wise activation functions, SiLU is I/O bound due to its low compute intensity relative to data movement.

## Source Files Overview

1. `silu.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.bin for the NPU in Ryzen™ AI).

1. `silu_bf16` kernel: A vectorized C++ implementation of SiLU for AIE cores, compiled into `kernels.a`. The kernel operates on 1024-element `bfloat16` chunks per invocation.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the testbench verifies the SiLU results against a CPU reference.

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
