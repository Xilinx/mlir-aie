<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# SwiGLU

SwiGLU (Swish-Gated Linear Unit) is a gated activation function used in large language models such as LLaMA and PaLM. It is defined as:

$$\text{SwiGLU}(x, W, V) = \text{SiLU}(xW) \otimes (xV)$$

where $\otimes$ denotes element-wise multiplication, $W$ and $V$ are two separate weight projections, and $\text{SiLU}(x) = x \cdot \sigma(x)$ is the Sigmoid Linear Unit. In practice the input gate and the linear projection are stored as two halves of a single weight matrix.

This design implements a `bfloat16` based SwiGLU on a vector, distributed in parallel across multiple AIE cores and NPU columns. The design accepts two input vectors (the gated and linear projections) and produces one output vector. Unlike single-input activation functions such as ReLU or GeLU, SwiGLU requires two simultaneous input streams per core, reflected in the two-ObjectFIFO input structure of this design.

## Source Files Overview

1. `swiglu.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.bin for the NPU in Ryzen™ AI).

1. `swiglu_bf16` kernel: A vectorized C++ implementation of SwiGLU for AIE cores, compiled into `kernels.a`. The kernel accepts two 1024-element `bfloat16` weight chunks and one activation chunk per invocation.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the testbench verifies the SwiGLU results against a CPU reference.

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
