<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Rotary Position Embedding (ROPE) Example

This design implements a `bfloat16`-based Rotary Position Embedding (RoPE) operation, a technique commonly used to encode positional information in transformer models. RoPE applies position-dependent rotations to the input embeddings using precomputed sine and cosine values, enabling the model to capture relative and absolute positions efficiently.

## Files

- `rope.py` : A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using aiecc.py to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI).

- `rope.cc` : A C++ implementation of a RoPE kernel for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics.  The source can be found [here](../../../aie_kernels/aie2p/rope.cc).

- `test.cpp` : C++ testbench that initializes input and LUT buffers, runs the AIE kernel, and verifies the output against a software reference implementation.

## Parameters

- `sequence_length` / `ROWS`: Number of rows in the input tensor
- `embedding_dim` / `COLS`: Embedding dimension (number of columns)
- `trace_size`: Size of the trace buffer (for debugging/profiling)

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

## Notes

- The ROPE kernel uses bfloat16 data types for both input and output.
- The cosine and sine values for each position and dimension are streamed in as inputs.
