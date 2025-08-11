<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Layer Normalization (LayerNorm) Example

This design implements a `float32`-based Layer Normalization (LayerNorm) operation, a widely used technique in deep learning models, especially in transformer architectures. LayerNorm normalizes the activations of each input across the features (columns) for every row, stabilizing and accelerating the training process by reducing internal covariate shift. This implementation utilizes Welford's pairwise algorithm, a numerically stable, one-pass method for computing mean and variance, making it especially well-suited for lower-precision data types.

## Files

- `layernorm.py` : A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using aiecc.py to produce design binaries (i.e., XCLBIN and inst.bin for the NPU in Ryzen™ AI).

- `layernorm.cc` : A C++ implementation of a LayerNorm kernel for AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics. The source can be found [here](../../../aie_kernels/aie2p/layer_norm.cc).

- `test.cpp` : C++ testbench that initializes input buffers, runs the AIE kernel, and verifies the output against a software reference implementation.

## Parameters

- `rows` / `ROWS`: Number of rows in the input tensor (typically the batch or sequence length)
- `cols` / `COLS`: Number of columns in the input tensor (feature or embedding dimension)
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

- The LayerNorm kernel is implemented here, currently supporting only smaller input sizes and a single AIE core.
- For parallel Welford computation within a core, data is accessed in column-major order so each vector register holds one column’s values, and fully vectorized mean/variance updates for multiple rows.

## To-Do

- Extend the LayerNorm implementation to support multi-core execution on the AIE array.
- Partition the input tensor so that multiple AIE cores can process different blocks of rows or columns in parallel.
- Extend the support for `bfloat16` datatype.
