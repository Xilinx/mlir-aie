<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Core Programming Examples</ins>

These programming examples provide a good starting point to illustrate how to build commonly used compute kernels (both single core and multicore data processing pipelines). They serve to highlight how designs can be described in python and lowered through the mlir-aie tool flow to an executable that runs on the IPU. 

* [Add One (with ObjectFIFOs)](./add_one_objFifo) - Single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.
* [Hello World (Log version)](./log_hello_world) - Single tile performs a self-query and `printf` function where printed data is moved from local buffers to external memory to be read by the host processor.
* [Matrix Multiplication](./matrix_multiplication) - Single tile performs a `matrix * matrix` multiply on int16 data type where `MxKxN` is `128x128x128`. The kernel itself computes `64x32x64 (MxKxN)` so it is invoked multiple times to complete the full matmul compute.
* [Vector Scalar](./vector_scalar) - Single tile performs `vector * scalar` of size `4096`. The kernel does a `1024` vector multiply and is invoked multiple times to complete the full vector*scalar compute.

