<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Dynamic Object FIFO

Contains an example of what a ObjectFIFO lowering may look like that does not unroll loops, but instead chooses the buffers dynamically.

`aie2.mlir` shows what the high-level ObjectFIFO **input** looks like.

`aie2-objfifo-lowered.mlir` shows what the **output** should be after lowering with `aie-opt --aie-objectFifo-stateful-transform aie2.mlir`.

Once implemented, this can be used as a test case.

**TODO**: Dynamic lock value calculation, see [here](https://github.com/Xilinx/mlir-aie/commit/01f3642ecfe3b39708d31b42bba7adf179935deb) for an example.