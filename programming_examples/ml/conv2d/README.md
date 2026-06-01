<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# 1x1 Conv2D (int8, optionally fused with ReLU)

A 1x1 int8 Conv2D in the IRON API, with an optional fused-ReLU mode selected at compile time via the `fuse_relu` knob.

* `fuse_relu=0` (default): `kernels.conv2dk1_i8`; signed int8 output.
* `fuse_relu=1`: `kernels.conv2dk1(act_dtype=int8)`; uint8 output. The unsigned saturation IS the fused ReLU.

The `scale` runtime parameter is lifted to a `Compile[int]` so the design skips the RTP buffer + barrier dance.


## Source Files Overview

1. `conv2d.py`: IRON design with the `fuse_relu` knob. Kernel factory, output dtype, default scale, and worker stack size all branch on this single flag.

1. `test.py`: torch-based host harness. Pass `--fuse_relu` to match the compiled design — picks the ReLU layer and uint8 output in the reference model.


## Usage

```shell
make && make run_py                          # plain Conv2d
make clean && make fuse_relu=1 && make fuse_relu=1 run_py   # fused ReLU
```
