<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Scalar Addition</ins>

This design shows an extremely simple single-AIE design: incrementing every value in an input matrix.

It demonstrates a number of features that scale to more realistic designs:

* A 2D DMA pattern (`TensorTiler2D.simple_tiler`) accesses `8x16` subtiles from a `16x128` input/output matrix. Thinking about input/output spaces as large grids with smaller grids of work dispatched to individual AIE cores is a fundamental, reusable concept.
* The body of work each AIE core does combines data movement (object-FIFO acquire and release) with compute.
* The overall structural design combines a static description (cores, connections, parts of the data movement) with a runtime sequence that controls dispatch.

## Source Files Overview

`matrix_scalar_add.py`: An `@iron.jit`-decorated design covering both the Ryzen AI NPU pipeline and the aiecc-based VCK5000 (Versal AIE1) flow. Three invocation modes:

* standalone — `python3 matrix_scalar_add.py`
* compile-only — `... --xclbin-path=PATH --insts-path=PATH` (used by the NPU `Makefile`)
* emit-MLIR — `... -d xcvc1902 --emit-mlir` (used by the VCK5000 path; aiecc consumes the printed MLIR with `--link_against_hsa`)

## Usage

### NPU

```shell
make
make run
```

For NPU2 (Strix): `make devicename=npu2 && make run devicename=npu2`.

### VCK5000

```shell
make vck5000
./test.elf
```
