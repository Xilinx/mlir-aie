<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Matrix Scalar Addition</ins>

This design shows an extremely simple single-AIE design: incrementing every value in one top-left tile of an input matrix.

It demonstrates a number of features that scale to more realistic designs:

* A 2D DMA pattern (`TensorTiler2D.simple_tiler`) accesses `8x16` subtiles from a `16x128` input/output matrix. Thinking about input/output spaces as large grids with smaller grids of work dispatched to individual AIE cores is a fundamental, reusable concept.
* The body of work each AIE core does combines data movement (object-FIFO acquire and release) with compute.
* The overall structural design combines a static description (cores, connections, parts of the data movement) with a runtime sequence that controls dispatch.
* The output buffer is initialized by the host. The design writes the selected output tile and leaves the remaining output positions at their initial values.

## Source Files Overview

`matrix_scalar_add.py`: An `@iron.jit`-decorated design targeting the Ryzen AI NPU pipeline. Two invocation modes:

* standalone — `python3 matrix_scalar_add.py`
* compile-only — `... --xclbin-path=PATH --insts-path=PATH` (used by the NPU `Makefile`)

## Usage

### NPU

```shell
python3 matrix_scalar_add.py
```

The standalone command detects the attached NPU family automatically. Pass
`-d npu2` only when selecting the NPU2 target explicitly.

A Makefile is available for the native C++ host flow:

```shell
make
make run
```

For NPU2 (Strix): `make devicename=npu2 && make run devicename=npu2`.
