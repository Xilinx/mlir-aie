<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Matrix-Vector Multiplication

A single AI Engine compute core computes `c = A @ b`, where `A` is `M`x`K` and `b` is a length-`K` vector.  Default config: `int16` inputs / `int32` outputs, `M`=`K`=`288`, kernel tile `m`=`k`=`32`.

> Built on the same data-movement concepts as the [whole-array design](../whole_array/README.md); see that README for the IRON walkthrough.

## Differences from the [Whole-Array Design](../whole_array/README.md)

- A specialized matrix-*vector* microkernel (`kernels.mv`) is used instead of the general matrix-matrix microkernel.  Defaults to the vectorized path; pass `--scalar` to fall back to the scalar variant.
- Data movement: an identical `K`-element chunk of `b` is broadcast; subsequent `m`x`k` tiles of `A` are distributed.  This is a single-core design; multi-core extension is left for a future revision.

## Building and Running the Design

You need C++23 for `bfloat16_t` support — `g++-13` works: [https://lindevs.com/install-g-on-ubuntu](https://lindevs.com/install-g-on-ubuntu).

`matrix_vector.py` is `@iron.jit`-decorated.  The Makefile drives the JIT pipeline via `--xclbin-path` so artifacts land in `build/` for `test.cpp` to consume:

```shell
make
make run
```

For direct Python run + numpy verify:

```shell
python3 matrix_vector.py                  # default M=K=288 i16/i32
python3 matrix_vector.py --use-chess 1    # chess kernel build
python3 matrix_vector.py --help           # full flag list
```
