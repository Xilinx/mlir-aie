<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Cascade Design

A 4xN_cols AIE array computes `C = A @ B` using AIE hardware **cascade streams** to accumulate partial products vertically within each column.  Each column's bottom row puts onto its column's cascade stream; mid rows read+put; the top row reads, accumulates into a local C tile, and writes it out.

Default config: `int16` inputs / `int32` outputs, `M`=`K`=`N` = `512`, kernel tile `m`=`k`=`n` = `64`, scalar cascade kernel.

> Different from the [whole-array design](../whole_array/README.md): cascade distributes the K accumulation across the four cores in a column (each row does `K // n_aie_rows` iterations), reducing per-core work but adding cascade-stream coordination.

The cascade kernel is currently scalar-only and the design is single-buffered (`fifo_depth=1` to avoid CDO program-memory blowup).  Compared to the vectorized whole-array design, cascade has a structurally lower performance ceiling — see the [perf comparison gist](https://gist.github.com/Yu-Zhewen/da3fed9feb278b973f35fb78c2d3a484).

## Building and Running the Design

You need C++23 for `bfloat16_t` support — `g++-13` works: [https://lindevs.com/install-g-on-ubuntu](https://lindevs.com/install-g-on-ubuntu).

`cascade.py` is `@iron.jit`-decorated.  The Makefile drives the JIT pipeline via `--xclbin-path` so artifacts land in `build/` for `test.cpp` to consume:

```shell
make
make run
```

For direct Python run + numpy verify:

```shell
python3 cascade.py                     # default i16/i32 4-col 512x512x512
python3 cascade.py --n-aie-cols 1      # single-column variant
python3 cascade.py --help              # full flag list
```
