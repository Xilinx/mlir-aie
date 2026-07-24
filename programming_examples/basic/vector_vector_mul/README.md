<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Multiply</ins>

A simple binary operator: a single AIE compute tile multiplies two `int32` vectors element-wise. The default vector size is `256`, fed into the core in sub-tiles of `16` via three depth-2 ObjectFifos (two consumer-side, one producer-side). Because the multiply is expressed as an inline Python loop on `int32`, no external compiled C++ kernel is bound — the operation lives entirely inside the IRON design.

The example targets the Ryzen™ AI NPU through the IRON `@iron.jit` host runtime.

## Source Files

1. [`vector_vector_mul.py`](vector_vector_mul.py) — IRON structural design plus host-side test driver. Decorated with `@iron.jit`; on first call it compiles the design and runs it on the NPU, then verifies the result against `a * b` computed on the host.

## Design Overview

1. ObjectFifos `in1` and `in2` connect a Shim Tile to a Compute Tile; `out` connects the Compute Tile back to the Shim Tile.
2. The runtime moves `256` `int32` from each input host buffer into the compute tile and drains the result back.
3. The compute tile acquires one tile of `16` elements from each input fifo, multiplies them element-wise, releases the result through `out`, and repeats for `256 / 16 = 16` tiles.
4. ObjectFifos are double-buffered (default depth `2`), so Shim and Compute DMAs run concurrently with the AIE core.

## Ryzen™ AI Usage

```shell
python3 vector_vector_mul.py
```

The IRON JIT runtime detects the attached NPU generation automatically. The
script compiles the design on its first invocation, runs it, verifies the
result, and reports both NPU latency and end-to-end Python wall-clock time.

For finer-grained benchmarking or a different vector size:

```shell
python3 vector_vector_mul.py --num-elements 256 --warmup 20 --iters 100
```

Run `python3 vector_vector_mul.py --help` for the full flag list.
