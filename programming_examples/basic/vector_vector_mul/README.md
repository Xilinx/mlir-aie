<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Vector Vector Multiply</ins>

A simple binary operator: a single AIE compute tile multiplies two `int32` vectors element-wise. The default vector size is `256`, fed into the core in sub-tiles of `16` via three depth-2 ObjectFifos (two consumer-side, one producer-side). Because the multiply is expressed as an inline Python loop on `int32`, no external compiled C++ kernel is bound — the operation lives entirely inside the IRON design.

The example targets the Ryzen™ AI NPU through the IRON `@iron.jit` host runtime, and the VCK5000 PCIe card through the legacy `aiecc` + HSA toolchain. Both paths share the same design body.

## Source Files

1. [`vector_vector_mul.py`](vector_vector_mul.py) — IRON structural design plus host-side test driver. Decorated with `@iron.jit`; on first call it compiles the design and runs it on the NPU, then verifies the result against `a * b` computed on the host. The same design body is reused by the `--emit-mlir-vck5000` flag, which prints MLIR for the VCK5000 toolchain.
1. [`test_vck5000.cpp`](test_vck5000.cpp) — host testbench for the VCK5000 path; loads the HSA-linked `test.elf`, drives the AIE array, and verifies output.

## Design Overview

1. ObjectFifos `in1` and `in2` connect a Shim Tile to a Compute Tile; `out` connects the Compute Tile back to the Shim Tile.
2. The runtime moves `256` `int32` from each input host buffer into the compute tile and drains the result back.
3. The compute tile acquires one tile of `16` elements from each input fifo, multiplies them element-wise, releases the result through `out`, and repeats for `256 / 16 = 16` tiles.
4. ObjectFifos are double-buffered (default depth `2`), so Shim and Compute DMAs run concurrently with the AIE core.

## Ryzen™ AI Usage

```shell
make run        # compile + execute on the attached NPU (auto-detected)
make clean
```

The NPU generation (NPU1 / NPU2) is auto-detected by the IRON runtime at JIT time, so no device flag is needed. `make run` reports both NPU latency (from the runtime) and end-to-end Python wall-clock so the host-side overhead delta is visible.

For finer-grained benchmarking, invoke the script directly:

```shell
python3 vector_vector_mul.py -n 256 -w 20 -i 100   # vector size, warmup, iters
```

Run `python3 vector_vector_mul.py --help` for the full flag list.

## VCK5000 Usage

The VCK5000 path bypasses `@iron.jit` (which is NPU-only) and uses the legacy MLIR-to-HSA flow:

```shell
make vck5000    # emits MLIR via --emit-mlir-vck5000, then aiecc + test_vck5000.cpp
./test.elf      # run on a VCK5000-attached host
```

`make vck5000` invokes `python3 vector_vector_mul.py --emit-mlir-vck5000 > build/aie.mlir` and feeds the result to `aiecc` together with `test_vck5000.cpp`.
