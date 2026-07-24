<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->


# Vector $e^x$

Demonstrates how the AIE's lookup-table capability is used to approximate
$e^x$.  Four cores each operate on `1024` `bfloat16` numbers; each core uses
a LUT approximation of $e^x$.  $e^x$ is typically used in machine learning
on relatively small inputs (≈ 0..1) and overflows to infinity for inputs
larger than ~89, so a small LUT approximation is usually accurate enough
compared to a Taylor-series evaluation.

## Source Files

1. [`vector_exp.py`](vector_exp.py) — IRON structural design plus the host
   driver. Decorated with `@iron.jit`; the design uses
   [`aie.iron.kernels.bf16_exp`](../../../python/iron/kernels/activation.py),
   which wraps [`aie_kernels/aie2/bf16_exp.cc`](../../../aie_kernels/aie2/bf16_exp.cc)
   and bundles the AIE runtime's
   [`lut_based_ops.cpp`](../../../aie_runtime_lib/AIE2/lut_based_ops.cpp)
   automatically.  No per-example `.cc` / `kernels.a` / xclbin step is
   needed.
2. [`bf16_exp.cc`](../../../aie_kernels/aie2/bf16_exp.cc) — vectorized table-lookup
   implementation for AIE cores.  Operates on vectors of size 16, loading
   the vectorized accumulator with LUT results before storing back.

## Usage

```shell
python3 vector_exp.py
```

The IRON JIT runtime detects the attached NPU generation automatically. The
script compiles the design on its first invocation, runs it, and verifies every
bfloat16 input value.

The host driver tests every possible bfloat16 value (every uint16
reinterpreted as bf16, 65536 inputs total) and verifies the LUT output
against `numpy.exp` within the same 0.128 absolute tolerance the original
C++ testbench used.
