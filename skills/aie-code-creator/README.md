<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# AIE Code Creator

## Description

This skill teaches how to write efficient designs for AMD XDNA NPUs (AIE2 in Phoenix / Hawk Point, AIE2P in Strix / Krackan Point) using the IRON Python API together with the C++ AIE kernels that run on the compute tiles.

An IRON design has two halves, and this skill covers both:

1. **Structural / data-movement code (Python)** - describes the AIE-array topology: which tiles run which tasks, what `ObjectFifo`s connect them, and what the host-side `Runtime` does to feed and drain data.
2. **Compute kernel code (C++)** - the per-tile functions written against the AIE API (`aie::vector`, `aie::mmul`, ...) and compiled into `.o` files referenced by the Python design.

The skill is **library-first**: mlir-aie ships `aie.iron.kernels` (maintained vectorized kernels for element-wise ops, reductions, activations, matmul, conv2d, and vision) and `aie.iron.algorithms` (whole-design templates such as `transform_parallel` and `reduce`), so many designs need no hand-written C++ at all. The workflow checks those first and drops to hand-written code only where nothing fits.

The skill provides:

- **Defaults** for kernel source, API level, target device, data type, and parallelism so designs can be produced without exhaustive clarification
- A **step-by-step workflow** from computation shape to complete design (built-in check -> pattern -> kernel template -> wiring -> sizing -> pitfalls check)
- **Reference files** loaded on demand covering the built-in kernel/algorithm libraries and `In`/`Out`/`CompileTime[T]` jit signatures, hardware architecture, the Python API (`Program`, `Worker`, `ObjectFifo`, `Runtime`, `Kernel`, `Buffer`, `iron.jit`), copy-ready design patterns, C++ kernel intrinsics, build/test harness, and known pitfalls
- A pointer to upstream mlir-aie's phased skill family (`aie-model-baseline`, `aie-dataflow-presim`, `aie-hw-bringup`, `aie-kernel-opt`, `aie-dataflow-opt`) for the phases before and after design creation

When asked for a complete design, it delivers the Python design file, a build/run snippet, and the C++ kernel file where one had to be written.

## Usage

Invoke when writing or modifying IRON / NPU code:

```text
Using the aie-code-creator skill, write a multi-core bfloat16 element-wise add for NPU2
```

```text
Using the aie-code-creator skill, generate an MMUL-based matmul kernel and the matching Python design
```

```text
Using the aie-code-creator skill, add a broadcast ObjectFifo to this design so all four workers see the same weights
```
