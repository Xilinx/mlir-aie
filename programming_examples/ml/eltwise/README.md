<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Eltwise (Add | Mul)

This design implements a `bfloat16` element-wise binary op (addition or multiplication) between two vectors, performed in parallel on two cores in a single column.  Element-wise ops usually end up being I/O bound due to the low compute intensity. In a practical ML implementation, this is the kind of kernel best fused onto a more compute-dense kernel (e.g., a convolution or GEMM).

The op is selected at compile time via the `op` parameter (`add` or `mul`); the structural design and host harness are shared.


## Source Files Overview

1. `eltwise.py`: A Python script that defines the AIE array structural design using the IRON API. `op` is a `CompileTime[str]` parameter so the body picks `kernels.add` or `kernels.mul` accordingly; everything else (placement, fifos, runtime sequence) is shared.

1. `add.cc` / `mul.cc`: Vectorized AIE kernels for vector add / multiply, pulled from the IRON kernel library. Sources live under [`aie_kernels/aie2/add.cc`](../../../aie_kernels/aie2/add.cc) and [`mul.cc`](../../../aie_kernels/aie2/mul.cc).

1. `test.cpp`: C++ testbench that loads the compiled XCLBIN + `insts.bin`, runs the kernel, and verifies the output against a CPU reference. Pass `--op add` or `--op mul` to match the compiled design.


## Usage

### Standalone JIT verification

```shell
python3 eltwise.py --op add
python3 eltwise.py --op mul
```

Pass `--dev npu2` for Strix.

### C++ Testbench

Build and run the add variant:
```shell
make op=add
make run op=add
```

Build and run the mul variant:
```shell
make op=mul
make run op=mul
```
