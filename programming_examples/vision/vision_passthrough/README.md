<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Vision Passthrough</ins>

A single AIE tile copies a `width x height` 8-bit image one line at a time using the `passThroughLine` kernel from `aie_kernels/generic/passThrough.cc`.  This pipeline mainly serves to test whether the data movement between a Shim tile and an AIE tile works correctly.

`vision_passthrough.py` is a single `@iron.jit`-decorated design that the `Makefile` drives in compile-only mode (`--xclbin-path` / `--insts-path`) for the OpenCV-based C++ host (`test.cpp`).  The same script also runs standalone for a quick in-Python identity check.

## Usage

### Standalone (no Makefile, no OpenCV needed)

```shell
python3 vision_passthrough.py
```

`-d npu2` for Strix; `-W` / `-H` override the image dimensions.

### Makefile + C++ testbench (OpenCV required)

```shell
make
make run
```

For NPU2 (Strix): `make device=npu2 && make run device=npu2`.
