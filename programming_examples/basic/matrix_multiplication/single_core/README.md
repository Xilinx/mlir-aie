<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Single Core Design

A single AI Engine compute core performs `C = A @ B`.  Default config: `int16` inputs / `int32` outputs, `M`&times;`K`&times;`N` = `512`&times;`512`&times;`512`, kernel tile `m`&times;`k`&times;`n` = `32`&times;`32`&times;`32`.  The host streams (m, k) x (k, n) tile pairs through one ObjectFifo per direction; the core multiply-accumulates into an (m, n) output tile and the runtime drains rows-of-tiles back to L3.

> This is a simplification of the [whole-array design](../whole_array/README.md): one compute core instead of the full 4xN_cols grid.  See that README for the broader IRON walkthrough.

## Building and Running the Design

You need C++23 for `bfloat16_t` support — `g++-13` works: [https://lindevs.com/install-g-on-ubuntu](https://lindevs.com/install-g-on-ubuntu).

`single_core.py` is `@iron.jit`-decorated.  The Makefile drives the JIT pipeline via `--xclbin-path` so artifacts land in `build/` for `test.cpp` to consume:

```shell
make
make run
```

For direct Python run + numpy verify (skips `test.cpp` entirely):

```shell
python3 single_core.py                            # default i16/i32 512x512x512
python3 single_core.py --b-col-maj 1              # column-major B input
python3 single_core.py --use-chess 1              # chess kernel build
python3 single_core.py --dtype_in bf16 --dtype_out bf16
python3 single_core.py --help                     # full flag list
```

Both paths share one design body; the makefile path is for the existing lit/sweep/test.cpp infrastructure.

## Tracing

```shell
make trace
```

Builds a trace-enabled xclbin (`build/trace_*.xclbin`) and runs `test.cpp` with `-t ${trace_size}`.  Trace events are captured to `trace.txt` and parsed into `trace_mm.json` for visualization.

`trace_size` defaults to `65536` and can be overridden:

```shell
make trace trace_size=32768
```
