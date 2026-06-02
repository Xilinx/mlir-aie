<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Magika</ins>
## Introduction
This implements parts of the magika AI-powered file type detection network described under https://github.com/google/magika. 

NOTE: Currently, the design supports standalone group0 and group2 blocks. The group1 component is not yet implemented.


## Source Files Overview
```
.
+-- c                           # C reference
+-- data                        # Input stimulus and output reference
+-- inc                         # Include files, such as LUT headers and sub kernel functions
+-- kernels                     # group kernel functions
+-- py                          # python utilities for extrapolating stimulus from onnx file
+-- group0.py                   # Group 0 design (IRON / @iron.jit)
+-- group2.py                   # Group 2 design (IRON / @iron.jit)
+-- Makefile                    #
+-- README.md                   # This file.
+-- run.lit                     # For LLVM Integrated Tester (LIT) of the design.
+-- test_group0.py              # Python code testbench for the group 0 design example
+-- test_group2.py              # Python code testbench for the group 2 design example
```

## Compilation
To compile the design for group 0 and run it.
```shell
make run_py
```

To compile the design for group 2:
```shell
make targetname=group2 run_py
```

To build the design to generate trace, replace `run_y` with `trace_py`
```shell
make targetname=group0 trace_py
```
