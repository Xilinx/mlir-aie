<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Quick Reference</ins>

## Python Bindings

| Syntax | Definition | Example | Notes |
|--------|------------|---------|-------|
| \<name\> = tile(column, row) | Declare AI Engine tile | ComputeTile = tile(1,3) | The actual tile coordinates run on the device may deviate from the ones declared here. In Ryzen AI, for example, these coordinates tend to be relative corodinates as the runtime scheduler may assign it to a different available column. |

## Object FIFO Bindings


## Python helper functions
| Function | Description |
|----------|-------------|
| print(ctx.module) | Converts our ctx wrapped structural code to mlir and prints to stdout|
| print(ctx.module.operation.verify()) | Runs additional structural verficiation on the python binded source code and prints to stdout |


