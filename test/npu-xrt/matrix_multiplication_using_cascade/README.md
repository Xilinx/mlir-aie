<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

## MM Cascade Design Example
This is a matrix multiply example with the sizes of (16 * 16) * (16 * 16) and i32 data type, where three different versions are compared to examine the possibility of distributing K dim accross multiple cores.

### Plain Version<br>
Generated from IREE end-to-end flow, using one core only.

### Buffer Version<br>
With four cores chained horizontally, the intermediate accumulations are passed through shared buffers implemented as ObjectFIFO.

### Cascade Version<br>
Still having four cores but the intermediate accumulations are communicated through the cascade port.

### Results<br>
From the trace files, 

|         | Total | Init | Compute |
|---------|-------|------|---------|
| Plain   | 26us  | 7us  | 19us    |
| Buffer  | 32us  | 7us  | 25us    |
| Cascade | 14us  | 7us  | 7us     |

The Buffer version is slow because of frequent lock-related operations.

The Cascade version almost halves the latency but with 4x cores. The performance gain is constrained by the initialization time of the accumulation buffer (depends on MxN only).
