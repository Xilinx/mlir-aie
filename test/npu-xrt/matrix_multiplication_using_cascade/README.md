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
This is a matrix multiply example with the sizes of (16 * 16) * (16 * 16) and i32 data type, where four different versions are compared to examine the possibility of distributing K dim accross multiple cores.

### Plainx1 Version<br>
Generated from IREE end-to-end flow, using one core only.

### Plainx4 Version<br>
Using four cores, as output stationary 

### Bufferx4 Version<br>
With four cores chained horizontally, the intermediate accumulations are passed through shared buffers implemented as ObjectFIFO.

### Cascadex4 Version<br>
Still having four cores but the intermediate accumulations are communicated through the cascade port.

### Results<br>
From the trace files, 

|           | Total  | Init  | Compute |
|-----------|--------|-------|---------|
| Plainx1   | 25.6us | 7.6us | 18.0us  |
| Plainx4   | 6.7us  | 2.0us | 4.7us   |
| Bufferx4  | 32.0us | 7.6us | 24.4us  |
| Cascadex4 | 13.9us | 7.6us | 6.3us   |

The Buffer version is slow because of frequent lock-related operations.

The Cascade version almost halves the latency but with 4x cores. The performance gain is constrained by the initialization time of the accumulation buffer (depends on MxN only).
