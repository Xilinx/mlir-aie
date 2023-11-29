<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Color Threshold</ins>

The Color Threshold pipeline design consists of 4 threshold blocks in separate AIE tiles that process a different region of an input image. 

The input image is brought into the array via Shim tile (0, 0) and first sent to Mem tile (0, 1) where it is then split into 4 blocks and distributed to the 4 AIE tiles (0, 2) to (0, 5). Each AIE tile applies a threshold kernel on its data. The results are then merged back together in the Mem tile and sent back to the output through the Shim tile.

To compile desing in Windows:
```
make
make colorThreshold.exe
```

To run the design:
```
make run
```
