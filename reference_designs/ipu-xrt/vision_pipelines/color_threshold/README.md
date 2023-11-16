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

The Color Threshold pipeline design consists of a 4 threshold blocks in separate tiles that process a different region of an input image. The results are then merged back together and sent to the output.

To compile desing in Windows:
```
make
make colorThreshold.exe
```

To run the design:
```
make run
```
