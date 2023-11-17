<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Edge Detect</ins>

The Edge Detect pipeline design consists of the following blocks arranged in a pipeline fashion for the detection of edges in a sequence of images : `rgba2gray`, `filter2D`, `threshold`, `addWeighted`, `gray2rgba`.

To compile desing in Windows:
```
make
make edgeDetect.exe
```

To run the design:
```
make run
```
