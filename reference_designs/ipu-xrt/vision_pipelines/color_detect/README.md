<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Edge Detect</ins>

The Color Detect pipeline design consists of the following blocks arranged in a pipeline fashion for the detecting 2 colors in a sequence of images : `rgba2hue`, `threshold`, `threshold`, `bitwiseOR`, `gray2rgba`, `bitwiseAND`.

To compile desing in Windows:
```
make
make colorDetect.exe
```

To run the design:
```
make run
```
