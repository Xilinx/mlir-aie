<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vision Passthrough</ins>

Single tile applies a pass through kernel on data from local memory. There are three versions of this pipeline that differ in the sizes of input and output data tensors. This pipeline mainly serves to test whether the data movement between a Shim tile and an AIE tile works correctly.

To compile the design:
```shell
make
make vision_passthrough.exe
```

To compile the placed design:
```shell
env use_placed=1 make
make vision_passthrough.exe
```

To run the design:
```shell
make run
```

