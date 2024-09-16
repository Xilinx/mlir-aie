<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Distribute Repeat</ins>

This reference design can be run on a Ryzen™ AI NPU.

In the [design](./aie2.py) data is brought from external memory via the `ShimTile` to the `MemTile`. The data is then split between two `ComputeTile`s using a distribute operation specified via an `object_fifo_link`. Each input objectfifo between the `MemTile` and each `ComputeTile` is further configured to repeat the data so as to send multiple copies of that data to each `ComputeTile`.

For more information, the `object_fifo_link` operation as well as the concept of distribution and its functionality is described in more depth in [Section-2b](../../../programming_guide/section-2/section-2b/03_Link_Distribute_Join/README.md#object-fifo-link) of the programming guide.

To compile and run the design for NPU:
```
make
make run
```