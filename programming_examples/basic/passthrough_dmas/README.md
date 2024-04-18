<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Passthrough DMAs</ins>

This reference design can be run on a RyzenAI NPU.

In the [design](./aie2.py) data is brought from external memory to `ComputeTile2` and back, without modification from the tile, by using an implicit copy via the compute tile's Data Movement Accelerator (DMA). The data is read from and written to external memory through Shim tile (`col`, 0).

The implicit copy is performed using the `object_fifo_link` operation that specifies how input data arriving via `of_in` should be sent further via `of_out` by specifically leveraging the compute tile's DMA. This operation and its functionality are described in more depth in [Section-2b](../../../programming_guide/section-2/section-2b/README.md/#object-fifo-link) of the programming guide.


To compile and run the design for NPU:
```
make
make run
```
