<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Passthrough DMAs</ins>

This reference design can be run on a Ryzen™ AI NPU.

In the [design](./passthrough_dmas.py) data is brought from external memory to a compute tile and back, without modification from the tile, by using an implicit copy via the compute tile's Direct Memory Access (DMA). The data is read from and written to external memory through a shim tile.

The implicit copy is performed using the ObjectFifo `forward()` function that specifies how input data arriving via `of_in` should be sent further via `of_out` by leveraging the fowarding tile's DMA. 

There are two versions of this design:
* [passthrough_dmas.py](./passthrough_dmas.py)
* [passthrough_dmas_placed.py](./passthrough_dmas_placed.py): This version of the design supports VCK500 and is written in a lower-level version of IRON. Instead of `forward()`, this version explicitly uses an `object_fifo_link` operation which is described in more depth in [Section-2b](../../../programming_guide/section-2/section-2b/03_Link_Distribute_Join/README.md#object-fifo-link) of the programming guide.


To compile and run the design for NPU:
```shell
make
make run
```

To compile and run the placed design for NPU:
```shell
env use_placed=1 make
make run
```