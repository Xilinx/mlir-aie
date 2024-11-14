<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Simple Repeat</ins>

This reference design can be run on a Ryzen™ AI NPU.

In the [design](./aie2.py) data is brought from external memory via the `ShimTile` to the `MemTile` and back by using an implicit copy via the mem tile's Data Movement Accelerator (DMA). Furthermore, the input data is repeated by the `MemTile` four times which results in the output data consisting of four instances of the input data.

The implicit copy is performed using the `object_fifo_link` operation that specifies how input data arriving via `of_in` should be sent further via `of_out` by specifically leveraging the mem tile's DMA. This operation and its functionality are described in more depth in [Section-2b](../../../../programming_guide/section-2/section-2b/03_Link_Distribute_Join/README.md#object-fifo-link) of the programming guide.

The repeat count is specified as follows:
```python
of_out.set_repeat_count(memtile_repeat_count)
```
Specifically, the instruction above specifies the number of repetitions that the producer side of the `of_out` objectfifo should do.
