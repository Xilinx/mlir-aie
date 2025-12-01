<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>To Stream Data Layout Transformations</ins>

In the [to_stream.py](./to_stream.py) design we first bring `24xi32` data from external memory to L2 memory (i.e., a Mem tile) with `of_in0`. We then use `of_in1` to forward the data from the `MemTile` to `my_worker`. Two FIFOs then move the data:
- first to L2 via `of_out1`, applying a data layout transformation as the data is pushed onto the AXI stream by the Worker tile's DMA,
- then to external memory via `of_out0` as `24xi32` tensors.
All FIFOs use double buffers.

```python
# Dataflow with ObjectFifos
# Input
of_in0 = ObjectFifo(tile24_ty, name="in0")
of_in1 = of_in0.cons().forward(name="in1", obj_type=tile24_ty)

# Output
of_out1 = ObjectFifo(tile24_ty, name="out1", dims_to_stream=[(8, 1), (3, 8)])
of_out0 = of_out1.cons().forward(name="out0", obj_type=tile24_ty)
```

The process on the Worker acquires one object from `of_in1` to consume and one object from `of_out1` to produce into. It then reads the value of the input object and loads it into the output one before releasing both objects.

It is possible to compile, run and test this design with the following commands:
```bash
make
make run
```

The [test.cpp](./test.cpp) as well as the `# To/from AIE-array data movement` section of the design code will be described in detail in [Section 2d](../../section-2d/).
