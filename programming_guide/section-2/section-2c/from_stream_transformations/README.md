<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>From Stream Data Layout Transformations</ins>

In the [from_stream.py](./from_stream.py) design we first bring `24xi32` data from external memory to L2 memory (i.e., a Mem tile) with `of_in0`. We then use `of_in1` to forward the data from the `MemTile` to `my_worker` and apply a data layout transformation as the data is read from the AXI stream into local memory by the Worker tile's DMA. Two FIFOs then move the output data from the Worker first to L2 via `of_out1`, then to external memory via `of_out0` as `24xi32` tensors. All FIFOs use double buffers.

```python
# Dataflow with ObjectFifos
# Input
of_in0 = ObjectFifo(data_ty, name="in0")
of_in1 = of_in0.cons().forward(
    name="in1", obj_type=data_ty, dims_from_stream=[(3, 1), (8, 3)]
)

# Output
of_out1 = ObjectFifo(data_ty, name="out1")
of_out0 = of_out1.cons().forward(name="out0", obj_type=data_ty)
```

The process on the Worker acquires one object from `of_in1` to consume and one object from `of_out1` to produce into. It then reads the value of the input object and loads it into the output one before releasing both objects.

The data layout transformation `dims_from_stream=[(3, 1), (8, 3)]` expresses the access pattern in which the Worker will write the data from the AXI stream into a local `24xi32` tensor. This access pattern can also be expressed with `for` loops as follows:
```python
for i in range(3):
    for j in range(8):
        # write data at index
        index = (i * 1 + j * 3)
```
If we imagine the 24-element wide tensor as 3 rows of 8 elements, the transformation above stores the data from the stream in column-major order.

> **NOTE:**  While the end-result is the same, the pattern in this example differs from the one in the [to_stream_transformations](../to_stream_transformations/) design. This is because the Worker has no control over how the data arrives from the AXI stream, whereas in the [to_stream.py](../to_stream_transformations/to_stream.py) example, the Worker can access the data directly in the order in which it will push it onto the stream.

It is possible to compile, run and test this design with the following commands:
```bash
make
make run
```

The [test.cpp](./test.cpp) as well as the `# To/from AIE-array data movement` section of the design code will be described in detail in [Section 2d](../../section-2d/).
