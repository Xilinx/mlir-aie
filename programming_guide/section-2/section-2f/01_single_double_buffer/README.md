<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Single / Double Buffer</ins>

The design in [single_buffer.py](./single_buffer.py) uses an Object FIFO `of_in` to transfer the output of `my_worker` to `my_worker2` and an Object FIFO `of_out` to transfer the output of `my_worker2` to external memory. `of_in` has a depth of `2` which describes a double, or ping-pong, buffer between the two Workers as shown in the figure below.

<img src="../../../assets/DoubleBuffer.svg" height=200 width="500">

> **NOTE:**  The image above assumes that the Workers are already mapped to `ComputeTile2` and `ComputeTile3`. However, this is not the only possible mapping and when creating a Worker, its placement can be left to the compiler.

Both the producer and the consumer processes in this design have trivial tasks. The producer process running on `my_worker` acquires one ping-pong buffer and writes `1` into all its entries before releasing it for consumption. The consumer process running on `my_worker2` acquires one buffer from `of_in` as well as one buffer from `of_out`, copies the data from the input Object FIFO to the output Object FIFO, and releases both objects for other processes. The Object FIFO lowering takes care of properly cycling between the ping and pong buffers.

> **NOTE:** To enforce a single buffer (no prefetching), specify the depth as an array with explicit per-endpoint depths:
> ```python
> of_in = ObjectFifo(data_ty, name="in", depth=[1, 1])  # single buffer
> of_out = ObjectFifo(data_ty, name="out", depth=[1, 1])  # single buffer
> ```
> This uses the array-depth form to explicitly limit each endpoint to one buffer. The producer and consumer must then strictly alternate access with no overlap.

All examples available in the [programming_examples](../../../../programming_examples/) contain this data movement pattern.

It is possible to compile, run and test this design with the following commands:
```bash
make
make run
```

The explicitly placed level of IRON programming for this design is available in [single_buffer_placed.py](./single_buffer_placed.py). It can be compiled, run and tested with the following commands:
```bash
env use_placed=1 make
make run
```

-----
[[Up](..)] [[Next](../02_external_mem_to_core/)]
