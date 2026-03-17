<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Object FIFO Repeat Pattern</ins>

At the closer-to-metal API level, the Object FIFO provides users with two ways to specify how data from the producer should be repeated. 

Both repeat features are achieved using the Direct Memory Access (DMA) unit of the Object FIFO's producer tile. In particular, data movement for each DMA channel is described as a chain of buffer descriptors, where each buffer descriptor (BD) indicates what data should be pushed to the AXI stream. The data movement generated for Object FIFOs follows a cyclic pattern of First In First Out where each object in the Object FIFO is moved by one BD in a chain.

The first repeat features enables users to repeat the entire BD chain using the following syntax:
```python
of0 = object_fifo("objfifo0", A, B, 2, np.ndarray[(256,), np.dtype[np.int32]])
of0.set_iter_count(2)
```
The code snippet above results in the repetition of each object in the Object FIFO following the pattern `buff_ping - buff_pong - buff_ping - buff_pong`. This is shown in the figure below with the red arrow representing the repeat value of the entire BD chain:

<img src="./../../../assets/RepeatSharedTile.png" height="300">

> **NOTE:**  Repeating BD chains is only available on Mem tiles.

Users may also repeat each individual BD in the chain. This feature is available using the following syntax:
```python
of0 = object_fifo("objfifo0", A, B, 2, np.ndarray[(256,), np.dtype[np.int32]])
of0.set_repeat_count(2)
```
The specified repeat value is applied to all BDs. It is currently not possible to set different repeat values per BD.
The code snippet above results in the repetition of each object in the Object FIFO following the pattern `buff_ping - buff_ping - buff_pong - buff_pong`. This is shown in the figure below with the red arrows representing the repeat values of each BD in the chain:

<img src="./../../../assets/RepeatSharedTile_2.png" height="300">

The `repeat_count` feature may also be used with a Compute tile producer. As synchronization logic is leveraged for object accesses between a Compute tile core and its DMA, the Object FIFO lowering will use available information to modify the values of Object FIFO ```acquire``` and ```release``` operations based on the repeat value. This is to ensure that enough tokens are produced by the compute core to allow the DMA to repeat and that these tokens are accounted for by the first ```acquire``` operation post DMA repetition. Doing this adjustement for Object FIFOs of depth larger than 1 is non-trivial and currently not supported.

> **NOTE:**  The two repeat features can be combined.

> **NOTE:**  No additional memory is allocated when repeating. Instead, DMAs are programmed to push existing data buffers multiple times.

For more information into DMAs and their buffer descriptors you can refer to the [Advanced Topic of Section 2a](../../section-2a/README.md#advanced-topic-data-movement-accelerators) and [Section 2g](../../section-2g/).

### Link & Repeat

It is also possible to use repetition with the link, described in the previous section, using the following syntax:
```python
of0 = object_fifo("objfifo0", A, B, 2, np.ndarray[(256,), np.dtype[np.int32]])
of1 = object_fifo("objfifo1", B, C, 2, np.ndarray[(256,), np.dtype[np.int32]])
object_fifo_link(of0, of1)
of1.set_repeat_count(2) # the data in each object is sent to the consumer C twice
```

<img src="./../../../assets/Repeat.png" height="150">

In this case the repetition is achieved using the Direct Memory Access (DMA) of the Object FIFO link's shared tile.

In particular, the repeat functionality can be used in conjunction with the distribute pattern introduced in the previous section. Currently, the repeat value specified for each distribute destination must be the same to ensure functional correctness. Additionally, the syntax currently doesn't support both output Object FIFOs with repeat and without at the same time, in the same distribute pattern. The code below shows how the two output Object FIFOs of a distribute pattern can be set to each repeat three times:
```python
of0 = object_fifo("objfifo0", A, B, 2, np.ndarray[(256,), np.dtype[np.int32]])
of1 = object_fifo("objfifo1", B, C, 2, np.ndarray[(256,), np.dtype[np.int32]])
of2 = object_fifo("objfifo2", B, D, 2, np.ndarray[(256,), np.dtype[np.int32]])
object_fifo_link(of0, [of1, of2])
of1.set_repeat_count(3)
of2.set_repeat_count(3)
```
The code snippet above is part of a test that can be found [here](../../../../test/npu-xrt/objectfifo_repeat/distribute_repeat/).

-----
[[Prev](../03_Implicit_Copy/)] [[Up](..)] [[Next - Section 2c](../../section-2c/)]
