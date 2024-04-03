<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3b - Key Object FIFO Patterns</ins>

The Object FIFO primitive supports several data movement patterns through its inputs and its member functions. We will now describe each of the currently supported patterns and provide links to more in-depth practical code examples that showcase each of them. 

#### Reuse

#### Broadcast

As was explained in the (TODO: add section with link) section, the `consumerTiles` input can be either a single tile or an array of tiles. When the input is specified as an array of tiles, this creates a broadcast communication from a single producer tile to multiple consumer tiles. The same data is transferred to each of the consumer tiles via the AXI stream interconnect, which handles the back-pressure from consumers with different execution times.

TODO: specify where the data is for each consumer and what will happen to the data over the stream (is copied at a low level and sent to each destination and the DMAs pick it up to put it in each of the memory modules)

Below is an example of an Object FIFO with three consumer tiles.

* normal and with skip connection (depth can be either a number or an array) - the OF primtiive is not responsible for keeping track and identifying dependencies between multiple objectfifos and that is why the ability to specify multiple depths is important

#### Link

```
class object_fifo_link(ObjectFifoLinkOp):
    def __init__(
        self,
        fifoIns,
        fifoOuts,
    )
```

#### Link & Distribute

#### Link & Join

## Data Layout Transformations

* dimensionsToStream, dimensionsFromStreamPerConsumer

* TODO: Remind: Point to more detailed, low-level objectfifo material in Tutorial

