<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3 - Data Movement (objectfifos)</ins>

* Introduce topic of objectfifos and how they abstract connections between objects in the AIE array

In this section of the programming guide, we introduce the Object FIFO high-level abstraction used to describe the data movement within the AIE array. At the end of this guide you will:
1. have a high-level understanding of the abstracion,
2. have learned how to initialize and access an Object FIFO through meaningful design examples,
3. understand the design decisions which led to current limitations and/or restrictions in the Object FIFO design,
4. know where to find more in-depth material of the Object FIFO implementation and lower-level lowering.

To understand the need for a data movement abastraction we must first understand the hardware architecture with which we are working. The AIE array is a spatial compute architecture with explicit data movement requirements. Each compute unit of the array works on data that is stored within its L1 memory module and that data needs to be explicitly moved there as part of the AIE's array global data movement configuration. This configuration involves several specialized hardware resources which handle the data movement over the entire array in such a way that data arrives at its destination without loss. The Object FIFO provides users with a way to specify the data movement in a more human comprehensible and accessible manner without sacrificing some of the more advanced control possibilities which the hardware provides.

To initialize an Object FIFO, users can use the `object_fifo` class constructor (TODO: add link to location):

```
class object_fifo:
    def __init__(
        self,
        name,
        producerTile,
        consumerTiles,
        depth,
        datatype,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    )
```

We will go over each of the inputs, what it represents and why it is required by the abstraction. We will first focus on the mandatory inputs and later on the default valued ones. 



* Point to more detailed objectfifo material in Tutorial
* Introduce key objectfifo connection patterns (link/ broadcast, join/ distribute)
