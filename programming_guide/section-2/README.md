<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 2 - Data Movement (ObjectFifos)

In this section of the programming guide, we introduce the ObjectFifo high-level communication primitive used to describe the data movement within the AIE array. At the end of this guide you will:
1. have a high-level understanding of the communication primitive API,
2. have learned how to initialize and access an ObjectFifo through meaningful design examples,
3. understand the design decisions, which led to current limitations and/or restrictions in the ObjectFifo design,
4. know where to find more in-depth material of the ObjectFifo implementation and lower-level lowering.

To understand the need for a data movement abstraction we must first understand the hardware architecture with which we are working. The AIE array is a [spatial compute architecture](../README.md) with explicit data movement requirements. Each compute unit of the array works on data that is stored within its L1 memory module and that data needs to be explicitly moved there as part of the AIE's array global data movement configuration. This configuration involves several specialized hardware resources that handle the data movement over the entire array in such a way that data arrives at its destination without loss. The ObjectFifo provides users with a way to specify the data movement in a more human-comprehensible and accessible manner, without sacrificing some of the more advanced control possibilities which the hardware provides.

> **NOTE:**  For more in-depth, low-level material on the data movement the ObjectFifo lowers into — DMAs, buffer descriptors, locks, and stream routing — see [Section 2g - Data Movement Without ObjectFifos](./section-2g/).

This guide is split into eight subsections, where each builds on top of the previous ones:

* **[Section 2a - Introduction](./section-2a)**
    * Initializing an ObjectFifo
    * Accessing the objects of an ObjectFifo
    * ObjectFifos with same producer / consumer
* **[Section 2b - Key ObjectFifo Patterns](./section-2b)**
    * Data movement patterns supported by the ObjectFifo: Reuse, Broadcast, Distribute, Join, Repeat
* **[Section 2c - Data Layout Transformations](./section-2c)**
    * Data layout transformation capabilities
* **[Section 2d - Runtime Data Movement](./section-2d)**
    * Managing runtime data movement between host memory and the AIE array
* **[Section 2e - Programming for multiple cores](./section-2e)**
    * Efficiently upgrading to designs with multiple cores
* **[Section 2f - Practical Examples](./section-2f)**
    * Common design patterns using ObjectFifos: single / double buffer, external memory to core (with and without L2), distribute in L2, join in L2
* **[Section 2g - Data Movement Without ObjectFifos](./section-2g)**
    * Programming DMA regions directly
* **[Section 2h - Advanced ObjectFifo + Cross-Tile Buffer](./section-2h)**
    * Asymmetric producer/consumer transfer granularity (`consumer_obj_type=`)
    * Direct AIE-stream connections (`aie_stream=(end, port)`)
    * Cross-tile Buffer placement in `Worker.fn_args` for neighbor-L1 access

> **NOTE:** Section 2f contains several practical code examples with common design patterns using the ObjectFifo which can be quickly picked up and tweaked for your own use.

-----
[Prev](../section-1/) &middot; [Top](..) &middot; [Next](../section-3/)
