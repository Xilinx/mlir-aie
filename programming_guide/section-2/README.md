<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2 - Data Movement (Object FIFOs)</ins>

In this section of the programming guide, we introduce the Object FIFO high-level communication primitive used to describe the data movement within the AIE array. At the end of this guide you will:
1. have a high-level understanding of the communication primitive API,
2. have learned how to initialize and access an Object FIFO through meaningful design examples,
3. understand the design decisions which led to current limitations and/or restrictions in the Object FIFO design,
4. know where to find more in-depth material of the Object FIFO implementation and lower-level lowering.

To understand the need for a data movement abstraction we must first understand the hardware architecture with which we are working. The AIE array is a [spatial compute architecture](../README.md) with explicit data movement requirements. Each compute unit of the array works on data that is stored within its L1 memory module and that data needs to be explicitly moved there as part of the AIE's array global data movement configuration. This configuration involves several specialized hardware resources which handle the data movement over the entire array in such a way that data arrives at its destination without loss. The Object FIFO provides users with a way to specify the data movement in a more human comprehensible and accessible manner without sacrificing some of the more advanced control possibilities which the hardware provides.

> **NOTE:**  For more in-depth, low-level material on Object FIFO programming in MLIR please see the MLIR-AIE [tutorials](../mlir_tutorials).

This guide is split into five sections, where each section builds on top of the previous ones:
> **NOTE:**  Section 2e contains several practical code examples with common design patterns using the Object FIFO which can be quickly picked up and tweaked for desired use.

<details><summary><a href="./section-2a">Section 2a - Introduction</a></summary>

* Initializing an Object FIFO
* Accessing the objects of an Object FIFO
* Object FIFOs with same producer / consumer
</details>
<details><summary><a href="./section-2b">Section 2b - Key Object FIFO Patterns</a></summary>

* Introduce data movement patterns supported by the Object FIFO
    * Reuse
    * Broadcast
    * Distribute
    * Join
</details>
<details><summary><a href="./section-2c">Section 2c - Data Layout Transformations</a></summary>

* Introduce data layout transformation capabilities
</details>
<details><summary><a href="./section-2d">Section 2d - Programming for multiple cores</a></summary>

* Walkthrough of the process of efficiently upgrading to designs with multiple cores
</details>
<details><summary><a href="./section-2e">Section 2e - Practical Examples</a></summary>

* Practical examples using Object FIFOs
    * Single / Double buffer
    * External memory to core
    * External memory to core using L2
    * Distribute in L2
    * Join in L2
</details>
<details><summary><a href="./section-2f">Section 2f - Data Movement Without Object FIFOs</a></summary>

* Walkthrough of the process of programming DMA regions
</details>
<details><summary><a href="./section-2g">Section 2g - Runtime Data Movement</a></summary>

* Walkthrough of the process of managing runtime data movement from/to host memory to/from the AIE array
</details>

-----
[[Prev - Section 1](../section-1/)] [[Top](..)] [[Next - Section 3](../section-3/)]
