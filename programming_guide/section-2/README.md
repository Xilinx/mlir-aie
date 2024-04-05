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

To understand the need for a data movement abstraction we must first understand the hardware architecture with which we are working. The AIE array is a spatial compute architecture with explicit data movement requirements. Each compute unit of the array works on data that is stored within its L1 memory module and that data needs to be explicitly moved there as part of the AIE's array global data movement configuration. This configuration involves several specialized hardware resources which handle the data movement over the entire array in such a way that data arrives at its destination without loss. The Object FIFO provides users with a way to specify the data movement in a more human comprehensible and accessible manner without sacrificing some of the more advanced control possibilities which the hardware provides.

*Note: For more in-depth, low-level material on Object FIFO programming in MLIR please see the MLIR-AIE [tutorials](/mlir-aie/tutorials/).*

This guide is split into three sections, where each section builds on top of the previous ones:

<details><summary><a href="./section-2a">Section 2a - Introduction</a></summary>

</details>
<details><summary><a href="./section-2b">Section 2b - Key Object FIFO Patterns</a></summary>

</details>
<details><summary><a href="./section-2c">Section 2c - Data Layout Transformations</a></summary>

</details>
<details><summary><a href="./section-2d">Section 2d - Programming for multiple cores</a></summary>
</details>
