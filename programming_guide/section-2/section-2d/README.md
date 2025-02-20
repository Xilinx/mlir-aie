<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2d - Runtime Data Movement</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * Section 2d - Runtime Data Movement
    * [Section 2e - Programming for multiple cores](../section-2e/)
    * [Section 2f - Practical Examples](../section-2f/)
    * [Section 2g - Data Movement Without Object FIFOs](../section-2g/)

-----

In the preceding sections, we looked at how we can describe data movement between tiles *within* the AIE-array. However, to do anything useful, we need to get data from outside the array, i.e., from the "host", into the AIE-array and back. On NPU devices, we can achieve this with the operations described in this section. 

The operations that will be described in this section must be placed in a separate `sequence()` of a `Runtime` class, or `aie.runtime_sequence` operation at the explicitly placed IRON level. The arguments to this function describe buffers that will be available on the host side; the body of the function describes how those buffers are moved into the AIE-array. [Section 3](../../section-3/) contains an example.

### Guide to Managing Runtime Data Movement to/from Host Memory

In high-performance computing applications, efficiently managing data movement and synchronization is crucial. This guide provides a comprehensive overview of how to utilize IRON to manage data movement at runtime from/to host memory to/from the AIE array (for example, in the Ryzen™ AI NPU).

For high-level IRON constructs like `RuntimeTasks`, please continue with this [reading](./RuntimeTasks.md).

For explicitly placed, closer-to-metal IRON API functions like `npu_dma_memcpy_nd` and `dma_wait` please continue reading [here](./DMATasks.md).

-----
[[Prev - Section 2c](../section-2c/)] [[Up](..)] [[Next - Section 2e](../section-2e/)]
