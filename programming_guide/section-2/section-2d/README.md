<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 2d - Runtime Data Movement

* [Section 2 - Data Movement (ObjectFifos)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key ObjectFifo Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * Section 2d - Runtime Data Movement
    * [Section 2e - Programming for multiple cores](../section-2e/)
    * [Section 2f - Practical Examples](../section-2f/)
    * [Section 2g - Data Movement Without ObjectFifos](../section-2g/)

-----

In the preceding sections, we looked at how we can describe data movement between tiles *within* the AIE-array. However, to do anything useful, we need to get data from outside the array, i.e., from the "host", into the AIE-array and back. On NPU devices, we can achieve this with the operations described in this section. 

The operations that will be described in this section must be placed in a separate `sequence()` of a `Runtime` class, or `aie.runtime_sequence` operation in the AIE dialect. The arguments to this function describe buffers that will be available on the host side; the body of the function describes how those buffers are moved into the AIE-array. [Section 3](../../section-3/) contains an example.

### Guide to Managing Runtime Data Movement to/from Host Memory

In high-performance computing applications, efficiently managing data movement and synchronization is crucial. This guide provides a comprehensive overview of how to utilize IRON to manage data movement at runtime from/to host memory to/from the AIE array (for example, in the Ryzen™ AI NPU).

For IRON constructs like `RuntimeTasks`, please continue with this [reading](./RuntimeTasks.md).

For the AIE dialect functions like `npu_dma_memcpy_nd` and `dma_wait` please continue reading [here](./DMATasks.md).

-----
[Prev](../section-2c/) &middot; [Top](..) &middot; [Next](../section-2e/)
