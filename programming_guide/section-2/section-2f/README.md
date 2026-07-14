<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 2f - Practical Examples

* [Section 2 - Data Movement (ObjectFifos)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key ObjectFifo Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Runtime Data Movement](../section-2d/)
    * [Section 2e - Programming for multiple cores](../section-2e/)
    * Section 2f - Practical Examples
    * [Section 2g - Data Movement Without ObjectFifos](../section-2g/)

-----

This section introduces several examples with common ObjectFifo data movement patterns. These examples are intended to be simple enough so as to be easily imported and adapted into other designs.

* **[Example 01 - Single / Double Buffer](./01_single_double_buffer/)**
    * Core to core data movement using single / double buffer
* **[Example 02 - External Memory to Core](./02_external_mem_to_core/)**
    * External memory to core and back using double buffers
* **[Example 03 - External Memory to Core through L2](./03_external_mem_to_core_L2/)**
    * External memory to core and back through L2 using double buffers
* **[Example 04 - Distribute from L2](./04_distribute_L2/)**
    * Distribute data from external memory to cores through L2
* **[Example 05 - Join in L2](./05_join_L2/)**
    * Join data from cores to external memory through L2

-----
[Prev](../section-2e/) &middot; [Top](..) &middot; [Next](../section-2g/)
