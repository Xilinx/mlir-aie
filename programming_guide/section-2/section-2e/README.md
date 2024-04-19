<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2e - Practical Examples</ins>

This section introduces several examples with common Object FIFO data movement patterns. These examples are intended to be simple enough so as to be easily imported and adapted into other designs.

<details><summary><a href="./01_single_double_buffer/">Example 01 - Single / Double Buffer</a></summary>

* Core to core data movement using single / double buffer
</details>
<details><summary><a href="./02_external_mem_to_core/">Example 02 - External Memory to Core</a></summary>

* External memory to core and back using double buffers
</details>
<details><summary><a href="./03_external_mem_to_core_L2/">Example 03 - External Memory to Core through L2</a></summary>

* External memory to core and back through L2 using double buffers
</details>
<details><summary><a href="./04_distribute_L2/">Example 04 - Distribute from L2</a></summary>

*  Distribute data from external memory to cores through L2
</details>
<details><summary><a href="./05_join_L2/">Example 05 - Join in L2</a></summary>

* Join data from cores to external memory through L2
</details>

-----
[[Prev - Section 2d](../section-2d/)] [[Up](..)] [[Next - Section 2f](../section-2f/)]
