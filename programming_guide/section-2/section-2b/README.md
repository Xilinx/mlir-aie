<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2b - Key Object FIFO Patterns</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * Section 2b - Key Object FIFO Patterns
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Programming for multiple cores](../section-2d/)
    * [Section 2e - Practical Examples](../section-2e/)
    * [Section 2f - Data Movement Without Object FIFOs](../section-2f/)
    * [Section 2g - Runtime Data Movement](../section-2g/)

-----

The Object FIFO primitive supports several data movement patterns. We will now describe each of the currently supported patterns in three subsections and provide links to more in-depth practical code examples that showcase each of them.

<details><summary><a href="./01_Reuse/">Object FIFO Reuse Pattern</a></summary>

* Reuse the unreleased objects of an Object FIFO
</details>
<details><summary><a href="./02_Broadcast/">Object FIFO Broadcast Pattern</a></summary>

* Broadcast data from one producer to multiple consumers
</details>
<details><summary><a href="./03_Link_Distribute_Join/">Object FIFO Distribute &amp; Join Patterns with Object FIFO Link</a></summary>

* Implicit copy of data from one Object FIFO to another via an Object FIFO Link
* Distribute different pieces of the input data to multiple consumers 
* Join outputs from different consumers into a bigger data tensor
</details>

-----
[[Prev - Section 2a](../section-2a/)] [[Up](..)] [[Next - Section 2c](../section-2c/)]
