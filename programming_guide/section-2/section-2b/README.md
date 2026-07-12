<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 2b - Key ObjectFifo Patterns { #key-object-fifo-patterns }

* [Section 2 - Data Movement (ObjectFifos)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * Section 2b - Key ObjectFifo Patterns
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Runtime Data Movement](../section-2d/)
    * [Section 2e - Programming for multiple cores](../section-2e/)
    * [Section 2f - Practical Examples](../section-2f/)
    * [Section 2g - Data Movement Without ObjectFifos](../section-2g/)


-----

The ObjectFifo primitive supports several data movement patterns. We will now describe each of the currently supported patterns in four subsections and provide links to more in-depth practical code examples that showcase each of them.

* **[ObjectFifo Reuse Pattern](./01_Reuse/)**
    * Reuse the unreleased objects of an ObjectFifo
* **[ObjectFifo Broadcast Pattern](./02_Broadcast/)**
    * Broadcast data from one producer to multiple consumers
* **[Implicit Copy Across ObjectFifos: Distribute & Join Patterns](./03_Implicit_Copy/)**
    * Implicit copy of data from one ObjectFifo to another
    * Distribute different pieces of the input data to multiple consumers
    * Join outputs from different producers into a bigger data tensor
* **[ObjectFifo Repeat Pattern](./04_Repeat/)**
    * Leverage ObjectFifo Link to repeat data from the producer

-----
[Prev](../section-2a/) &middot; [Top](..) &middot; [Next](../section-2c/)
