<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 4 - Performance Measurement & Vector Programming

Now that you've had a chance to walk through the components of compiling and running a program on the Ryzen™ AI hardware in [section-3](../section-3), we will start looking at how we measure performance and utilize vector programming techniques to fully leverage the power of the AI Engines for parallel compute.

It is helpful to first examine performance measurement before we delve into vector programming in order to get a baseline for where our application performance is. There are many factors that contribute to performance including latency, throughput and power efficiency. Performance measurement is an active area of research to provide more powerful tools for users to measure the speedup of their application on AIEs. In [section-4a](./section-4a) and [section-4b](./section-4b/), we look at performance from the perspective of timers and trace. Then in [section-4c](./section-4c), we look more closely at how to vectorize AIE kernel code.

* [Section 4a - Timers](./section-4a)
* [Section 4b - Trace](./section-4b)
* [Section 4c - Kernel vectorization and optimization](./section-4c)

-----
[Prev](../section-3/) &middot; [Top](..) &middot; [Next](../section-5/)