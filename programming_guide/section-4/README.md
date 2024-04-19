<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 4 - Vector Programming & Performance Measurement</ins>

Now that you've had a chance to walk through the components of compiling and running a program on the Ryzen AI hardware in [section-3](../section-3), we will start looking at how we measure performance and utilize vector programming technqiues to fully leverage the power of the AI Engines for parallel compute.

It's helpful to first examine perfomance measurement before we delve into vector programming in order to get a baseline for where our application performance is. There are many factors that contribute to performance including latency, throughput and power efficiency. Performance measurement is an active area of research to provide more powerful tools for users to measure the speedup of their appication on AIEs. In [section-4a](./section-4a) and [section-4b](./section-4b/), we look a performance from the perspective of timers and trace. Then in [section-4c](./section-4c), we look more closely at how to vectorize AIE kernel code.

* [Section 4a - Timers](./section-4a)
* [Section 4b - Trace](./section-4b)
* [Section 4c - Kernel vectorization](./section-4c)

-----
[[Prev - Section 3](../section-3/)] [[Top](..)] [[Next - Section 5](../section-5/)]