<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 4 - Vector programming & Peformance Measurement</ins>

Now that you've had a chance to walk through the components of compiling and running a program on the Ryzen AI hardware in [section-3](../section-3), we will start looking at how we measure performance and utilize vector programming technqiues to fully leverage the power of the AI Engines for parallel compute.


## <ins>Performance Measurement</ins>
It's helpful to first examine perfomance measurement before we delve into vector programming in order to get a baseline for where our application performance is. There are many factors that contribute to performance including latency, throughput and power efficiency. Performance measurement is an active area of research to provide more powerful tools for users to measure the acceleration of their appication on AIEs. 

* [Section 4a - Timers](./section-4a)
* [Section 4b - Trace](./section-4b)
* [Section 4c - Kernel vectorization](./section-4c)
* [Section 4d - Automated vectorization](./section-4d)


* Section-4a
    * wall clock
* Section-4b
    * trace - breakdown mechnisms
* Section-4c
    * vectorization - vector-scalar
* Sectin-4d
    * auto-vectorizer

* Introduce performance measurement (trace) and how we measure cycle count and efficiency
    * Example application and view trace
    * Talk about throughput with wall clock
* Vector Scalar design example
    * how to vectorize
* Auto-vectorizer



* Discuss topic of vector programming at the kernel level
    * Using vector scalar example