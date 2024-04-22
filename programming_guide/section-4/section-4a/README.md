<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 4a - Timers</ins>

* [Section 4 - Vector Programming & Peformance Measurement](../../section-4)
    * Section 4a - Timers
    * [Section 4b - Trace](../section-4b)
    * [Section 4c - Kernel Vectorization](../section-4c)

-----

We begin by first looking at timers for measuring application performance and what that tells us. The performance of an accelerated AI Engine application involves a number of components on the software stack, from invoking the application at the OS level, to passing control on to the kernel drivers, moving and dispatching work to the AIE array, running the accelerated application on AIE cores, and finally returning the data to the application for next-step processing. The most straightforward way to capture the performance of this entire stack of communication and processing is with an application timer, also known as the "wall clock" time. This gives us the upper bounds for how long an AIE accelerated application takes but adds to it the OS and kernel driver overhead. This is something that can be minimized when running multiple iterations of an acclerated program or running a sufficiently compute intensive application. Let's take a look at how we add the "wall clock" timer to an example program.

## <ins>Application timer - Modifying [test.cpp](./test.cpp)</ins>
Adding the application timer is as simple as noting a start and stop time surrounding the calling of the kernel function. We can use the clock timer from the chrono library which is imported via `import <chrono>` but this may already be imported by other libraries (in our `test.cpp`, this is the case). Then we record the start and stop time of our chrono timer with function calls surrounding our kernel function call like the following:

```c++
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(bo_instr, instr_v.size(), bo_inout0, bo_inout1, bo_inout2);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << "NPU time: " << npu_time << "us." << std::endl;
```
This provides us with a good baseline for how long our accelerated kernel function takes.

## <ins>Multiple iterations</ins>
A timer for a single kernel function call is a useful starting data point for understanding performance but there can be a lot of variability and overhead for a single call that is smoothed out when run multiple times. In order to benchmark the steady-state kernel run time, we can add code around our kernel call to execute multiple times and capture the minimium, maximize and average time that our kernel takes.

In our example [test.cpp](./test.cpp), we wrap our calls within a for loop (based on `num_iter`/ number of iterations). 

```c++
  unsigned num_iter = n_iterations + n_warmup_iterations;
  for (unsigned iter = 0; iter < num_iter; iter++) {
    <... kernel run code ...>
  }
```
It is also useful to run the kernel a number of times prior to recording the steady-state average. This hides initial startup timing overhead that sometimes occurs during the first few runs. We call these initial loops warmup iterations which do not include verifying results and measuring the kernel function time.
```c++
  for (unsigned iter = 0; iter < num_iter; iter++) {
    <... kernel run code ...>    
    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }
    <... verify and measure timers ...>
  }
```
Finally, we accumulate relevant timer data to calculate and track average, minimum, and maximum times.
```c++
  for (unsigned iter = 0; iter < num_iter; iter++) {
    <... kernel run code, warmup conditional, verify, and measure timers ...>
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }
```
We can then compute and print the actual average, minimum and maximum run times.
```c++
  std::cout << "Avg NPU time: " << npu_time_total / n_iterations << "us." << std::endl;
  std::cout << "Min NPU time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU time: " << npu_time_max << "us." << std::endl;
```

## <u>Exercises</u>
1. Take a look at the timer code in our example [test.cpp](./test.cpp). Then build and run the design by calling `make; make run` and note the reported average "wall clock" time. What value did you see? <img src="../../../mlir_tutorials/images/answer1.jpg" title="Answer can be anywhere from 300-600us" height=25>

1. Our design was run once with a single iteration and no warmup. Let's run our design again by calling `make run` again. What reported Avg NPU time did you see this time? <img src="../../../mlir_tutorials/images/answer1.jpg" title="Answer can still be anywhere from 300-600us but is likely different than before" height=25>

1. Let's set our iterations to 10 and run again with `make run` which recompiles our host code for `test.cpp`. What reported Avg NPU time do you see this time? <img src="../../../mlir_tutorials/images/answer1.jpg" title="Answer can be anywhere from 430-480us but is likely different than before" height=25>

1. Let's change our design and increase the loop size of our kernel by a factor of 10. This involves changing the outer loop from 8 to 80. What reported times do you see now? <img src="../../../mlir_tutorials/images/answer1.jpg" title="? us" height=25>


-----
[[Up]](../../section-4) [[Next]](../section-4b)

