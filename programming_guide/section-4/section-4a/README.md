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

* [Section 4 - Performance Measurement & Vector Programming](../../section-4)
    * Section 4a - Timers
    * [Section 4b - Trace](../section-4b)
    * [Section 4c - Kernel Vectorization and Optimization](../section-4c)

-----

We begin by first looking at timers for measuring application performance and what that tells us. The performance of an accelerated AI Engine application involves a number of components on the software stack, from invoking the application at the OS level, to passing control on to the kernel drivers, moving and dispatching work to the AIE array, running the accelerated application on AIE cores, and finally returning the data to the application for next-step processing. The most straightforward way to capture the performance of this entire stack of communication and processing is with an application timer, also known as the "wall clock" time. This gives us the upper bounds for how long an AIE accelerated application takes but adds to it the OS and kernel driver overhead. This is something that can be minimized when running multiple iterations of an acclerated program or running a sufficiently compute intensive application. Let's take a look at how we add the "wall clock" timer to an example program.

## <ins>Application timer - Modifying [test.cpp](./test.cpp)</ins>
Adding the application timer is as simple as noting a start and stop time surrounding the calling of the kernel function. We can use the clock timer from the chrono library which is imported via `import <chrono>` but this may already be imported by other libraries (this is the case in our `test.cpp`). Then we record the start and stop time of our chrono timer with timer function calls surrounding our kernel function as follows:

```c++
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inFactor, bo_outC);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << "NPU time: " << npu_time << "us." << std::endl;
```
This provides us with a good baseline for how long our accelerated kernel function takes.

## <ins>Multiple iterations</ins>
A timer for a single kernel function call is a useful starting point for understanding performance but there can be a lot of variability and overhead for a single call that is smoothed out when run multiple times. In order to benchmark the steady-state kernel run time, we can add code around our kernel call to execute multiple times and capture the minimium, maximum, and average time that our kernel takes.

In our example [test.cpp](./test.cpp), we wrap our calls within a for loop (based on `num_iter` or number of iterations).

```c++
  unsigned num_iter = n_iterations + n_warmup_iterations;
  for (unsigned iter = 0; iter < num_iter; iter++) {
    <... kernel run code ...>
  }
```
It is also useful to run the kernel a number of times prior to calculating the steady-state run times. This hides initial startup timing overhead that sometimes occurs during the first few runs. We call these initial loops warmup iterations, where we do not verify the results or measure the run time during warmup.
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
We can then compute and print the actual average, minimum and maximum run times at the end of our host code.
```c++
  std::cout << "Avg NPU time: " << npu_time_total / n_iterations << "us." << std::endl;
  std::cout << "Min NPU time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU time: " << npu_time_max << "us." << std::endl;
```

In addition, if you have an estimate of the number of MACs each kernel execution takes, you can report additional performance data such as GFLOPs as can be seen in the matrix multiplication example [test.cpp](../../../programming_examples/basic/matrix_multiplication/test.cpp#L170).

## <u>Exercises</u>
1. Take a look at the timer code in our example [test.cpp](./test.cpp). Then build and run the design by calling `make run` and note the reported average "wall clock" time. What value did you see? <img src="../../../mlir_tutorials/images/answer1.jpg" title="Answer can be anywhere from 300-600us" height=25>

1. Our design was run once with a single iteration and no warmup. Let's run our design again by calling `make run` again. What reported Avg NPU time did you see this time? <img src="../../../mlir_tutorials/images/answer1.jpg" title="Answer can still be anywhere from 300-600us but is likely different than before" height=25>

1. Let's set our iterations to 10 and run again with `make run-10` which passes in the argument `--iters 10` to our executable. What reported Avg NPU time do you see this time? <img src="../../../mlir_tutorials/images/answer1.jpg" title="This time, we see a narrower range between 300-400 us" height=25>

1. Finally, let's add a 4 warmup iterations to cut higher outliers when the application is first run by calling `make run-10-warmup`. This passes in the `--warmup 4` to our executable. What reported Avg NPU time do you see this time? <img src="../../../mlir_tutorials/images/answer1.jpg" title="This time, we see an lower average range between 200-300 us" height=25>

-----
[[Up]](../../section-4) [[Next]](../section-4b)

