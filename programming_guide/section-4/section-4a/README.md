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

We begin by first looking at timers for measuring application performance and what that tells us. The performance of an accelerated AI Engine application involves a number of components on the software stack, from invoking the application at the OS level, to passing control on to the kernel drivers, moving and dispatching work to the AIE array, running the accelerated application on AIE cores, and finally returning the data to the application for next-step processing. The most straightforward way to capture the performance of this entire stack of communication and processing is with an application timer, also known as the "wall clock" time. This gives us the upper bounds for how long an AIE accelerated application takes but adds to it the OS and kernel driver overhead. This is something that can be minimized when running multiple iterations of an accelerated program or running a sufficiently compute intensive application. Let's take a look at how we add the "wall clock" timer to an example program.

## <ins>The Compact Form: ``@iron.jit`` + ``run_iters``</ins>

In a `@iron.jit`-decorated design, calling the design at the host side is what runs the kernel.  The IRON helper [`aie.utils.benchmark.run_iters`](../../../python/utils/benchmark.py) wraps that call site in a `warmup + iters` loop, captures NPU-side time (from the kernel result) and end-to-end host time, and returns a `BenchmarkResult`.  The pair is exactly what the explicit chrono-loop in `test.cpp` does — in one line:

```python
from aie.utils.benchmark import print_benchmark, run_iters

bench = run_iters(
    vector_scalar_mul, a_in, f_in, c_out,
    warmup=opts.warmup, iters=opts.iters,
)
print_benchmark(bench)
```

`[vector_scalar_mul.py](./vector_scalar_mul.py)` uses this directly; running `make run` (or `python3 vector_scalar_mul.py --iters 10 --warmup 4`) prints the canonical `avg/min/max us` pair for both NPU and end-to-end timings.

The rest of this section walks down through what's happening underneath, using the explicit chrono-loop in [test.cpp](./test.cpp) and the manual accumulator loop in [test.py](./test.py).

## <ins>Application timer - Modifying [test.cpp](./test.cpp)</ins>
Adding the application timer is as simple as noting a start and stop time surrounding the calling of the kernel function. We can use the clock timer from the chrono library which is included via `#include <chrono>` but this may already be imported by other libraries (this is the case in our `test.cpp`). Then we record the start and stop time of our chrono timer with timer function calls surrounding our kernel function as follows:

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

In addition, if you have an estimate of the number of MACs each kernel execution takes, you can report additional performance data such as GFLOPs as can be seen in the matrix multiplication example [test.cpp](../../../programming_examples/basic/matrix_multiplication/test.cpp#L295).

## <ins>Verifying NPU output: `aie.utils.verify`</ins>

Benchmarking is paired with a correctness check.  AIE kernels are
often LUT approximations or use saturating arithmetic, so the canonical
host comparator is not `np.array_equal` — too strict for bf16 or LUT
outputs — but a tolerance-aware sibling in
[`aie.utils.verify`](../../../python/utils/verify.py):

```python
from aie.utils.verify import count_mismatches, nearly_equal

# Count tolerance violations; stop on the first inf/nan from either side
# (the LUT's behaviour outside its defined input range is not part of
# the contract).
errors, n_checked = count_mismatches(actual, ref, rtol=0.05)
print(f"{errors} / {n_checked} samples outside tolerance")

# Same comparator, boolean mask form — useful when you want to inspect
# exactly which entries failed.
mask = nearly_equal(actual, ref, rtol=0.05)
```

`nearly_equal(a, b, *, rtol=0.128, atol=None)` returns a boolean
ndarray with `True` where `|a - b| < max(atol, rtol * (|a| + |b|))`.
Both inputs are coerced to `float32` (enough headroom for bf16); `NaN`
on either side yields `False` (matches IEEE semantics).  Defaults to
`rtol=0.128` (12.8%) to match the `test_utils::nearly_equal` C++
helper that `test.cpp` testbenches use, so Python and C++ host harnesses
agree on what counts as a passing run.

`count_mismatches(actual, ref, *, rtol=0.128, atol=None, stop_at_nonfinite=True)`
returns `(errors, n_checked)`.  `n_checked` is less than `len(ref)`
when `stop_at_nonfinite=True` (the default) and an `inf`/`nan` appears
— the verification halts there rather than counting every saturated
sample as a violation.  Pass `stop_at_nonfinite=False` to count
everything.

`aie.utils.verify` also exposes `assert_pass(actual, expected, *, rtol=None, atol=None, fail_msg=None)`
— a one-liner that runs the comparison, prints `PASS!` on success, and
`sys.exit("FAIL!")` on mismatch.  With both `rtol` and `atol` set to
`None` (the default), it uses `np.array_equal` for exact compare (the
right choice for integer / bit-exact pipelines); passing `rtol=` opts
into the tolerance comparator above.

## <u>Exercises</u>
1. Build + run the `@iron.jit` design with `make run` and note the reported NPU + end-to-end averages. (The defaults are `--warmup 4 --iters 10`; override via `--warmup N --iters M` on the python command line.) What did you see?

1. Build the xclbin + insts pair via `make all`, then run the explicit C++ host with `make run_cpp` and note the reported "wall clock" time. <img src="../../../mlir_exercises/images/answer1.jpg" title="Answer can be anywhere from 300-600us" height=25>

1. The design was run once with a single iteration and no warmup. Run again with `make run_cpp` — same defaults. What reported Avg NPU time did you see this time? <img src="../../../mlir_exercises/images/answer1.jpg" title="Answer can still be anywhere from 300-600us but is likely different than before" height=25>

1. Set iterations to 10 with `make run_cpp-10` (passes `--iters 10`). What reported Avg NPU time do you see this time? <img src="../../../mlir_exercises/images/answer1.jpg" title="This time, we see a narrower range between 300-400 us" height=25>

1. Finally, add 4 warmup iterations to cut higher outliers when the application is first run by calling `make run_cpp-10-warmup`. This passes `--warmup 4` to the executable. What reported Avg NPU time do you see this time? <img src="../../../mlir_exercises/images/answer1.jpg" title="This time, we see an lower average range between 200-300 us" height=25>

-----
[[Up]](../../section-4) [[Next]](../section-4b)

