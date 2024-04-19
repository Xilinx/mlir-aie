<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 4c - Kernel vectorization</ins>

* [Section 4 - Vector Programming & Peformance Measurement](../../section-4)
    * [Section 4a - Timers](./section-4a)
    * [Section 4b - Trace](./section-4b)
    * [Section 4c - Kernel vectorization](./section-4c)

Now that we are able to measure the general application time ([section-4a](../section-4a/)) and have measured the kernel performance via tracing ([section-4b](../section-4b)), we will take a closer look at kernel vectorization. We will be using the [vector-scalar multiply example](../../../programming_examples/basic/vector_scalar_mul/) to illustrate kernel vectorization concepts. Go ahead and read the design example summary [here](../../../programming_examples/basic/vector_scalar_mul/) first to get an idea of the different components of our example design. We will then zoom into the examining the [scale.cc](../../../aie_kernels/aie2/scale.cc) kernel source file.

In [scale.cc](../../../aie_kernels/aie2/scale.cc), we see that the scalar code is relatively straight forward:
```C++
template <typename T>
void scale_scalar(T *a, T *c, T factor, const int32_t N) {
  event0();
  for (int i = 0; i < N; i++) {
    c[i] = factor * a[i];
  }
  event1();
}
```

Here, the code iterate over the input vector (`a`) and multiplies each element from the vector with a scalar value (`factor`) before storing the results in output vector (`c`). The simple C/C++ code for this consists of a for-loop, with a simple read and scalar multiply operation inside the loop.

To vectorize this, we first need to familiarize ourselves with the AIE API which abstracts the underlying AIE hardware and the optimal intrinsics that execute on thathardware with an abstracted C++ API. Documentation for AIE API for 2023.2 Vitis tools can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/modules.html). On the left pane, navigate to <u>AI Engine PI User Guide -> API Reference -> Arithmetic</u> and select the first `aie::mul` which shows a `Vec * E` where `E` is an elementary data type like a scalar int. 

To be able to use this AIE API function in our kernel code, we first need to include the AIE API headers.
```C++
#include <aie_api/aie.hpp>
```

Then, we declare vectors as follows:
```C++
aie::vector<T, vec_factor> my_vector
```
* T - type, such as `int32_t`
* vec_factor - vector size, such as 16. 

The size of the vector depends on the type. For example, the standard vector register in AIE2 is 512 bits. For int32_t, that means we can store 16 of them. Extending this for other supported data types, we have the following table:

| Data type | Vector size |
|-----------|-------------|
| int32_t   | 16 |
| int16_t   | 32 |
| int8_t   | 64 |
| int4_t   | 128 |

A more complete table of supported vectors can be found in the AIE API User Guide [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__basic__types.html). Note that if the listed data types * vector size ends up being larger than 512-bits, that just means it's stored in 2+ vector registers instead of just one.

We can load the vector register from local L1 memory with the `aie::load_v` function, defeined as follows:
```C++
  T *__restrict pA1 = a;
  ...
      aie::vector<T, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
```
Here, `__restict` is used to qualify the pointer to indicate that it's restrict pointer and can therefore be more optimally scheduled as sequential access to the pointer within the code will not access the same memory location and is therefore independent.

The vector load has a template argument `vec_factor` to match the one used in the `aie::vecor` declaration.

Finally, we get to the `aie::mul` call which takes the vector and scalar as arguments and stores the result in an accumulator desginated by:
```C++
      aie::accum<acc64, vec_factor> cout
```
The accumulator data type in this case is 16x 64-bit accumulator. 
Finally, we store the results using the vector store function `aie::store_v` as shown:
```C++
      aie::store_v(pC1, cout.to_vector<T>(0)):
```
Here, the accumulator type can be shift-round-saturated back to a vector register with the `.to_vector<T>(0)` call where `T` is the vector register type and the argument is the shift amount.

The entire vector block is then:
```C++
template <typename T>
void scale_vectorized(T *a, T *c, T factor, const int32_t N) {
  constexpr int vec_factor = 16;
  event0();
  T *__restrict pA1 = a;
  T *__restrict pC1 = c;
  const int F = N / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::accum<acc64, vec_factor> cout = aie::mul(A0, factor);
      aie::store_v(pC1, cout.to_vector<T>(0));
      pC1 += vec_factor;
    }
  event1();
}
```
In this first example, the vectorization strategy was straight forward. Intead of iterating over a vector of values and doing a single scalar multiply, we load a vector of input values, perfrom a vector*scalar operation using the AIE API functions, and then store the vector of results.

## <u>Exercises</u>
1. Let's take a look at the trace for our vector scalar design. First, let's make sure our [vector_scalar_mul design](../../../programming_examples/basic/vector_scalar_mul/) has `vectorized=False` in the [aie2.py](../../../programming_examples/basic/vector_scalar_mul/aie2.py) source file. Then run `make trace`. Then measure the time between event 0 and event 1 in Perfetto. Note that in the waveform, 1 ms is about 1 cycle. How many cycles did you measure? <img src="../../../mlir_tutorials/images/answer1.jpg" title="~10,284 cycles" height=25> 

1. Now let's set the `vectorized=True` and rerun our design in trace mode and measure the distance bewteen event 0 and 1. What value do you see now? <img src="../../../mlir_tutorials/images/answer1.jpg" title="~625 cycles" height=25>

## Multiplier efficiency

Let's take a closer look at hardware efficiency. In particular, we examine how often we are maximally utilizaing all multipliers in our fixed-point vector datapath. The AI Engine fixed-point vector datapath operates at 100% efficiency when we can schedule a vector MAC every clock cycle. The MAC itself operates at 100% efficiency if all the MAC units are being used. Because the hardware is optimized for matrix multiplication, our fixed-point vector datapath has post-additions with a smaller number of output lanes. What this means is that element-wise MACs generally run less efficiently since the hardware does not have enough output lanes for all MACs. So for 16-bit x 16-bit, we can do 64x MACs but with 32 output lanes, element-wise multiply for 16-bit x 16-bit is at 50% per cycle MAC utilization efficiency. 

If we examine the Matrix Multiplication mode table in the AIE API User Guide [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html), we see that for 16-bit x 16-bit real matmul in AIE2 (aka AIE-ML), we support 4x4x4 mode which is 64x MACs. Another way to see the total number of MACs for different bit precisions is the `Table: Supported Precision Width of the Vector Data Path` in the [AM020 spec](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Functional-Overview). There, we see again see 64x MACs for 16-bit x 16-bit. For 32-bit x 32-bit, we have 16x MACs. So element-wise multily fo 32-bit x 32-bit does achieve 100% per cycle MAC utilization efficiency.

Going back to our vector-scalar design, we are iterating over 4096 samples in our test case but only 1024 samples every invocation of our kernel. The scalar implementation takes about ~10,000 cycles to process while the vector implementaion only takes ~600 cycles. This is an speedup factor of about 16x! That's a signficant gain but we'd like to look more closely to see if we can squeeze out even more performance.

Looking at the optimal hardware utilziation for 32-bit x 32-bit MAC, we expect 1024 cycles to take 64 cycles (1024/ 16) if it were possible to do a vector MAC every cycle. This gives our vector implementation a MAC utilization efficiency of about 11%. Whiel this does includes loop preamble and postamble overhead along with function overhead and it is true that these overheads would make up a smaller percentage of the total compute time if we are processing a larger set of data, this doesn't fully account for the trace results we see.

## Data movement

The other consideration when looking at vectorization and performance is data movement. The time to compute must be balanced with the time to move data so that neither becomes the bottleneck. Looking at our example once again, we are moving data via objectFifos through the stream switch which moves 32-bits of data every clock cycle. This means to move 1024 32-bit data, it would require 1024 cycles. But we are actually ping-ponging our data movement so that we're moving the next set of 1024 words while the first set is being computed on. So we would expect the kernel to be able to compute the data in a fraction of the data movement time (though we are admittedly limited to 1024 cycles for throughput). The real reason our compute is larger than 64 cycles is because both objects in our objectFifo are in the same local memory bank and thus access conflicts are occuring, which increases our total compute time. Also, the actual micrcode schedule that the compiler achieved is not 100% efficient in the inner loop. All these factors then are at play, preventing us from getting closer to the ideal schedule. However, this does not prevent us from seeing signficant gains over the scalar implementation. 

## Conclusions

Having walked through this example, we can see the importance of matching the algorithm to the hardware in order to ahieve maximum utilization efficiency. This means matmul style ops work best, not only because it matches the built-in matmul in the fixed-point vector datapath, but also because matmul has higher data re-use which brings down the data movement time so that data movement time and compute time are more closely aligned. 


-----
[[Prev]](../section-4b) [[Up]](../../section-4) [[Next - Section 5]](../../section-5)

