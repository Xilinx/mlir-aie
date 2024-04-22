<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 4c - Kernel Vectorization</ins>

* [Section 4 - Vector Programming & Peformance Measurement](../../section-4)
    * [Section 4a - Timers](../section-4a)
    * [Section 4b - Trace](../section-4b)
    * Section 4c - Kernel Vectorization

-----

Now that we are able to measure the total application time ([section-4a](../section-4a/)) and have examined the kernel performance via tracing ([section-4b](../section-4b)), we will take a closer look at kernel vectorization. We will be using the [vector-scalar multiply example](../../../programming_examples/basic/vector_scalar_mul/) again to illustrate kernel vectorization concepts. Go ahead and read the design example summary for [vector-scalar multiply](../../../programming_examples/basic/vector_scalar_mul/) first to get an idea of the different components of this example design. Then, let's take a closer look at the kernel source file ([scale.cc](../../../aie_kernels/aie2/scale.cc)).

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

Here, the code iterates over the input vector (`a`) and multiplies each element from the vector with a scalar value (`factor`) before storing the results in output vector (`c`). The simple C/C++ code for this consists of a for-loop, with a simple read and scalar multiply operation inside the loop.

To vectorize this, we first need to familiarize ourselves with the AIE API which abstracts the underlying AIE processor and associated low-level intrinsics with an higher level C++ API. Documentation for AIE API (2023.2 Vitis tools) can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/modules.html). To view details on the vector x scalar mutlipler, on the left pane, navigate to *AI Engine API User Guide -> API Reference -> Arithmetic* and select the first `aie::mul` which shows a `Vec * E` where `E` is an elementary data type like a scalar int. 

To be able to use this AIE API function in our kernel code, we first need to include the AIE API headers.
```C++
#include <aie_api/aie.hpp>
```

Then, we declare a vector as follows:
```C++
aie::vector<T, vec_factor> my_vector
```
* T - data type, such as `int32_t`
* vec_factor - vector size, such as 16. 

The size of the vector depends on the type. For example, the standard vector register in AIE2 is 512 bits. For `int32_t`, that means we can store 16 of them. Extending this to the other supported data types, we have the following abbreviated table:

| Data type | Vector size |
|-----------|-------------|
| int32_t   | 16 |
| int16_t   | 32 |
| int8_t   | 64 |
| int4_t   | 128 |

A more complete table of supported vectors can be found in the AIE API User Guide [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__basic__types.html). Note that if the listed data types * vector size ends up being larger than 512-bits, that just means it's stored in 2+ vector registers instead of just one.

We can load the vector register from local L1 memory with the `aie::load_v` function, defined as follows:
```C++
      T *__restrict pA1 = a;

      aie::vector<T, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
```
Here, `__restict` is used to qualify the pointer to indicate that it's a restrict pointer and therefore memory access to that pointer can be more optimally arranged by the scheduler. This is because restrict says that sequential access to the pointer will not access the same memory location and can therefore be treated as independent.

The vector load has a template argument `vec_factor` to match the one used in the `aie::vector` declaration.

Finally, we get to the `aie::mul` call which takes a vector and a scalar as arguments and stores the result in an accumulator register desginated by:
```C++
      aie::accum<acc64, vec_factor> cout
```
The accumulator data type in this case is 16x 64-bit accumulator. We store the computed results back to local memory using the vector store function `aie::store_v` as shown:
```C++
      T *__restrict pC1 = c;

      aie::store_v(pC1, cout.to_vector<T>(0)):
```
Here, the accumulator type can be shift-round-saturated back to a vector register with the `.to_vector<T>(0)` call where `T` is the vector register type and the single integer argument `(0)` is the shift amount.

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
In this first example, the vectorization strategy was relatively straight forward. Instead of iterating over a vector of values and doing a single scalar multiply, we load a vector of input values, iterate over a smaller loop to perfrom a vector*scalar operation using the AIE API functions, and then store the vector of results back to local memory.

## <u>Exercises</u>
1. Let's take a look at the trace for our vector scalar design. First, let's edit our [vector_scalar_mul design](../../../programming_examples/basic/vector_scalar_mul/) so that the [aie2.py](../../../programming_examples/basic/vector_scalar_mul/aie2.py) source file has `vectorized=False`. In the soruce code, we simply select the scalar version of the kernel function. Then run `make trace`. After the trace compilation is complete, open `parse_eventIR_vs.json` in https://ui.perfetto.dev and measure the delta between `event 0` and `event 1`. Note that in the Perfetto waveform, 1 ms is equal to 1 clock cycle. How many cycles did you measure? <img src="../../../mlir_tutorials/images/answer1.jpg" title="~10,284 cycles" height=25> 

1. Now let's turn vectorization back on by changing `vectorized=True` and rerun our build (`make clean; make trace`). Measure the delta between `event 0` and `event 1` again. What value do you see now? <img src="../../../mlir_tutorials/images/answer1.jpg" title="~625 cycles" height=25>

## Multiplier Efficiency

Let's take a closer look at hardware efficiency. In particular, we examine how often we are maximally utilizing all the multipliers in our fixed-point vector datapath. The AI Engine fixed-point vector datapath operates at 100% efficiency when we can schedule a vector MAC every clock cycle. The MAC itself operates at 100% efficiency if all the MAC units are being used. The overall MAC utilization efficiency then is a product of these two percentages. For example, if we have 1x vector MAC every 2 cycles, and we use 50% of our vector MACs each cycle, then 50% * 50% is a total mac efficiency of 25%.

The AIE fixed-point vector datapath is optimized for matrix multiplication, and as a result, has a post-addition block with a smaller number of output lanes. What this means is that element-wise MACs generally run less efficiently since the hardware does not have enough output lanes for all MACs. So for 16-bit x 16-bit, we can do 64x MACs but with 32 output lanes, element-wise multiply for 16-bit x 16-bit has a MAC utilization efficiency of 50%. 

If we examine the matrix multiplication mode table in the AIE API User Guide [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html), we see that for 16-bit x 16-bit matmul in AIE2 (aka AIE-ML), we support 4x4x4 mode which does 64 MACs each cycle. Another way to see the total number of MACs for different bit precisions is the `Table: Supported Precision Width of the Vector Data Path` in the [AM020 spec](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Functional-Overview). There, we again see 64 MACs for 16-bit x 16-bit. For 32-bit x 32-bit, we have 16 MACs per cycle. So element-wise multiply for 32-bit x 32-bit does in fact have a per cycle MAC utilization efficiency of 100% which is not the case for element-wie multiplies of smaller bit precisions.

Going back to our vector-scalar design, we are processing 4096 samples totak, but only 1024 samples every iteration of our kernel. From the previous exercises, you saw that the scalar implementation takes about ~10,000 cycles to process while the vector implementaion only takes ~600 cycles. This is an speedup factor of over **16X**! That's a signficant gain but we'd like to look more closely to see if we can squeeze out even more performance.

### Crunching the numbers
Looking at the optimal MAC utilziation for 32-bit x 32-bit, we expect 1024 cycles to actually only take 64 cycles (1024/ 16) if it were possible to do a vector MAC every cycle. This seemingly gives our vector implementation a total MAC utilization efficiency of ~11%. In a non-ideal kernel, we have cycle overheads in the way of loop preamble and postamble as well as general function overhead. It is also true that these overheads become a smaller percentage of the total compute time if we are process a larger set of data. But this static overhead does not fully explain the unexpectedly small MAC utilization efficiency we see at first glance.

### Data movement

The other consideration when looking at vectorization and performance is data movement. The time to compute must be balanced with the time to move data so that neither becomes the bottleneck. Looking at our example once again, we are moving data via objectFifos through the stream switch. Each stream switch channel moves 32-bits of data every clock cycle. This means to move 1024 32-bit data, it would require 1024 cycles. This makes it seem like our kernel throughput should be 1024 cycles. 

But in our example, we are actually ping-ponging our data movement so that we are moving the next set of 1024 words while the first set is being computed on. So we would expect the kernel to be able to compute the data in a fraction of the data movement time (though we are still limited to 1024 cycles throughput). The real reason our compute is larger than 64 cycles is because both sets of data in our objectFifo are in the same local memory bank and thus access conflicts are occuring, which increases our total compute time. Also, the actual inner loop microcode schedule from the compiler is closer to 80% efficient instead of 100%. All these factors then add up to a total compute cycle count that's larger than our back-of-the-envelope ideal number.

## Conclusions

Having walked through this example, we can see the importance of matching the algorithm to the hardware in order to achieve maximum utilization efficiency. This generally means matmul style ops gives us the best MAC efficiency, not only because it matches the built-in vector matmul in the fixed-point vector datapath, but also because matmul has higher data re-use which lowers the data movement component so that data movement time and compute time are more closely aligned. 


-----
[[Prev]](../section-4b) [[Up]](../../section-4) [[Next - Section 5]](../../section-5)

