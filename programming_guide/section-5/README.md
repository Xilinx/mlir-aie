<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 5 - Example Vector Designs</ins>

The [programming examples](../../programming_examples) are a number of sample designs that further help explain many of the unique features of AI Engines and the NPU array in Ryzen™ AI.

## Simplest

#### Passthrough

The [passthrough](../../programming_examples/basic/passthrough_kernel/) example is the simplest "getting started" example.  It copies 4096 bytes from the input to output using vectorized loads and stores.  The design example shows a typical project organization which is easy to reproduce with other examples.  There are only really 3 important files here.
1. [`passthrough_kernel.py`](../../programming_examples/basic/passthrough_kernel/passthrough_kernel.py) The IRON structural design plus the host-side test driver. Decorated with `@iron.jit` so the first call compiles the design and runs it on the NPU, then verifies the result against the input. Also shows a simple use of the Object FIFOs described in [section 2](../section-2).
1. [`passThrough.cc`](../../aie_kernels/generic/passThrough.cc)  This is a C++ file which performs the vectorized copy operation.
1. [`Makefile`](../../programming_examples/basic/passthrough_kernel/Makefile) A Makefile documenting (and implementing) the build process for the various artifacts.

The [passthrough DMAs](../../programming_examples/basic/passthrough_dmas/) example shows an alternate method of performing a copy without involving the cores, and instead performing a loopback.

## Basic

| Design name | Data type | Description |
|-|-|-|
| [Vector Scalar Add](../../programming_examples/basic/vector_scalar_add/) | i32 | Adds 1 to every element in  vector |
| [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) | i32 | Returns a vector multiplied by a scale factor |
| [Vector Vector Add](../../programming_examples/basic/vector_vector_add/) | i32 | Returns a vector summed with another vector |
| [Vector Vector Modulo](../../programming_examples/basic/vector_vector_modulo/) | i32 | Returns vector % vector |
| [Vector Vector Multiply](../../programming_examples/basic/vector_vector_mul/) | i32 | Returns a vector multiplied by a vector |
| [Vector Reduce Add](../../programming_examples/basic/vector_reduce_add/) | bfloat16 | Returns the sum of all elements in a vector |
| [Vector Reduce Max](../../programming_examples/basic/vector_reduce_max/) | bfloat16 | Returns the maximum of all elements in a vector |
| [Vector Reduce Min](../../programming_examples/basic/vector_reduce_min/) | bfloat16 | Returns the minimum of all elements in a vector |
| [Vector Exp](../../programming_examples/basic/vector_exp/) | bfloat16 | Returns a vector representing e<sup>x</sup> of the inputs |
| [DMA Transpose](../../programming_examples/basic/transposes/) (using `--strategy=dma`) | i32 | Transposes a matrix with the Shim DMA using `npu_dma_memcpy_nd` |
| [Matrix Scalar Add](../../programming_examples/basic/matrix_scalar_add/) | i32 | Returns a matrix multiplied by a scalar |
| [Single core GEMM](../../programming_examples/basic/matrix_multiplication/single_core/) | bfloat16 | A single core matrix-matrix multiply |
| [Multi core GEMM](../../programming_examples/basic/matrix_multiplication/whole_array/) | bfloat16 | A matrix-matrix multiply using 16 AIEs with operand broadcast.  Uses a simple "accumulate in place" strategy |
| [GEMV](../../programming_examples/basic/matrix_multiplication/matrix_vector/) | bfloat16 | A vector-matrix multiply returning a vector |

## Machine Learning Kernels

| Design name | Data type | Description | 
|-|-|-|
| [Eltwise (Add / Mul)](../../programming_examples/ml/eltwise/) | bfloat16 | Element-wise addition or multiplication of two vectors (`op={add,mul}` knob). |
| [Eltwise Unary (ReLU / SiLU / GELU)](../../programming_examples/ml/eltwise_unary/) | bfloat16 | Element-wise ReLU, SiLU, or GELU activation on a vector (`op={relu,silu,gelu}` knob). |
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  |
| [Conv2D (optional fused ReLU)](../../programming_examples/ml/conv2d/) | i8 | 1x1 Conv2D for CNNs; `fuse_relu=1` swaps the output to uint8 saturation, fusing ReLU at the vector register level. |

## Exercises

1. Can you modify the [passthrough](../../programming_examples/basic/passthrough_kernel/) design to copy more (or less) data? <img src="../../mlir_exercises/images/answer1.jpg" title="Check the Makefile...in1_size and out_size" height=25>

1. Take a look at the host driver in our [Vector Exp](../../programming_examples/basic/vector_exp/) example [vector_exp.py](../../programming_examples/basic/vector_exp/vector_exp.py). Take note of the data type and the size of the test vector. What do you notice? <img src="../../mlir_exercises/images/answer1.jpg" title="We are testing 65536 values or 2^16, therefore testing all possible bfloat16 values through the approximation." height=25>

1. What is the communication-to-computation  ratio in [Eltwise Unary (ReLU)](../../programming_examples/ml/eltwise_unary/)? <img src="../../mlir_exercises/images/answer1.jpg" title="~6 as reported by the Trace. This is why it is a good candidate for kernel fusion with Conv2D or GEMMs for ML." height=25>

1. **HARD** Which basic example is a component in [Softmax](../../programming_examples/ml/softmax/)? <img src="../../mlir_exercises/images/answer1.jpg" title="[Vector Exp](../../programming_examples/basic/vector_exp/)" height=25>

-----
[[Prev - Section 4](../section-4/)] [[Top](..)] [[Next - Section 6](../section-6/)]
