<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Basic Programming Examples</ins>

These programming examples provide a good starting point to illustrate how to build commonly used compute kernels (both single-core and multi-core data processing pipelines). They serve to highlight how designs can be described in Python and lowered through the mlir-aie tool flow to an executable that runs on the NPU. [Passthrough Kernel](./passthrough_kernel) and [Vector Scalar Mul](./vector_scalar_mul) are good designs to get started with. Please see [section 3](../../programming_guide/section-3/) of the [programming guide](../../programming_guide/) for a more detailed guide on developing designs.

* [Passthrough DMAs](./passthrough_dmas) - Data movement memcpy using object FIFOs via DMAs only, without involving the AIE core.
* [Passthrough Kernel](./passthrough_kernel) - Vectorized memcpy via a single AIE core kernel.
* [Passthrough PyKernel](./passthrough_pykernel) - Memcpy where the AIE kernel is written as an inline Python function rather than a C++ external function.
* [Passthrough DMAs PLIO](./passthrough_dmas_plio) - **Targets the Xilinx VCK5000, not Ryzen AI NPU.** Demonstrates PLIO-connected soft DMAs in programmable logic.
* [DMA Transpose](./dma_transpose) - Matrix transpose using the Shim DMA with `npu_dma_memcpy_nd`.
* [DMA Transpose Packet](./dma_transpose_packet) - Matrix transpose using packet-switched DMA flows.
* [Chaining Channels](./chaining_channels) - Demonstrates chaining multiple DMA buffer descriptors in sequence on a single channel.
* [Combined Transpose](./combined_transpose) - Matrix transpose combining Shim DMA strides with AIE core VSHUFFLE instructions.
* [Shuffle Transpose](./shuffle_transpose) - Matrix transpose using only AIE core VSHUFFLE instructions.
* [Vector Scalar Add](./vector_scalar_add) - Single tile increments every element of a vector by `1`.
* [Vector Scalar Mul](./vector_scalar_mul) - Single tile performs `vector * scalar` of size `4096` in `1024`-element chunks.
* [Vector Scalar Add Runlist](./vector_scalar_add_runlist) - Vector scalar add using the run-list execution model.
* [Vector Vector Add](./vector_vector_add) - Single tile performs `vector + vector` of size `1024`.
* [Vector Vector Add BDs Init Values](./vector_vector_add_BDs_init_values) - Vector addition with buffer descriptors pre-initialized with values.
* [Vector Vector Modulo](./vector_vector_modulo) - Single tile performs `vector % vector` of size `1024`.
* [Vector Vector Multiply](./vector_vector_mul) - Single tile performs `vector * vector` of size `1024`.
* [Vector Reduce Add](./vector_reduce_add) - Single tile reduction returning the `sum` of a vector.
* [Vector Reduce Max](./vector_reduce_max) - Single tile reduction returning the `max` of a vector.
* [Vector Reduce Min](./vector_reduce_min) - Single tile reduction returning the `min` of a vector.
* [Vector Exp](./vector_exp) - Element-wise $e^x$ using the AIE look-up table capability.
* [Matrix Scalar Add](./matrix_scalar_add) - Single tile adds a scalar constant to every element of a `16x8` matrix.
* [Matrix Multiplication](./matrix_multiplication) - Single-core, multi-core (whole array), and matrix-vector multiply designs, plus sweep benchmarking infrastructure.
* [Row Wise Bias Add](./row_wise_bias_add) - Adds a bias vector to each row of a matrix using DMA tiling.
* [Event Trace](./event_trace) - Demonstrates the AIE hardware trace unit for measuring kernel cycle counts and stall events. See also [Section 4b](../../programming_guide/section-4/section-4b/) of the programming guide.
* [Packet Switch](./packet_switch) - Demonstrates packet-switched routing for multiplexing multiple data streams over shared interconnect.
* [Tiling Exploration](./tiling_exploration) - Interactive exploration of `TensorAccessPattern` and `TensorTiler2D` for n-dimensional DMA tiling. Includes visualization tools.
* [Memcpy](./memcpy) - **Exercise design.** A parameterized multi-column memcpy with an intentionally unoptimized runtime sequence. The goal is to add task groups to achieve peak bandwidth. See [getting_started/00_memcpy](../getting_started/00_memcpy/) for the reference solution.
