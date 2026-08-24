<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# AIE Kernels

These kernels are provided as example building blocks for larger designs, and also as illustrations of how to write single core programs for AIEs which can then be duplicated or mixed into multi-core designs using the structural IRON API.

In some cases, the kernels are just generic C code, and will run on any family of AI Engines with varying performance.  Other kernels are then optimized for the AIE1 and AIE2 architectures.  Finally, some kernels use the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics, and whose documentation can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html), while others use the architecture specific low-level intrinsics directly

> **NOTE:** this set of AIE kernels are meant for demonstration along with the programming examples. The goal is not to be 100% performant, there may be room for further improvement. The kernels are provided as-is with no guarantees of support of AMD or AMD Research and Advanced Development.

## Generic
| Class | Name | Coding style | Purpose | Datatypes |
|-|-|-|-|-|
| basic | [passThrough.cc](./generic/passThrough.cc) | AIE API | A simple memcpy operation | `uint8_t`, `int16_t`, `int32_t` |
| data movement | [transpose.cc](./generic/transpose.cc) | AIE API | Blocked matrix transpose (4×4 / 8×8 sub-tiles, VSHUFFLE) | `bfloat16` |
| data movement | [expand.cc](./generic/expand.cc) | AIE API | int4→bf16 dequant with per-group scale factors | `int4`→`bfloat16` |
| gemv | [mv.cc](./generic/mv.cc) | AIE API | Matrix/Vector multiply | `bfloat16` |
| blas | [axpy.cc](./generic/axpy.cc) | AIE API | `z = a*x + y` (SAXPY) | `bfloat16` |

## AIE1
| Name | Coding style | Purpose |
|-|-|-|

## AIE2
| Class | Name | Coding style | Purpose | Datatypes |
|-|-|-|-|-|
| basic | [zero.cc](./aie2/zero.cc) | AIE API | Fill a tensor with zeroes | template |
| basic | [add.cc](./aie2/add.cc) | AIE API | Pointwise addition of 2 tensors | `bfloat16` |
| basic | [mul.cc](./aie2/mul.cc) | AIE API | Pointwise multiplication of 2 tensors | `bfloat16` |
| basic | [scale.cc](./aie2/scale.cc) | AIE API | Scale all elements of a tensor with a scale factor | `int32_t` |
| basic | [scale_shift.cc](./aie2/scale_shift.cc) | AIE API | Scale-and-shift | `int32_t` |
| basic | [bitwiseOR.cc](./aie2/bitwiseOR.cc) | AIE API | Bitwise OR of fixed point tensors | `uint8_t`,`int16_t`,`int32_t`|
| basic | [bitwiseAND.cc](./aie2/bitwiseAND.cc) | AIE API | Bitwise AND of fixed point tensors | `uint8_t`,`int16_t`,`int32_t` |
| gemm  | [mm.cc](./aie2/mm.cc) | AIE API | Matrix/Matrix multiplication | `int8_t`,`int16_t`,`bfloat16` |
| gemm  | [mv.cc](./aie2/mv.cc) | AIE API | Matrix/Vector multiplication | `int16_t`→`int32_t` |
| gemm  | [cascade_mm.cc](./aie2/cascade_mm.cc) | AIE API | Cascade Matrix/Matrix multiply (multi-core) | `int16_t`,`bfloat16` |
| |
| reduction | [reduce_add.cc](./aie2/reduce_add.cc) | Intrinsics | Sum of elements in a tensor | `int32_t` |
| reduction| [reduce_max.cc](./aie2/reduce_max.cc) | Intrinsics | Max value across a tensor | `int32_t` |
| reduction | [reduce_min.cc](./aie2/reduce_min.cc) | Intrinsics | Min value across a tensor | `int32_t` |
| |
| activation | [relu.cc](./aie2/relu.cc) | Intrinsics | ReLU activation | `bfloat16` |
| activation | [leaky_relu.cc](./aie2/leaky_relu.cc) | AIE API | Leaky ReLU activation | `bfloat16` |
| activation | [gelu.cc](./aie2/gelu.cc) | AIE API | GELU activation (tanh approx) | `bfloat16` |
| activation | [silu.cc](./aie2/silu.cc) | AIE API | SiLU / Swish activation | `bfloat16` |
| activation | [swiglu.cc](./aie2/swiglu.cc) | AIE API | SwiGLU gated activation | `bfloat16` |
| activation | [tanh.cc](./aie2/tanh.cc) | AIE API | Tanh activation (LUT) | `bfloat16` |
| activation | [sigmoid.cc](./aie2/sigmoid.cc) | AIE API | Sigmoid activation (LUT) | `bfloat16` |
| activation | [softmax.cc](./aie2/softmax.cc) | AIE API | Softmax | `bfloat16` |
| activation | [bf16_softmax.cc](./aie2/bf16_softmax.cc) | AIE API | Softmax (bf16 variant) | `bfloat16` |
| activation | [bf16_exp.cc](./aie2/bf16_exp.cc) | AIE API | Element-wise `e^x` | `bfloat16` |
| |
| ml | [conv2dk1_i8.cc](./aie2/conv2dk1_i8.cc) | AIE API | 1x1 Conv2D | `int8_t` |
| ml | [conv2dk1.cc](./aie2/conv2dk1.cc) | AIE API | 1x1 Conv2D with fused ReLU | `int8_t`, `uint8_t` |
| ml | [conv2dk3.cc](./aie2/conv2dk3.cc) | AIE API | 3x3 Conv2D with fused ReLU | `int8_t`, `uint8_t` |
| ml | [conv2dk1_skip.cc](./aie2/conv2dk1_skip.cc) | AIE API| 1x1 Conv2D with fused skip addition | `int8_t`, `uint8_t` |
| ml | [conv2dk1_skip_init.cc](./aie2/conv2dk1_skip_init.cc) | AIE API | 1x1 Conv2D with fused 1x1 Conv2D skip addition | `int8_t`, `uint8_t` |
| ml | [bottleneck/](./aie2/bottleneck) | AIE API | BatchNorm-fused bottleneck conv set (`bn_*`) | `int8_t`, `uint8_t` |
| |
| vision | [gray2rgba.cc](./aie2/gray2rgba.cc) | AIE API | Convert from grayscale to RGBA format | `uint8_t` |
| vision |[rgba2gray.cc](./aie2/rgba2gray.cc) | AIE API | Convert from RGBA format to grayscale | `uint8_t` |
| vision | [rgba2hue.cc](./aie2/rgba2hue.cc) | AIE API | Convert from RGBA to hue | `uint8_t` |
| vision | [addWeighted.cc](./aie2/addWeighted.cc) | AIE API | Fixed point weighted sum of two tensors | `uint8_t` |
| vision | [threshold.cc](./aie2/threshold.cc) | AIE API | Clipping | `uint8_t` |
| vision | [filter2d.cc](./aie2/filter2d.cc) | AIE API | Fixed point 2D image processing filter | `uint8_t` |

## AIE2P
| Class | Name | Coding style | Purpose | Datatypes |
|-|-|-|-|-|
| basic | [zero.cc](./aie2p/zero.cc) | AIE API | Fill a tensor with zeroes (512-bit stores) | template |
| gemm | [mm.cc](./aie2p/mm.cc) | AIE API | Matrix/Matrix multiplication | `int8_t`,`int16_t`,`bfloat16` |
| gemm | [mm_bfp.cc](./aie2p/mm_bfp.cc) | AIE API | Block-floating-point matmul | `bfp16` |
| gemm | [mm_bfp_mixed.cc](./aie2p/mm_bfp_mixed.cc) | AIE API | Mixed-precision BFP matmul | `bfp16` |
| gemm | [mm_activation_epilogue.cc](./aie2p/mm_activation_epilogue.cc) | AIE API | Matmul with fused activation epilogue | `bfloat16` |
| |
| activation | [gelu.cc](./aie2p/gelu.cc) | AIE API | GELU activation | `bfloat16` |
| activation | [silu.cc](./aie2p/silu.cc) | AIE API | SiLU / Swish activation | `bfloat16` |
| activation | [swiglu.cc](./aie2p/swiglu.cc) | AIE API | SwiGLU gated activation | `bfloat16` |
| activation | [tanh.cc](./aie2p/tanh.cc) | AIE API | Tanh activation (native) | `bfloat16` |
| activation | [sigmoid.cc](./aie2p/sigmoid.cc) | AIE API | Sigmoid activation | `bfloat16` |
| activation | [leaky_relu.cc](./aie2p/leaky_relu.cc) | AIE API | Leaky ReLU activation | `bfloat16` |
| activation | [softmax.cc](./aie2p/softmax.cc) | AIE API | Softmax + `partial_softmax` (flash-attn) + `mask` | `bfloat16` |
| activation | [bf16_exp.cc](./aie2p/bf16_exp.cc) | AIE API | Element-wise `e^x` (LUT) | `bfloat16` |
| activation | [exp2f_vec.cc](./aie2p/exp2f_vec.cc) | AIE API | Element-wise `2^x` (degree-5 minimax poly; higher accuracy on negatives) | `float32` |
| |
| norm | [layer_norm.cc](./aie2p/layer_norm.cc) | AIE API | Layer normalization (+ affine/cast f32 path) | `bfloat16`, `float32` |
| norm | [rms_norm.cc](./aie2p/rms_norm.cc) | AIE API | RMS normalization | `bfloat16` |
| |
| data movement | [cast_f32_bf16.cc](./aie2p/cast_f32_bf16.cc) | AIE API | f32→bf16 narrowing cast (host-matching `conv_even` rounding) | `float32`→`bfloat16` |
| |
| attention | [mha.cc](./aie2p/mha.cc) | AIE API | Flash-attention toolkit (matmul_PV, partial_softmax, rescale_O, …); composes `softmax.cc` + `mm.cc` | `bfloat16` |
| positional | [rope.cc](./aie2p/rope.cc) | AIE API | RoPE — `rope` (interleaved / Llama) + `rope_two_halves` (HF) | `bfloat16` |
| |
| ml | [conv2dk1_i8.cc](./aie2p/conv2dk1_i8.cc) | AIE API | 1x1 Conv2D | `int8_t` |
| ml | [conv2dk14.cc](./aie2p/conv2dk14.cc) | AIE API | 1x14 / 14x1 Conv2D | `int8_t` |
| ml | [dwconv1d.cc](./aie2p/dwconv1d.cc) | AIE API | Depthwise 1D convolution | `bfloat16` |
