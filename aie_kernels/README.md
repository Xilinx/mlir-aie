<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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

## AIE1
| Name | Coding style | Purpose |
|-|-|-|

## AIE2
| Class | Name | Coding style | Purpose | Datatypes |
|-|-|-|-|-|
| basic | [zero.cc](../../aie_kernels/aie2/zero.cc) | AIE API | Fill a tensor with zeroes | template |
| basic | [add.cc](../../aie_kernels/aie2/add.cc) | AIE API | Pointwise addition of 2 tensors | `bfloat16` |
| basic | [mul.cc](../../aie_kernels/aie2/mul.cc) | AIE API | Pointwise multiplication of 2 tensors | `bfloat16` |
| basic | [scale.cc](../../aie_kernels/aie2/scale.cc) | AIE API | Scale all elements of a tensor with a scale factor | `int32_t` |
| basic | [bitwiseOR.cc](../../aie_kernels/aie2/bitwiseOR.cc) | AIE API | Bitwise OR of fixed point tensors | `uint8_t`,`int16_t`,`int32_t`|
| basic | [bitwiseAND.cc](../../aie_kernels/aie2/bitwiseAND.cc) | AIE API | Bitwise AND of fixed point tensors | `uint8_t`,`int16_t`,`int32_t` |
| gemm  | [mm.cc](../../aie_kernels/aie2/mm.cc) | AIE API | Matrix/Matrix multiplication | `int16_t`,`bfloat16_t` |
| gemm  | [mv.cc](../../aie_kernels/aie2/mv.cc) | AIE API | Matrix/Vector multiplication | `bfloat16_t` |
| |
| reduction | [reduce_add.cc](../../aie_kernels/aie2/reduce_add.cc) | Intrinsics | Find the sum of elements in a tensor | `int32 _t` |
| reduction| [reduce_max.cc](../../aie_kernels/aie2/reduce_max.cc) | Intrinsics | Find max value across a tensor | `int32 _t` |
| reduction | [reduce_min.cc](../../aie_kernels/aie2/reduce_min.cc) | Intrinsics | Find min value across a tensor | `int32 _t` |
||
| ml | [conv2dk1_i8.cc](../../aie_kernels/aie2/conv2dk1_i8.cc) | AIE API | 1x1 Conv2D | `int8_t` |
| ml | [conv2dk1.cc](../../aie_kernels/aie2/conv2dk1.cc) | AIE API | 1x1 Conv2D with fused ReLU | `int8_t`, `uint8_t` |
| ml | [conv2dk3.cc](../../aie_kernels/aie2/conv2dk3.cc) | AIE API | 3x3 Conv2D with fused ReLU | `int8_t`, `uint8_t` |
| ml | [conv2dk1_skip.cc](../../aie_kernels/aie2/conv2dk1_skip.cc) | AIE API| 1x1 Conv2D with fused skip addition | `int8_t`, `uint8_t` |
| ml | [conv2dk1_skip_init.cc](../../aie_kernels/aie2/conv2dk1_skip_init.cc) | AIE API | 1x1 Conv2D with fused 1x1 Conv2D skip addition | `int8_t`, `uint8_t` |
| ml |[relu.cc](../../aie_kernels/aie2/relu.cc) | Intrinsics | ReLU activation function | `bfloat16_t` |
| ml |  [bf16_exp.cc](../../aie_kernels/aie2/bf16_exp.cc) | AIE API | Raise all elements in a `bfloat` tensor to $e^x$ | `bfloat16_t` |
| |
| vision | [gray2rgba.cc](../../aie_kernels/aie2/gray2rgba.cc) | AIE API | Convert from grayscale to RGBA format | `uint8_t` |
| vision |[rgba2gray.cc](../../aie_kernels/aie2/rgba2gray.cc) | AIE API | Convert from RGBA format to grayscale | `uint8_t` |
| vision | [rgba2hue.cc](../../aie_kernels/aie2/rgba2hue.cc) | AIE API | Convert from RGBA to hue | `uint8_t` |
| vision | [addWeighted.cc](../../aie_kernels/aie2/addWeighted.cc) | AIE API | Fixed point weighted sum of two tensors | `uint8_t` |
| vision | [threshold.cc](../../aie_kernels/aie2/threshold.cc) | AIE API | Clipping | `uint8_t` |  
| vision | [filter2d.cc](../../aie_kernels/aie2/filter2d.cc) | AIE API | Fixed point 2D image processing filter | `uint8_t` |
