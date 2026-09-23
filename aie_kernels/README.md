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
| data movement | [expand.cc](./generic/expand.cc) | AIE API | uint4→bf16 dequant with per-group scale factors (zero-extended, no zero point) | `uint4`→`bfloat16` |
| gemv | [mv_bf16.cc](./generic/mv_bf16.cc) | AIE API | Matrix/Vector multiply, row-major A (IRON GEMV) | `bfloat16` |
| gemv | [mv_i16.cc](./generic/mv_i16.cc) | AIE API | Matrix/Vector multiply, A word-transposed | `int16_t`→`int32_t` |
| blas | [axpy.cc](./generic/axpy.cc) | AIE API | `z = a*x + y` (SAXPY) | `bfloat16` |
| positional | [rope.cc](./generic/rope.cc) | AIE API | RoPE — `rope` (interleaved / Llama) + `rope_two_halves` (HF) | `bfloat16` |
| gemm | [mm_fused.cc](./generic/mm_fused.cc) | AIE API | Fused GEMM with in-L1 f32 accumulate and activation epilogue (`acc_init` / `k_step` / `epilogue_chunk`); tile geometry via `-DMM_FUSED_*`, activation mode and clamp bounds as runtime arguments to `epilogue_chunk` | `bfloat16` |
| quantization | [q4nx_dequant.cc](./generic/q4nx_dequant.cc) | AIE API (AIE2P) | Dequantize packed q4nx scales, minima and 4-bit codes into GEMM-ordered BFP blocks; geometry via `-DQ4NX_*` | `uint8_t` → `bfp16ebs8` |

## AIE1
| Name | Coding style | Purpose |
|-|-|-|

## AIE2
| Class | Name | Coding style | Purpose | Datatypes |
|-|-|-|-|-|
| basic | [zero.cc](./generic/zero.cc) | AIE API | Fill a tensor with zeroes | template |
| basic | [add.cc](./generic/add.cc) | AIE API | Pointwise addition of 2 tensors (16-wide here, 32 on AIE2P) | `bfloat16` |
| basic | [mul.cc](./generic/mul.cc) | AIE API | Pointwise multiplication of 2 tensors (16-wide here, 32 on AIE2P) | `bfloat16` |
| basic | [scale.cc](./aie2/scale.cc) | AIE API | Scale all elements of a tensor with a scale factor | `int32_t` |
| basic | [scale_shift.cc](./aie2/scale_shift.cc) | AIE API | Scale-and-shift | `int32_t` |
| basic | [bitwiseOR.cc](./aie2/bitwiseOR.cc) | AIE API | Bitwise OR of fixed point tensors | `uint8_t`,`int16_t`,`int32_t`|
| basic | [bitwiseAND.cc](./aie2/bitwiseAND.cc) | AIE API | Bitwise AND of fixed point tensors | `uint8_t`,`int16_t`,`int32_t` |
| gemm  | [mm.cc](./aie2/mm.cc) | AIE API | Matrix/Matrix multiplication | `int8_t`,`int16_t`,`bfloat16` |
| gemm  | [cascade_mm.cc](./aie2/cascade_mm.cc) | AIE API | Cascade Matrix/Matrix multiply (multi-core) | `int16_t`,`bfloat16` |
| |
| reduction | [reduce_add.cc](./aie2/reduce_add.cc) | Intrinsics | Sum of elements in a tensor | `int32_t` |
| reduction| [reduce_max.cc](./aie2/reduce_max.cc) | Intrinsics | Max value across a tensor | `int32_t` |
| reduction | [reduce_min.cc](./aie2/reduce_min.cc) | Intrinsics | Min value across a tensor | `int32_t` |
| |
| activation | [relu.cc](./aie2/relu.cc) | Intrinsics | ReLU activation | `bfloat16` |
| activation | [leaky_relu.cc](./generic/leaky_relu.cc) | AIE API | Leaky ReLU activation (16 lanes here, 32 on AIE2P) | `bfloat16` |
| activation | [gelu.cc](./aie2/gelu.cc) | AIE API | GELU activation (tanh approx) | `bfloat16` |
| activation | [silu.cc](./generic/silu.cc) | AIE API | SiLU / Swish activation (shared; tanh path from `activations.h`) | `bfloat16` |
| activation | [swiglu.cc](./generic/swiglu.cc) | AIE API | SwiGLU gated activation (shared; tanh path from `activations.h`) | `bfloat16` |
| activation | [tanh.cc](./generic/tanh.cc) | AIE API | Tanh activation (shared; tanh path from `activations.h`: LUT here, native on AIE2P) | `bfloat16` |
| activation | [sigmoid.cc](./generic/sigmoid.cc) | AIE API | Sigmoid activation (shared; tanh path from `activations.h`) | `bfloat16` |
| activation | [softmax.cc](./aie2/softmax.cc) | AIE API | Softmax | `bfloat16` |
| activation | [bf16_exp.cc](./aie2/bf16_exp.cc) | AIE API | Element-wise `e^x` | `bfloat16` |
| norm | [rms_norm.cc](./aie2/rms_norm.cc) | AIE API | RMS normalization — `rms_norm` (eps=1e-5) + `rms_norm_eps` (runtime eps) | `bfloat16` |
| |
| ml | [conv2dk1_i8.cc](./generic/conv2dk1_i8.cc) | AIE API | 1x1 Conv2D (8 accumulators of M=4 here, 4 of M=8 on AIE2P) | `int8_t` |
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
| basic | [zero.cc](./generic/zero.cc) | AIE API | Fill a tensor with zeroes (512-bit stores) | template |
| basic | [add.cc](./generic/add.cc) | AIE API | Pointwise addition of 2 tensors (32-wide here, 16 on AIE2) | `bfloat16` |
| basic | [mul.cc](./generic/mul.cc) | AIE API | Pointwise multiplication of 2 tensors (32-wide here, 16 on AIE2) | `bfloat16` |
| gemm | [mm.cc](./aie2p/mm.cc) | AIE API | Matrix/Matrix multiplication | `int8_t`,`int16_t`,`bfloat16` |
| gemm | [mm_bfp.cc](./aie2p/mm_bfp.cc) | AIE API | Block-floating-point matmul | `bfp16` |
| gemm | [mm_bfp_mixed.cc](./aie2p/mm_bfp_mixed.cc) | AIE API | Mixed-precision BFP matmul | `bfp16` |
| gemm | [mm_activation_epilogue.cc](./aie2p/mm_activation_epilogue.cc) | AIE API | Matmul with fused activation epilogue | `bfloat16` |
| |
| activation | [gelu.cc](./aie2p/gelu.cc) | AIE API | GELU activation. Kept separate from the AIE2 copy: this one is MAC-fused with an `s*beta` precompute and post-RA pipelining (II=18), which is tuning, not an arch constant | `bfloat16` |
| activation | [silu.cc](./generic/silu.cc) | AIE API | SiLU / Swish activation (shared; tanh path from `activations.h`) | `bfloat16` |
| activation | [swiglu.cc](./generic/swiglu.cc) | AIE API | SwiGLU gated activation (shared; tanh path from `activations.h`) | `bfloat16` |
| activation | [tanh.cc](./generic/tanh.cc) | AIE API | Tanh activation (shared; tanh path from `activations.h`: native here, LUT on AIE2) | `bfloat16` |
| activation | [sigmoid.cc](./generic/sigmoid.cc) | AIE API | Sigmoid activation (shared; tanh path from `activations.h`) | `bfloat16` |
| activation | [leaky_relu.cc](./generic/leaky_relu.cc) | AIE API | Leaky ReLU activation (32 lanes here, 16 on AIE2) | `bfloat16` |
| activation | [softmax.cc](./aie2p/softmax.cc) | AIE API | Softmax + `partial_softmax` (flash-attn) + `mask` | `bfloat16` |
| activation | [bf16_exp.cc](./aie2p/bf16_exp.cc) | AIE API | Element-wise `e^x` (LUT) | `bfloat16` |
| activation | [exp2f_vec.cc](./aie2p/exp2f_vec.cc) | AIE API | Element-wise `2^x` (degree-5 minimax poly; higher accuracy on negatives) | `float32` |
| |
| norm | [layer_norm.cc](./aie2p/layer_norm.cc) | AIE API | Layer normalization (+ affine/cast f32 path) | `bfloat16`, `float32` |
| norm | [rms_norm.cc](./aie2p/rms_norm.cc) | AIE API | RMS normalization — `rms_norm` (eps=1e-5) + `rms_norm_eps` (runtime eps) | `bfloat16` |
| |
| data movement | [cast_f32_bf16.cc](./aie2p/cast_f32_bf16.cc) | AIE API | f32→bf16 narrowing cast (host-matching `conv_even` rounding) | `float32`→`bfloat16` |
| |
| attention | [mha.cc](./aie2p/mha.cc) | AIE API | Flash-attention **decode** toolkit (matmul_PV, partial_softmax, rescale_O, …); composes `softmax.cc` + `mm.cc` | `bfloat16` |
| attention | [flash_attn_prefill.cc](./aie2p/flash_attn_prefill.cc) | AIE API | Flash-attention **prefill** with online softmax, as five per-step entry points an ObjectFifo design drives (`round_begin`, `qk_step`, `block_mid`, `fv_step`, `epilogue`). `-DPREFILL_HEAD_DIM` picks the geometry: 512 global, 256 sliding-window | `bfloat16` |
| |
| ml | [conv2dk1_i8.cc](./generic/conv2dk1_i8.cc) | AIE API | 1x1 Conv2D (4 accumulators of M=8 here, 8 of M=4 on AIE2) | `int8_t` |
| ml | [conv2dk14.cc](./aie2p/conv2dk14.cc) | AIE API | 1x14 / 14x1 Conv2D | `int8_t` |
| ml | [dwconv1d_channels_first.cc](./aie2p/dwconv1d_channels_first.cc) | AIE API | Depthwise 1D convolution, **channels-first** — one channel per call, vectorizes along time; runtime length, `'same'` padding, optional bias. The general-purpose one | `bfloat16` |
| ml | [dwconv1d_channels_last.cc](./aie2p/dwconv1d_channels_last.cc) | AIE API | Depthwise 1D convolution, **channels-last** — one timestep per call, vectorizes across channels with per-channel taps and an optional clamp. See [_Choosing a depthwise conv1d_](#choosing-a-depthwise-conv1d) | `bfloat16` |

## Choosing a depthwise conv1d

The two `dwconv1d` kernels compute the same thing over transposed tensors, and the layout is the whole distinction: it picks the vectorization axis, the tap representation, and which one is faster.

| | channels-first | channels-last |
|-|-|-|
| one call emits | one channel, all timesteps | one timestep, all channels |
| contiguous axis | time | channels |
| taps | `K` scalars, broadcast | `K` vectors of `C`, one per channel |
| vectorizes over | time, via `sliding_mul` | channels, via plain `mac` |
| `K` | runtime-templated, 1–17 | fixed per entry point (5 today) |
| sequence length | runtime | n/a |
| MACs per instruction slot | 4.2 | 12.2 |

Channels-last retires nearly 3x the MACs per slot because `sliding_mul` spends half its vector on the window halo and then rebuilds each tap's operand with a `vshift`, while the channels-last form's operands are already aligned and every lane is a real MAC.

**That is not a reason to prefer it.** It only pays when the data is already channels-last — transposing to reach it costs more than it saves — and it wants `C * K` resident weights, a compile-time `C`, and program memory linear in `C`. Reach for channels-first by default; reach for channels-last when the producer already emits that layout.
