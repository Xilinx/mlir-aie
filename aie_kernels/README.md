<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# AIE Kernels

These kernels are provided as example building blocks for larger designs, and also as illustrations of how to write single core programs for AIEs which can then be duplicated or mixed into multi-core designs using the structural IRON API.

Most kernels use the AIE API, a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics, and whose documentation can be found [here](https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/index.html). A few use the architecture-specific low-level intrinsics directly.

> **NOTE:** this set of AIE kernels are meant for demonstration along with the programming examples. The goal is not to be 100% performant, there may be room for further improvement. The kernels are provided as-is with no guarantees of support of AMD or AMD Research and Advanced Development.

## Layout

Kernels are grouped by family, and each family directory matches a module under [`aie.iron.kernels`](../python/iron/kernels/). One source serves every architecture:

- Most `.cc` files build for both AIE2 and AIE2P. They choose widths and paths from [`aie_arch.h`](./aie_arch.h), which names what each architecture offers (`AIE_BF16_LANES`, `AIE_HAS_NATIVE_TANH`, ...) and which code was tuned for it (`AIE_TUNED_AIE2`), rather than testing `__AIE_ARCH__`. It has one row per architecture and stops the build on an architecture without one.
- Code tuned for one architecture sits beside an untuned branch that tests only those capabilities, which is what a new architecture's row builds first. `AIE_KERNELS_PORTABLE=1` in the environment selects the untuned branch on AIE2 and AIE2P too, and pairs the factories' stack sizes, tolerances and reference models with it, so it can be checked on hardware that exists. `mm.cc` is the exception: its headers follow the architecture, since AIE2 lacks some of AIE2P's `mmul` shapes.
- When the two architectures need different code, the family holds `X_aie2.h` and `X_aie2p.h` and a small `X.cc` that includes the right one.
- [`common/`](./common) holds helpers shared across families.

The tables below describe the sources. Which kernels each NPU builds, and whether they passed the nightly hardware sweep, is in the [kernels view](https://xilinx.github.io/mlir-aie/kernel-checks/#view=kernels) of the Nightly Kernel Checks page, generated from `aie.iron.kernels` each night.

## activation
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [leaky_relu.cc](./activation/leaky_relu.cc) | AIE API | Leaky ReLU activation | `bfloat16` |
| [gelu.cc](./activation/gelu.cc) | AIE API | GELU activation (tanh approx). The per-arch headers differ in tuning: the AIE2P one is MAC-fused with an `s*beta` precompute and post-RA pipelining | `bfloat16` |
| [silu.cc](./activation/silu.cc) | AIE API | SiLU / Swish activation (tanh path from `activations.h`) | `bfloat16` |
| [swiglu.cc](./activation/swiglu.cc) | AIE API | SwiGLU gated activation (tanh path from `activations.h`) | `bfloat16` |
| [tanh.cc](./activation/tanh.cc) | AIE API | Tanh activation (tanh path from `activations.h`: native on AIE2P, LUT on AIE2) | `bfloat16` |
| [sigmoid.cc](./activation/sigmoid.cc) | AIE API | Sigmoid activation (tanh path from `activations.h`) | `bfloat16` |
| [softmax.cc](./activation/softmax.cc) | AIE API | Softmax; on AIE2P also `partial_softmax` (flash-attn) and `mask` | `bfloat16` |
| [bf16_exp.cc](./activation/bf16_exp.cc) | AIE API | Element-wise `e^x` | `bfloat16` |
| [exp2f_vec.cc](./activation/exp2f_vec.cc) | AIE API | Element-wise `2^x` (degree-5 minimax poly; higher accuracy on negatives) | `float32` |

## conv
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [conv2dk1_i8.cc](./conv/conv2dk1_i8.cc) | AIE API | 1x1 Conv2D | `int8_t` |
| [conv2dk1.cc](./conv/conv2dk1.cc) | AIE API | 1x1 Conv2D with fused ReLU | `int8_t`, `uint8_t` |
| [conv2dk3.cc](./conv/conv2dk3.cc) | AIE API | 3x3 Conv2D with fused ReLU | `int8_t`, `uint8_t` |
| [conv2dk1_skip.cc](./conv/conv2dk1_skip.cc) | AIE API| 1x1 Conv2D with fused skip addition | `int8_t`, `uint8_t` |
| [conv2dk1_skip_init.cc](./conv/conv2dk1_skip_init.cc) | AIE API | 1x1 Conv2D with fused 1x1 Conv2D skip addition | `int8_t`, `uint8_t` |
| [conv2dk14.cc](./conv/conv2dk14.cc) | AIE API | 1x14 / 14x1 Conv2D | `int8_t` |
| [bn_*.cc](./conv) | AIE API | BatchNorm-fused bottleneck conv set | `int8_t`, `uint8_t` |
| [dwconv1d_channels_first.cc](./conv/dwconv1d_channels_first.cc) | AIE API | Depthwise 1D convolution, **channels-first** — one channel per call, vectorizes along time; runtime length, `'same'` padding, optional bias. The general-purpose one | `bfloat16` |
| [dwconv1d_channels_last.cc](./conv/dwconv1d_channels_last.cc) | AIE API | Depthwise 1D convolution, **channels-last** — one timestep per call, vectorizes across channels with per-channel taps and an optional clamp. See [_Choosing a depthwise conv1d_](#choosing-a-depthwise-conv1d) | `bfloat16` |

## core
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [set_rounding.cc](./core/set_rounding.cc) | Intrinsics | Set the core's rounding mode | — |

## datamovement
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [transpose.cc](./datamovement/transpose.cc) | AIE API | Blocked matrix transpose (4×4 / 8×8 sub-tiles, VSHUFFLE) | `bfloat16` |
| [expand.cc](./datamovement/expand.cc) | AIE API | uint4→bf16 dequant with per-group scale factors (zero-extended, no zero point) | `uint4`→`bfloat16` |
| [axpy.cc](./datamovement/axpy.cc) | AIE API | `z = a*x + y` (SAXPY) | `bfloat16` |
| [rope.cc](./datamovement/rope.cc) | AIE API | RoPE — `rope` (interleaved / Llama) + `rope_two_halves` (HF) | `bfloat16` |
| [cast_f32_bf16.cc](./datamovement/cast_f32_bf16.cc) | AIE API | f32→bf16 narrowing cast (host-matching `conv_even` rounding) | `float32`→`bfloat16` |

## eltwise
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [passThrough.cc](./eltwise/passThrough.cc) | AIE API | A simple memcpy operation | `uint8_t`, `int16_t`, `int32_t` |
| [add.cc](./eltwise/add.cc) | AIE API | Pointwise addition of 2 tensors | `bfloat16` |
| [mul.cc](./eltwise/mul.cc) | AIE API | Pointwise multiplication of 2 tensors | `bfloat16` |
| [scale.cc](./eltwise/scale.cc) | AIE API | Scale all elements of a tensor with a scale factor | `int32_t` |
| [scale_shift.cc](./eltwise/scale_shift.cc) | AIE API | Scale-and-shift | `int32_t` |
| [relu.cc](./eltwise/relu.cc) | Intrinsics | ReLU activation | `bfloat16` |

## fused
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [fused_mm_tile.cc](./fused/fused_mm_tile.cc) | AIE API | Fused GEMM with in-L1 f32 accumulate and activation epilogue (`acc_init` / `k_step` / `epilogue_chunk`, from [mm_fused.h](./fused/mm_fused.h)); tile geometry via `-DMM_FUSED_*`, activation mode and clamp bounds as runtime arguments to `epilogue_chunk` | `bfloat16` |

## linalg
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [mm.cc](./linalg/mm.cc) | AIE API | Matrix/Matrix multiplication | `int8_t`,`int16_t`,`bfloat16` |
| [cascade_mm.cc](./linalg/cascade_mm.cc) | Scalar, cascade intrinsics | Cascade Matrix/Matrix multiply (multi-core) | `int16_t`,`bfloat16` |
| [mm_bfp.cc](./linalg/mm_bfp.cc) | AIE API | Block-floating-point matmul (AIE2P only) | `bfp16` |
| [mm_bfp_mixed.cc](./linalg/mm_bfp_mixed.cc) | AIE API | Mixed-precision BFP matmul (AIE2P only) | `bfp16` |
| [mv_bf16.cc](./linalg/mv_bf16.cc) | AIE API | Matrix/Vector multiply, row-major A (IRON GEMV) | `bfloat16` |
| [mv_i16.cc](./linalg/mv_i16.cc) | AIE API | Matrix/Vector multiply, A word-transposed | `int16_t`→`int32_t` |
| [mha.cc](./linalg/mha.cc) | AIE API | Flash-attention **decode** toolkit (matmul_PV, partial_softmax, rescale_O, …); composes `softmax_aie2p.h` + `mm_aie2p.h` | `bfloat16` |
| [flash_attn_prefill.cc](./linalg/flash_attn_prefill.cc) | AIE API | Flash-attention **prefill** with online softmax, as five per-step entry points an ObjectFifo design drives (`round_begin`, `qk_step`, `block_mid`, `fv_step`, `epilogue`). `-DPREFILL_HEAD_DIM` picks the geometry: 512 global, 256 sliding-window | `bfloat16` |

## norm
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [layer_norm.cc](./norm/layer_norm.cc) | AIE API | Layer normalization | `bfloat16` |
| [rms_norm.cc](./norm/rms_norm.cc) | AIE API | RMS normalization — `rms_norm` (eps=1e-5) + `rms_norm_eps` (runtime eps) | `bfloat16` |

## quant
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [q4nx_dequant.cc](./quant/q4nx_dequant.cc) | AIE API | Dequantize packed q4nx scales, minima and 4-bit codes into GEMM-ordered BFP blocks (AIE2P only); geometry via `-DQ4NX_*` | `uint8_t` → `bfp16ebs8` |

## reduce
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [reduce_add.cc](./reduce/reduce_add.cc) | Intrinsics | Sum of elements in a tensor | `int32_t` |
| [reduce_max.cc](./reduce/reduce_max.cc) | Intrinsics | Max value across a tensor | `int32_t`, `bfloat16` |
| [reduce_min.cc](./reduce/reduce_min.cc) | Intrinsics | Min value across a tensor | `int32_t` |

## transformer
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [layer_norm_f32.cc](./transformer/layer_norm_f32.cc) | AIE API | f32 layer normalization: `layer_norm_f32` (identity affine) and `layer_norm_affine_cast` (affine, then cast to bf16) | `float32` |
| [mm_activation_epilogue.cc](./transformer/mm_activation_epilogue.cc) | AIE API | Matmul with fused activation epilogue | `float32` |

## vision
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [bitwiseOR.cc](./vision/bitwiseOR.cc) | AIE API | Bitwise OR of fixed point tensors | `uint8_t`,`int16_t`,`int32_t`|
| [bitwiseAND.cc](./vision/bitwiseAND.cc) | AIE API | Bitwise AND of fixed point tensors | `uint8_t`,`int16_t`,`int32_t` |
| [gray2rgba.cc](./vision/gray2rgba.cc) | AIE API | Convert from grayscale to RGBA format | `uint8_t` |
| [rgba2gray.cc](./vision/rgba2gray.cc) | AIE API | Convert from RGBA format to grayscale | `uint8_t` |
| [rgba2hue.cc](./vision/rgba2hue.cc) | AIE API | Convert from RGBA to hue | `uint8_t` |
| [addWeighted.cc](./vision/addWeighted.cc) | AIE API | Fixed point weighted sum of two tensors | `uint8_t` |
| [threshold.cc](./vision/threshold.cc) | AIE API | Clipping | `uint8_t` |
| [filter2d.cc](./vision/filter2d.cc) | AIE API | Fixed point 2D image processing filter | `uint8_t` |

## zero
| Name | Coding style | Purpose | Datatypes |
|-|-|-|-|
| [zero.cc](./zero/zero.cc) | AIE API | Fill a tensor with zeroes | template |

## Choosing a depthwise conv1d

The two `dwconv1d` kernels compute the same thing over transposed tensors, and the layout is the whole distinction: it picks the vectorization axis, the tap representation, and which one is faster.

| | channels-first | channels-last |
|-|-|-|
| one call emits | one channel, all timesteps | one timestep, all channels |
| contiguous axis | time | channels |
| taps | `K` scalars, broadcast | `K` vectors of `C`, one per channel |
| vectorizes over | time, via `sliding_mul` (`mac_elem_16_2` on AIE2) | channels, via plain `mac` |
| `K` | runtime-templated, 1–17 | fixed per entry point (5 today) |
| sequence length | runtime | n/a |
| MACs per instruction slot (AIE2P) | 4.2 | 12.2 |

Channels-last retires nearly 3x the MACs per slot because `sliding_mul` spends half its vector on the window halo and then rebuilds each tap's operand with a `vshift`, while the channels-last form's operands are already aligned and every lane is a real MAC.

**That is not a reason to prefer it.** It only pays when the data is already channels-last — transposing to reach it costs more than it saves — and it wants `C * K` resident weights, a compile-time `C`, and program memory linear in `C`. Reach for channels-first by default; reach for channels-last when the producer already emits that layout.
