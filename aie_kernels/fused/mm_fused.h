//===- mm_fused.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../aie_kernel_utils.h"
#include "../common/activations.h"
#include "../common/zero.h"
#include "mm_fused_mmul.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// bf16 GEMM compute kernel: each tile owns an m x n slice of C, accumulated
// over K in an f32 accumulator kept in L1 for the whole reduction. Three entry
// points the design's loop nest drives once per iteration: mm_fused_acc_init
// (zero the accumulator), mm_fused_k_step (multiply one A band by one B chunk
// into it), mm_fused_epilogue_chunk (drain one chunk to a bf16 C object). The
// nest lives in the design (not here) so each level has an ObjectFifo acquire
// point. Tile geometry arrives as -DMM_FUSED_* flags, the single source of
// truth shared with the design's buffer sizing and unroll factors.

#if !defined(MM_FUSED_TILE_M) || !defined(MM_FUSED_TILE_K) ||                  \
    !defined(MM_FUSED_TILE_N) || !defined(MM_FUSED_CT_K)
#error "design.py must pass -DMM_FUSED_TILE_M / _TILE_K / _TILE_N / _CT_K"
#endif
#if !defined(MM_FUSED_OUT_CHUNK) || !defined(MM_FUSED_C_DEPTH)
#error "design.py must pass -DMM_FUSED_OUT_CHUNK / -DMM_FUSED_C_DEPTH"
#endif

// Which epilogue modes to compile in, as a bitmask over 1 << mode: 0 = none,
// 1 = gelu, 2 = silu, 3 = sigmoid, matching Epilogue.mode in design.py. The
// mode itself is a runtime argument; the mask only bounds program memory.
#ifndef MM_FUSED_EPILOGUE_MODE_MASK
#define MM_FUSED_EPILOGUE_MODE_MASK 0xF
#endif

namespace {
constexpr int M = MM_FUSED_TILE_M;
// Asymmetric tile buffering (arXiv:2511.16041; ref impl in mlir-aie
// programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering): the
// A tile spans MA rows while the accumulator spans M, so the core folds RHO = M
// / MA A bands into one C tile. A dies when consumed but C lives across the
// whole K reduction, so sizing both to M pays the peak L1 cost twice (MA == M
// is the symmetric case). With an f32 accumulator the freed L1 buys a deeper k
// slice rather than the wider C tile that bf16/bfp16 accumulation affords.
constexpr int MA = MM_FUSED_TILE_MA;
constexpr int K = MM_FUSED_TILE_K;
constexpr int N = MM_FUSED_TILE_N;
// Register tiling, and how much of K one compute tile holds at a time. Both are
// design.py's to choose -- CT_K in particular trades against the n width for a
// fixed L1 budget.
constexpr int R = MM_FUSED_R;
constexpr int S = MM_FUSED_S;
constexpr int T = MM_FUSED_T;
constexpr int CT_K = MM_FUSED_CT_K;

// Each entry point brackets its own call in event0/event1. fused_mm_tile.cc
// calls them all from one call and brackets that instead, so it defines
// MM_FUSED_WHOLE_TILE_MARKERS to compile these out.
#ifdef MM_FUSED_WHOLE_TILE_MARKERS
constexpr bool step_markers = false;
#else
constexpr bool step_markers = true;
#endif

// Output stage geometry.
constexpr int CHUNK = MM_FUSED_OUT_CHUNK;
constexpr int C_DEPTH = MM_FUSED_C_DEPTH;
// The epilogue is lane-wise, so its width changes no result. aie2 stays at 16
// lanes because its LUT tanh is 16 lanes.
#if AIE_TUNED_AIE2
constexpr int V = 16;
#else
constexpr int V = CHUNK % 32 == 0 ? 32 : 16;
#endif
static_assert(CHUNK % V == 0, "output chunk must be a whole number of vectors");

// Same divisibility conditions mm.cc asserts for its own 2x2 mmul, plus the
// two the k blocking adds.
static_assert(M % MA == 0, "tile_m must be a whole number of A bands");
static_assert(MA % (2 * R) == 0,
              "tile_ma must be a multiple of 2*r (2x2 mmul)");
static_assert(N % (2 * T) == 0, "tile_n must be a multiple of 2*t (2x2 mmul)");
static_assert(K % CT_K == 0, "tile_k must be a multiple of the k slice");
static_assert(CT_K % S == 0, "k slice must be a multiple of s");

// The core powers up in rounding_mode::floor, so a kernel that converts must
// choose explicitly. Truncation biases every conversion the same direction, so
// the error accumulates over the K reduction instead of cancelling -- ~1% of
// the result, against ~0.02% for round-to-nearest-even, which is far more than
// the bfp16 emulation itself costs. Every entry point that converts sets it:
// the mmul and the epilogue's f32->bf16 store, both below.
//
// Flag name and polarity follow mm.cc, so the two kernels are configured the
// same way; the operator passes -DROUND_CONV_EVEN by default.
#ifdef ROUND_CONV_EVEN
constexpr aie::rounding_mode round_mode = aie::rounding_mode::conv_even;
#else
constexpr aie::rounding_mode round_mode = aie::rounding_mode::floor;
#endif

// One activation's inner loop. Templated so each mode compiles branch-free;
// mm_fused_epilogue_chunk selects between them once per chunk.
//
// The clamp is unconditional. An unclamped caller sends (-inf, +inf), which
// leaves every finite value bit-identical, so there is no unclamped
// instantiation to compile and no clamped-versus-not fork in the build.
template <int MODE>
static inline void epilogue_body(bfloat16 *__restrict y_out,
                                 const float *__restrict src, float clamp_min,
                                 float clamp_max) {
  // The clamp runs on the bf16 result, against bounds rounded the same way.
  // Rounding is monotone and fixes representable values, so for finite
  // inputs round(clamp(f, lo, hi)) == clamp(round(f), round(lo), round(hi))
  // and the output is bit-identical to clamping in f32. aie2p has no native
  // f32 min/max.
  aie::accum<accfloat, V> bound;
  bound.from_vector(aie::broadcast<float, V>(clamp_min));
  const aie::vector<bfloat16, V> lo = bound.template to_vector<bfloat16>();
  bound.from_vector(aie::broadcast<float, V>(clamp_max));
  const aie::vector<bfloat16, V> hi = bound.template to_vector<bfloat16>();

  // Walking cursors rather than src + j * V, which Peano recomputes each
  // trip.
  AIE_LOOP_MAX_ITERATION_COUNT(CHUNK / V)
  AIE_LOOP_UNROLL(2)
  for (int j = 0; j < CHUNK / V; j++) {
    // The accumulator stays f32 through the activation and is converted to
    // bf16 exactly once. Converting first would round twice and let the
    // activation's slope amplify the first rounding -- see activations.h.
    aie::vector<float, V> f = aie::load_v<V>(src);
    src += V;
    if constexpr (MODE == 1)
      f = gelu_vec<V>(f);
    else if constexpr (MODE == 2)
      f = silu_vec<V>(f);
    else if constexpr (MODE == 3)
      f = sigmoid_vec<V>(f);
    aie::accum<accfloat, V> out;
    out.from_vector(f);
    aie::vector<bfloat16, V> v = out.template to_vector<bfloat16>();
    aie::store_v(y_out, aie::max(aie::min(v, hi), lo));
    y_out += V;
  }
}
} // namespace

extern "C" {

// Zero the f32 accumulator, before the k loop starts accumulating into it.
//
// A bias is deliberately not supported: initialising the accumulator from one
// would mean consuming an extra object through the handshake the B ObjectFifo
// owns, which desynchronises that fifo and hangs rather than mis-computing.
void mm_fused_acc_init(float *y_acc) {
  zero_vectorized<float, M, N, step_markers>(y_acc);
}

// One step of the k loop: one B chunk multiplied against one A band,
// accumulated into y_acc.
//
// Takes no locks. A is a single object spanning every z slice of the mmul, and
// the A and B fifos own the handshake, so the core body acquires around this
// call rather than the kernel acquiring inside it.
// mm_fused_b_elem_t is bfp16ebs8 or bfloat16 depending on how B is stored,
// which mm_fused_mmul.h selects from the architecture. One signature either
// way, so the design's Kernel declaration does not have to care.
void mm_fused_k_step(bfloat16 *a_buf, mm_fused_b_elem_t *b_buf, float *y_acc,
                     int32_t band) {
  if constexpr (step_markers)
    event0();
  ::aie::set_rounding(round_mode);
  // The accumulator is [row-block][col-block][r*t], so band b starts at
  // b * MA * N -- b*(MA/R) row-blocks in, each colB*(r*t) wide.
  mm_fused_mmul_2x2<(MA / R), (CT_K / S), (N / T), R, S, T>(
      a_buf, b_buf, y_acc + band * (MA * N));
  if constexpr (step_markers)
    event1();
}

// Output stage: convert chunk (outer * C_DEPTH + half) of the f32 accumulator
// to a bf16 C object, applying an activation and clamp on the way out. Fusing
// here is the point -- the values are already in registers, so the activation
// costs one more vector op per 16 elements instead of a separate pass over L1.
//
// The mode is runtime, tested once per chunk so the inner loops stay
// branch-free; the cost is program memory, since every mode in the mask is
// compiled in. The bounds arrive as raw int32 because npu_write_rtp only
// writes i32 words. The chunk index is split (outer, half) because the core
// body unrolls the drain by the C fifo depth to keep the acquired buffer index
// a compile-time constant.
void mm_fused_epilogue_chunk(bfloat16 *y_out, float *y_acc, int32_t outer,
                             int32_t half, int32_t mode, int32_t clamp_min_bits,
                             int32_t clamp_max_bits) {
  // The store below is a conversion, so it obeys the same rounding mode the
  // mmul does and must agree with it.
  if constexpr (step_markers)
    event0();
  ::aie::set_rounding(round_mode);
  const float *__restrict src = y_acc + (outer * C_DEPTH + half) * CHUNK;
  // __builtin_bit_cast, not memcpy: memcpy leaves an unresolved external
  // call here rather than folding to a register move.
  const float clamp_min = __builtin_bit_cast(float, clamp_min_bits);
  const float clamp_max = __builtin_bit_cast(float, clamp_max_bits);

  switch (mode) {
#if MM_FUSED_EPILOGUE_MODE_MASK & 2
  case 1:
    epilogue_body<1>(y_out, src, clamp_min, clamp_max);
    break;
#endif
#if MM_FUSED_EPILOGUE_MODE_MASK & 4
  case 2:
    epilogue_body<2>(y_out, src, clamp_min, clamp_max);
    break;
#endif
#if MM_FUSED_EPILOGUE_MODE_MASK & 8
  case 3:
    epilogue_body<3>(y_out, src, clamp_min, clamp_max);
    break;
#endif
  // Mode 0 is always compiled, so a mode the mask leaves out yields an
  // unactivated result rather than an unwritten buffer.
  default:
    epilogue_body<0>(y_out, src, clamp_min, clamp_max);
    break;
  }
  if constexpr (step_markers)
    event1();
}
}
