//===- yolo_m9_attn_score_fused_vec.cc -----------------------*- C++ -*-===//
//
// Fused qk_row + attn_scale + softmax_row kernel for the PSA attention.
// Per (head, query_idx) call:
//   1. qk:    scores_i32[j] = sum_{k} Q[k, query_idx] * K[k, j]
//   2. scale: scaled_i32[j] = scores_i32[j] * mul_int    (constexpr 91)
//   3. SRS:   scaled_i8[j]  = clip(srs(scaled_i32[j], rs_qk + mul_shift))
//   4. softmax over scaled_i8 → final i8 (probabilities × 2^-out_log2_scale)
//
// The fusion saves three round-trips through the chunk_io row in L1
// (qk-out write, scale read/write, softmax read/write → just softmax
// write at the end) and collapses the two separate SRS steps (qk's
// requant + scale's requant) into a single combined srs by
// (rs_qk + mul_shift). With mul_int=91 a compile-time constant, the
// "* 91" folds into the combined accum-shift step at near-zero cost.
//
// Phase 1+2 (qk + scale combined): vector-broadcast pattern, 16-wide
// j-groups with aie::accum<acc32, 16>, broadcast Q[k] vmac against
// K[k, j_base..j_base+15] strip. After inner k loop, `aie::mul(acc, 91)`
// scales in i32 and `to_vector<int8>(rs_qk + mul_shift)` does the
// combined SRS in one vector op.
//
// Phase 3 (softmax): 3-pass scalar-FP algorithm matching the original
// softmax_row.cc — peano's aie2p libc++ doesn't provide expf so we use
// the precomputed fp32 LUT. Pass 1 (row max) is vector-reduce; passes 2
// and 3 stay scalar to avoid the 1KB float[] scratch that overflows
// the per-tile stack.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_QK_KD
#error "YOLO_M9_QK_KD must be defined at compile time"
#endif
#ifndef YOLO_M9_QK_N
#error "YOLO_M9_QK_N must be defined at compile time"
#endif

static constexpr int kKd      = YOLO_M9_QK_KD;
static constexpr int kN       = YOLO_M9_QK_N;
static constexpr int kJVec    = 16;
static constexpr int kJGroups = kN / kJVec;

static_assert(kKd > 0,            "QK_KD must be > 0");
static_assert(kN % kJVec == 0,    "QK_N must be a multiple of 16");

static constexpr int32_t I8_SMAX = 127;
static constexpr int32_t I8_SMIN = -128;
static constexpr int32_t I8_UMIN = 0;    // softmax probs in [0, 1] → i8 in [0, 127]

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_attn_score_fused_i8_i8(
    int8_t *qk_frame,            // (2*kd, N) Q || K
    int8_t *chunk_io,            // (chunk_rows, N) — only chunk_io[chunk_row] is touched
    float *exp_lut,              // (256,) softmax exp LUT
    const int32_t chunk_row,
    const int32_t query_idx,
    const int32_t /*kd*/,
    const int32_t /*N*/,
    const int32_t rs_qk,         // qk right_shift (e.g. 3 for m9)
    const int32_t mul_int,       // attn scale mul (e.g. 91 for m9)
    const int32_t mul_shift,     // attn scale shift (e.g. 7 for m9)
    const int32_t out_log2_scale) {  // softmax out scale (e.g. -7 for m9; out = prob * 128)
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *__restrict scores_row = chunk_io + chunk_row * kN;
  const int8_t *__restrict q_col = qk_frame + query_idx;       // stride kN per k
  const int8_t *__restrict k_base = qk_frame + kKd * kN;       // K rows start here

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  // === Phase 1+2: qk + scale → i8 scores row in chunk_io ===
  //
  // mmul accumulate is vector; the two SRS+clip steps run scalar in
  // the tail to match the project's established `banker_srs` rounding
  // (vector `to_vector<int8>(shift)` uses `positive_inf` rounding,
  // which gives LSB-off results vs ORT's banker rounding — chain stays
  // bit-exact either way but stage-10 ORT compare flags 1-LSB diffs).
  AIE_LOOP_RANGE(kJGroups, kJGroups)
  for (int g = 0; g < kJGroups; ++g) {
    aie::accum<acc32, kJVec> acc;
    acc.from_vector(aie::zeros<int32, kJVec>());

    const int8_t *__restrict k_strip = k_base + g * kJVec;

    AIE_LOOP_RANGE(kKd, kKd)
    for (int k = 0; k < kKd; ++k) {
      const int8_t q_scalar = q_col[k * kN];
      aie::vector<int8, kJVec> q_v = aie::broadcast<int8, kJVec>(q_scalar);
      aie::vector<int8, kJVec> k_v = aie::load_v<kJVec>(k_strip + k * kN);
      acc = aie::mac(acc, q_v, k_v);
    }

    aie::vector<int32, kJVec> sum_v = acc.template to_vector<int32>();
    int8_t *__restrict out_p = scores_row + g * kJVec;
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < kJVec; ++i) {
      // Step 1: qk SRS + clip to i8.
      int32_t qk_s = banker_srs(sum_v[i], rs_qk);
      if (qk_s > I8_SMAX) qk_s = I8_SMAX;
      if (qk_s < I8_SMIN) qk_s = I8_SMIN;
      // Step 2: scale by mul_int (constexpr 91), SRS + clip.
      int32_t scaled = qk_s * mul_int;
      int32_t sr = banker_srs(scaled, mul_shift);
      if (sr > I8_SMAX) sr = I8_SMAX;
      if (sr < I8_SMIN) sr = I8_SMIN;
      out_p[i] = (int8_t)sr;
    }
  }

  // === Phase 3a: softmax pass 1 — row max via vector reduce ===
  // Find max over 256 i8 values. Vector reduce-max over 16-lane chunks.
  aie::vector<int8, kJVec> max_v = aie::load_v<kJVec>(scores_row);
  AIE_LOOP_RANGE(kJGroups - 1, kJGroups - 1)
  for (int g = 1; g < kJGroups; ++g) {
    aie::vector<int8, kJVec> v = aie::load_v<kJVec>(scores_row + g * kJVec);
    max_v = aie::max(max_v, v);
  }
  int32_t row_max = (int32_t)aie::reduce_max(max_v);

  // === Phase 3b: pass 2 — sum of exps (scalar fp; LUT-driven) ===
  // 2^-out_log2_scale via integer shift (avoids libm). For out_log2_scale=-7
  // this is 1<<7 = 128.0f.
  const int32_t out_scale_int = 1 << (-out_log2_scale);
  const float out_scale = (float)out_scale_int;

  float sum = 0.0f;
  for (int j = 0; j < kN; ++j) {
    int32_t shifted = (int32_t)scores_row[j] - row_max;
    if (shifted < -128) shifted = -128;
    sum += exp_lut[shifted + 128];
  }

  // === Phase 3c: pass 3 — normalize + quantize back into scores_row ===
  const float inv_sum = 1.0f / sum;
  for (int j = 0; j < kN; ++j) {
    int32_t shifted = (int32_t)scores_row[j] - row_max;
    if (shifted < -128) shifted = -128;
    float e = exp_lut[shifted + 128];
    float p = e * inv_sum;
    int32_t q = (int32_t)(p * out_scale + 0.5f);
    if (q > I8_SMAX) q = I8_SMAX;
    if (q < I8_UMIN) q = I8_UMIN;
    scores_row[j] = (int8_t)q;
  }

  event1();
}

} // extern "C"
