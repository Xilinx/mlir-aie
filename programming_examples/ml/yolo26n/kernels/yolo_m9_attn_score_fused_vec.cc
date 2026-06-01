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
// Phase 1+2 (qk + scale combined): vector-broadcast pattern, 64-wide
// j-groups with aie::accum<acc32, 64>, broadcast Q[k] vmac against
// K[k, j_base..j_base+63] strip. After inner k loop, vec
// to_vector<int8>(rs_qk) -> aie::mul(qk_v, mul_int) -> to_vector<int8>
// (mul_shift) emits the combined SRS as two vec ops. 64-wide because
// peano AIE2P backend crashes on accum<acc32, {16,32}>::to_vector<int8>.
//
// Phase 3 (softmax): 3-pass algorithm matching the original
// softmax_row.cc — peano's aie2p libc++ doesn't provide expf so we use
// the precomputed fp32 LUT. Pass 1 (row max) is vector-reduce. Pass 2
// gathers the exp values into a 1 KB float[256] scratch (worker stack
// bumped to 4 KB to fit) and reduces sum. Pass 3 normalizes via
// `aie::inv(sum)` (HW reciprocal, no __divsf3) + a 16-wide vec FP mul
// + saturating cast back to i8.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_QK_KD
#error "YOLO_M9_QK_KD must be defined at compile time"
#endif
#ifndef YOLO_M9_QK_N
#error "YOLO_M9_QK_N must be defined at compile time"
#endif

static constexpr int kKd = YOLO_M9_QK_KD;
static constexpr int kN = YOLO_M9_QK_N;
static constexpr int kJVec =
    64; // peano accum::to_vector<int8> crashes at 16 + 32, OK at 64
static constexpr int kJGroups = kN / kJVec;

static_assert(kKd > 0, "QK_KD must be > 0");
static_assert(kN % kJVec == 0, "QK_N must be a multiple of 16");

static constexpr int32_t I8_SMAX = 127;
static constexpr int32_t I8_SMIN = -128;
static constexpr int32_t I8_UMIN =
    0; // softmax probs in [0, 1] → i8 in [0, 127]

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_attn_score_fused_i8_i8(
    int8_t *qk_frame, // (2*kd, N) Q || K
    int8_t *chunk_io, // (chunk_rows, N) — only chunk_io[chunk_row] is touched
    float *exp_lut,   // (256,) softmax exp LUT
    const int32_t chunk_row, const int32_t query_idx, const int32_t /*kd*/,
    const int32_t /*N*/,
    const int32_t rs_qk,            // qk right_shift (e.g. 3 for m9)
    const int32_t mul_int,          // attn scale mul (e.g. 91 for m9)
    const int32_t mul_shift,        // attn scale shift (e.g. 7 for m9)
    const int32_t out_log2_scale) { // softmax out scale (e.g. -7 for m9; out =
                                    // prob * 128)
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *__restrict scores_row = chunk_io + chunk_row * kN;
  const int8_t *__restrict q_col = qk_frame + query_idx; // stride kN per k
  const int8_t *__restrict k_base = qk_frame + kKd * kN; // K rows start here

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs (round-half-to-even). Lets us use
  // vector to_vector<int8>(rs) instead of the per-element scalar SRS/clip
  // tail without LSB drift.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // === Phase 1+2: qk + scale → i8 scores row in chunk_io ===
  // Inner mac is vector. SRS+clip + scale+SRS+clip vectorized via two
  // to_vector<int8>(rs) calls bracketing a vec `aie::mul(qk_v, mul_int)`.
  AIE_LOOP_RANGE(kJGroups, kJGroups)
  for (int g = 0; g < kJGroups; ++g) {
    aie::accum<acc32, kJVec> acc;
    acc.from_vector(aie::zeros<int32, kJVec>());

    const int8_t *__restrict k_strip = k_base + g * kJVec;

    AIE_LOOP_RANGE(kKd, kKd)
    AIE_LOOP_UNROLL(4)
    for (int k = 0; k < kKd; ++k) {
      const int8_t q_scalar = q_col[k * kN];
      aie::vector<int8, kJVec> q_v = aie::broadcast<int8, kJVec>(q_scalar);
      aie::vector<int8, kJVec> k_v = aie::load_v<kJVec>(k_strip + k * kN);
      acc = aie::mac(acc, q_v, k_v);
    }

    aie::vector<int8, kJVec> qk_v = acc.template to_vector<int8>(rs_qk);
    aie::accum<acc32, kJVec> scaled_acc = aie::mul(qk_v, (int16)mul_int);
    aie::vector<int8, kJVec> sr_v =
        scaled_acc.template to_vector<int8>(mul_shift);
    aie::store_v(scores_row + g * kJVec, sr_v);
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

  // === Phase 3b: pass 2 — sum of exps, with per-element exp cached ===
  // Cache the gathered exp values so phase 3c becomes pure vec mul-shift
  // (no re-gather, no per-element fdiv). 1 KB stack — fits on (7,3)
  // alongside attn_score's tiny locals.
  const int32_t out_scale_int = 1 << (-out_log2_scale); // 128 for m9

  alignas(8) float exp_cache[kN];
  // 4 parallel float accumulators break the serial fp-add dependency chain,
  // letting peano overlap 4 independent (lda + fadd) issue streams instead
  // of stalling on the 1 fadd per iter.
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  static_assert(kN % 4 == 0, "kN must be multiple of 4 for 4-way sum unroll");
  for (int j = 0; j < kN; j += 4) {
    int32_t s0 = (int32_t)scores_row[j + 0] - row_max;
    int32_t s1 = (int32_t)scores_row[j + 1] - row_max;
    int32_t s2 = (int32_t)scores_row[j + 2] - row_max;
    int32_t s3 = (int32_t)scores_row[j + 3] - row_max;
    if (s0 < -128) s0 = -128;
    if (s1 < -128) s1 = -128;
    if (s2 < -128) s2 = -128;
    if (s3 < -128) s3 = -128;
    float e0 = exp_lut[s0 + 128];
    float e1 = exp_lut[s1 + 128];
    float e2 = exp_lut[s2 + 128];
    float e3 = exp_lut[s3 + 128];
    exp_cache[j + 0] = e0;
    exp_cache[j + 1] = e1;
    exp_cache[j + 2] = e2;
    exp_cache[j + 3] = e3;
    sum0 += e0;
    sum1 += e1;
    sum2 += e2;
    sum3 += e3;
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);

  // === Phase 3c: pass 3 — vec FP normalize + quantize ===
  // aie::inv(float) = HW reciprocal (single op) — replaces __divsf3.
  // Fold out_scale into the broadcast scale so per-element work is one
  // vec fpmul. Clamp + narrow via aie::min/max (no scalar if/else).
  const float scale = (float)out_scale_int * aie::inv(sum);

  constexpr int kFVec = 16;
  aie::vector<float, kFVec> scale_v = aie::broadcast<float, kFVec>(scale);
  aie::vector<int32, kFVec> smax_v = aie::broadcast<int32, kFVec>(I8_SMAX);
  aie::vector<int32, kFVec> smin_v = aie::broadcast<int32, kFVec>(I8_UMIN);

  AIE_LOOP_RANGE(kN / kFVec, kN / kFVec)
  for (int j = 0; j < kN; j += kFVec) {
    aie::vector<float, kFVec> e_v = aie::load_v<kFVec>(exp_cache + j);
    aie::accum<accfloat, kFVec> p_acc = aie::mul(e_v, scale_v);
    aie::vector<int32, kFVec> q_v =
        aie::to_fixed<int32>(p_acc.template to_vector<float>(), 0);
    q_v = aie::min(q_v, smax_v);
    q_v = aie::max(q_v, smin_v);
    // Narrow i32 → i8. UNROLL_FULL lets peano pipeline 16 stores (and the
    // scalar-extracts) instead of running them as a serial loop.
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < kFVec; ++i)
      scores_row[j + i] = (int8_t)q_v[i];
  }

  event1();
}

} // extern "C"
