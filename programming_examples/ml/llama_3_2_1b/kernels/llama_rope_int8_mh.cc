//===- llama_rope_int8_mh.cc --------------------------------*- C++ -*-===//
// Multi-head GQA rope: one call rotates all N_HEADS_Q Q heads (32 for
// Llama 3.2 1B at HEAD_DIM=64) and passes a per-Q-head scale tail through
// unchanged (q_scale_h, sv_inv_out_scale_h fp32 pairs, 8 B/head) so the
// downstream q-splitter + flowkv_mh kernel can find them. Body+tail
// layout is input == output. Math mirrors llama_rope_int8_dyn's
// simplified form: act_scale and 1/act_scale cancel through rope, and
// dropping them removes 1-ULP noise relative to including them.
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_ROPE_MH_HEAD_DIM
#define LLAMA_ROPE_MH_HEAD_DIM 64
#endif
#ifndef LLAMA_ROPE_MH_N_HEADS
#define LLAMA_ROPE_MH_N_HEADS 32
#endif

static constexpr int kHD = LLAMA_ROPE_MH_HEAD_DIM;
static constexpr int kNHeads = LLAMA_ROPE_MH_N_HEADS;
static constexpr int kHalf = kHD / 2;
static constexpr int kBody = kHD * kNHeads;
static constexpr int kTail = kNHeads * 8;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static_assert(kHD % 2 == 0, "head_dim must be even");

static inline int8_t round_to_i8(float v) {
  int32_t sign = (v >= 0.0f) ? 1 : -1;
  float scaled = (float)sign * v * 2.0f + 1.0f;
  int32_t doubled = (int32_t)scaled;
  int32_t r = sign * (doubled / 2);
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

extern "C" void llama_rope_int8_mh_dyn(int8_t *restrict x,
                                       bfloat16 *restrict cs_packed,
                                       int8_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);

  const bfloat16 *cos = cs_packed;
  const bfloat16 *sin = cs_packed + kHD;

  for (int h = 0; h < kNHeads; h++) {
    int base = h * kHD;
    for (int i = 0; i < kHalf; i++) {
      float x1f = (float)x[base + i];
      float x2f = (float)x[base + kHalf + i];
      float c1 = (float)cos[i];
      float s1 = (float)sin[i];
      float c2 = (float)cos[kHalf + i];
      float s2 = (float)sin[kHalf + i];

      float o1 = x1f * c1 - x2f * s1;
      float o2 = x2f * c2 + x1f * s2;

      out[base + i] = round_to_i8(o1);
      out[base + kHalf + i] = round_to_i8(o2);
    }
  }

  // Passthrough the per-head scale tail unchanged.
  memcpy(out + kBody, x + kBody, kTail);

  event1();
}
