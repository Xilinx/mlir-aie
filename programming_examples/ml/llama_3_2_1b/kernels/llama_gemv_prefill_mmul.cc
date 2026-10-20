//===- llama_gemv_prefill_mmul.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array's M-row prefill cores on aie::mmul<4, 8, 8> (int8, int32
// accumulation): a 4x8 tile of a slot's weights against an 8x8 tile of
// eight activations' transposed columns. Sums and per-element epilogue are
// the GEMV's, so every row is byte-identical to a GEMV row. Slot, activation
// and output layouts: test_gemv_array.pack_lane_prefill / prefill_act_object.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

#ifndef LLAMA_GEMV_ARRAY_M
#define LLAMA_GEMV_ARRAY_M 8
#endif

static constexpr int kM = LLAMA_GEMV_ARRAY_M;
static constexpr int kGroups = (kM + 7) / 8;
static constexpr int kGroupBytes = 2048 * 8;
static constexpr int kScalesOff = kGroups * kGroupBytes;
static constexpr int kOutRows = 64;
using MMUL = ::aie::mmul<4, 8, 8, int8, int8, acc32>;

// Activation group g's sums against the slot's four rows: lane n*8 + m holds
// row n against activation 8g + m. This and group_epilogue stay out of line
// so the entry points share one copy in the 16 KB program memory.
__attribute__((noinline)) static ::aie::vector<int32, 32>
group_sums(const int8_t *restrict act, const int8_t *restrict w, int g) {
  const int8_t *b = act + g * kGroupBytes;
  MMUL c;
  c.mul(::aie::load_v<32>(w), ::aie::load_v<64>(b));
  for (int kt = 1; kt < 2048 / 8; kt++)
    c.mac(::aie::load_v<32>(w + kt * 32), ::aie::load_v<64>(b + kt * 64));
  return c.template to_vector<int32>();
}

// The f32 or (i8) gate epilogue on one group's 4x8 sums, written to
// out[m][sic*4 + n]: the GEMV epilogue's arithmetic per element, on all 32
// lanes at once where the GEMV's guards hold for every lane (see
// vec_epilogue_exact and gemm_tile_perchan_acttail_impl), else its scalar
// code per element.
// One copy serves both (i8 is a runtime flag) to save program memory.
__attribute__((noinline)) static void
group_epilogue(::aie::vector<int32, 32> sums, const int8_t *restrict act,
               const int8_t *restrict w, int8_t *restrict out, int g,
               int32_t sic, bool i8, float inv_out_scale) {
  const int32_t *bias = reinterpret_cast<const int32_t *>(w + 4 * 2048);
  const float *w_scales = reinterpret_cast<const float *>(w + 4 * 2048 + 16);
  const float *act_scales =
      reinterpret_cast<const float *>(act + kScalesOff) + 8 * g;
  const int m_end = kM - 8 * g < 8 ? kM - 8 * g : 8;
  const uint32_t kLo = i8 ? 96 : 80, kHi = i8 ? 158 : 174;
  bool ok =
      !i8 ||
      (uint32_t)(((__builtin_bit_cast(int32_t, inv_out_scale) >> 23) & 0xff) -
                 kLo) <= kHi - kLo;
  for (int i = 0; i < 8 + 4; i++) {
    float f = i < 8 ? act_scales[i] : w_scales[i - 8];
    ok &= (uint32_t)(((__builtin_bit_cast(int32_t, f) >> 23) & 0xff) - kLo) <=
          kHi - kLo;
  }
  ::aie::vector<float, 8> a8 = ::aie::load_v<8>(act_scales);
  ::aie::vector<float, 32> a_v = ::aie::concat(a8, a8, a8, a8);
  ::aie::vector<float, 32> w_v =
      ::aie::concat(::aie::broadcast<float, 8>(w_scales[0]),
                    ::aie::broadcast<float, 8>(w_scales[1]),
                    ::aie::broadcast<float, 8>(w_scales[2]),
                    ::aie::broadcast<float, 8>(w_scales[3]));
  sums = ::aie::add(sums, ::aie::concat(::aie::broadcast<int32, 8>(bias[0]),
                                        ::aie::broadcast<int32, 8>(bias[1]),
                                        ::aie::broadcast<int32, 8>(bias[2]),
                                        ::aie::broadcast<int32, 8>(bias[3])));
  ok &=
      (::aie::ge(sums, int32(-(1 << 23) + 1)) & ::aie::lt(sums, int32(1 << 23)))
          .count() == 32;
  ::aie::vector<float, 32> fp;
  if (ok) {
    fp = vec_mul_rne(vec_mul_rne(::aie::to_float<float>(sums, 0), a_v), w_v);
    if (i8) {
      fp = vec_mul_rne(fp, ::aie::broadcast<float, 32>(inv_out_scale));
      ok = ::aie::reduce_max(::aie::bit_and(
               int32(0x7fffffff), fp.template cast_to<int32>())) < 0x4f000000;
    }
  }
  ::aie::vector<int8, 32> r8;
  if (i8 && ok)
    r8 = vec_round_to_i8(fp);
  for (int m = 0; m < m_end; m++)
    for (int n = 0; n < 4; n++) {
      const int lane = n * 8 + m;
      if (i8) {
        int8_t v;
        if (ok) {
          v = r8.get(lane);
        } else {
          float f = (float)sums.get(lane) * act_scales[m] * w_scales[n];
          float scaled = f * inv_out_scale;
          int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
          v = (int8_t)(r > I8_MAX ? I8_MAX : r < I8_MIN ? I8_MIN : r);
        }
        out[(8 * g + m) * kOutRows * 4 + sic * 4 + n] = v;
      } else {
        reinterpret_cast<float *>(out)[(8 * g + m) * kOutRows + sic * 4 + n] =
            ok ? fp.get(lane)
               : (float)sums.get(lane) * act_scales[m] * w_scales[n];
      }
    }
}

extern "C" {

void llama_gemv_mmul_f32(int8_t *restrict act, int8_t *restrict w,
                         int8_t *restrict out, int32_t sic) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int g = 0; g < kGroups; g++)
    group_epilogue(group_sums(act, w, g), act, w, out, g, sic, false, 0.0f);
  event1();
}

void llama_gemv_mmul_i8(int8_t *restrict act, int8_t *restrict w,
                        int8_t *restrict out, int32_t sic,
                        float inv_out_scale) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int g = 0; g < kGroups; g++)
    group_epilogue(group_sums(act, w, g), act, w, out, g, sic, true,
                   inv_out_scale);
  event1();
}

// Down's first three column chunks: the slot's sums for every activation go
// into (chunk 0) or onto (chunks 1, 2) partial, int32[kM x rows], laid out
// [m][slot*4 + n] with `rows` the down rows this pass covers.
void llama_gemv_mmul_kchunk(int8_t *restrict act, int8_t *restrict w,
                            int32_t *restrict partial, int32_t slot,
                            int32_t chunk, int32_t rows) {
  event0();
  for (int g = 0; g < kGroups; g++) {
    ::aie::vector<int32, 32> sums = group_sums(act, w, g);
    const int m_end = kM - 8 * g < 8 ? kM - 8 * g : 8;
    for (int m = 0; m < m_end; m++)
      for (int n = 0; n < 4; n++) {
        int32_t *p = partial + (8 * g + m) * rows + slot * 4 + n;
        *p = chunk == 0 ? sums.get(n * 8 + m) : *p + sums.get(n * 8 + m);
      }
  }
  event1();
}

// Down's last chunk: partial plus this chunk's sums, then the f32 epilogue.
void llama_gemv_mmul_kfinal(int8_t *restrict act, int8_t *restrict w,
                            int32_t *restrict partial, int8_t *restrict out,
                            int32_t slot, int32_t sic, int32_t rows) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int g = 0; g < kGroups; g++) {
    ::aie::vector<int32, 32> sums = group_sums(act, w, g);
    const int m_end = kM - 8 * g < 8 ? kM - 8 * g : 8;
    for (int m = 0; m < m_end; m++)
      for (int n = 0; n < 4; n++)
        sums.set(sums.get(n * 8 + m) +
                     partial[(8 * g + m) * rows + slot * 4 + n],
                 n * 8 + m);
    group_epilogue(sums, act, w, out, g, sic, false, 0.0f);
  }
  event1();
}

} // extern "C"
