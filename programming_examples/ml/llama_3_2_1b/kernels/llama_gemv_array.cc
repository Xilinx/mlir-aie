//===- llama_gemv_array.cc -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared GEMV array cores: every decode projection on one slot format,
// [kRows*K int8 weights | kRows int32 bias | kRows fp32 w_scales], with the
// per-token act_scale in the activation tail act[K..K+4]. Each row's
// arithmetic is the per-channel acttail kernels' in
// llama_gemm_int8_srs_tiled_layer{,_mh}.cc, so outputs are bit-identical.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "llama_gemv_array.h"

#ifndef LLAMA_GEMV_ARRAY_KSET
#define LLAMA_GEMV_ARRAY_KSET 8
#endif

// Residual units (o, down): the activation carries the residual int8[2048]
// and its fp32 scale at kResidOff; each core owns 256 of the 2048 rows and
// emits a kResidBlock-byte block [x_fp fp32 x256 | absmax bits i32 | pad].
// Both stay 64-byte aligned for the vector loads and stores.
static constexpr int kResidOff = 8256;
static constexpr int kResidRows = 256;
static constexpr int kResidBlock = kResidRows * 4 + 64;

extern "C" {

void llama_gemv_array_k2048_f32(int8_t *restrict act, int8_t *restrict w,
                                int8_t *restrict out, int32_t slot) {
  event0();
  gemv_f32<2048, 4>(act, w, reinterpret_cast<float *>(out) + slot * 4);
  event1();
}

void llama_gemv_array_k8192_f32(int8_t *restrict act, int8_t *restrict w,
                                int8_t *restrict out, int32_t slot) {
  event0();
  gemv_f32<8192, 1>(act, w, reinterpret_cast<float *>(out) + slot);
  event1();
}

void llama_gemv_array_k2048_i8(int8_t *restrict act, int8_t *restrict w,
                               int8_t *restrict out, int32_t slot,
                               float inv_out_scale) {
  event0();
  gemv_i8<2048, 4>(act, w, out + slot * 4, inv_out_scale);
  event1();
}

// One KV group's combined chunk, llama_qkv_combine's per-head layout, from
// the core's q, k and v rows (Q heads 4c..4c+3 and KV head c on core c):
//   fp  = [q fp32 x256 | k fp32 x64 | v fp32 x64]   (the core's scratch)
//   out = [q body i8 x256 | q tail 4x8 B | k fp32 x64 | v fp32 x64 | cs 256 B]
// The q body and tail are llama_q_requant's for those four heads. RoPE (S2)
// goes after the requant, on the q body in place; the cs region is reserved
// for S2's per-position table and written as zeros until then.
void llama_gemv_array_qkv_comb(int8_t *restrict fp, int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  q_requant_heads<4, 64>(reinterpret_cast<const float *>(fp), out, out + 256);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  memcpy(out + 288, fp + 1024, 512);
  memset(out + 800, 0, 256);
  event1();
}

// Core `core`'s half of llama_rescale_add_D: x_fp = resid * resid_scale +
// proj on its rows (proj is the unit's fp32 output in fp[0..256)), and the
// rows' magnitude max as int32 bits, which the glue core's
// llama_gemv_array_rescale_finalize turns into the global one.
void llama_gemv_array_resid_partial(int8_t *restrict act, int8_t *restrict fp,
                                    int8_t *restrict blk, int32_t core) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const int8_t *resid = act + kResidOff + core * kResidRows;
  ::aie::vector<float, 16> resid_scale_v = ::aie::broadcast<float, 16>(
      *reinterpret_cast<const float *>(act + kResidOff + 2048));
  const float *proj = reinterpret_cast<const float *>(fp);
  float *x = reinterpret_cast<float *>(blk);
  ::aie::vector<int32, 16> mag = ::aie::zeros<int32, 16>();
  for (int c = 0; c < kResidRows / 16; c++) {
    ::aie::vector<float, 16> xf =
        vec_residual_sum(resid + c * 16, proj + c * 16, resid_scale_v);
    ::aie::store_v(x + c * 16, xf);
    mag = ::aie::max(
        mag, ::aie::bit_and(int32(0x7fffffff), xf.template cast_to<int32>()));
  }
  *reinterpret_cast<int32_t *>(blk + kResidRows * 4) = ::aie::reduce_max(mag);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// The glue core's half of llama_rescale_add_D over the joined blocks (lo:
// cores 0-3, hi: cores 4-7): the global absmax is the integer max of the
// eight partials, then the same scale, reciprocal and requant. out is
// int8[2048 + 8] with the new scale at out[2048].
void llama_gemv_array_rescale_finalize(int8_t *restrict lo, int8_t *restrict hi,
                                       int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  int32_t absmax_bits = 0x2b8cbccc;
  for (int c = 0; c < 8; c++) {
    const int8_t *b = (c < 4 ? lo : hi) + (c % 4) * kResidBlock;
    int32_t m = *reinterpret_cast<const int32_t *>(b + kResidRows * 4);
    absmax_bits = m > absmax_bits ? m : absmax_bits;
  }
  float x_scale = __builtin_bit_cast(float, absmax_bits) * (1.0f / 127.0f);
  ::aie::vector<float, 16> inv_v = ::aie::broadcast<float, 16>(
      vec_sw_recip(::aie::broadcast<float, 16>(x_scale)).get(0));
  for (int c = 0; c < 8; c++) {
    const float *x = reinterpret_cast<const float *>((c < 4 ? lo : hi) +
                                                     (c % 4) * kResidBlock);
    for (int i = 0; i < kResidRows / 16; i++)
      ::aie::store_v(
          out + c * kResidRows + i * 16,
          vec_round_to_i8(vec_mul_rne(::aie::load_v<16>(x + i * 16), inv_v)));
  }
  *reinterpret_cast<float *>(out + 2048) = x_scale;
  *reinterpret_cast<int32_t *>(out + 2052) = 0;
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

void llama_gemv_array_topk_reset(int32_t *restrict set_len) { *set_len = 0; }

// Candidate `i` of this core's top-k set for the merging glue core
// (llama_topk_merge_insert): [logit | gidx | embed_sc | valid | pad to 64 |
// row int8 x 2048].
void llama_gemv_array_topk_cand(float *restrict set_logit,
                                int32_t *restrict set_gidx,
                                float *restrict set_scale,
                                int8_t *restrict set_row,
                                int32_t *restrict set_len, int8_t *restrict out,
                                int32_t i) {
  int32_t *hdr = reinterpret_cast<int32_t *>(out);
  hdr[0] = __builtin_bit_cast(int32_t, set_logit[i]);
  hdr[1] = set_gidx[i];
  hdr[2] = __builtin_bit_cast(int32_t, set_scale[i]);
  hdr[3] = i < *set_len;
  for (int c = 0; c < 2048 / 64; c++)
    ::aie::store_unaligned_v(
        out + 64 + c * 64,
        ::aie::load_unaligned_v<64>(set_row + i * 2048 + c * 64));
}

void llama_gemv_array_topk_pack(float *restrict set_logit,
                                int32_t *restrict set_gidx,
                                float *restrict set_scale,
                                int32_t *restrict set_len,
                                int8_t *restrict out) {
  memcpy(out, set_logit, 4 * LLAMA_GEMV_ARRAY_KSET);
  memcpy(out + 4 * LLAMA_GEMV_ARRAY_KSET, set_gidx, 4 * LLAMA_GEMV_ARRAY_KSET);
  memcpy(out + 8 * LLAMA_GEMV_ARRAY_KSET, set_scale, 4 * LLAMA_GEMV_ARRAY_KSET);
  memcpy(out + 12 * LLAMA_GEMV_ARRAY_KSET, set_len, 4);
}

} // extern "C"
