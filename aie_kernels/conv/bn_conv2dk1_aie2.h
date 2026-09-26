//===- bn_conv2dk1_aie2.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_CONV_BN_CONV2DK1_AIE2_H
#define AIE_KERNELS_CONV_BN_CONV2DK1_AIE2_H

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// AIE2 unrolls the loops over a group's N chunks by itself; AIE2P keeps them,
// and with them acc[] on the stack, unless told to unroll.
#if AIE_TUNED_AIE2P
#define K1_UNROLL_CHUNKS AIE_LOOP_UNROLL_FULL
#else
#define K1_UNROLL_CHUNKS
#endif

// 1x1 conv on [C/8][W][8] rows: each mmul<P,8,8> takes P pixels x 8 input
// channels against one [8][8] weight block. A row is covered by P-pixel
// chunks; when input_width is not a multiple of P the last chunk starts at
// input_width - P and overlaps the one before it.
template <bool Aligned, unsigned E = 32, typename T>
static inline aie::vector<T, E> k1_load(const T *p) {
  if constexpr (Aligned)
    return aie::load_v<E>(p);
  else
    return aie::load_unaligned_v<E>(p, 8);
}

// An unaligned store rewrites the enclosing 64-byte window. Buffers are
// 32-byte aligned, so storing a 32-byte aligned vector directly keeps that
// window from reaching past the end of the buffer.
template <bool Aligned, typename T>
static inline void k1_store(T *p, aie::vector<T, 32> v) {
  if (Aligned || ((uintptr_t)p & 31) == 0)
    aie::store_v(p, v);
  else
    aie::store_unaligned_v(p, v, 8);
}

template <bool Aligned, typename T>
static inline void k1_store(T *p, aie::vector<T, 64> v) {
  if constexpr (Aligned) {
    aie::store_v(p, v);
  } else {
    k1_store<false>(p, v.template extract<32>(0));
    k1_store<false>(p + 32, v.template extract<32>(1));
  }
}

// AIE2P loads each [8][8] weight block as one 64-byte aligned vector, and
// designs that pack several layers' weights into one buffer can hand in
// weights that are not 64-byte aligned.
static inline bool k1_wts_aligned(const int8_t *kernels) {
#if AIE_TUNED_AIE2P
  return ((uintptr_t)kernels & 63) == 0;
#else
  return true;
#endif
}

// On AIE2P every walker a kernel links in does not fit next to the depthwise
// conv in a mobilenet core's program memory. When the factory names the conv
// dimensions, a kernel compiles in only the walker its width takes: 8-pixel
// chunks unless their unaligned loads meet a shallow input-channel loop, with
// aligned loads when the chunks tile the row. Without them it vectorizes only
// rows the aligned 4-pixel walker takes.
#if AIE_TUNED_AIE2P && defined(CONV_INPUT_WIDTH)
#define K1_WIDTH CONV_INPUT_WIDTH
constexpr int K1_P =
    K1_WIDTH >= 8 && (K1_WIDTH % 8 == 0 || CONV_INPUT_CHANNELS >= 64) ? 8 : 4;
constexpr bool K1_ALIGNED = K1_WIDTH % K1_P == 0;
#endif

template <typename... T>
static inline bool k1_fits(const int32_t input_width, const int8_t *kernels,
                           const T *...p) {
#if defined(K1_WIDTH)
  return input_width == K1_WIDTH && k1_wts_aligned(kernels) &&
         (!K1_ALIGNED || (((uintptr_t)p | ...) & (8 * K1_P - 1)) == 0);
#elif AIE_TUNED_AIE2P
  return input_width % 4 == 0 && k1_wts_aligned(kernels) &&
         (((uintptr_t)p | ...) & 31) == 0;
#else
  return true;
#endif
}

// N chunks of one output channel block. Chunk j is at byte offset 8 * P * j
// from in, except the last at last_off; epi(acc, side + offset...) gives the
// vector stored at out + offset.
template <bool Aligned, int P, int N, typename TI, typename TO, typename Epi,
          typename... S>
static inline void
k1_chunks(const TI *__restrict in, const int8_t *__restrict wts,
          TO *__restrict out, const int32_t row, const int32_t ic_blocks,
          const int32_t last_off, Epi epi, const S *__restrict... side) {
  constexpr int E = 8 * P;
  using MMUL = aie::mmul<P, 8, 8, TI, int8>;
  MMUL acc[N];
  aie::vector<int8, 64> b = aie::load_v<64>(wts);
  K1_UNROLL_CHUNKS
  for (int j = 0; j < N; j++)
    acc[j].mul(k1_load<Aligned, E>(in + (j == N - 1 ? last_off : E * j)), b);
#pragma clang loop min_iteration_count(1)
  for (int ic = 1; ic < ic_blocks; ic++) {
    in += row;
    wts += 64;
    b = aie::load_v<64>(wts);
    K1_UNROLL_CHUNKS
    for (int j = 0; j < N; j++)
      acc[j].mac(k1_load<Aligned, E>(in + (j == N - 1 ? last_off : E * j)), b);
  }
  K1_UNROLL_CHUNKS
  for (int j = 0; j < N; j++) {
    const int32_t o = j == N - 1 ? last_off : E * j;
    k1_store<Aligned>(out + o, epi(acc[j], (side + o)...));
  }
}

// Walks the whole [OC/8][W][8] output; side buffers share its layout.
// Needs input_width >= P.
template <bool Aligned, int P = 4, typename TI, typename TO, typename Epi,
          typename... S>
static void k1_rows(const TI *input, const int8_t *kernels, TO *output,
                    const int32_t input_width, const int32_t input_channels,
                    const int32_t output_channels, Epi epi, const S *...side) {
  constexpr int N = 4;
  constexpr int E = 8 * P;
  const int32_t row = input_width * 8;
  const int32_t ic_blocks = input_channels / 8;
  const int32_t chunks = (input_width + P - 1) / P;
  const int32_t groups = chunks / N;
  const int32_t rem = chunks % N;
  const int32_t tail = (input_width - P) * 8;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *wts = kernels + oc * ic_blocks * 64;
    TO *out = output + oc * row;
    for (int g = 0; g < groups; g++) {
      const int32_t x = g * N * E;
      const int32_t last =
          (rem == 0 && g == groups - 1) ? tail - x : E * (N - 1);
      k1_chunks<Aligned, P, N>(input + x, wts, out + x, row, ic_blocks, last,
                               epi, (side + oc * row + x)...);
    }
    const int32_t x = groups * N * E;
    switch (rem) {
    case 1:
      k1_chunks<Aligned, P, 1>(input + x, wts, out + x, row, ic_blocks,
                               tail - x, epi, (side + oc * row + x)...);
      break;
    case 2:
      k1_chunks<Aligned, P, 2>(input + x, wts, out + x, row, ic_blocks,
                               tail - x, epi, (side + oc * row + x)...);
      break;
    case 3:
      k1_chunks<Aligned, P, 3>(input + x, wts, out + x, row, ic_blocks,
                               tail - x, epi, (side + oc * row + x)...);
      break;
    }
  }
}

#if AIE_TUNED_AIE2P
// The cascade-split pairs: a call is one output channel block of 7 pixels,
// as 4-pixel chunks at pixels 0 and 3, and each pixel's 8 partial sums cross
// the cascade as the low lanes of a v16acc64.
constexpr int K1_CAS_PIXELS = 7;
constexpr int K1_CAS_LAST = 8 * (K1_CAS_PIXELS - 4);

template <typename TI>
static inline void
k1_cas_conv(const TI *__restrict in, const int8_t *__restrict wts,
            const int32_t row, const int32_t ic_blocks,
            aie::accum<acc32, 32> &lo, aie::accum<acc32, 32> &hi) {
  using MMUL = aie::mmul<4, 8, 8, TI, int8>;
  MMUL a, b;
  aie::vector<int8, 64> w = aie::load_v<64>(wts);
  a.mul(k1_load<false>(in), w);
  b.mul(k1_load<false>(in + K1_CAS_LAST), w);
  if (ic_blocks > 1) {
#pragma clang loop min_iteration_count(1)
    for (int ic = 1; ic < ic_blocks; ic++) {
      in += row;
      wts += 64;
      w = aie::load_v<64>(wts);
      a.mac(k1_load<false>(in), w);
      b.mac(k1_load<false>(in + K1_CAS_LAST), w);
    }
  }
  lo = a.to_accum();
  hi = b.to_accum();
}

template <typename TI>
static void k1_cas_put(const TI *in, const int8_t *wts, const int32_t row,
                       const int32_t ic_blocks) {
  aie::accum<acc32, 32> lo, hi;
  k1_cas_conv(in, wts, row, ic_blocks, lo, hi);
  const aie::vector<int32, 32> a = lo.template to_vector<int32>();
  const aie::vector<int32, 32> b = hi.template to_vector<int32>();
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < 4; p++)
    put_mcd(lups(a.extract<8>(p).template grow<16>(), 0));
  AIE_LOOP_UNROLL_FULL
  for (int p = 1; p < 4; p++)
    put_mcd(lups(b.extract<8>(p).template grow<16>(), 0));
}

// epi(acc, side...) gives a chunk's 4 requantized pixels; side buffers share
// the output's layout. Each pixel is stored as two words, as a vector store
// here could rewrite bytes past the end of the output. The saturation mode
// goes back as it was: the scalar path reads the cascade with an unsigned
// lsrs.
template <typename TI, typename TO, typename Epi, typename... S>
static void k1_cas_get(const TI *in, const int8_t *wts, TO *out,
                       const int32_t row, const int32_t ic_blocks, Epi epi,
                       const S *...side) {
  const aie::saturation_mode sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::accum<acc32, 32> lo, hi;
  k1_cas_conv(in, wts, row, ic_blocks, lo, hi);
  aie::vector<int32, 8> c[K1_CAS_PIXELS];
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < K1_CAS_PIXELS; p++)
    c[p] = aie::vector<int32, 16>(lsrs(get_scd_v16acc64(), 0, 1)).extract<8>(0);
  lo = aie::add(lo, aie::concat(c[0], c[1], c[2], c[3]));
  hi = aie::add(hi, aie::concat(c[3], c[4], c[5], c[6]));
  const aie::vector<uint32, 8> a = aie::vector_cast<uint32>(epi(lo, side...));
  const aie::vector<uint32, 8> b =
      aie::vector_cast<uint32>(epi(hi, (side + K1_CAS_LAST)...));
  uint32_t *o = (uint32_t *)out;
  AIE_LOOP_UNROLL_FULL
  for (int i = 0; i < 8; i++)
    o[i] = a[i];
  AIE_LOOP_UNROLL_FULL
  for (int i = 2; i < 8; i++)
    o[6 + i] = b[i];
  aie::set_saturation(sat);
}

// Channels per split for a power-of-two split. A runtime divide is a
// libcall, and a kernel that calls anything saves registers on every call,
// so the cascade wrappers try the vector path, which calls nothing, before
// the scalar helpers.
static inline int32_t k1_per_split(const int32_t c, const int32_t split) {
  return c >> __builtin_ctz(split);
}
#endif

static inline bool k1_cas_fits(const int8_t *kernels, const int32_t in_split,
                               const int32_t out_split) {
#if AIE_TUNED_AIE2P
  const int32_t pow2 =
      (in_split & (in_split - 1)) | (out_split & (out_split - 1));
  return k1_wts_aligned(kernels) && in_split > 0 && out_split > 0 && pow2 == 0;
#else
  return false;
#endif
}

template <typename TI>
static inline bool k1_cas_put_new(const TI *input, const int8_t *kernels,
                                  const int32_t input_width,
                                  const int32_t input_channels,
                                  const int32_t input_split, const int32_t oc) {
#if AIE_TUNED_AIE2P
  if (!k1_cas_fits(kernels, input_split, 1))
    return false;
  event0();
  const int32_t blocks = k1_per_split(input_channels, input_split) / 8;
  k1_cas_put(input, kernels + oc * blocks * 64, input_width * 8, blocks);
  event1();
  return true;
#else
  return false;
#endif
}

#endif
