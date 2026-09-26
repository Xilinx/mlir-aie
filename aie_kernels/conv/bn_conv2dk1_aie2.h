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

// k1_rows_fixed walks an even-width row with the conv's shape as constants;
// the pipeliner does not take the reduction loop k1_chunks runs once per
// group, nor a loop with k1_store's alignment branch. An even-width row puts
// every chunk on a boundary of K1_FIXED_ALIGN bytes, the most a row's length
// keeps. A kernel without k1_rows_fixed defines K1_NO_FIXED.
constexpr int K1_IC_BLOCKS = CONV_INPUT_CHANNELS / 8;
constexpr int K1_OC_BLOCKS = CONV_OUTPUT_CHANNELS / 8;
#ifdef K1_NO_FIXED
constexpr bool K1_EVEN = false;
#else
constexpr bool K1_EVEN = K1_WIDTH >= 8 && K1_WIDTH % 2 == 0;
#endif
constexpr unsigned K1_FIXED_ALIGN = K1_WIDTH % 8 == 0   ? 64
                                    : K1_WIDTH % 4 == 0 ? 32
                                                        : 16;
constexpr int K1_ROW = K1_WIDTH * 8;
constexpr int K1_CHUNKS = (K1_WIDTH + 7) / 8;
constexpr int K1_U = 448 % K1_ROW == 0 ? 448 / K1_ROW : 1;
constexpr int K1_VALID = K1_ROW < 64 ? K1_ROW : 64;
// k1_rows_deep also beats k1_fixed_reduce where both apply. Kernels that link
// it define K1_DEEP_WALKER before including this header.
#if defined(K1_DEEP_WALKER)
constexpr bool K1_DEEP = K1_WIDTH % 7 == 0 && K1_U >= 2 &&
                         K1_IC_BLOCKS >= K1_U &&
                         (!K1_EVEN || K1_IC_BLOCKS > 10);
#else
constexpr bool K1_DEEP = false;
#endif
constexpr bool K1_FIXED = K1_EVEN && !K1_DEEP;
// Output channel blocks per pass: more than four accumulators spill.
constexpr int K1_DEEP_M = K1_CHUNKS >= 4 ? 1 : 2;

constexpr uintptr_t K1_PTR_MASK = K1_FIXED     ? K1_FIXED_ALIGN - 1
                                  : K1_DEEP    ? 63
                                  : K1_ALIGNED ? 8 * K1_P - 1
                                               : 0;
#endif

template <typename... T>
static inline bool k1_fits(const int32_t input_width, const int8_t *kernels,
                           const T *...p) {
#if defined(K1_WIDTH)
  return input_width == K1_WIDTH && k1_wts_aligned(kernels) &&
         (((uintptr_t)p | ...) & K1_PTR_MASK) == 0;
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

#if defined(K1_WIDTH)
template <typename T>
static inline aie::vector<T, 64> k1_load_fixed(const T *p) {
  if constexpr (K1_FIXED_ALIGN == 64)
    return aie::load_v<64>(p);
  else if constexpr (K1_FIXED_ALIGN == 32)
    return aie::concat(aie::load_v<32>(p), aie::load_v<32>(p + 32));
  else
    return aie::load_unaligned_v<64>(p, K1_FIXED_ALIGN);
}

template <typename T>
static inline void k1_store_fixed(T *p, aie::vector<T, 64> v) {
  constexpr unsigned A = K1_FIXED_ALIGN;
  AIE_LOOP_UNROLL_FULL
  for (unsigned i = 0; i < 64 / A; i++)
    aie::store_v(p + A * i, v.template extract<A>(i));
}

// The pipelined loop steps through one operand while the other's vectors stay
// in registers: G output channel blocks' weights while it walks the row's
// 8-pixel chunks, or a chunk's inputs while it walks the output channel
// blocks GO at a time. The last group of GO blocks ends at the last block, so
// it recomputes some of the group before it.
template <int G, typename TI, typename TO, typename Epi>
static void k1_fixed_hold_wts(const TI *__restrict input,
                              const int8_t *__restrict kernels,
                              TO *__restrict output, Epi epi) {
  constexpr int row = K1_WIDTH * 8;
  constexpr int chunks = K1_WIDTH / 8;
  using MMUL = aie::mmul<8, 8, 8, TI, int8>;
  for (int oc = 0; oc < K1_OC_BLOCKS; oc += G) {
    aie::vector<int8, 64> b[G][K1_IC_BLOCKS];
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < G; g++)
      AIE_LOOP_UNROLL_FULL
    for (int ic = 0; ic < K1_IC_BLOCKS; ic++)
      b[g][ic] = aie::load_v<64>(kernels + (g * K1_IC_BLOCKS + ic) * 64);
    kernels += G * K1_IC_BLOCKS * 64;
    const TI *__restrict in = input;
    TO *__restrict out = output + oc * row;
    for (int c = 0; c < chunks; c++) {
      aie::vector<TI, 64> a[K1_IC_BLOCKS];
      AIE_LOOP_UNROLL_FULL
      for (int ic = 0; ic < K1_IC_BLOCKS; ic++)
        a[ic] = k1_load_fixed(in + ic * row);
      in += 64;
      AIE_LOOP_UNROLL_FULL
      for (int g = 0; g < G; g++) {
        MMUL acc;
        AIE_LOOP_UNROLL_FULL
        for (int ic = 0; ic < K1_IC_BLOCKS; ic++) {
          if (ic == 0)
            acc.mul(a[ic], b[g][ic]);
          else
            acc.mac(a[ic], b[g][ic]);
        }
        k1_store_fixed(out + g * row, epi(acc));
      }
      out += 64;
    }
  }
}

template <int GO, typename TI, typename TO, typename Epi>
static void k1_fixed_hold_in(const TI *__restrict input,
                             const int8_t *__restrict kernels,
                             TO *__restrict output, Epi epi) {
  constexpr int row = K1_WIDTH * 8;
  constexpr int last = row - 64;
  constexpr int chunks = (K1_WIDTH + 7) / 8;
  constexpr int step = K1_IC_BLOCKS * 64;
  constexpr int groups = (K1_OC_BLOCKS + GO - 1) / GO;
  constexpr int back = groups * GO - K1_OC_BLOCKS;
  using MMUL = aie::mmul<8, 8, 8, TI, int8>;
  for (int c = 0; c < chunks; c++) {
    const int x = c < chunks - 1 ? 64 * c : last;
    aie::vector<TI, 64> a[K1_IC_BLOCKS];
    AIE_LOOP_UNROLL_FULL
    for (int ic = 0; ic < K1_IC_BLOCKS; ic++)
      a[ic] = k1_load_fixed(input + ic * row + x);
    const int8_t *__restrict wts = kernels;
    TO *__restrict out = output + x;
    for (int k = 0; k < groups; k++) {
      MMUL acc[GO];
      AIE_LOOP_UNROLL_FULL
      for (int ic = 0; ic < K1_IC_BLOCKS; ic++) {
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < GO; o++) {
          const aie::vector<int8, 64> b = aie::load_v<64>(wts + o * step);
          if (ic == 0)
            acc[o].mul(a[ic], b);
          else
            acc[o].mac(a[ic], b);
        }
        wts += 64;
      }
      AIE_LOOP_UNROLL_FULL
      for (int o = 0; o < GO; o++)
        k1_store_fixed(out + o * row, epi(acc[o]));
      const int adv = back && k == groups - 2 ? GO - back : GO;
      wts += (adv - 1) * step;
      out += adv * row;
    }
  }
}

// A deeper reduction does not fit in registers, so the pipelined loop is the
// input-channel loop, for one chunk of G output channel blocks.
template <int G, typename TI, typename TO, typename Epi>
static void k1_fixed_reduce(const TI *__restrict input,
                            const int8_t *__restrict kernels,
                            TO *__restrict output, Epi epi) {
  constexpr int row = K1_WIDTH * 8;
  constexpr int last = row - 64;
  constexpr int chunks = (K1_WIDTH + 7) / 8;
  constexpr int step = K1_IC_BLOCKS * 64;
  using MMUL = aie::mmul<8, 8, 8, TI, int8>;
  for (int oc = 0; oc < K1_OC_BLOCKS; oc += G) {
    const int o = oc + G <= K1_OC_BLOCKS ? oc : K1_OC_BLOCKS - G;
    for (int c = 0; c < chunks; c++) {
      const int x = c < chunks - 1 ? 64 * c : last;
      const TI *__restrict in = input + x;
      const int8_t *__restrict wts = kernels + o * step;
      MMUL acc[G];
      aie::vector<TI, 64> a = k1_load_fixed(in);
      AIE_LOOP_UNROLL_FULL
      for (int g = 0; g < G; g++)
        acc[g].mul(a, aie::load_v<64>(wts + g * step));
      for (int ic = 1; ic < K1_IC_BLOCKS; ic++) {
        in += row;
        wts += 64;
        a = k1_load_fixed(in);
        AIE_LOOP_UNROLL_FULL
        for (int g = 0; g < G; g++)
          acc[g].mac(a, aie::load_v<64>(wts + g * step));
      }
      TO *__restrict out = output + o * row + x;
      AIE_LOOP_UNROLL_FULL
      for (int g = 0; g < G; g++)
        k1_store_fixed(out + g * row, epi(acc[g]));
    }
  }
}

// n is the fewest cycles per mac measured for each shape.
template <typename TI, typename TO, typename Epi>
static void k1_rows_fixed(const TI *__restrict input,
                          const int8_t *__restrict kernels,
                          TO *__restrict output, Epi epi) {
  constexpr int n = K1_IC_BLOCKS > 10   ? 4
                    : K1_IC_BLOCKS <= 5 ? 1
                    : K1_OC_BLOCKS > 6  ? 3
                                        : 2;
  constexpr int g = n < K1_OC_BLOCKS ? n : K1_OC_BLOCKS;
  if constexpr (K1_IC_BLOCKS > 10)
    k1_fixed_reduce<g>(input, kernels, output, epi);
  else if constexpr (K1_WIDTH % 8 == 0 && K1_WIDTH / 8 >= K1_OC_BLOCKS)
    k1_fixed_hold_wts<K1_IC_BLOCKS <= 2 && K1_OC_BLOCKS % 2 == 0 ? 2 : 1>(
        input, kernels, output, epi);
  else
    k1_fixed_hold_in<g>(input, kernels, output, epi);
}

// As in k1_fixed_reduce, with accumulators for every 8-pixel chunk of a row
// times M. K1_U rows are 448 bytes, seven aligned 64-byte windows; one step
// loads those once and shifts each chunk out of them. A row narrower than 8
// pixels takes one chunk whose last pixel belongs to the next row and is never
// stored.
constexpr int k1_deep_x(int j) {
  return j < K1_CHUNKS - 1 ? 64 * j : K1_ROW - K1_VALID;
}

// The 64 bytes at offset s of the windows w, of which the first K1_VALID are
// used.
template <typename T>
static inline aie::vector<T, 64> k1_win(const aie::vector<T, 64> *w,
                                        const int s) {
  const int i = s / 64, f = s % 64;
  if (f == 0)
    return w[i];
  return ::shift_bytes(w[i], f + K1_VALID <= 64 ? w[i] : w[i + 1], f);
}

template <typename T>
static inline aie::vector<T, 64> k1_row_load(const T *__restrict p,
                                             const int s) {
  aie::vector<T, 64> w[2];
  w[0] = aie::load_v<64>(p + s / 64 * 64);
  if (s % 64 + K1_VALID > 64)
    w[1] = aie::load_v<64>(p + s / 64 * 64 + 64);
  return k1_win(w, s % 64);
}

template <int M, typename TI, typename TO, typename Epi, typename... S>
static inline void k1_deep_group(const TI *__restrict input,
                                 const int8_t *__restrict kernels,
                                 TO *__restrict output, const int oc, Epi epi,
                                 const S *__restrict... side) {
  constexpr int N = K1_CHUNKS;
  constexpr int U = K1_U;
  constexpr int G = K1_IC_BLOCKS / U;
  using MMUL = aie::mmul<8, 8, 8, TI, int8>;
  MMUL acc[N][M];
  // One input-channel block r: a(j) gives chunk j of its row, and the first
  // block multiplies instead of accumulating.
  const auto step = [&](const int8_t *__restrict const *wts, const int r,
                        const auto &a, const bool first) {
    aie::vector<int8, 64> b[M];
    AIE_LOOP_UNROLL_FULL
    for (int m = 0; m < M; m++)
      b[m] = aie::load_v<64>(wts[m] + 64 * r);
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < N; j++) {
      const aie::vector<TI, 64> x = a(j);
      AIE_LOOP_UNROLL_FULL
      for (int m = 0; m < M; m++)
        if (first)
          acc[j][m].mul(x, b[m]);
        else
          acc[j][m].mac(x, b[m]);
    }
  };
  const int8_t *__restrict wp[M];
  AIE_LOOP_UNROLL_FULL
  for (int m = 0; m < M; m++)
    wp[m] = kernels + (oc + m) * K1_IC_BLOCKS * 64;
  // The rows after the last whole group go first, so the loop only
  // accumulates.
  AIE_LOOP_UNROLL_FULL
  for (int r = G * U; r < K1_IC_BLOCKS; r++)
    step(
        wp, r,
        [&](int j) { return k1_row_load(input, r * K1_ROW + k1_deep_x(j)); },
        r == G * U);
  if constexpr (G * U == K1_IC_BLOCKS) {
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < N; j++)
      AIE_LOOP_UNROLL_FULL
    for (int m = 0; m < M; m++)
      acc[j][m] = MMUL(aie::zeros<acc32, 64>());
  }
  AIE_LOOP_RANGE(G, G)
  for (int g = 0; g < G; g++) {
    aie::vector<TI, 64> w[7];
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 7; i++)
      w[i] = aie::load_v<64>(input + g * 448 + 64 * i);
    AIE_LOOP_UNROLL_FULL
    for (int u = 0; u < U; u++)
      step(
          wp, u, [&](int j) { return k1_win(w, u * K1_ROW + k1_deep_x(j)); },
          false);
    AIE_LOOP_UNROLL_FULL
    for (int m = 0; m < M; m++)
      wp[m] += 64 * U;
  }
  const auto out = [&](const int m, const int j) {
    const int o = (oc + m) * K1_ROW + k1_deep_x(j);
    if constexpr (K1_WIDTH < 8) {
      if (m % 2) {
        const auto side_load = [](const auto *p) {
          const auto v = k1_load_fixed(p - 8);
          return ::shift_bytes(v, v, 8);
        };
        return epi(acc[j][m], side_load(side + o)...);
      }
    }
    return epi(acc[j][m], k1_load_fixed(side + o)...);
  };
  if constexpr (K1_WIDTH >= 8) {
    AIE_LOOP_UNROLL_FULL
    for (int m = 0; m < M; m++)
      AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < N; j++)
      k1_store_fixed(output + (oc + m) * K1_ROW + k1_deep_x(j), out(m, j));
  } else {
    // Pairs of 7-pixel rows from an even block are 112 bytes on a 16-byte
    // boundary; a lone last row ends in an 8-byte tail.
    AIE_LOOP_UNROLL_FULL
    for (int m = 0; m < M; m += 2) {
      TO *__restrict p = output + (oc + m) * K1_ROW;
      const aie::vector<TO, 64> a = out(m, 0);
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < 3; i++)
        aie::store_v(p + 16 * i, a.template extract<16>(i));
      if (m + 1 < M) {
        const aie::vector<TO, 64> v =
            ::shift_bytes(::shift_bytes(a, a, 56), out(m + 1, 0), 56);
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < 4; i++)
          aie::store_v(p + 48 + 16 * i, v.template extract<16>(i));
      } else {
        const auto t = aie::vector_cast<int32>(a.template extract<16>(3));
        int32_t *__restrict q = (int32_t *)(p + 48);
        q[0] = t[0];
        q[1] = t[1];
      }
    }
  }
}

template <typename TI, typename TO, typename Epi, typename... S>
static void
k1_rows_deep(const TI *__restrict input, const int8_t *__restrict kernels,
             TO *__restrict output, Epi epi, const S *__restrict... side) {
  constexpr int M = K1_DEEP_M;
  constexpr int R = K1_OC_BLOCKS % M;
  for (int oc = 0; oc < K1_OC_BLOCKS - R; oc += M)
    k1_deep_group<M>(input, kernels, output, oc, epi, side...);
  if constexpr (R)
    k1_deep_group<R>(input, kernels, output, K1_OC_BLOCKS - R, epi, side...);
}
#endif

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
