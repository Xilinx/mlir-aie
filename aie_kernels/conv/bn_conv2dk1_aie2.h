//===- bn_conv2dk1_aie2.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_CONV_BN_CONV2DK1_AIE2_H
#define AIE_KERNELS_CONV_BN_CONV2DK1_AIE2_H

#include <aie_api/aie.hpp>
#include <stdint.h>

// 1x1 conv on [C/8][W][8] rows: each mmul<4,8,8> takes 4 pixels x 8 input
// channels against one [8][8] weight block. A row is covered by 4-pixel
// chunks; when input_width is not a multiple of 4 the last chunk starts at
// input_width - 4 and overlaps the one before it.
template <bool Aligned, typename T>
static inline aie::vector<T, 32> k1_load(const T *p) {
  if constexpr (Aligned)
    return aie::load_v<32>(p);
  else
    return aie::load_unaligned_v<32>(p, 8);
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

// N chunks of one output channel block. Chunk j is at byte offset 32 * j
// from in, except the last at last_off; epi(acc, side + offset...) gives the
// vector stored at out + offset.
template <bool Aligned, int N, typename TI, typename TO, typename Epi,
          typename... S>
static inline void
k1_chunks(const TI *__restrict in, const int8_t *__restrict wts,
          TO *__restrict out, const int32_t row, const int32_t ic_blocks,
          const int32_t last_off, Epi epi, const S *__restrict... side) {
  using MMUL = aie::mmul<4, 8, 8, TI, int8>;
  MMUL acc[N];
  aie::vector<int8, 64> b = aie::load_v<64>(wts);
  for (int j = 0; j < N; j++)
    acc[j].mul(k1_load<Aligned>(in + (j == N - 1 ? last_off : 32 * j)), b);
#pragma clang loop min_iteration_count(1)
  for (int ic = 1; ic < ic_blocks; ic++) {
    in += row;
    wts += 64;
    b = aie::load_v<64>(wts);
    for (int j = 0; j < N; j++)
      acc[j].mac(k1_load<Aligned>(in + (j == N - 1 ? last_off : 32 * j)), b);
  }
  for (int j = 0; j < N; j++) {
    const int32_t o = j == N - 1 ? last_off : 32 * j;
    k1_store<Aligned>(out + o, epi(acc[j], (side + o)...));
  }
}

// Walks the whole [OC/8][W][8] output; side buffers share its layout.
template <bool Aligned, typename TI, typename TO, typename Epi, typename... S>
static void k1_rows(const TI *input, const int8_t *kernels, TO *output,
                    const int32_t input_width, const int32_t input_channels,
                    const int32_t output_channels, Epi epi, const S *...side) {
  constexpr int N = 4;
  const int32_t row = input_width * 8;
  const int32_t ic_blocks = input_channels / 8;
  const int32_t chunks = (input_width + 3) / 4;
  const int32_t groups = chunks / N;
  const int32_t rem = chunks % N;
  const int32_t tail = (input_width - 4) * 8;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *wts = kernels + oc * ic_blocks * 64;
    TO *out = output + oc * row;
    for (int g = 0; g < groups; g++) {
      const int32_t x = g * N * 32;
      const int32_t last =
          (rem == 0 && g == groups - 1) ? tail - x : 32 * (N - 1);
      k1_chunks<Aligned, N>(input + x, wts, out + x, row, ic_blocks, last, epi,
                            (side + oc * row + x)...);
    }
    const int32_t x = groups * N * 32;
    switch (rem) {
    case 1:
      k1_chunks<Aligned, 1>(input + x, wts, out + x, row, ic_blocks, tail - x,
                            epi, (side + oc * row + x)...);
      break;
    case 2:
      k1_chunks<Aligned, 2>(input + x, wts, out + x, row, ic_blocks, tail - x,
                            epi, (side + oc * row + x)...);
      break;
    case 3:
      k1_chunks<Aligned, 3>(input + x, wts, out + x, row, ic_blocks, tail - x,
                            epi, (side + oc * row + x)...);
      break;
    }
  }
}

#endif
