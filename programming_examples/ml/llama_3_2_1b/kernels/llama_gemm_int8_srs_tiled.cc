//===- llama_gemm_int8_srs_tiled.cc -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tiled decode-GEMV kernel for production-shape Llama 3.2 1B.
//
// The monolithic llama_gemm_int8_srs.cc keeps the entire (N, K) weight
// matrix in compute-tile L1 -- fine at toy D=64, dies at D=2048
// (4 MB / matrix >> 64 KB tile L1). Here we tile on N (output
// channels): each call processes N_TILE rows from the (N, K) matrix,
// against the FULL K-length activation that stays resident in L1
// across the N/N_TILE iterations the IRON design fires.
//
// Activation stays in L1 once acquired; weights stream in tile-by-tile
// from a pingpong-depth ObjectFifo fed by a single BD-chained shim DMA.
// Output streams out tile-by-tile, consolidated into one host buffer.
//
// We do NOT touch the legacy llama_gemm_int8_srs.cc -- Phase 4/5/6b
// still depend on it for toy shapes.
//
// ATB doesn't apply here: ATB is a prefill (M >> 1) optimization for
// asymmetric A/C tile reuse; decode is M=1 so C is a single row, no
// reuse benefit. This kernel is plain "chunk N" GEMV streaming.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LLAMA_GEMM_TILED_K
#define LLAMA_GEMM_TILED_K 2048
#endif
#ifndef LLAMA_GEMM_TILED_N_TILE
#define LLAMA_GEMM_TILED_N_TILE 8
#endif

static constexpr int kK = LLAMA_GEMM_TILED_K;
static constexpr int kNTile = LLAMA_GEMM_TILED_N_TILE;
static constexpr int kVec = 64;
static_assert(kK % kVec == 0,
              "K must be a multiple of 64 (one aie::mac lane group)");
static constexpr int kGroups = kK / kVec;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

// Identical arithmetic to llama_gemm_int8_srs.cc -- a single shared
// numpy reference is bit-exact for both kernels.
static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// Shared core: kNTile rows of (i8 weights, i32 bias) -> kNTile i8 out.
static inline void gemm_tile_core(const int8_t *__restrict act,
                                  const int8_t *__restrict weights,
                                  const int32_t *__restrict bias,
                                  int8_t *__restrict out_tile,
                                  int32_t right_shift) {

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }

    int32_t sum = aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX)
      s = I8_MAX;
    if (s < I8_MIN)
      s = I8_MIN;
    out_tile[n] = (int8_t)s;
  }
}

extern "C" {

// Inlined-bias entry. Per-call w_tile layout:
//   [kNTile*kK i8 weights | kNTile i32 bias]
// Works when N_TILE*K + N_TILE*4 fits cleanly in one L1 bank
// (16 KB). Used by K=2048 N_TILE=4 (slot=8208 B). Hits Peano Bug 3a
// at K=8192 because slot then spans an awkward (2-bank + 16 B) layout.
void llama_gemm_int8_srs_tiled(int8_t *restrict act, int8_t *restrict w_tile,
                               int8_t *restrict out_tile, int32_t right_shift) {
  event0();
  const int8_t *weights = w_tile;
  const int32_t *bias = reinterpret_cast<const int32_t *>(w_tile + kNTile * kK);
  gemm_tile_core(act, weights, bias, out_tile, right_shift);
  event1();
}

// Split-bias entry. Weights and bias arrive on separate ObjectFifos so
// the weight buffer can be sized to an exact bank multiple. Required
// for K=8192 N_TILE=4 (weight slot = 32 KB = exactly 2 banks).
void llama_gemm_int8_srs_tiled_split(int8_t *restrict act,
                                     int8_t *restrict w_tile,
                                     int32_t *restrict b_tile,
                                     int8_t *restrict out_tile,
                                     int32_t right_shift) {
  event0();
  gemm_tile_core(act, w_tile, b_tile, out_tile, right_shift);
  event1();
}

} // extern "C"
