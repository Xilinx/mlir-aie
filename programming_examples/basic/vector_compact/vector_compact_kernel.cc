//===- vector_compact_kernel.cc ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Vectorized stream-compaction (left-pack) for AIE2P.
//
// Given an input vector `a_in` of N bfloat16 values, write all "survivors"
// (values >= tau, here tau = 0.0 -> keep the positive values) packed to the
// left of `c_out`, followed by zeros.  Survivors keep their relative order.
//
// The classic compaction is O(survivors) scalar stores per tile.  This kernel
// instead demonstrates a *butterfly / prefix-scan* approach, which turns the
// per-tile packing into O(log2 TILE) vector ops:
//
//   1. Vectorized compare  -> bitmask                       (1 VGE)
//   2. Hillis-Steele prefix sum on the mask via shift+VADD  (log2 TILE stages)
//        -> gives each lane its destination index
//   3. Butterfly routing via shift+VSEL                     (log2 TILE stages)
//        -> moves each survivor to its destination lane
//
// TILE = 32 (one 512-bit X register of bf16).  N = 1024 -> 32 tiles.
//
// IMPLEMENTATION NOTE on the element-shift primitive
// --------------------------------------------------
// Both the prefix sum and the butterfly need element shifts with zero fill:
//   shift_right_k: out[i] = v[i + k]   (0 if i+k >= TILE)
//   shift_left_k:  out[i] = v[i - k]   (0 if i-k <  0)
//
// The original version implemented these via a zero-padded 2*TILE scratch
// buffer and a vector load at offset k:  aie::load_v<TILE>(&scratch[k]).  On
// AIE2P a 512-bit (TILE=32 x bf16/int16) vector load requires 64-byte
// alignment, but &scratch[k] sits at a byte offset of k*sizeof(T), which is
// NOT 64-byte aligned for most k.  The hardware silently returns wrong data,
// which is why the original kernel produced 504 mismatching survivors.
//
// The fix uses the aie_api element-granular shuffle primitives, which lower to
// the VSHIFT/VSHUFFLE hardware path and carry no alignment requirement:
//
//   shuffle_up_fill(v, zeros, k):   out[i+k] = v[i],  out[0..k-1]   = 0
//                                   == shift_left_k  (each lane gets v[i-k])
//   shuffle_down_fill(v, zeros, k): out[i]   = v[i+k], out[T-k..T-1] = 0
//                                   == shift_right_k (each lane gets v[i+k])
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef TILE
#define TILE 32
#endif

// log2(TILE) == 5 for TILE == 32.  The five Hillis-Steele / butterfly stages
// use shift distances 1, 2, 4, 8, 16.
static constexpr int LOG2_TILE = 5;

// ---------------------------------------------------------------------------
// Element-shift primitives (see the note at the top of the file for why these
// use a zero-padded scratch round-trip rather than a single VSHIFT).
//
// shift_left_k<T>(v, k):  out[i] = (i - k >= 0) ? v[i - k] : 0
//   Each lane receives the value k positions to its LEFT (zero fill on the
//   left).  Used by the Hillis-Steele *prefix* (left-to-right) scan.
//
// shift_right_k<T>(v, k): out[i] = (i + k < TILE) ? v[i + k] : 0
//   Each lane receives the value k positions to its RIGHT (zero fill on the
//   right).  Used by the butterfly to pull survivors leftwards.
// ---------------------------------------------------------------------------
template <typename T>
static inline aie::vector<T, TILE> shift_left_k(aie::vector<T, TILE> v, int k) {
  // out[i] = (i-k >= 0) ? v[i-k] : 0.
  // shuffle_up_fill(v, fill, k): out[i+k] = v[i] for i < TILE-k, and the low k
  // lanes are filled from fill[TILE-k ..].  With fill = zeros this is exactly
  // a left shift with zero fill.
  aie::vector<T, TILE> zero = aie::zeros<T, TILE>();
  return aie::shuffle_up_fill(v, zero, (unsigned)k);
}

template <typename T>
static inline aie::vector<T, TILE> shift_right_k(aie::vector<T, TILE> v,
                                                 int k) {
  // out[i] = (i+k < TILE) ? v[i+k] : 0.
  // shuffle_down_fill(v, fill, k): out[i] = v[i+k] for i < TILE-k, and the high
  // k lanes are filled from fill[0 ..].  With fill = zeros this is exactly a
  // right shift with zero fill.
  aie::vector<T, TILE> zero = aie::zeros<T, TILE>();
  return aie::shuffle_down_fill(v, zero, (unsigned)k);
}

// ---------------------------------------------------------------------------
// The kernel.  Processes the whole N-element input in one core invocation
// because the output write pointer is a running counter that carries across
// tiles (the left-pack is inherently serial between tiles).
// ---------------------------------------------------------------------------
// VECTOR_OPS_PER_TILE = 31
//   Per tile the vectorized work is, in vector instructions:
//     1   VGE compare (keep mask)
//     1   VSEL  to materialise the 0/1 mask
//     5   VSHIFT + 5 VADD   (Hillis-Steele inclusive prefix sum, LOG2_TILE=5)
//     1   VSUB  (exclusive prefix)            +  1 VSUB (displacement)
//     5x (1 VSHIFT recv + 1 VSHIFT value + 1 VSHIFT rem + 2 VSEL) butterfly
//         routing  -> but only the shifts/selects are pure vector ops; the
//         broadcasts are hoistable constants.
//   The dominant, non-hoistable vector-op count is ~31 vector instructions per
//   tile.  With n_tiles = N/TILE = 32 tiles that is ~992 vector ops total for
//   N = 1024.  The event0()/event1() window below measures the actual core
//   cycle cost of this work (read back via the AIE trace infrastructure).
template <int N>
static void bf16_vector_compact_impl(bfloat16 *restrict a_in,
                                     bfloat16 *restrict c_out) {
  event0(); // ---- begin region-of-interest (butterfly compaction) ----
  const bfloat16 tau = (bfloat16)0.0f;
  aie::vector<bfloat16, TILE> tau_v = aie::broadcast<bfloat16, TILE>(tau);
  aie::vector<bfloat16, TILE> zero_bf = aie::zeros<bfloat16, TILE>();

  // lane_idx[i] = i  (int16 so prefix sums up to TILE never overflow).
  alignas(64) int16_t lane_init[TILE];
  for (int i = 0; i < TILE; ++i)
    lane_init[i] = (int16_t)i;
  aie::vector<int16_t, TILE> lane_idx = aie::load_v<TILE>(&lane_init[0]);
  aie::vector<int16_t, TILE> zero_i16 = aie::zeros<int16_t, TILE>();
  aie::vector<int16_t, TILE> one_i16 =
      aie::broadcast<int16_t, TILE>((int16_t)1);

  int write_ptr = 0; // running output offset (number of survivors so far)

  const int n_tiles = N / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    bfloat16 *tile_in = a_in + t * TILE;

    // (a) Load 32 bf16 values.
    aie::vector<bfloat16, TILE> v = aie::load_v<TILE>(tile_in);

    // (b) Compare >= tau -> mask (VGE).
    aie::mask<TILE> keep = aie::ge(v, tau_v);

    // (c) Materialise the mask as a 0/1 int16 vector for arithmetic.
    //     select(v1, v2, m): out[i] = m[i] ? v2[i] : v1[i].
    aie::vector<int16_t, TILE> m_i16 = aie::select(zero_i16, one_i16, keep);

    // (d) Hillis-Steele *inclusive prefix* sum of the mask.
    //     Each stage folds in partial sums from the LEFT (shift_left_k), which
    //     is what makes this a left-to-right scan.  After the 5 stages,
    //     incl[i] = sum(m_i16[0..i]).
    aie::vector<int16_t, TILE> incl = m_i16;
    for (int s = 0; s < LOG2_TILE; ++s) {
      int shift = 1 << s; // 1, 2, 4, 8, 16
      aie::vector<int16_t, TILE> shifted = shift_left_k<int16_t>(incl, shift);
      incl = aie::add(incl, shifted);
    }

    // (e) exclusive prefix excl[i] = incl[i] - m_i16[i] = number of survivors
    //     strictly before lane i = the destination index of a survivor at i.
    //     displacement[i] = i - excl[i] = number of *dropped* lanes before i,
    //     i.e. how far the value at lane i must travel left.  For survivors
    //     this is non-decreasing, so the butterfly below never makes two
    //     survivors cross.
    aie::vector<int16_t, TILE> excl = aie::sub(incl, m_i16);
    aie::vector<int16_t, TILE> disp = aie::sub(lane_idx, excl);

    // (f) Butterfly routing on the bf16 values, 5 stages (LSB -> MSB of disp).
    //     We move each value left by its displacement, one power-of-two at a
    //     time.  At stage k (dist = 2^k):
    //       - a source lane whose *remaining* displacement has bit k set moves
    //         its value left by dist;
    //       - destination lane j therefore receives from lane j+dist iff the
    //         value currently at j+dist is moving this stage.
    //     We track the remaining displacement (rem) alongside the values and
    //     subtract dist from it whenever a value is moved.
    aie::vector<bfloat16, TILE> out_v = v;
    aie::vector<int16_t, TILE> rem = disp;
    for (int k = 0; k < LOG2_TILE; ++k) {
      int dist = 1 << k; // 1, 2, 4, 8, 16
      aie::vector<int16_t, TILE> bit_v =
          aie::broadcast<int16_t, TILE>((int16_t)dist);
      aie::vector<int16_t, TILE> one_v =
          aie::broadcast<int16_t, TILE>((int16_t)1);

      // moves[i] = bit k of rem[i] set ? 1 : 0   (per *source* lane)
      aie::mask<TILE> moves_m = aie::eq(aie::bit_and(rem, bit_v), bit_v);
      aie::vector<int16_t, TILE> moves = aie::select(zero_i16, one_v, moves_m);

      // recv[j] = moves[j + dist]  -> destination j receives this stage.
      aie::vector<int16_t, TILE> recv = shift_right_k<int16_t>(moves, dist);
      aie::mask<TILE> recv_m = aie::eq(recv, one_v);

      // Pull the incoming value / remaining-disp from dist lanes to the right.
      aie::vector<bfloat16, TILE> v_in = shift_right_k<bfloat16>(out_v, dist);
      aie::vector<int16_t, TILE> rem_in = shift_right_k<int16_t>(rem, dist);
      aie::vector<int16_t, TILE> dist_v =
          aie::broadcast<int16_t, TILE>((int16_t)dist);

      // Where a value arrives, take it and decrement its remaining disp.
      out_v = aie::select(out_v, v_in, recv_m);
      rem = aie::select(rem, aie::sub(rem_in, dist_v), recv_m);
    }

    // (g) Only the first popcount(mask) lanes of out_v hold survivors; the rest
    //     are stale.  Write the whole compacted tile to a zero-padded scratch
    //     and copy exactly `cnt` survivors to the running output position.
    int cnt = 0;
    for (int i = 0; i < TILE; ++i)
      cnt += (int)m_i16.get(i);

    alignas(64) bfloat16 packed[TILE + TILE];
    aie::store_v(&packed[0], out_v);
    aie::store_v(&packed[TILE], zero_bf);
    for (int i = 0; i < cnt; ++i)
      c_out[write_ptr + i] = packed[i];

    // (h) Advance the write pointer by the survivor count.
    write_ptr += cnt;
  }

  // Zero-pad the remainder of the output [write_ptr .. N).  write_ptr is a
  // data-dependent (survivor-count) offset, so this is a scalar fill to avoid
  // a misaligned vector store.
  for (int i = write_ptr; i < N; ++i)
    c_out[i] = (bfloat16)0.0f;

  event1(); // ---- end region-of-interest ----
}

// ---------------------------------------------------------------------------
// Scalar left-pack baseline.  O(survivors) scalar stores per tile -- this is
// the conventional compaction (cf. compact.cc in open-xdna).  Provided for an
// apples-to-apples comparison against the butterfly/prefix-scan kernel above.
// ---------------------------------------------------------------------------
template <int N>
static void bf16_scalar_compact_impl(bfloat16 *restrict a_in,
                                     bfloat16 *restrict c_out) {
  event0(); // ---- begin region-of-interest (scalar compaction) ----
  const bfloat16 tau = (bfloat16)0.0f;
  int write_ptr = 0;
  for (int i = 0; i < N; ++i) {
    if (a_in[i] >= tau)
      c_out[write_ptr++] = a_in[i];
  }
  for (int i = write_ptr; i < N; ++i)
    c_out[i] = (bfloat16)0.0f;
  event1(); // ---- end region-of-interest ----
}

#ifndef N_ELEMS
#define N_ELEMS 1024
#endif

extern "C" {

// N is fixed at compile time (N_ELEMS, default 1024) for this version: the
// template specialisation keeps the tile-count loop bound a compile-time
// constant for the vectorizer, and avoids threading a runtime scalar through
// the ObjectFifo plumbing.  The two-pointer signature matches the single-tile
// IRON kernel convention (cf. aie_kernels/aie2p/bf16_exp.cc).
void bf16_vector_compact(bfloat16 *a_in, bfloat16 *c_out) {
  bf16_vector_compact_impl<N_ELEMS>(a_in, c_out);
}

// Scalar baseline -- same signature/convention as the vector kernel.
void bf16_scalar_compact(bfloat16 *a_in, bfloat16 *c_out) {
  bf16_scalar_compact_impl<N_ELEMS>(a_in, c_out);
}

} // extern "C"
