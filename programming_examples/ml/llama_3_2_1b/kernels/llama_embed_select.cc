//===- llama_embed_select.cc --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// On-chip embed gather via STREAM + SELECT (no dynamic DMA, no host).
//
// The lm_head streams the whole tied embed table (int8[V,D]) through the array
// every token. This kernel watches that stream tile-by-tile and, when the
// sampled token id falls in the current tile, copies the matching row out as
// the next token's layer-0 input (int8[D] + fp32 scale tail).
//
// The host embed math (bench_quality_mh.py) is:
//   row   = embed_i8[id] * embed_sc[id]            (fp32[D])
//   scale = max(|row|, 1e-12) / 127
//   xin   = round(row / scale)                     (int8[D])
// Because embed_sc[id] > 0 is a scalar it CANCELS in row/scale:
//   xin   = round(embed_i8[id] * 127 / max(|embed_i8[id]|))
//   scale = max(|embed_i8[id]|) * embed_sc[id] / 127
// So we only need the int8 row + embed_sc[id] (one scalar) -- no fp32 dequant.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

#ifndef LLAMA_EMBED_D
#define LLAMA_EMBED_D 2048
#endif
#ifndef LLAMA_EMBED_NTILE
#define LLAMA_EMBED_NTILE 4
#endif

static constexpr int kD = LLAMA_EMBED_D;
static constexpr int kNTile = LLAMA_EMBED_NTILE;

static inline int8_t round_to_i8(float v) {
  float r = v >= 0.0f ? (v + 0.5f) : (v - 0.5f);
  int32_t i = (int32_t)r;
  if (i > 127)
    i = 127;
  if (i < -128)
    i = -128;
  return (int8_t)i;
}

// IEEE fp32 reciprocal (NR), matches the rest of the design (Peano `/` is HW
// approximate). See peano_aie2p_bugs.
static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

extern "C" {

// Process one streamed tile of kNTile embed rows.
//   tile     : int8[kNTile * kD]   -- kNTile consecutive embed rows (the lm_head
//              GEMM weight bytes for this tile)
//   scales   : fp32[kNTile]        -- embed_sc for these kNTile rows
//   state    : carry buffer. state[0..kD-1] reserved? No -- we write directly to
//              `out`. state holds [found i32 | sel_embed_sc f32] across tiles.
//   out      : int8[kD + 8]        -- selected row int8[D] + fp32 scale + pad
//   token    : the sampled token id (int32)
//   tile_idx : 0-based tile index in the stream (row base = tile_idx*kNTile)
//
// When token in [base, base+kNTile): copy that row's int8 into a temp, track
// absmax, then (after the tile) requant -> out + write scale. We do it in one
// call when the row is in this tile; otherwise no-op. `out`/state persist.
void llama_embed_select(int8_t *restrict tile, float *restrict scales,
                        int8_t *restrict out, int32_t token, int32_t tile_idx) {
  int32_t base = tile_idx * kNTile;
  int local = token - base;
  if (local < 0 || local >= kNTile)
    return; // token not in this tile

  const int8_t *row = tile + local * kD;
  float embed_sc = scales[local];

  // absmax of the int8 row
  int32_t amax = 0;
  for (int i = 0; i < kD; i++) {
    int32_t a = row[i];
    if (a < 0)
      a = -a;
    if (a > amax)
      amax = a;
  }
  if (amax < 1)
    amax = 1; // avoid div-by-zero (all-zero row)

  // xin = round(row * 127 / amax); scale = amax * embed_sc / 127
  float inv = sw_recip((float)amax) * 127.0f; // 127/amax
  for (int i = 0; i < kD; i++)
    out[i] = round_to_i8((float)row[i] * inv);

  float scale = (float)amax * embed_sc * (1.0f / 127.0f);
  memcpy(out + kD, &scale, 4);
  int32_t zero = 0;
  memcpy(out + kD + 4, &zero, 4);
}

// Slot-based variant: the whole lm_head-style weight slot
//   [kPrefix pad | kNTile*kD int8 rows | kNTile i32 bias | kNTile fp32 scale]
// is passed; rows + scales are indexed within it. Mirrors how the lm_head GEMM
// consumes its weight slot, so the real fused version reuses the lmw stream.
#ifndef LLAMA_EMBED_PREFIX
#define LLAMA_EMBED_PREFIX 64
#endif
void llama_embed_select_slot(int8_t *restrict slot, int8_t *restrict out,
                             int32_t *restrict token, int32_t tile_idx) {
  int8_t *rows = slot + LLAMA_EMBED_PREFIX;
  float *scales = reinterpret_cast<float *>(slot + LLAMA_EMBED_PREFIX +
                                            kNTile * kD + kNTile * 4);
  llama_embed_select(rows, scales, out, token[0], tile_idx);
}

} // extern "C"
