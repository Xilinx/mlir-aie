//===- llama_embed_select.cc --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// On-chip embed gather via STREAM + SELECT (no dynamic DMA, no host). The
// lm_head streams the whole tied embed table (int8[V,D]) every token;
// this kernel watches tile-by-tile and, when the sampled token id falls
// in the current tile, copies that row out as the next layer-0 input.
// embed_sc[id] (a positive scalar) cancels in the host's row/scale
// requant math, so only the int8 row + embed_sc[id] are needed here --
// no fp32 dequant of the row itself.
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

// Process one streamed tile of kNTile embed rows (the lm_head GEMM
// weight bytes for this tile) + their embed_sc. state carries [found i32
// | sel_embed_sc f32] across tiles; out is the selected row int8[D] +
// fp32 scale + pad, written directly (no separate accumulation buffer).
// When token falls in [tile_idx*kNTile, +kNTile): copy that row, track
// absmax, requant into out -- one call, in the tile the token is in;
// otherwise a no-op. out/state persist across calls.
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

// Slot-based variant: takes the whole lm_head GEMM weight slot (same
// layout llama_gemm_int8_srs_tiled_layer.cc's slots use) so the real
// fused version can reuse the lmw stream directly.
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
