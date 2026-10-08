//===- resize.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A uint8 RGB image resized as torch's antialiased bicubic resize does it on
// the CPU, across then down, each pass rounded to uint8, and written as 16x16
// patches of u8 / 255 in bf16. The filters are resample_quantize's tables of
// WORDS-int32 chunks, a patch's 16 outputs to a chunk. CORES cores receive
// every CHUNK-byte image chunk (a row is whole chunks); core c owns patch
// columns c, c + CORES, ..., at most COLS of them.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef WORDS
#define WORDS 356
#endif
#ifndef CHUNK
#define CHUNK 4096
#endif
#ifndef COLS
#define COLS 8
#endif
#ifndef CORES
#define CORES 16
#endif

namespace {
using taps_t = aie::vector<int16_t, 32>;
using sum_t = aie::accum<acc32, 32>;

constexpr int HEADER = 4;
constexpr int SIDE = 16;
constexpr int WIN = 2 * ((WORDS - HEADER) / SIDE - 2) - 1;
static_assert(WIN <= 64, "a window is at most two vectors of taps");
// The input pixels under one patch column's 16 outputs, at any scale a
// WIN-tap window allows, and the 64 a window's vectors read past them.
constexpr int SPAN = 5 * WIN + 8;
constexpr int PLANE = (SPAN + 64 + 63) / 64 * 64;
// Whole 32-byte vectors, so every row of mid and hold is vector-aligned.
constexpr int LINE = (COLS * SIDE * 3 + 31) / 32 * 32;

// The bf16 bits of u8 / 255, computed in float32 and rounded to bf16 once.
const uint16_t PIXEL[256] = {
    0x0000, 0x3b81, 0x3c01, 0x3c41, 0x3c81, 0x3ca1, 0x3cc1, 0x3ce1, 0x3d01,
    0x3d11, 0x3d21, 0x3d31, 0x3d41, 0x3d51, 0x3d61, 0x3d71, 0x3d81, 0x3d89,
    0x3d91, 0x3d99, 0x3da1, 0x3da9, 0x3db1, 0x3db9, 0x3dc1, 0x3dc9, 0x3dd1,
    0x3dd9, 0x3de1, 0x3de9, 0x3df1, 0x3df9, 0x3e01, 0x3e05, 0x3e09, 0x3e0d,
    0x3e11, 0x3e15, 0x3e19, 0x3e1d, 0x3e21, 0x3e25, 0x3e29, 0x3e2d, 0x3e31,
    0x3e35, 0x3e39, 0x3e3d, 0x3e41, 0x3e45, 0x3e49, 0x3e4d, 0x3e51, 0x3e55,
    0x3e59, 0x3e5d, 0x3e61, 0x3e65, 0x3e69, 0x3e6d, 0x3e71, 0x3e75, 0x3e79,
    0x3e7d, 0x3e81, 0x3e83, 0x3e85, 0x3e87, 0x3e89, 0x3e8b, 0x3e8d, 0x3e8f,
    0x3e91, 0x3e93, 0x3e95, 0x3e97, 0x3e99, 0x3e9b, 0x3e9d, 0x3e9f, 0x3ea1,
    0x3ea3, 0x3ea5, 0x3ea7, 0x3ea9, 0x3eab, 0x3ead, 0x3eaf, 0x3eb1, 0x3eb3,
    0x3eb5, 0x3eb7, 0x3eb9, 0x3ebb, 0x3ebd, 0x3ebf, 0x3ec1, 0x3ec3, 0x3ec5,
    0x3ec7, 0x3ec9, 0x3ecb, 0x3ecd, 0x3ecf, 0x3ed1, 0x3ed3, 0x3ed5, 0x3ed7,
    0x3ed9, 0x3edb, 0x3edd, 0x3edf, 0x3ee1, 0x3ee3, 0x3ee5, 0x3ee7, 0x3ee9,
    0x3eeb, 0x3eed, 0x3eef, 0x3ef1, 0x3ef3, 0x3ef5, 0x3ef7, 0x3ef9, 0x3efb,
    0x3efd, 0x3eff, 0x3f01, 0x3f02, 0x3f03, 0x3f04, 0x3f05, 0x3f06, 0x3f07,
    0x3f08, 0x3f09, 0x3f0a, 0x3f0b, 0x3f0c, 0x3f0d, 0x3f0e, 0x3f0f, 0x3f10,
    0x3f11, 0x3f12, 0x3f13, 0x3f14, 0x3f15, 0x3f16, 0x3f17, 0x3f18, 0x3f19,
    0x3f1a, 0x3f1b, 0x3f1c, 0x3f1d, 0x3f1e, 0x3f1f, 0x3f20, 0x3f21, 0x3f22,
    0x3f23, 0x3f24, 0x3f25, 0x3f26, 0x3f27, 0x3f28, 0x3f29, 0x3f2a, 0x3f2b,
    0x3f2c, 0x3f2d, 0x3f2e, 0x3f2f, 0x3f30, 0x3f31, 0x3f32, 0x3f33, 0x3f34,
    0x3f35, 0x3f36, 0x3f37, 0x3f38, 0x3f39, 0x3f3a, 0x3f3b, 0x3f3c, 0x3f3d,
    0x3f3e, 0x3f3f, 0x3f40, 0x3f41, 0x3f42, 0x3f43, 0x3f44, 0x3f45, 0x3f46,
    0x3f47, 0x3f48, 0x3f49, 0x3f4a, 0x3f4b, 0x3f4c, 0x3f4d, 0x3f4e, 0x3f4f,
    0x3f50, 0x3f51, 0x3f52, 0x3f53, 0x3f54, 0x3f55, 0x3f56, 0x3f57, 0x3f58,
    0x3f59, 0x3f5a, 0x3f5b, 0x3f5c, 0x3f5d, 0x3f5e, 0x3f5f, 0x3f60, 0x3f61,
    0x3f62, 0x3f63, 0x3f64, 0x3f65, 0x3f66, 0x3f67, 0x3f68, 0x3f69, 0x3f6a,
    0x3f6b, 0x3f6c, 0x3f6d, 0x3f6e, 0x3f6f, 0x3f70, 0x3f71, 0x3f72, 0x3f73,
    0x3f74, 0x3f75, 0x3f76, 0x3f77, 0x3f78, 0x3f79, 0x3f7a, 0x3f7b, 0x3f7c,
    0x3f7d, 0x3f7e, 0x3f7f, 0x3f80};

int rows, columns, cpr, owned, core;
// part: the next image chunk's in its row; ring: that row's in mid.
int consumed, part, ring, arrived, opened, next, hp, hwin;
bool ok;
int hstart[COLS][SIDE];
alignas(64) int16_t hweight[COLS][SIDE][64];
int span0[COLS], span[COLS];
// A row's pixels under each patch column, a plane to a channel.
alignas(64) uint8_t line[COLS][3][PLANE];
// The last WIN rows across, row r at r % WIN, and the band's rows down: patch
// column i's channel ch is bytes (3 * i + ch) * SIDE on.
alignas(64) uint8_t mid[WIN][LINE];
alignas(64) uint8_t hold[SIDE][LINE];

taps_t widen(const uint8_t *x) {
  return aie::load_unaligned_v<32>(x).unpack().cast_to<int16_t>();
}

// The sums of outputs o..o+N-1 of patch column i over channel plane x,
// output o + j in lanes j * 32 / N on, which add up to it.
// Inlined whole and branch-free, so the leaves' loads overlap.
template <int N, bool WIDE>
__attribute__((always_inline)) aie::vector<int32_t, 32> tree(const uint8_t *x,
                                                             int i, int o) {
  if constexpr (N == 1) {
    const int16_t *w = hweight[i][o];
    x += hstart[i][o] - span0[i];
    sum_t a = aie::mul<acc32>(widen(x), aie::load_v<32>(w));
    if constexpr (WIDE)
      a = aie::mac(a, widen(x + 32), aie::load_v<32>(w + 32));
    return a.to_vector<int32_t>(0);
  } else {
    auto [even, odd] = aie::interleave_unzip(
        tree<N / 2, WIDE>(x, i, o), tree<N / 2, WIDE>(x, i, o + N / 2), 32 / N);
    return aie::add(even, odd);
  }
}

// (bias + sum) >> p, clipped to [0, 255], as srs gives it.
struct Srs {
  aie::rounding_mode rounding = aie::swap_rounding(aie::rounding_mode::floor);
  aie::saturation_mode saturation = aie::get_saturation();
  Srs() { aie::set_saturation(aie::saturation_mode::saturate); }
  ~Srs() {
    aie::set_rounding(rounding);
    aie::set_saturation(saturation);
  }
};

template <bool WIDE>
void across() {
  Srs srs;
  uint8_t *m = mid[ring];
  const auto bias = aie::broadcast<int32_t, 32>(1 << (hp - 1));
  const auto zero = aie::zeros<int32_t, 32>();
  for (int i = 0; i < owned; i++)
    for (int ch = 0; ch < 3; ch++) {
      // A half at a time, so the leaves' vectors fit the registers.
      aie::vector<int32_t, 32> half[2];
#pragma clang loop unroll(disable)
      for (int h = 0; h < 2; h++)
        half[h] = tree<SIDE / 2, WIDE>(line[i][ch], i, h * SIDE / 2);
      auto [lo, hi] = aie::interleave_unzip(half[0], half[1], 32 / SIDE);
      auto [even, odd] = aie::interleave_unzip(aie::add(lo, hi), zero, 1);
      sum_t a(aie::add(aie::add(even, odd), bias));
      aie::store_v(m + (3 * i + ch) * SIDE,
                   a.to_vector<uint8_t>(hp).extract<16>(0));
    }
}

// Peano cannot legalize this loop bounded by a pointer, `src + 3 <= stop`.
// A byte store is a read-modify-write that serializes with the next, so
// whole pixels go four to a word store.
void split(const uint8_t *__restrict src, uint8_t *__restrict red,
           uint8_t *__restrict green, uint8_t *__restrict blue, int n) {
  int k = 0;
  for (; k < n && (uintptr_t)(red + k) % 4; k++) {
    red[k] = src[3 * k];
    green[k] = src[3 * k + 1];
    blue[k] = src[3 * k + 2];
  }
  int words = (n - k) / 4;
  if (words >= 4) {
    const uint8_t *s = src + 3 * k;
    uint8_t *r = (uint8_t *)__builtin_assume_aligned(red + k, 4);
    uint8_t *g = (uint8_t *)__builtin_assume_aligned(green + k, 4);
    uint8_t *b = (uint8_t *)__builtin_assume_aligned(blue + k, 4);
#pragma clang loop min_iteration_count(4)
#pragma clang loop hint(aie - gpr - realloc, 1)
    for (int w = 0; w < words; w++, s += 12) {
      uint32_t x = s[0] | s[3] << 8 | s[6] << 16 | (uint32_t)s[9] << 24;
      uint32_t y = s[1] | s[4] << 8 | s[7] << 16 | (uint32_t)s[10] << 24;
      uint32_t z = s[2] | s[5] << 8 | s[8] << 16 | (uint32_t)s[11] << 24;
      __builtin_memcpy(r + 4 * w, &x, 4);
      __builtin_memcpy(g + 4 * w, &y, 4);
      __builtin_memcpy(b + 4 * w, &z, 4);
    }
    k += 4 * words;
  }
  for (; k < n; k++) {
    red[k] = src[3 * k];
    green[k] = src[3 * k + 1];
    blue[k] = src[3 * k + 2];
  }
}

// The band's output rows whose input rows have all come, in order.
void down(const int32_t *vt) {
  if (!ok)
    return;
  int p = vt[0], slot = 2 + (vt[1] + 1) / 2, n = owned * SIDE * 3;
  Srs srs;
  const auto bias = aie::broadcast<int32_t, 32>(1 << (p - 1));
  const uint8_t *in[WIN];
  for (; next < SIDE; next++) {
    const int32_t *t = vt + HEADER + next * slot;
    int s = t[0], c = t[1];
    if (s + c > arrived)
      break;
    if (s < 0 || s < arrived - WIN || c < 0 || c > WIN) {
      ok = false;
      break;
    }
    const int16_t *w = (const int16_t *)(t + 2);
    // Row s's slot: ring is arrived's, and s is at most WIN rows before it.
    int m = ring - (arrived - s);
    m += m < 0 ? WIN : 0;
    for (int k = 0; k < c; k++, m = m + 1 == WIN ? 0 : m + 1)
      in[k] = mid[m];
    for (int b = 0; b < n; b += 32) {
      sum_t a(bias);
      for (int k = 0; k < c; k++)
        a = aie::mac(a, aie::load_v<32>(in[k] + b).unpack().cast_to<int16_t>(),
                     w[k]);
      aie::store_v(hold[next] + b, a.to_vector<uint8_t>(p));
    }
  }
}
} // namespace

// counts: [width chunks, bands, image chunks now, patches a band, image
// chunks left]. Every count is of what all cores share, never of `ok`, so
// the broadcast streams stay in step on any input.
extern "C" void resize_setup(int32_t *counts, int32_t h, int32_t w, int32_t ho,
                             int32_t wo, int32_t c) {
  rows = h;
  columns = w;
  core = c;
  cpr = w > 0 ? (3 * w + CHUNK - 1) / CHUNK : 0;
  int strips = wo > 0 ? wo / SIDE : 0;
  // A column past the patches at least, so the last of every band is zeros.
  int nmax = strips / CORES + 1;
  owned = c < strips ? (strips - c + CORES - 1) / CORES : 0;
  ok = h >= 1 && w >= 1 && ho >= SIDE && wo >= SIDE && ho % SIDE == 0 &&
       wo % SIDE == 0 && nmax <= COLS;
  consumed = part = ring = arrived = opened = next = 0;
  counts[0] = strips;
  counts[1] = ho > 0 ? ho / SIDE : 0;
  counts[2] = 0;
  counts[3] = nmax;
  counts[4] = 0;
}

// Width chunk k is patch column k, this core's when k % CORES == core.
extern "C" void resize_take(int32_t *chunk, int32_t k) {
  if (!ok || k % CORES != core)
    return;
  int i = k / CORES, p = chunk[0], win = chunk[1];
  if (p < 1 || p > 22 || win < 1 || win > WIN || chunk[2] != k * SIDE ||
      chunk[3] != SIDE) {
    ok = false;
    return;
  }
  hp = p;
  hwin = win;
  int slot = 2 + (win + 1) / 2;
  const int32_t *last = chunk + HEADER + (SIDE - 1) * slot;
  int first = chunk[HEADER], end = last[0] + last[1];
  for (int o = 0; o < SIDE; o++) {
    const int32_t *t = chunk + HEADER + o * slot;
    const int16_t *w = (const int16_t *)(t + 2);
    hstart[i][o] = t[0];
    if (t[0] < first || t[1] < 0 || t[1] > win || t[0] + t[1] > end)
      ok = false;
    for (int j = 0; j < 64; j++)
      hweight[i][o][j] = j < t[1] ? w[j] : 0;
  }
  span0[i] = first;
  span[i] = end - first;
  if (first < 0 || end > columns || span[i] > SPAN)
    ok = false;
}

// The next height chunk opens: the image chunks up to the last input row its
// outputs read.
extern "C" void resize_band(int32_t *vt, int32_t *counts) {
  int win = vt[1], e = rows;
  if (vt[0] >= 1 && vt[0] <= 22 && win >= 1 && win <= WIN &&
      vt[2] == opened * SIDE && vt[3] == SIDE) {
    const int32_t *t = vt + HEADER + (SIDE - 1) * (2 + (win + 1) / 2);
    e = t[0] + t[1];
    e = e < 0 ? 0 : e > rows ? rows : e;
  } else {
    ok = false;
  }
  int need = e * cpr - consumed;
  counts[2] = need > 0 ? need : 0;
  opened++;
  next = 0;
  down(vt);
}

extern "C" void resize_consume(uint8_t *chunk, int32_t *vt) {
  event0();
  consumed++;
  if (ok) {
    int at = part * CHUNK;
    for (int i = 0; i < owned; i++) {
      int lo = 3 * span0[i], hi = lo + 3 * span[i];
      int from = lo > at ? lo : at;
      int to = hi < at + CHUNK ? hi : at + CHUNK;
      // x * 43691 >> 17 is x / 3 for 0 <= x < 98304, which Peano calls
      // __divsi3 for; past this column's bytes px is not used.
      int px = (unsigned)(from - lo) * 43691u >> 17, ch = from - lo - 3 * px;
      const uint8_t *src = chunk + (from - at), *stop = chunk + (to - at);
      uint8_t *red = line[i][0], *green = line[i][1], *blue = line[i][2];
      // The pixels a chunk boundary splits, then whole ones.
      if (ch == 1 && src < stop)
        green[px] = *src++, ch = 2;
      if (ch == 2 && src < stop)
        blue[px++] = *src++;
      int whole = stop > src ? (stop - src) * 43691 >> 17 : 0;
      split(src, red + px, green + px, blue + px, whole);
      src += 3 * whole;
      px += whole;
      if (src < stop)
        red[px] = *src++;
      if (src < stop)
        green[px] = *src;
    }
  }
  if (++part != cpr) {
    event1();
    return;
  }
  if (ok && hwin > 32)
    across<true>();
  else if (ok)
    across<false>();
  part = 0;
  ring = ring + 1 == WIN ? 0 : ring + 1;
  arrived++;
  down(vt);
  event1();
}

// Patch column core + i * CORES of the band, zeros past the image.
extern "C" void resize_emit(uint16_t *out, int32_t i) {
  if (next < SIDE)
    ok = false;
  if (!ok || i >= owned) {
    for (int k = 0; k < SIDE * SIDE * 3; k += 32)
      aie::store_v(out + k, aie::zeros<uint16_t, 32>());
    return;
  }
  // Four pixels, a word of each plane in and six out. Byte loads here
  // miscompile: llvm-aie's post-increment combine over a chained pointer reads
  // row for row + SIDE.
  uint16_t *o = (uint16_t *)__builtin_assume_aligned(out, 4);
  for (int y = 0; y < SIDE; y++) {
    const uint8_t *row = hold[y] + 3 * i * SIDE;
    for (int x = 0; x < SIDE; x += 4, o += 12) {
      uint32_t c[3], w[6];
      for (int ch = 0; ch < 3; ch++)
        __builtin_memcpy(&c[ch],
                         __builtin_assume_aligned(row + ch * SIDE + x, 4), 4);
      for (int v = 0; v < 12; v += 2)
        w[v / 2] = PIXEL[c[v % 3] >> 8 * (v / 3) & 255] |
                   (uint32_t)PIXEL[c[(v + 1) % 3] >> 8 * ((v + 1) / 3) & 255]
                       << 16;
      __builtin_memcpy(o, w, 24);
    }
  }
}

extern "C" void resize_finish(int32_t *counts) {
  counts[4] = rows * cpr - consumed;
}
