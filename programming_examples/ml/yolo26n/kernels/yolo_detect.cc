//===- yolo_detect.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// YOLO26n's end-to-end postprocess, bit-exact with its ONNX graph; see
// SegmentDesign.detect. Scores compare by their rank among the sigmoid
// tables' float32 values, ties to the lower index as in ONNX TopK. Anchors
// run level by level, row by row (80x80 at stride 8, 40x40 at 16, 20x20 at
// 32). Class logits come CHUNK anchors at a time, [CLASSES / 8][CHUNK][8];
// box distances [CHUNK][8], the first 4 of each anchor's 8 bytes.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#ifndef YOLO_DETECT_CHUNK
#error "YOLO_DETECT_CHUNK (anchors per record chunk) must be defined"
#endif

constexpr int CLASSES = 80;
constexpr int KEEP = 300;
constexpr int RANKS = 768;
constexpr int ANCHORS = 6400 + 1600 + 400;
// Bytes of picked bits, with room for a chunk's word read past the last.
constexpr int PICKED = ANCHORS / 8 + 4;
constexpr int CHUNK = YOLO_DETECT_CHUNK;
static_assert(6400 % CHUNK == 0 && 8000 % CHUNK == 0,
              "a chunk stays within one level");

static int level_of(int anchor) {
  return anchor < 6400 ? 0 : (anchor < 8000 ? 1 : 2);
}

// value = num * 2**exp2 as float32 bits; |num| < 2**24, so exact.
static int32_t f32_bits(int32_t num, int exp2) {
  if (num == 0)
    return 0;
  const uint32_t sign = num < 0 ? 0x80000000u : 0;
  const uint32_t mag = num < 0 ? -num : num;
  const int msb = 31 - __builtin_clz(mag);
  const uint32_t mantissa = (mag << (23 - msb)) & 0x7fffff;
  return sign | uint32_t(msb + exp2 + 127) << 23 | mantissa;
}

// Places, from the highest rank down and stable within a rank, of the
// entries for_each(visit) visits, as visit(index, rank) in index order;
// writes to `first` the entries taking the top KEEP places. `hist` is
// RANKS words of scratch.
template <typename ForEach>
static void top_keep(ForEach for_each, int32_t *__restrict hist,
                     int16_t *__restrict first) {
  for (int r = 0; r < RANKS; r++)
    hist[r] = 0;
  for_each([&](int, int key) { hist[key]++; });
  // hist[r] becomes the first place of rank r, ranks taken from the top.
  int32_t place = 0;
  for (int r = RANKS - 1; r >= 0; r--) {
    const int32_t count = hist[r];
    hist[r] = place;
    place += count;
  }
  for_each([&](int i, int key) {
    const int32_t at = hist[key]++;
    if (at < KEEP)
      first[at] = i;
  });
}

// x / 80 for 0 <= x < 2**17, without a division libcall.
static int div80(int x) { return (x * 52429) >> 22; }

static int popcount8(uint32_t byte) {
  byte = byte - ((byte >> 1) & 0x55);
  byte = (byte & 0x33) + ((byte >> 2) & 0x33);
  return (byte + (byte >> 4)) & 0x0f;
}

extern "C" {

// best: per anchor, the rank of its best class score; hist counts them per
// rank (zeroed at chunk 0). ONNX Runtime's float32 sigmoid is not monotonic
// where it saturates, so the scores' ranks, not the logits, are compared.
// `rank` holds each level's ranks of the 256 logits, then their peaks (the
// highest rank of any logit up to each): an anchor's highest logit has its
// best score when its rank is the peak, otherwise every class is ranked.
void yolo_detect_max(const int8_t *__restrict records,
                     const int16_t *__restrict rank, int16_t *__restrict best,
                     int32_t *__restrict hist, int32_t chunk) {
  event0();
  static_assert(CHUNK * 8 % 32 == 0, "a chunk's group is whole vectors");
  if (chunk == 0)
    for (int r = 0; r < RANKS; r++)
      hist[r] = 0;
  const int level = level_of(chunk * CHUNK);
  const int16_t *ranks = rank + level * 256 + 128;
  const int16_t *peaks = rank + (3 + level) * 256 + 128;
  // Lane 8a of `most` ends up the highest logit of anchor a.
  alignas(32) int8_t most[CHUNK * 8];
  for (int j = 0; j < CHUNK * 8 / 32; j++) {
    aie::vector<int8_t, 32> m = aie::load_v<32>(records + j * 32);
    AIE_LOOP_UNROLL_FULL
    for (int g = 1; g < CLASSES / 8; g++)
      m = aie::max(m, aie::load_v<32>(records + g * CHUNK * 8 + j * 32));
    m = aie::max(m, aie::shuffle_down_rotate(m, 4));
    m = aie::max(m, aie::shuffle_down_rotate(m, 2));
    m = aie::max(m, aie::shuffle_down_rotate(m, 1));
    aie::store_v(most + j * 32, m);
  }
  for (int a = 0; a < CHUNK; a++) {
    const int8_t m = most[a * 8];
    int16_t top = ranks[m];
    if (top != peaks[m])
      for (int g = 0; g < CLASSES / 8; g++) {
        const int8_t *group = records + (g * CHUNK + a) * 8;
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 8; l++) {
          const int16_t r = ranks[group[l]];
          top = r > top ? r : top;
        }
      }
    best[chunk * CHUNK + a] = top;
    hist[top]++;
  }
  event1();
}

// first: the KEEP anchors in score order, from max's ranks and their
// counts. picked: one bit per anchor. row: for the k-th picked anchor by
// index, its place in `first`. count: the gather's running count of picked
// anchors, reset here.
void yolo_detect_select(const int16_t *__restrict best,
                        int32_t *__restrict hist, int16_t *__restrict first,
                        uint8_t *__restrict picked, int16_t *__restrict row,
                        int32_t *__restrict count) {
  event0();
  static_assert(ANCHORS / 8 <= 2 * RANKS, "the prefix counts reuse hist");
  static_assert(ANCHORS % 16 == 0, "ranks are scanned 16 at a time");
  // hist[r] becomes the first place of rank r, ranks taken from the top;
  // only anchors of rank `low` or higher take a place below KEEP.
  int32_t place = 0;
  int16_t low = 0;
  for (int r = RANKS - 1; r >= 0; r--) {
    const int32_t n = hist[r];
    hist[r] = place;
    if (place < KEEP)
      low = r;
    place += n;
  }
  for (int i = 0; i < ANCHORS; i += 16) {
    uint32_t bits =
        aie::ge(aie::load_v<16>(best + i), aie::broadcast<int16_t, 16>(low))
            .to_uint32();
    while (bits) {
      const int a = i + __builtin_ctz(bits);
      bits &= bits - 1;
      const int32_t at = hist[best[a]]++;
      if (at < KEEP)
        first[at] = a;
    }
  }
  for (int i = 0; i < PICKED; i++)
    picked[i] = 0;
  for (int k = 0; k < KEEP; k++)
    picked[first[k] / 8] |= 1 << (first[k] % 8);
  // The picked anchors before each byte of `picked`, then each one's
  // number in index order, as the gather meets them.
  int16_t *before = reinterpret_cast<int16_t *>(hist);
  int16_t n = 0;
  for (int i = 0; i < ANCHORS / 8; i++) {
    before[i] = n;
    n += popcount8(picked[i]);
  }
  for (int place = 0; place < KEEP; place++) {
    const int a = first[place];
    const int k =
        before[a / 8] + popcount8(picked[a / 8] & ((1u << (a % 8)) - 1));
    row[k] = place;
  }
  count[0] = 0;
  event1();
}

void yolo_detect_gather(const int8_t *__restrict records,
                        const int8_t *__restrict boxes,
                        const uint8_t *__restrict picked,
                        const int16_t *__restrict row,
                        int8_t *__restrict candidates,
                        int8_t *__restrict distances, int32_t *__restrict count,
                        int32_t chunk) {
  event0();
  static_assert(CHUNK <= 24, "a chunk's bits are within one word");
  const int base = chunk * CHUNK;
  const uint8_t *bytes = picked + base / 8;
  uint32_t bits =
      (bytes[0] | bytes[1] << 8 | bytes[2] << 16 | uint32_t(bytes[3]) << 24) >>
      (base % 8);
  bits &= (1u << CHUNK) - 1;
  // A picked anchor's classes, eight at a time, and its four distances.
  uint64_t *cand = reinterpret_cast<uint64_t *>(candidates);
  const uint64_t *rec = reinterpret_cast<const uint64_t *>(records);
  uint32_t *dist = reinterpret_cast<uint32_t *>(distances);
  const uint32_t *box = reinterpret_cast<const uint32_t *>(boxes);
  while (bits) {
    const int i = __builtin_ctz(bits);
    bits &= bits - 1;
    const int r = row[count[0]++];
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < CLASSES / 8; g++)
      cand[r * CLASSES / 8 + g] = rec[g * CHUNK + i];
    dist[r] = box[i * 2];
  }
  event1();
}

// out: KEEP rows of x1, y1, x2, y2, score, class as float32 bits.
// box_shift: per level, log2 of the box distances' scale.
void yolo_detect_finish(const int8_t *__restrict candidates,
                        const int8_t *__restrict distances,
                        const int16_t *__restrict first,
                        const int16_t *__restrict rank,
                        const int32_t *__restrict sigmoid,
                        int32_t *__restrict hist, int16_t *__restrict best,
                        int32_t *__restrict out, int32_t box_shift0,
                        int32_t box_shift1, int32_t box_shift2) {
  event0();
  // Each picked anchor's best class ranks at least as high as the last
  // one's, so the best KEEP pairs do too: only logits that can reach that
  // rank at some level are ranked.
  const int16_t *last = rank + level_of(first[KEEP - 1]) * 256 + 128;
  int16_t floor = 0;
  for (int c = 0; c < CLASSES; c++) {
    const int16_t r = last[candidates[(KEEP - 1) * CLASSES + c]];
    floor = r > floor ? r : floor;
  }
  // Per level, the lowest logit of rank at least `floor`.
  int8_t lowest[3] = {127, 127, 127};
  for (int level = 0; level < 3; level++)
    for (int v = -128; v < 127; v++)
      if (rank[level * 256 + v + 128] >= floor) {
        lowest[level] = v;
        break;
      }
  static_assert(KEEP * CLASSES % 32 == 0, "candidates are whole vectors");
  static_assert(CLASSES >= 32, "a vector spans at most two candidates");
  alignas(32) static const int8_t lane[32] = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  top_keep(
      [&](auto visit) {
        const aie::vector<int8_t, 32> lanes = aie::load_v<32>(lane);
        int r = 0, end = CLASSES;
        for (int i = 0; i < KEEP * CLASSES / 32; i++) {
          if (i * 32 >= end) {
            r++;
            end += CLASSES;
          }
          // Lanes before `end` are candidate r's, the rest r + 1's.
          const int next = r + 1 < KEEP ? r + 1 : r;
          const auto low = aie::select(
              aie::broadcast<int8_t, 32>(lowest[level_of(first[next])]),
              aie::broadcast<int8_t, 32>(lowest[level_of(first[r])]),
              aie::lt(lanes, int8_t(end - i * 32 < 32 ? end - i * 32 : 32)));
          uint32_t bits =
              aie::ge(aie::load_v<32>(candidates + i * 32), low).to_uint32();
          while (bits) {
            const int e = i * 32 + __builtin_ctz(bits);
            bits &= bits - 1;
            const int row = e >= end ? next : r;
            visit(e, rank[level_of(first[row]) * 256 + candidates[e] + 128]);
          }
        }
      },
      hist, best);
  const int32_t shifts[3] = {box_shift0, box_shift1, box_shift2};
  for (int k = 0; k < KEEP; k++) {
    static_assert(CLASSES == 80, "div80");
    const int r = div80(best[k]);
    const int c = best[k] - r * CLASSES;
    const int a = first[r];
    const int level = level_of(a);
    const int local = a - (level == 0 ? 0 : (level == 1 ? 6400 : 8000));
    const int width = 80 >> level;
    const int log2_stride = 3 + level;
    // In units of 2**-frac: an anchor centre is x + 1/2, a distance d * 2**s.
    const int s = shifts[level];
    const int frac = -s > 1 ? -s : 1;
    const int y = div80(local << level);
    const int32_t ax = (2 * (local - y * width) + 1) << (frac - 1);
    const int32_t ay = (2 * y + 1) << (frac - 1);
    const int8_t *d = distances + r * 4;
    const int up = frac + s;
    out[k * 6 + 0] = f32_bits(ax - d[0] * (1 << up), log2_stride - frac);
    out[k * 6 + 1] = f32_bits(ay - d[1] * (1 << up), log2_stride - frac);
    out[k * 6 + 2] = f32_bits(ax + d[2] * (1 << up), log2_stride - frac);
    out[k * 6 + 3] = f32_bits(ay + d[3] * (1 << up), log2_stride - frac);
    out[k * 6 + 4] = sigmoid[level * 256 + candidates[r * CLASSES + c] + 128];
    out[k * 6 + 5] = f32_bits(c, 0);
  }
  event1();
}

} // extern "C"
