//===- resize_linear.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenCV's INTER_LINEAR resize of a uint8 RGB image, bit for bit as its
// fixed-point path computes it on the CPU, one source row in at a time and
// each output row out once both its source rows are in.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

namespace {
constexpr int CN = 3;
constexpr int COEF = 2048;
constexpr int OUT_W = OUT_BYTES / CN;

int in_h, in_w, out_h, out_w, taken, emitted, xmax;
bool ok;
double scale_y;
int16_t xofs[OUT_W];
int16_t alpha[2 * OUT_W];
int32_t hor[2][OUT_BYTES];

// OpenCV's saturate_cast<short>(float) of a weight in [0, 2048]: cvRound,
// to nearest, ties to even.
int16_t weight(float v) {
  int n = (int)v;
  const float d = v - (float)n;
  if (d > 0.5f || (d == 0.5f && (n & 1)))
    n++;
  return (int16_t)n;
}

int floor_int(float v) {
  int n = (int)v;
  return (float)n > v ? n - 1 : n;
}

// Output row `dy`'s source rows, clamped to the image, and its weights.
void row_of(int dy, int &r0, int &r1, int32_t &b0, int32_t &b1) {
  const float fy = (float)((dy + 0.5) * scale_y - 0.5);
  const int sy = floor_int(fy);
  const float f = fy - (float)sy;
  b0 = weight((1.f - f) * COEF);
  b1 = weight(f * COEF);
  r0 = sy < 0 ? 0 : sy >= in_h ? in_h - 1 : sy;
  r1 = sy + 1 < 0 ? 0 : sy + 1 >= in_h ? in_h - 1 : sy + 1;
}
} // namespace

extern "C" {

// counts[0]: the source rows a call takes; counts[1]: the output rows to
// emit now. Sizes the kernel refuses (past the build's rows, or below 1)
// take every row and emit zero rows, so the streams stay in step.
void resize_linear_setup(int32_t *counts, int32_t rows, int32_t columns,
                         int32_t out_rows, int32_t out_columns) {
  in_h = rows;
  in_w = columns;
  out_h = out_rows;
  out_w = out_columns;
  taken = emitted = 0;
  ok = rows >= 1 && columns >= 1 && out_rows >= 1 && out_columns >= 1 &&
       CN * columns <= IN_BYTES && out_columns <= OUT_W;
  counts[0] = rows > 0 ? rows : 0;
  counts[1] = 0;
  if (!ok)
    return;
  const double scale_x = 1. / ((double)out_w / in_w);
  scale_y = 1. / ((double)out_h / in_h);
  xmax = out_w;
  for (int dx = 0; dx < out_w; dx++) {
    float fx = (float)((dx + 0.5) * scale_x - 0.5);
    int sx = floor_int(fx);
    fx -= (float)sx;
    if (sx < 0)
      fx = 0, sx = 0;
    if (sx + 1 >= in_w) {
      if (dx < xmax)
        xmax = dx;
      if (sx >= in_w - 1)
        fx = 0, sx = in_w - 1;
    }
    xofs[dx] = (int16_t)(CN * sx);
    alpha[2 * dx] = weight((1.f - fx) * COEF);
    alpha[2 * dx + 1] = weight(fx * COEF);
  }
}

// Source row `taken` in: its horizontal pass, exact in int32; then how many
// output rows have both their source rows.
void resize_linear_take(const uint8_t *__restrict row, int32_t *counts) {
  event0();
  int n = 0;
  if (ok) {
    int32_t *h = hor[taken & 1];
    for (int dx = 0; dx < out_w; dx++) {
      const uint8_t *s = row + xofs[dx];
      for (int c = 0; c < CN; c++)
        h[CN * dx + c] =
            dx < xmax ? s[c] * alpha[2 * dx] + s[c + CN] * alpha[2 * dx + 1]
                      : s[c] * COEF;
    }
    for (int dy = emitted; dy < out_h; dy++, n++) {
      int r0, r1;
      int32_t b0, b1;
      row_of(dy, r0, r1, b0, b1);
      if (r1 > taken)
        break;
    }
  }
  taken++;
  counts[1] = n;
  event1();
}

// The next output row: OpenCV's vertical pass on each element, its
// weights at 16 bits and each product's low 16 bits dropped; zeros past
// the image.
void resize_linear_emit(uint8_t *__restrict out) {
  event0();
  int width = 0;
  if (ok && emitted < out_h) {
    int r0, r1;
    int32_t b0, b1;
    row_of(emitted, r0, r1, b0, b1);
    const int32_t *s0 = hor[r0 & 1], *s1 = hor[r1 & 1];
    width = CN * out_w;
    for (int x = 0; x < width; x++)
      out[x] = (uint8_t)((((b0 * (s0[x] >> 4)) >> 16) +
                          ((b1 * (s1[x] >> 4)) >> 16) + 2) >>
                         2);
  }
  for (int x = width; x < OUT_BYTES; x++)
    out[x] = 0;
  emitted++;
  event1();
}

// The output rows still owed once every source row is in: none for an
// image the kernel resized, every row (zeros) for one it refused.
void resize_linear_finish(int32_t *counts) {
  counts[1] = out_h > emitted ? out_h - emitted : 0;
}

} // extern "C"
