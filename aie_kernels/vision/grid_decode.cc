//===- grid_decode.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An anchor-free detector's boxes in float32 from one row of a prediction
// map: each cell's centre offset by its grid position and its size
// exponentiated, both times the map's stride; objectness and class scores
// copied after them.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include "../common/exp_f32.h"
#include <aie_api/aie.hpp>

namespace {
constexpr int N = 16;
constexpr int OUT = 5 + CLASSES;
} // namespace

extern "C" {

// in: WIDTH cells of CHANNELS bf16, the box first, objectness at OBJ, the
// CLASSES scores from CLS; out: WIDTH rows of OUT float32, the row at
// grid row y of a map of stride `stride`, a power of two.
void grid_decode(const bfloat16 *__restrict in, float *__restrict out,
                 int32_t y, int32_t stride) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, N> one = aie::broadcast<bfloat16, N>(1.0f);
  const float s = (float)stride, gy = (float)y;
  for (int x0 = 0; x0 < WIDTH; x0 += N) {
    aie::vector<bfloat16, N> dw, dh;
    for (int i = 0; i < N; i++) {
      const int x = x0 + i < WIDTH ? x0 + i : WIDTH - 1;
      dw[i] = in[x * CHANNELS + 2];
      dh[i] = in[x * CHANNELS + 3];
    }
    const aie::vector<float, N> w =
        exp_f32(aie::mul(dw, one)).template to_vector<float>();
    const aie::vector<float, N> h =
        exp_f32(aie::mul(dh, one)).template to_vector<float>();
    for (int i = 0; i < N && x0 + i < WIDTH; i++) {
      const bfloat16 *cell = in + (x0 + i) * CHANNELS;
      float *row = out + (x0 + i) * OUT;
      // A power of two scales each exactly; the sum ahead of it is one
      // native float add, not correctly rounded on AIE2P (nms.cc says
      // why), which the contract's 1e-6 relative tolerance covers.
      row[0] = ((float)cell[0] + (float)(x0 + i)) * s;
      row[1] = ((float)cell[1] + gy) * s;
      row[2] = w[i] * s;
      row[3] = h[i] * s;
      row[4] = (float)cell[OBJ];
      for (int c = 0; c < CLASSES; c++)
        row[5 + c] = (float)cell[CLS + c];
    }
  }
  event1();
}

} // extern "C"
