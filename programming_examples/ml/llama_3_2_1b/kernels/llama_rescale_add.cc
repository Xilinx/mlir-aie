//===- llama_rescale_add.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Residual rescale-add: adds an int8 residual (carrying a per-token fp32
// scale) to the TRUE fp32 projection output, requanting to a NEW
// per-token dynamic scale recomputed at every add -- replaces a
// catastrophic static-0.05 residual scale (numpy_layer_mh.py's
// residual_dyn add). Two-pass, no intermediate buffer (Peano AIE2P
// corrupts stack fp32 arrays kept across loops): pass A finds absmax,
// pass B requants, recomputing the add both times (negligible cost vs.
// the gemms feeding this). Buffers: resid/out int8[kD+8] (4 B scale + 4
// B pad); proj float[kD].
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_RESCALE_D
#define LLAMA_RESCALE_D 2048
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

// Software fp32 reciprocal (see llama_flowkv.cc for why Peano's `/` needs
// this workaround); identical to the other kernels' sw_recip so the numpy
// golden and device target share the function.
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

void llama_rescale_add_D(int8_t *restrict resid, float *restrict proj,
                         int8_t *restrict out) {
  event0();

  constexpr int kD = LLAMA_RESCALE_D;

  float resid_scale;
  memcpy(&resid_scale, resid + kD, 4);

  // Pass A: x_fp = resid_i8 * resid_scale + proj; track absmax.
  float absmax = 0.0f;
  for (int i = 0; i < kD; i++) {
    float xf = (float)resid[i] * resid_scale + proj[i];
    float a = xf >= 0.0f ? xf : -xf;
    if (a > absmax)
      absmax = a;
  }

  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float x_scale = absmax * (1.0f / 127.0f);
  float inv = sw_recip(x_scale);

  // Pass B: recompute the fp32 sum, requant to int8 with the new scale.
  for (int i = 0; i < kD; i++) {
    float xf = (float)resid[i] * resid_scale + proj[i];
    out[i] = round_to_i8(xf * inv);
  }

  memcpy(out + kD, &x_scale, 4);
  int32_t zero = 0;
  memcpy(out + kD + 4, &zero, 4);

  event1();
}

} // extern "C"
