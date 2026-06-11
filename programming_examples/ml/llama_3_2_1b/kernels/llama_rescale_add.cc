//===- llama_rescale_add.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase B residual rescale-add: combine an int8 residual stream carrying a
// per-token fp32 scale with the TRUE fp32 projection output (o_proj / down),
// requant to a NEW per-token dynamic scale. This is the device port of the
// numpy golden's residual_dyn add (numpy_layer_mh.py rescale-add): the
// catastrophic static-0.05 residual is replaced by a per-token dynamic scale
// recomputed at every add.
//
//   resid_fp = resid_i8 * resid_scale        (resid_scale from resid[D..D+4])
//   x_fp     = resid_fp + proj_fp
//   x_scale  = max(|x_fp|) / 127
//   x_i8     = round(x_fp / x_scale)          (1/x_scale via sw_recip)
//
// Two-pass-no-intermediate-buffer (Bug 2: Peano AIE2P corrupts stack fp32
// arrays kept across loops): pass A finds absmax, pass B requants. The fp32
// add is recomputed in both passes; the 2x scalar cost is negligible vs the
// gemms feeding this.
//
// Buffers: resid int8[kD + 8] (4 B scale + 4 B pad tail); proj float[kD];
// out int8[kD + 8] (new scale written to out[kD..kD+4], pad zeroed).
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

// Pure IEEE fp32 reciprocal: Peano AIE2P lowers `1.0f / x` to a HW reciprocal
// approximation, NOT IEEE-correct (Bug 1). NR over a bit-hack initial guess
// converges to IEEE fp32 precision (4 iters). Identical to the dyn-rmsnorm
// kernel's sw_recip so the numpy golden and device target share the function.
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
