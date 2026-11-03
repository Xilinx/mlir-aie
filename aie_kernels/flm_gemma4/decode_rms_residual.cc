//===- decode_rms_residual.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_geometry.h"
#include "rms_norm.h"
#include "utils.h"

// The Worker owns the three phases and their input/output synchronization.
// state keeps the residual across projections. y has the projection stream's
// 16-bf16 header followed by the normalized vector.
extern "C" {
void rms_begin(bf16 *x, bf16 *w, bf16 *state, bf16 *y) {
  *reinterpret_cast<uint32 *>(y + 14) = 0;
  copy_vectorized<bf16, MODEL_DIM>(state, x);
  rms_norm<MODEL_DIM>(y + 16, state, w);
}

void rms_middle(bf16 *x, bf16 *w, bf16 *state, bf16 *y) {
  bf16 *temp = state + MODEL_DIM;
  rms_norm<MODEL_DIM>(y + 16, x, w + MODEL_DIM);
  residual_add<MODEL_DIM>(temp, state, y + 16);
  copy_vectorized<bf16, MODEL_DIM>(state, temp);
  rms_norm<MODEL_DIM>(y + 16, temp, w + 2 * MODEL_DIM);
}

void rms_finish(bf16 *x, bf16 *w, bf16 *state, bf16 *y) {
  bf16 *temp = state + MODEL_DIM;
  rms_norm<MODEL_DIM>(temp, x, w + 3 * MODEL_DIM);
  residual_add<MODEL_DIM>(y, state, temp);
}
}
