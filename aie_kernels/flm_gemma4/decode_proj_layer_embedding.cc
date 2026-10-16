//===- decode_proj_layer_embedding.cc ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_bf16_proj.h"
#include "decode_geometry.h"
#include "decode_layout.h"
#include "rms_norm.h"
#include "utils.h"

extern "C" {

void proj_layer_embedding_finish(bf16 *norm_w, bf16 *x0_per_layer, bf16 *x_proj,
                                 bf16 *y) {
  scale_vectorized<PLI_D>(x_proj, (bf16)PER_LAYER_INPUT_SCALE);

  rms_norm<PLI_D>(x_proj, x_proj, norm_w);
  residual_add<PLI_D>(x_proj, x0_per_layer, x_proj);

  scale_vectorized<PLI_D>(x_proj, (bf16)PER_LAYER_MODEL_PROJECTION_SCALE);
  copy_vectorized<bf16, PLI_D>(y + MODEL_DIM + 32, x_proj);
  copy_vectorized<bf16, MODEL_DIM + 32>(y, norm_w + PLI_D);
}
}