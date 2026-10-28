//===- decode_attn_qk_common.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The online-softmax running-max update: rescale against the new max,
// exponentiate, and return the correction factor for the kv side. Shared by
// decode_attn_qk.cc and decode_attn_qk_kvh2.cc. It is not in
// decode_attn_kv_common.h because
// it needs the decode_lut_exp.h table, which costs the kv kernels 4 KB of
// core data memory.
#ifndef AIE_KERNELS_FLM_GEMMA4_DECODE_ATTN_QK_COMMON_H
#define AIE_KERNELS_FLM_GEMMA4_DECODE_ATTN_QK_COMMON_H

#include "decode_geometry.h"
#include "decode_lut_exp.h"

inline aie::vector<bf16, 16> update(bf16 *m, float *c,
                                    aie::vector<bf16, 16> &out,
                                    aie::mask<16> &mask, bool &is_first) {
  aie::vector<bf16, 16> masked_vm =
      aie::select((bf16)(-0x1.FEp127f), out, mask);
  bf16 vm = aie::reduce_max(masked_vm);
  bf16 vmax = aie::max(vm, (bf16)*m);
  aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);

  constexpr int min_clamp = -87.0f;
  constexpr int max_clamp = 88.0f;
  aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
  aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp);

  aie::vector<bf16, 16> Vec = aie::clamp(Vecsub, min_value, max_value);
  aie::accum<accfloat, 16> Outvec = getExpBf16(Vec);
  Vec = Outvec.template to_vector<bf16>(0);
  Vec = aie::select((bf16)0, Vec, mask);

  bf16 correct = aie::sub((bf16)*m, vmax);
  aie::vector<bf16, 16> correct_vec = aie::broadcast<bf16, 16>(correct);
  correct_vec = aie::clamp(correct_vec, min_value, max_value);
  aie::accum<accfloat, 16> correct_acc = getExpBf16(correct_vec);
  aie::vector<float, 16> correct_float =
      correct_acc.template to_vector<float>();

  *m = (float)vmax;
  *c = (float)correct_float.get(0);
  return Vec;
}

#endif // AIE_KERNELS_FLM_GEMMA4_DECODE_ATTN_QK_COMMON_H
