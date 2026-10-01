//===- decode_rope_core.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Lock-free entry points around decode_rope.cc's per-head arithmetic, for
// testing it in isolation.
#include "decode_rope.cc"

#if FLM_GEMMA4_DECODE_ROPE_SWA
constexpr int ROPE_CORE_DH = SWA_DH;
#else
constexpr int ROPE_CORE_DH = DH;
#endif

extern "C" {

// x is one q head then one k head; both are normalized in place, as in the
// kernel. rope_w is [cos | sin | q norm weight | k norm weight].
void rope_head_core(bf16 *restrict x, bf16 *restrict rope_w, bf16 *restrict y) {
  constexpr int Dh = ROPE_CORE_DH;
  event0();
  _rotate_t<Dh>(y, x, rope_w, rope_w + Dh);
  _rotate_t<Dh>(y + Dh, x + Dh, rope_w, rope_w + 2 * Dh);
  event1();
}

void v_norm_core(bf16 *restrict x, bf16 *restrict y) {
  event0();
  rms_norm_unweighted<ROPE_CORE_DH>(y, x);
  event1();
}
}
