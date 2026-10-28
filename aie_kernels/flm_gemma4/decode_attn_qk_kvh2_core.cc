//===- decode_attn_qk_kvh2_core.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// k holds the two KV heads' k objects back to back; m travels as in
// decode_attn_qk_core.cc.
#include "decode_attn_qk_kvh2.cc"

extern "C" void attn_qk_kvh2_round_core(bf16 *qm, bf16 *k, bf16 *s, float *c,
                                        int iter, int L0) {
  event0();
  bf16 *m_in = qm + Q_HEADS_PADDED_PER_CU * DH;
  alignas(aie::vector_decl_align) bf16 m[16];
  aie::store_v(m, aie::load_v<16>(m_in));

  int i = L0 - 16 * iter;
  bool is_first = (iter == 0);
  aie::vector<int, 16> idx;
  for (int z = 0; z < 16; z++)
    idx.set(z + 1, z);
  aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);

  for (int j = 0; j < 2; j++) {
    bool is_up = (j == 0);
    _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(qm, k + j * 16 * DH, s + j * 4 * 16,
                                          m + j * 4, c + j * 4, mask, is_first,
                                          is_up);
  }

  aie::store_v(s + Q_HEADS_PADDED_PER_CU * 16, aie::load_v<16>(m));
  event1();
}
