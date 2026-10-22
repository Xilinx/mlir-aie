//===- decode_glu.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_layout.h"
#include "lut_based_ops.h"
#include "utils.h"

template <int L>
void pseduo_glu(bf16 *y, const bf16 *x) {
  bf16 *gate_ptr = const_cast<bf16 *>(x) + (L / 2);
  bf16 *hid_ptr = const_cast<bf16 *>(x);
  bf16 *y_ptr = y;
  for (int i = 0; i < L / 2; i += 16) {
    aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr + i);
    aie::vector<bf16, 16> hid_vec = aie::load_v<16>(hid_ptr + i);
    gate_vec = getGeluBf16(gate_vec);

    aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, hid_vec);
    aie::store_v(y_ptr + i, y_vec);
  }
}

extern "C" {

// One GLU_SLICE of gate and up values into slice i of the hidden buffer.
void glu_chunk(bf16 *hid, bf16 *x, int i) {
  pseduo_glu<GLU_SLICE>(hid + i * (GLU_SLICE / 2), x);
}

// Slice j of the hidden buffer out through one object of the y stream.
void glu_emit(bf16 *y_out, bf16 *hid, int j) {
  copy_vectorized<bf16, GLU_SLICE / 2>(y_out, hid + j * (GLU_SLICE / 2));
}
}