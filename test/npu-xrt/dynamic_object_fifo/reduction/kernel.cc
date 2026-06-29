//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned long N>
void add(const T_in *__restrict inA, const T_in *__restrict inB,
         T_out *__restrict out) {
  for (int i = 0; i < N; i++) {
    out[i] = inA[i] + inB[i];
  }
}

extern "C" {

void add_10_i32(const int *__restrict inA, const int *__restrict inB,
                int *__restrict out) {
  add<int, int, 10>(inA, inB, out);
}
}
