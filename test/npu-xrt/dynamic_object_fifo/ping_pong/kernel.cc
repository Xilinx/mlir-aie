//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned long N>
void passthrough(const T_in *__restrict in, T_out *__restrict out) {
  for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }
}

extern "C" {

void passthrough_64_i32(const int *__restrict in, int *__restrict out) {
  passthrough<int, int, 64>(in, out);
}
}