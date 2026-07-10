// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

extern "C" {
void passthrough(int32_t *in, int32_t *out, int32_t sz) {
  for (int i = 0; i < sz; i++) {
    out[i] = in[i];
  }
}
}
