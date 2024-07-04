// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

extern "C" {
void passthrough(int32_t *in, int32_t *out, int32_t sz) {
  for (int i = 0; i < sz; i++) {
    out[i] = in[i];
  }
}
}
