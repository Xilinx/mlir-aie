// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.

extern "C" {
void passthrough(int32_t *in, int32_t *out, int32_t sz) {
  for (int i = 0; i < sz; i++) {
    out[i] = in[i];
  }
}
}
