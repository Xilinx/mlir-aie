//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

template <typename T_in, typename T_out, int N>
void scale(T_in *a, T_out *c, T_in factor) {
  for (int i = 0; i < N; i++) {
    c[i] = factor * a[i];
  }
}

extern "C" {

void scale_int32(int32_t *a_in, int32_t *c_out) {
  scale<int32_t, int32_t, 16>(a_in, c_out, 3);
}

} // extern "C"
