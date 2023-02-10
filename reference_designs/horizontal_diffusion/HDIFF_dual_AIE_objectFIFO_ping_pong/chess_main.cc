// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET

// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hdiff.h"

int main() {
  int32_t din1[10] = {};
  int32_t din2[10] = {};
  int32_t din3[10] = {};
  int32_t din4[10] = {};
  int32_t din5[10] = {};
  int32_t dout[10] = {};
  vec_hdiff(din1, din2, din3, din4, din5, dout);
  return 1;
}
