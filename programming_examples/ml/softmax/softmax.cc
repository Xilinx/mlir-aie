//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

// Softmax DUT generated from vector dialect
extern void dut(bfloat16 *a_in, bfloat16 *cout);

extern "C" {

void softmax_bf16_vector(bfloat16 *a_in, bfloat16 *c_out) {
  event0();
  dut(a_in, c_out);
  event1();
}

} // extern "C"
