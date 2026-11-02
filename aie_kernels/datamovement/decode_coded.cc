//===- decode_coded.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// m coded rows (aie.utils.coded) of DIM_K weights in, the bf16 rows out:
// common/coded_rows.h's decode as a kernel of its own.

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_K
#error Please define DIM_K, the weights in a row, at compile time.
#endif

#ifndef CAPACITY
#define CAPACITY 15
#endif

#include "../common/coded_rows.h"

extern "C" {

void decode_coded_bf16(uint32_t m, const uint8_t *__restrict coded_in,
                       bfloat16 *__restrict out) {
  event0();
  decode_coded_rows<DIM_K, CAPACITY>(m, coded_in, out);
  event1();
}

} // extern "C"
