//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// #include <stdint.h>

extern "C" {

void hdiff_lap (int32_t * restrict in, int32_t * restrict  flux_forward);
void hdiff_flux(int32_t * restrict in, int32_t * restrict  flux_forward,  int32_t * restrict out);

}

