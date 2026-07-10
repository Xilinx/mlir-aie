//===- conv2dk1.h -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CONV2DK1_H
#define _CONV2DK1_H

extern "C" {
void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale);

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale);
} // extern "C"

#endif
