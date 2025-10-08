//===- conv2dk3.h -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _CONV2DK3_H
#define _CONV2DK3_H

extern "C" {

void conv2dk14_i8(int8_t *input, int8_t *kernels, int8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int32_t kernel_width,
                  const int scale);

// void conv2dk14_i8(int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts,
//                  uint8_t *output, const int32_t input_width,
//                  const int32_t input_channels, const int32_t output_channels,
//                  const int32_t kernel_width, const int32_t kernel_height,
//                  const int32_t check, const int scale,
//                  const int channel_offset);

// void conv2dk14_ui8(uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t
// *wts,
//                   uint8_t *output, const int32_t input_width,
//                   const int32_t input_channels, const int32_t
//                   output_channels, const int32_t kernel_width, const int32_t
//                   kernel_height, const int32_t check, const int scale, const
//                   int channel_offset);

} // extern "C"

#endif
