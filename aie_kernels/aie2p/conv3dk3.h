//===- conv3dk3.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _CONV3DK3_H
#define _CONV3DK3_H

#include <stdint.h>

extern "C" {

#ifdef SCALAR

void conv3dk3_ui8_scalar(uint8_t *plane0, uint8_t *plane1, uint8_t *plane2,
                         int8_t *wts, uint8_t *output,
                         const int32_t input_width, const int32_t input_height,
                         const int32_t input_channels,
                         const int32_t output_channels,
                         const int32_t kernel_width, const int32_t kernel_height,
                         const int32_t kernel_depth, const int32_t check,
                         const int scale, const int channel_offset);

#else

void conv3dk3_ui8(uint8_t *plane0, uint8_t *plane1, uint8_t *plane2,
                  int8_t *wts, uint8_t *output, const int32_t input_width,
                  const int32_t input_height, const int32_t input_channels,
                  const int32_t output_channels, const int32_t kernel_width,
                  const int32_t kernel_height, const int32_t kernel_depth,
                  const int32_t check, const int scale,
                  const int channel_offset);

#endif

} // extern "C"

#endif // _CONV3DK3_H
