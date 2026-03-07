//===- passthrough_3d.cc ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Minimal passthrough kernel for testing 3D data flow
// Simply copies input plane to output plane without any computation

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../aie_kernel_utils.h"

extern "C" {

//*****************************************************************************
// passthrough_3d - minimal passthrough for testing
// Simply copies one input plane to output plane
//
// Input layout: HxWxC (height, width, channels)
// Output layout: HxWxC (same as input)
//*****************************************************************************
void passthrough_3d_ui8(uint8_t *input_plane, uint8_t *output_plane,
                        const int32_t input_width, const int32_t input_height,
                        const int32_t input_channels) {
  event0();

  // Calculate total plane size
  int32_t plane_size = input_height * input_width * input_channels;

  // Simple memcpy from input to output
  memcpy(output_plane, input_plane, plane_size);

  event1();
}

} // extern "C"
