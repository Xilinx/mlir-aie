//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _MY_KERNEL_H
#define _MY_KERNEL_H

extern "C" {

//void extern_kernel(int32_t *restrict buf);
//void my_threshold(int32_t *img_in, int32_t *img_out, 
//	const int32_t img_width, const int32_t img_height, 
//	const int32_t thesh_val, const int32_t max_val);
//void my_threshold(int32_t *img_in, int32_t *img_out); 
void conv2dk3(int8_t *line0, int8_t *line1,int8_t *line2,int8_t *wts, uint8_t *output,  const int32_t  input_width,  const int32_t  input_channels, const int32_t  output_channels,
                        const int32_t  kernel_width,  const int32_t  kernel_height,  const int32_t  check, const int scale, const int channel_offset);                   
       
} // extern "C"

#endif
