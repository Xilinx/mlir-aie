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
// void conv2dk1(int32_t *ifm, int32_t *wts, int32_t *ofm0,int32_t *ofm1,int32_t *ofm2);
void conv2dk1_skip(uint8_t *input0,uint8_t *input1,  int8_t *kernels, uint8_t *output, int8_t *skip, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels,const int scale,const int skip_scale )      ;  
} // extern "C"

#endif
