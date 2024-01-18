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
    
    void conv2dk1_skip_ui8(uint8_t *input0, uint8_t *input1,  int8_t *kernels, uint8_t *output, uint8_t *skip, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels,const int32_t scale,const int32_t skip_scale ) ;                   
} // extern "C"

#endif
