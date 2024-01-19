//===- pass_through.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _PASS_THROUGH_H
#define _PASS_THROUGH_H

extern "C" {

    void pass_through(uint8_t *input0,uint8_t *input1,  uint8_t *output, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels ); 
    
} // extern "C"

#endif
