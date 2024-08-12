//===- dwconv.h -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#ifndef KERNEL_H
#define KERNEL_H

#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>

#include "dw_conv_16.cc"
#include "dw_conv_16_s1.cc"

template<unsigned int M,
         unsigned int N,
         unsigned int C,
         unsigned int DW_S,
         unsigned int DW_W>
void dwconv_int8(input_window_uint8* restrict actsIn,
                 input_window_int8* restrict weightsIn,
                 output_window_uint8* restrict actsOut) {
    if(DW_S == 1) {
        S1::dw_conv_int8_3x3(actsIn, weightsIn, actsOut, M, N, C, DW_S);
    } else {
        SX::dw_conv_int8_3x3(actsIn, weightsIn, actsOut, M, N, C, DW_S);
    }
}

#endif
