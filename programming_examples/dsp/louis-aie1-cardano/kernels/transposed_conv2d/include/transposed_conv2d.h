//===- transposed_conv2d.h -----------------------------------000---*- C++ -*-===//
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
#include <stdio.h>
#include <cassert>

#include "transposed_conv2d.cc"

// TODO add padding

template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
void transposed_conv2d_int8_casc(input_window_int8* restrict actsIn,
                                 input_window_int8* restrict weightsIn,
                                 input_stream_acc48* accIn,
                                 output_stream_acc48* accOut) {

    // For free relu on WB
    set_sat();

    F2::transposed_conv2d_int8(actsIn, weightsIn, NULL, M, N, Cin, Cout, F, S, CASC_IN_EN, CASC_OUT_EN);
}

template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
void transposed_conv2d_int8_casc_out(input_window_int8* restrict actsIn,
                          input_window_int8* restrict weightsIn,
                          input_stream_acc48* accIn,
                          output_window_uint8* restrict actsOut) {

    // For free relu on WB
    set_sat();

    F2::transposed_conv2d_int8(actsIn, weightsIn, actsOut, M, N, Cin, Cout, F, S, CASC_IN_EN, CASC_OUT_EN);

}

template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
void transposed_conv2d_int8_casc_in(input_window_int8* restrict actsIn,
                         input_window_int8* restrict weightsIn,
                         output_stream_acc48* accOut) {

    // For free relu on WB
    set_sat();

    F2::transposed_conv2d_int8(actsIn, weightsIn, NULL, M, N, Cin, Cout, F, S, CASC_IN_EN, CASC_OUT_EN);
}


#endif
