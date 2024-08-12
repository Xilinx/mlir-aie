//===- conv2d.h -----------------------------------000---*- C++ -*-===//
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

#include "conv2d_s1.cc"
#include "conv2d_sx.cc"
#include "conv2d_s1_px.cc"

template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int P,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
    void conv2d_int8_casc(input_window_uint8* restrict actsIn,
                          input_window_int8* restrict weightsIn,
                          input_stream_acc48* accIn,
                          output_stream_acc48* accOut) {

    // For free relu on WB
    set_sat();

    if(S == 1) {
        S1::conv2d_int8_S1(actsIn, weightsIn, NULL, M, N, Cin, Cout, F, S, P, CASC_IN_EN, CASC_OUT_EN);
    } else {
        SX::conv2d_int8_S(actsIn, weightsIn, NULL, M, N, Cin, Cout, F, S, P, CASC_IN_EN, CASC_OUT_EN);
    }
}

/*template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int P,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
void conv2d_int8_casc_out(input_window_int8* restrict actsIn,
                          input_window_int8* restrict weightsIn,
                          input_stream_acc48* accIn,
                          output_window_uint8* restrict actsOut) {

    // For free relu on WB
    set_sat();

    if(S == 1) {
        S1::conv2d_int8_S1(actsIn, weightsIn, actsOut, M, N, Cin, Cout, F, S, P, CASC_IN_EN, CASC_OUT_EN);
    } else {
        SX::conv2d_int8_S(actsIn, weightsIn, actsOut, M, N, Cin, Cout, F, S, P, CASC_IN_EN, CASC_OUT_EN);
    }
    }*/

template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int P,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
void conv2d_int8_casc_out(input_window_uint8* restrict actsIn,
                          input_window_int8* restrict weightsIn,
                          input_stream_acc48* accIn,
                          output_window_uint8* restrict actsOut,
                          const unsigned int PE,
                          const unsigned int PW,
                          const unsigned int PN,
                          const unsigned int PS) {

    // For free relu on WB
    set_sat();

    if(S == 1 && (P==1)) {
        S1PX::conv2d_int8_S1(actsIn, weightsIn, actsOut,
                             M, N, Cin, Cout, F, S, P,
                             PE, PW, PN, PS,
                             CASC_IN_EN, CASC_OUT_EN);
    } else if (S == 1 && P == 0){
        S1::conv2d_int8_S1(actsIn, weightsIn, actsOut,
                           M, N, Cin, Cout, F, S, P,
                           CASC_IN_EN, CASC_OUT_EN);
    } else {
        SX::conv2d_int8_S(actsIn, weightsIn, actsOut,
                          M, N, Cin, Cout, F, S, P,
                          CASC_IN_EN, CASC_OUT_EN);
    }
}

template <unsigned int M,
          unsigned int N,
          unsigned int Cin,
          unsigned int Cout,
          unsigned int F,
          unsigned int S,
          unsigned int P,
          unsigned int CASC_IN_EN,
          unsigned int CASC_OUT_EN>
void conv2d_int8_casc_in(input_window_uint8* restrict actsIn,
                         input_window_int8* restrict weightsIn,
                         output_stream_acc48* accOut) {

    // For free relu on WB
    set_sat();

    if(S == 1) {
        S1::conv2d_int8_S1(actsIn, weightsIn, NULL, M, N, Cin, Cout, F, S, P, CASC_IN_EN, CASC_OUT_EN);
    } else {
        SX::conv2d_int8_S(actsIn, weightsIn, NULL, M, N, Cin, Cout, F, S, P, CASC_IN_EN, CASC_OUT_EN);
    }
}


#endif
