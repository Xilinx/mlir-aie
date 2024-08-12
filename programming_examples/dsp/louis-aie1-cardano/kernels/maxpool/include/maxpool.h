//===- maxpool.h -----------------------------------000---*- C++ -*-===//
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

#include "maxpool_2x2.cc"

template<unsigned int M,
         unsigned int N,
         unsigned int C,
         unsigned int MP_S,
         unsigned int MP_W>
void maxpool_int8(input_window_uint8* restrict actsIn,
                  output_window_uint8* restrict actsOut) {
    maxpool_int8_2x2(actsIn, actsOut, M, N, C);
}

#endif
