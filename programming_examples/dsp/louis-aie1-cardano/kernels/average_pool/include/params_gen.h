//===- params_gen.h -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#ifndef PARAMS_H
#define PARAMS_H

#define M 11
#define N 11
#define C 64

#define MR (((M%2) == 1) ? M + 1 : M)

#define MP_W 2
#define MP_S 2

#define inTileSize (MR * N * C)

#define outHeight (N/MP_S)
#define outWidth ((M / MP_S) + ((M / MP_S) % 2))

#define outTileSize  outWidth * outHeight * C
#define realOutTileSize  (M / MP_S) *  (N / MP_S) * C

#define AIn_FILENAME "../data/AIn.txt"
#define AOutRef_FILENAME "../data/AOutRef.txt"
#define AOutRefReg_FILENAME "../data/AOutRefReg.txt"

#endif
