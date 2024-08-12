//===- dw_params.h -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#ifndef MAXPOOL_PARAMS_H
#define MAXPOOL_PARAMS_H

#define MR (((M%2) == 1) ? M + 1 : M)

#define inTileSize (MR * N * C)

#define outHeight ((N - DW_W)/DW_S + 1)
#define outWidthR ((M - DW_W)/DW_S + 1)

// Because of our data layout ensure even width
#define outWidth (((outWidthR % 2) == 0) ? outWidthR : outWidthR + 1)

#define outTileSize  outWidth * outHeight * C
#define realOutTileSize  outHeight * outWidthR * C

#define weightTileSize C * ((DW_W + 3) / 4) * 4 * DW_W

#define AIn_FILENAME "../data/AIn.txt"
#define WIn_FILENAME "../data/WIn.txt"
#define AOutRefReg_FILENAME "../data/AOut.txt"

#define SHIFT 0

#endif
