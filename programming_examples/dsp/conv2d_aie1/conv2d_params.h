//===- test_vck5000.cpp -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#ifndef CONV2D_PARAMS_H
#define CONV2D_PARAMS_H

#define MR (((M%2) == 1) ? M + 1 : M)

#define CinUp (((Cin+7)/8) * 8)
#define CoutUp (((Cout+7)/8) * 8)

#define outHeight ((N - F + 2*P) / S + 1)
#define outWidthR ((M - F + 2*P) / S + 1)

#define outWidth ((outWidthR%2) == 1 ? outWidthR + 1 : outWidthR)

#define SHIFT 0


#endif
