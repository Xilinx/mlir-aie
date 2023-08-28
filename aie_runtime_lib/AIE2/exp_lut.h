//===---  exp_lut.h - get exponential values from loopup tables ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//
//===----------------------------------------------------------------------===//
// This is the implementation of getting exponential values for a bfloat16
// vector from exponential lookup tables.
//===----------------------------------------------------------------------===//
#ifndef __EXP_LUT_H__
#define __EXP_LUT_H__

__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x);
#endif //__EXP_LUT_H__
