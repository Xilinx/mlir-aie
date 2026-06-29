//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _MY_KERNEL_H
#define _MY_KERNEL_H

extern "C" {

void extern_kernel(int32_t *restrict A, int32_t *restrict B,
                   int32_t *restrict acc, int32_t *restrict C);
} // extern "C"

#endif
