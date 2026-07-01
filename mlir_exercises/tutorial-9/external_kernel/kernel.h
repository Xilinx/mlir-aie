//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _MY_KERNEL_H
#define _MY_KERNEL_H

extern "C" {

void extern_kernel(int32_t *restrict buf);

} // extern "C"

#endif
