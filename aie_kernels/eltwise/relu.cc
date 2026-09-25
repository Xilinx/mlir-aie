//===- relu.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __AIE_ARCH__ == 20
#include "relu_aie2.h"
#else
#include "relu_aie2p.h"
#endif
