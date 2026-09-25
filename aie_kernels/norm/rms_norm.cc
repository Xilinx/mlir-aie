//===- rms_norm.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __AIE_ARCH__ == 20
#include "rms_norm_aie2.h"
#else
#include "rms_norm_aie2p.h"
#endif
