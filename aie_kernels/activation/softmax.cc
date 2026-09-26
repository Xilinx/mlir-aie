//===- softmax.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_arch.h"

#if AIE_TUNED_AIE2
#include "softmax_aie2.h"
#else
#include "softmax_aie2p.h"
#endif
