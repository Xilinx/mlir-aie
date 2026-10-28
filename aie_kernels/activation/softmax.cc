//===- softmax.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_arch.h"

// The accurate exp2 is a 32-lane polynomial, which softmax_aie2p.h's walk fits.
#if AIE_TUNED_AIE2 && !defined(EXP2_BF16_ACCURATE)
#include "softmax_aie2.h"
#else
#include "softmax_aie2p.h"
#endif
