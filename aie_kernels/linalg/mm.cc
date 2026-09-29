//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_arch.h"

// Each header's micro-tiles are its architecture's mmul shapes, and AIE2 has
// not all of AIE2P's (no transpose of an 8x8 int32 C tile), so the choice
// follows the architecture rather than AIE_TUNED_*. _MM_MAC_DIMS in
// python/iron/kernels/linalg.py is keyed the same way.
#if AIE_ARCH_AIE2
#include "mm_aie2.h"
#else
#include "mm_aie2p.h"
#endif
