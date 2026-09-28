//===- gelu.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_arch.h"

#if AIE_TUNED_AIE2
#include "gelu_aie2.h"
#else
#include "gelu_aie2p.h"
#endif
