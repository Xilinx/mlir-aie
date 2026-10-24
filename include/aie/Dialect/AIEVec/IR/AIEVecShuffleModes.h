//===- AIEVecShuffleModes.h - Semantics of aievec.shuffle modes -*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// What each aievec.shuffle mode does, derived from the mode name.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECSHUFFLEMODES_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECSHUFFLEMODES_H

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

namespace xilinx::aievec {

// Whether the mode exists on AIE2P only.
bool isAIE2POnlyShuffleMode(ShuffleMode mode);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECSHUFFLEMODES_H
