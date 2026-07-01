//===- AIEVecAIE1Dialect.h - AIE1 Vector Dialect ----------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the AIE1 vector dialect.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIE1_IR_AIEVECAIE1DIALECT_H
#define AIE_DIALECT_AIEVEC_AIE1_IR_AIEVECAIE1DIALECT_H

namespace xilinx {
namespace aievec {
namespace aie1 {

class AIEVecAIE1Dialect;

} // end namespace aie1
} // end namespace aievec
} // end namespace xilinx

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1OpsDialect.h.inc"

#endif // AIE_DIALECT_AIEVEC_AIE1_IR_AIEVECAIE1DIALECT_H
