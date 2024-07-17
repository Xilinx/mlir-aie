//===- AIEVecAIE1Dialect.h - AIE1 Vector Dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file defines the AIE1 vector dialect.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIE1_IR_AIEVECAIE1DIALECT_H
#define AIE_DIALECT_AIEVEC_AIE1_IR_AIEVECAIE1DIALECT_H

// #include "aie/Dialect/AIEVec/IR/AIEVecTypes.h"

namespace xilinx {
namespace aievec {
namespace aie1 {

class AIEVecAIE1Dialect;
// Translation from AIE vector code to C++
// void registerAIEVecToCppTranslation();

} // end namespace aie1
} // end namespace aievec
} // end namespace xilinx

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1OpsDialect.h.inc"

#endif // AIE_DIALECT_AIEVEC_AIE1_IR_AIEVECAIE1DIALECT_H
