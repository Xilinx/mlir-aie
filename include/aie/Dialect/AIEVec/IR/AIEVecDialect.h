//===- AIEVecDialect.h - AIE Vector Dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file defines the AIE vector dialect.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H

#include "aie/Dialect/AIEVec/IR/AIEVecTypes.h"

namespace xilinx {
namespace aievec {

class AIEVecDialect;
// Translation from AIE vector code to C++
void registerAIEVecToCppTranslation();

} // end namespace aievec
} // end namespace xilinx

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.h.inc"

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H
