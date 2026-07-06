//===- AIEVecDialect.h - AIE Vector Dialect ---------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

} // end namespace aievec
} // end namespace xilinx

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.h.inc"

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H
