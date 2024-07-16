//===- AIEVecAIE1Ops.h - AIE1 Vector Dialect and Operations -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// This file defines the AIE1 vector dialect and the operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECAIE1OPS_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECAIE1OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// #include "aie/Dialect/AIEVec/IR/AIEVecEnums.h.inc"
// #define GET_ATTRDEF_CLASSES
// #include "aie/Dialect/AIEVec/IR/AIEVecAttributes.h.inc"

#include "AIEVecAIE1Dialect.h"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.h.inc"

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECAIE1OPS_H
