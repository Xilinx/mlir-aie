//===- AIEVecOps.h - AIE Vector Dialect and Operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file defines the AIE vector dialect and the operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H

#include "AIEVecDialect.h"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h.inc"

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H
