//===- AIECoreSymbols.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Name that the dialect and the aiecc driver share for a compiled core: the
// symbol of its outlined body. One definition keeps the two in step.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H
#define AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H

#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::AIE {

// Name that AIECoreToStandard gives the function it outlines a CoreOp's body
// into. Every reader of a compiled core object resolves the body through it.
inline std::string coreFrameSymbolName(int col, int row) {
  std::string name;
  llvm::raw_string_ostream(name) << "core_" << col << "_" << row;
  return name;
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H
