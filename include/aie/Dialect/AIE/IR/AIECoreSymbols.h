//===- AIECoreSymbols.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Names that the dialect and the aiecc driver have to agree on when they talk
// about a compiled core: the symbol its outlined body gets, and the attribute
// aiecc's stack analysis stamps on it. Formatting either of these in more
// than one place is how they drift.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H
#define AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::AIE {

// aiecc's computed stack requirement for a core, stamped on the CoreOp for
// later diagnostics; erased before the module is handed on.
inline constexpr llvm::StringLiteral kComputedStackRequirementAttrName =
    "aiecc.computed_stack_requirement";

// Name AIECoreToStandard gives the function it outlines a CoreOp's body
// into. Shared so every reader of a compiled core object agrees on it.
inline std::string coreFrameSymbolName(int col, int row) {
  std::string name;
  llvm::raw_string_ostream(name) << "core_" << col << "_" << row;
  return name;
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H
