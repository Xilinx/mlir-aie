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

// Stack requirement aiecc's call-graph analysis computed for a core, stamped
// on the CoreOp so later stages can report it alongside the stack region.
// aiecc erases it again before the module is handed on, so it never reaches
// user-visible IR.
inline constexpr llvm::StringLiteral kComputedStackRequirementAttrName =
    "aiecc.computed_stack_requirement";

// Name of the top-level function AIECoreToStandard outlines a CoreOp's body
// into. The canonical definition is there (where the function is actually
// created); every other reader of a compiled core object -- aiecc's
// post-build stack-size check reads this function's own frame size back out
// of the object -- must agree on the same name, so they share this rather
// than each formatting "core_<col>_<row>" independently.
inline std::string coreFrameSymbolName(int col, int row) {
  std::string name;
  llvm::raw_string_ostream(name) << "core_" << col << "_" << row;
  return name;
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_IR_AIECORESYMBOLS_H
