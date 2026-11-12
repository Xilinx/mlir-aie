//===- TargetInfo.h - AIEVec target parsing ---------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parsing of the `aie-target` option shared by the AIEVec passes.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_UTILS_TARGETINFO_H
#define AIE_DIALECT_AIEVEC_UTILS_TARGETINFO_H

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace xilinx::aievec {

// Named apart from xilinx::AIE::AIEArch, which several AIEVec sources also see.
enum class AIEVecArch { AIE1, AIE2, AIE2P, AIE2PS };

struct AIEVecTargetInfo {
  AIEVecArch arch = AIEVecArch::AIE1;

  bool isAIE1() const { return arch == AIEVecArch::AIE1; }
  bool isAIE2() const { return arch == AIEVecArch::AIE2; }
  bool isAIE2P() const { return arch == AIEVecArch::AIE2P; }
  bool isAIE2PS() const { return arch == AIEVecArch::AIE2PS; }

  // Whether vector-to-AIEVec conversion and AIEVec-to-LLVM lowering exist for
  // this target. AIE2PS is accepted as a target, but has neither.
  bool hasAIEVecLowering() const { return !isAIE2PS(); }
};

// Accepts "" and "aie" (AIE1), "aieml" and "aie2" (AIE2), "aie2p" and "aie2ps".
// Returns std::nullopt for any other spelling.
std::optional<AIEVecTargetInfo> parseAIEVecTarget(llvm::StringRef spelling);

// As above, but also reports an unknown spelling as an error on `op`.
std::optional<AIEVecTargetInfo> parseAIEVecTarget(llvm::StringRef spelling,
                                                  mlir::Operation *op);

// As above, and also reports an error on `op` if the target has no AIEVec
// lowering. Used by the vector-to-AIEVec passes.
std::optional<AIEVecTargetInfo>
parseAIEVecLoweringTarget(llvm::StringRef spelling, mlir::Operation *op);

llvm::StringRef getAIEVecTargetName(AIEVecTargetInfo info);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_UTILS_TARGETINFO_H
