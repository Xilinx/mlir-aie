//===- TargetInfo.cpp - AIEVec target parsing -----------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/Utils/TargetInfo.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace xilinx::aievec {

std::optional<AIEVecTargetInfo> parseAIEVecTarget(StringRef spelling) {
  return StringSwitch<std::optional<AIEVecTargetInfo>>(spelling)
      .Cases({"", "aie"}, AIEVecTargetInfo{AIEVecArch::AIE1})
      .Cases({"aieml", "aie2"}, AIEVecTargetInfo{AIEVecArch::AIE2})
      .Case("aie2p", AIEVecTargetInfo{AIEVecArch::AIE2P})
      .Case("aie2ps", AIEVecTargetInfo{AIEVecArch::AIE2PS})
      .Default(std::nullopt);
}

std::optional<AIEVecTargetInfo> parseAIEVecTarget(StringRef spelling,
                                                  Operation *op) {
  std::optional<AIEVecTargetInfo> info = parseAIEVecTarget(spelling);
  if (!info)
    op->emitError() << "unknown AIE target '" << spelling << "'";
  return info;
}

std::optional<AIEVecTargetInfo> parseAIEVecLoweringTarget(StringRef spelling,
                                                          Operation *op) {
  std::optional<AIEVecTargetInfo> info = parseAIEVecTarget(spelling, op);
  if (!info)
    return std::nullopt;
  if (!info->hasAIEVecLowering()) {
    op->emitError() << "AIEVec has no vector lowering for target '"
                    << getAIEVecTargetName(*info) << "'";
    return std::nullopt;
  }
  return info;
}

StringRef getAIEVecTargetName(AIEVecTargetInfo info) {
  switch (info.arch) {
  case AIEVecArch::AIE1:
    return "aie";
  case AIEVecArch::AIE2:
    return "aie2";
  case AIEVecArch::AIE2P:
    return "aie2p";
  case AIEVecArch::AIE2PS:
    return "aie2ps";
  }
  llvm_unreachable("unknown AIEVecArch");
}

} // namespace xilinx::aievec
