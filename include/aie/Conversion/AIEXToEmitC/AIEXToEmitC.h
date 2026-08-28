//===- AIEXToEmitC.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIEXTOEMITC_AIEXTOEMITC_H
#define AIE_CONVERSION_AIEXTOEMITC_AIEXTOEMITC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEXToEmitCPass();

// Same pass, with the fold-ddr-addr-offset option set explicitly (see
// ConvertAIEXToEmitC's option in Passes.td) -- for programmatic callers
// (e.g. AIETranslateNpuToCpp) that aren't going through a textual pass
// pipeline string.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEXToEmitCPass(bool foldDDRAddrOffset);

} // namespace xilinx

#endif // AIE_CONVERSION_AIEXTOEMITC_AIEXTOEMITC_H
