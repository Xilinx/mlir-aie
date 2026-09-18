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

/// \brief Configure the pass programmatically; see AIETranslateNpuToCpp.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEXToEmitCPass(bool foldDDRAddrOffset, bool emitDispatchShim);

} // namespace xilinx

#endif // AIE_CONVERSION_AIEXTOEMITC_AIEXTOEMITC_H
