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

/// \brief Same pass with its options set explicitly, for programmatic callers
/// (e.g. AIETranslateNpuToCpp) not going through a pipeline string.
/// \param foldDDRAddrOffset true for the xclbin + instruction-buffer runtime;
/// false for full-ELF and HRX, which translate host addresses themselves.
/// \param emitDispatchShim also emit the extern "C" entry points the JIT
/// dispatch bridge loads via ctypes.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEXToEmitCPass(bool foldDDRAddrOffset, bool emitDispatchShim);

} // namespace xilinx

#endif // AIE_CONVERSION_AIEXTOEMITC_AIEXTOEMITC_H
