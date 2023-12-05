//===- AIETargets.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx {
namespace AIE {

mlir::LogicalResult AIETranslateToXAIEV2(mlir::ModuleOp module,
                                         llvm::raw_ostream &output);
mlir::LogicalResult AIEFlowsToJSON(mlir::ModuleOp module,
                                   llvm::raw_ostream &output);
mlir::LogicalResult ADFGenerateCPPGraph(mlir::ModuleOp module,
                                        llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateSCSimConfig(mlir::ModuleOp module,
                                            llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateShimSolution(mlir::ModuleOp module,
                                             llvm::raw_ostream &);
mlir::LogicalResult AIETranslateGraphXPE(mlir::ModuleOp module,
                                         llvm::raw_ostream &);
mlir::LogicalResult AIETranslateToCDO(mlir::ModuleOp module,
                                      llvm::raw_ostream &output);

mlir::LogicalResult AIETranslateToIPU(mlir::ModuleOp module,
                                      llvm::raw_ostream &output);
std::vector<uint32_t> AIETranslateToIPU(mlir::ModuleOp);
mlir::LogicalResult AIETranslateToLdScript(mlir::ModuleOp module,
                                           llvm::raw_ostream &output,
                                           int tileCol, int tileRow);
} // namespace AIE
} // namespace xilinx
