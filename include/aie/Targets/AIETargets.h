//===- AIETargets.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_AIETARGETS_H
#define AIE_TARGETS_AIETARGETS_H

#ifdef AIE_ENABLE_GENERATE_CDO_DIRECT
extern "C" {
#include "cdo_driver.h"
}
#endif

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
mlir::LogicalResult AIETranslateToBCF(mlir::ModuleOp module,
                                      llvm::raw_ostream &output, int tileCol,
                                      int tileRow);
mlir::LogicalResult
AIELLVMLink(llvm::raw_ostream &output, std::vector<std::string> Files,
            bool DisableDITypeMap = false, bool NoVerify = false,
            bool Internalize = false, bool OnlyNeeded = false,
            bool PreserveAssemblyUseListOrder = false, bool Verbose = false);

#ifdef AIE_ENABLE_GENERATE_CDO_DIRECT
mlir::LogicalResult AIETranslateToCDODirect(mlir::ModuleOp m,
                                            llvm::StringRef workDirPath,
                                            byte_ordering endianness,
                                            bool emitUnified, bool axiDebug,
                                            bool aieSim);
#endif

mlir::LogicalResult AIETranslateToTargetArch(mlir::ModuleOp module,
                                             llvm::raw_ostream &output);
} // namespace AIE

namespace aievec {

/// Translates the AIE vector dialect MLIR to C++ code.
mlir::LogicalResult translateAIEVecToCpp(mlir::Operation *op, bool aieml,
                                         mlir::raw_ostream &os);

} // namespace aievec
} // namespace xilinx

#endif