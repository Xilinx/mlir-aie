//===- Translation.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_TRANSLATION_H
#define AIE_C_TRANSLATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef aieTranslateAIEVecToCpp(MlirOperation op,
                                                         bool aieml);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateModuleToLLVMIR(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToNPU(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToXAIEV2(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToHSA(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToBCF(MlirOperation op, int col,
                                                   int row);
MLIR_CAPI_EXPORTED MlirStringRef aieLLVMLink(MlirStringRef *modules,
                                             int nModules);
MLIR_CAPI_EXPORTED MlirLogicalResult aieTranslateToCDODirect(
    MlirOperation moduleOp, MlirStringRef workDirPath, bool bigEndian,
    bool emitUnified, bool cdoDebug, bool aieSim, bool xaieDebug,
    size_t partitionStartCol, bool enableCores);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_TRANSLATION_H
