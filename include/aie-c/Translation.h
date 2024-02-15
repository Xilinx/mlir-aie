//===- Translation.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_TRANSLATION_H
#define AIE_C_TRANSLATION_H

#ifdef AIE_ENABLE_GENERATE_CDO_DIRECT
extern "C" {
#include "cdo_driver.h"
}
#endif

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef aieTranslateAIEVecToCpp(MlirOperation op,
                                                         bool aieml);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateModuleToLLVMIR(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToIPU(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToXAIEV2(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToBCF(MlirOperation op, int col,
                                                   int row);
MLIR_CAPI_EXPORTED MlirStringRef aieLLVMLink(MlirStringRef *modules,
                                             int nModules);
#ifdef AIE_ENABLE_GENERATE_CDO_DIRECT
MLIR_CAPI_EXPORTED MlirLogicalResult aieTranslateToCDODirect(
    MlirOperation moduleOp, MlirStringRef workDirPath, bool bigEndian,
    bool emitUnified, bool axiDebug, bool aieSim, size_t partitionStartCol);
#endif

#ifdef __cplusplus
}
#endif

#endif // AIE_C_TRANSLATION_H
