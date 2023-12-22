//===- Translation.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_TRANSLATION_H
#define AIE_C_TRANSLATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED char *aieTranslateAIEVecToCpp(MlirOperation op, bool aieml);
MLIR_CAPI_EXPORTED char *translateModuleToLLVMIR(MlirOperation op);
MLIR_CAPI_EXPORTED char *aieTranslateToCDO(MlirOperation op);
MLIR_CAPI_EXPORTED char *aieTranslateToIPU(MlirOperation op);
MLIR_CAPI_EXPORTED char *aieTranslateToXAIEV2(MlirOperation op);
MLIR_CAPI_EXPORTED char *aieTranslateToBCF(MlirOperation op, int col, int row);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_TRANSLATION_H
