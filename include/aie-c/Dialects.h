//===- Dialects.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_DIALECTS_H
#define AIE_C_DIALECTS_H

#include "mlir-c/IR.h"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIE, aie);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIEVec, aievec);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIEX, aiex);

//===---------------------------------------------------------------------===//
// ObjectFifoType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool aieTypeIsObjectFifoType(MlirType type);
MLIR_CAPI_EXPORTED MlirType aieObjectFifoTypeGet(MlirType type);

//===---------------------------------------------------------------------===//
// ObjectFifoSubviewType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool aieTypeIsObjectFifoSubviewType(MlirType type);
MLIR_CAPI_EXPORTED MlirType aieObjectFifoSubviewTypeGet(MlirType type);

//===---------------------------------------------------------------------===//
// BlockFloatType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool aieTypeIsBlockFloatType(MlirType type);
MLIR_CAPI_EXPORTED MlirType aieBlockFloatTypeGet(MlirContext ctx,
                                                 const std::string &blockType);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_DIALECTS_H
