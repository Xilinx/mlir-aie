//===- Dialects.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_DIALECTS_H
#define AIE_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIE, aie);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIEVec, aievec);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIEX, aiex);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_DIALECTS_H
