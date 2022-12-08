//===- Registration.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_REGISTRATION_H
#define AIE_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all AIE dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
void aieRegisterAllDialects(MlirContext context);

/** Registers all AIE passes for symbolic access with the global registry. */
void aieRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // AIE_C_REGISTRATION_H
