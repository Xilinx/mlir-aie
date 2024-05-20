//===- TargetModel.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_TARGETMODEL_H
#define AIE_C_TARGETMODEL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque type declarations.
//
// Types are exposed to C bindings as structs containing opaque pointers. They
// are not supposed to be inspected from C. This allows the underlying
// representation to change without affecting the API users. The use of structs
// instead of typedefs enables some type safety as structs are not implicitly
// convertible to each other.
//
// Instances of these types may or may not own the underlying object. The
// ownership semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage d;                                                                 \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AieTargetModel, uint32_t);

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED AieTargetModel aieGetTargetModel(uint32_t device);

/// Returns the number of columns in the target model.
MLIR_CAPI_EXPORTED int aieTargetModelColumns(AieTargetModel targetModel);

/// Returns the number of rows in the target model.
MLIR_CAPI_EXPORTED int aieTargetModelRows(AieTargetModel targetModel);

/// Returns true if this is an NPU target model.
MLIR_CAPI_EXPORTED bool aieTargetModelIsNPU(AieTargetModel targetModel);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_TARGETMODEL_H
