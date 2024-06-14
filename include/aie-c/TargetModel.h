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

DEFINE_C_API_STRUCT(AieTargetModel, uint64_t);

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED AieTargetModel aieGetTargetModel(uint32_t device);

/// Returns the data bus width for the target model.
MLIR_CAPI_EXPORTED uint32_t
aieGetTargetModelAddressGenGranularity(AieTargetModel targetModel);

/// Returns the number of columns in the target model.
MLIR_CAPI_EXPORTED int aieTargetModelColumns(AieTargetModel targetModel);

/// Returns the number of rows in the target model.
MLIR_CAPI_EXPORTED int aieTargetModelRows(AieTargetModel targetModel);

/// Returns true if this is an NPU target model.
MLIR_CAPI_EXPORTED bool aieTargetModelIsNPU(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED bool aieTargetModelIsCoreTile(AieTargetModel targetModel,
                                                 int col, int row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsMemTile(AieTargetModel targetModel,
                                                int col, int row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsShimNOCTile(AieTargetModel targetModel,
                                                    int col, int row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsShimPLTile(AieTargetModel targetModel,
                                                   int col, int row);

MLIR_CAPI_EXPORTED bool
aieTargetModelIsShimNOCorPLTile(AieTargetModel targetModel, int col, int row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsInternal(AieTargetModel targetModel,
                                                 int src_col, int src_row,
                                                 int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsWest(AieTargetModel targetModel,
                                             int src_col, int src_row,
                                             int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsEast(AieTargetModel targetModel,
                                             int src_col, int src_row,
                                             int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsNorth(AieTargetModel targetModel,
                                              int src_col, int src_row,
                                              int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsSouth(AieTargetModel targetModel,
                                              int src_col, int src_row,
                                              int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsMemWest(AieTargetModel targetModel,
                                                int src_col, int src_row,
                                                int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsMemEast(AieTargetModel targetModel,
                                                int src_col, int src_row,
                                                int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsMemNorth(AieTargetModel targetModel,
                                                 int src_col, int src_row,
                                                 int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool aieTargetModelIsMemSouth(AieTargetModel targetModel,
                                                 int src_col, int src_row,
                                                 int dst_col, int dst_row);

MLIR_CAPI_EXPORTED bool
aieTargetModelIsLegalMemAffinity(AieTargetModel targetModel, int src_col,
                                 int src_row, int dst_col, int dst_row);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetMemSouthBaseAddress(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetMemNorthBaseAddress(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetMemEastBaseAddress(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetMemWestBaseAddress(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetLocalMemorySize(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetNumLocks(AieTargetModel targetModel, int col, int row);

MLIR_CAPI_EXPORTED uint32_t aieTargetModelGetNumBDs(AieTargetModel targetModel,
                                                    int col, int row);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetNumMemTileRows(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t
aieTargetModelGetMemTileSize(AieTargetModel targetModel);

/// Returns true if this is an NPU target model.
MLIR_CAPI_EXPORTED bool aieTargetModelIsNPU(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t aieTargetModelGetColumnShift(AieTargetModel targetModel);

MLIR_CAPI_EXPORTED uint32_t aieTargetModelGetRowShift(AieTargetModel targetModel);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_TARGETMODEL_H
