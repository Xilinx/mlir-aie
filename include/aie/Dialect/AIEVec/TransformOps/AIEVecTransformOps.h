//===- AIEVecTransformOps.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIEVECTRANSFORMOPS_H
#define AIE_DIALECT_AIEVEC_AIEVECTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"

namespace mlir {
namespace linalg {
class GenericOp;
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/TransformOps/AIEVecTransformOps.h.inc"

#endif // AIE_DIALECT_AIEVEC_AIEVECTRANSFORMOPS_H
