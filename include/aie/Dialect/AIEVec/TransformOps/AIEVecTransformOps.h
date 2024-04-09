//===- AIEVecTransformOps.h -------------------------------------*- C++ -*-===//
//
// Copyright (c) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIEVECTRANSFORMOPS_H
#define AIE_DIALECT_AIEVEC_AIEVECTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"

namespace mlir {
namespace linalg {
class GenericOp;
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/TransformOps/AIEVecTransformOps.h.inc"

#endif // AIE_DIALECT_AIEVEC_AIEVECTRANSFORMOPS_H
