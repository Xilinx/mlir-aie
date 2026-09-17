//===- AIEObjectFifoUtils.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEOBJECTFIFOUTILS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEOBJECTFIFOUTILS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

namespace xilinx::AIE {

/// The fifo whose objects form a link's shared pool. Placement and lowering
/// must agree on this: charging each participant separately double-counts
/// storage, while choosing the smaller participant can underestimate it.
inline ObjectFifoCreateOp getObjectFifoLinkPoolOwner(ObjectFifoLinkOp link) {
  auto ins = link.getInputObjectFifos();
  auto outs = link.getOutputObjectFifos();
  if (link.isJoin())
    return outs[0];
  if (link.isDistribute())
    return ins[0];

  auto inType = mlir::cast<mlir::MemRefType>(
      mlir::cast<AIEObjectFifoType>(ins[0].getElemType()).getElementType());
  auto outType = mlir::cast<mlir::MemRefType>(
      mlir::cast<AIEObjectFifoType>(outs[0].getElemType()).getElementType());
  // Padding is applied as the objects leave, so the pool holds what arrives.
  if (outs[0].getInitValues() ||
      (outType.getNumElements() > inType.getNumElements() &&
       !outs[0].getPadDimensions()))
    return outs[0];
  return ins[0];
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEOBJECTFIFOUTILS_H
