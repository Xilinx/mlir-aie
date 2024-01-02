//===- AIEDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx;

#include "aie/Dialect/AIEX/IR/AIEXDialect.cpp.inc"

namespace xilinx::AIEX {

// FIXME: use Tablegen'd dialect class
void AIEXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"
      >();
}

} // namespace xilinx::AIEX

#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"

LogicalResult AIEX::UseTokenOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (isa<func::FuncOp>(parentOp) || isa<AIE::CoreOp>(parentOp) ||
      isa<AIE::MemOp>(parentOp) || isa<AIE::ShimDMAOp>(parentOp))
    return success();
  return failure();
}

LogicalResult AIEX::MulticastOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front())
    if (!isa<MultiDestOp, AIE::EndOp>(ops))
      return ops.emitOpError("cannot be contained in a Multicast op");

  return success();
}

LogicalResult AIEX::BroadcastPacketOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front())
    if (!isa<BPIDOp, AIE::EndOp>(ops))
      return ops.emitOpError("cannot be contained in a BroadcastPacket op");

  return success();
}

LogicalResult AIEX::IpuDmaMemcpyNdOp::verify() {
  MemRefType buffer = getMemref().getType();
  if (!buffer.getElementType().isInteger(32))
    return emitOpError("must be used with memref type i32.");
  llvm::SmallVector<int32_t> strides(getStrides().rbegin(),
                                     getStrides().rend());
  llvm::SmallVector<int32_t> lengths(getLengths().rbegin(),
                                     getLengths().rend());

  if (lengths[3] > 64)
    return emitOpError("Length 3 exceeds the [1:64] range.");
  if (strides[1] && lengths[1] > 0x3FF)
    return emitOpError("Length 1 exceeds the [0:1023] range.");
  if (strides[0] && lengths[0] > 0x3FF)
    return emitOpError("Length 0 exceeds the [0:1023] range.");
  if (strides[2] > 0x100000)
    return emitOpError("Stride 3 exceeds the [1:1M] range.");
  if (strides[1] > 0x100000)
    return emitOpError("Stride 2 exceeds the [1:1M] range.");
  if (strides[0] > 0x100000)
    return emitOpError("Stride 1 exceeds the [1:1M] range.");
  return success();
}

LogicalResult AIEX::IpuShimTilePushQueueOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(0, 0); // assume shim
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getRepeatCount() > 255)
    return emitOpError("Repeat count exceeds the [0:255] range.");
  return success();
}

LogicalResult AIEX::IpuWriteBdExShimTileOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(0, 0); // assume shim
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getD0Wrap() > 0x3FF)
    return emitOpError("D0 Wrap exceeds the [0:1023] range.");
  if (getD0Stepsize() > 0xFFFFF)
    return emitOpError("D0 Stepsize exceeds the [0:1M-1] range.");
  if (getD1Wrap() > 0x3FF)
    return emitOpError("D1 Wrap exceeds the [0:1023] range.");
  if (getD1Stepsize() > 0xFFFFF)
    return emitOpError("D1 Stepsize exceeds the [0:1M-1] range.");
  if (getD2Stepsize() > 0xFFFFF)
    return emitOpError("D2 Stepsize exceeds the [0:1M-1] range.");
  if (getIterationWrap() > 0x3F)
    return emitOpError("Iteration Wrap exceeds the [0:63] range.");
  if (getIterationStepsize() > 0xFFFFF)
    return emitOpError("Iteration Stepsize exceeds the [0:1M-1] range.");
  return success();
}