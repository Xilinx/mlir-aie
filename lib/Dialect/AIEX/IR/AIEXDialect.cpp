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

LogicalResult AIEX::NpuDmaMemcpyNdOp::verify() {
  MemRefType buffer = getMemref().getType();
  if (buffer.getElementTypeBitWidth() != 32)
    return emitOpError("must be used with memref type with element width 32.");
  if (!llvm::all_of(getMixedStrides(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    llvm::report_fatal_error("Only constant strides currently supported.");
  if (!llvm::all_of(getMixedSizes(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    llvm::report_fatal_error("Only constant sizes currently supported.");
  if (!llvm::all_of(getMixedOffsets(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    llvm::report_fatal_error("Only constant offsets currently supported.");

  llvm::SmallVector<int64_t, 3> strides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> sizes =
      llvm::map_to_vector(llvm::reverse(getMixedSizes()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });

  if (sizes[3] > 64)
    return emitOpError("Size 3 exceeds the [1:64] range.");
  if (strides[1] && sizes[1] > 0x3FF)
    return emitOpError("Size 1 exceeds the [0:1023] range.");
  if (strides[0] && sizes[0] > 0x3FF)
    return emitOpError("Size 0 exceeds the [0:1023] range.");
  if (strides[2] > 0x100000)
    return emitOpError("Stride 3 exceeds the [1:1M] range.");
  if (strides[1] > 0x100000)
    return emitOpError("Stride 2 exceeds the [1:1M] range.");
  if (strides[0] > 0x100000)
    return emitOpError("Stride 1 exceeds the [1:1M] range.");
  return success();
}

LogicalResult AIEX::NpuDmaWaitOp::verify() {
  AIE::DeviceOp dev = (*this)->getParentOfType<AIE::DeviceOp>();
  // Some passes (e.g. aie-standard-lowering) use aiex ops outside a DeviceOp,
  // so we can't expect the device to always exist.
  if (dev && !dev.lookupSymbol(getSymbol()))
    return emitOpError("couldn't find symbol in parent device");
  return success();
}

LogicalResult AIEX::NpuShimTilePushQueueOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(0, 0); // assume shim
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getRepeatCount() > 255)
    return emitOpError("Repeat count exceeds the [0:255] range.");
  return success();
}

LogicalResult AIEX::NpuWriteBdExShimTileOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(0, 0); // assume shim
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getD0Size() > 0x3FF)
    return emitOpError("D0 Size exceeds the [0:1023] range.");
  if (getD0Stride() > 0xFFFFF)
    return emitOpError("D0 Stride exceeds the [0:1M-1] range.");
  if (getD1Size() > 0x3FF)
    return emitOpError("D1 Size exceeds the [0:1023] range.");
  if (getD1Stride() > 0xFFFFF)
    return emitOpError("D1 Stride exceeds the [0:1M-1] range.");
  if (getD2Stride() > 0xFFFFF)
    return emitOpError("D2 Stride exceeds the [0:1M-1] range.");
  if (getIterationSize() > 0x3F)
    return emitOpError("Iteration Size exceeds the [0:63] range.");
  if (getIterationStride() > 0xFFFFF)
    return emitOpError("Iteration Stride exceeds the [0:1M-1] range.");
  return success();
}