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
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx {
namespace AIEX {

// FIXME: use Tablegen'd dialect class
AIEXDialect::AIEXDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect("AIEX", ctx, ::mlir::TypeID::get<AIEXDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"
      >();
}

} // namespace AIEX
} // namespace xilinx

#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"

LogicalResult xilinx::AIEX::UseTokenOp::verify() {
  auto parentOp = (*this)->getParentOp();
  if (isa<func::FuncOp>(parentOp) || isa<xilinx::AIE::CoreOp>(parentOp) ||
      isa<xilinx::AIE::MemOp>(parentOp) ||
      isa<xilinx::AIE::ShimDMAOp>(parentOp))
    return success();
  return failure();
}

LogicalResult xilinx::AIEX::MulticastOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if (auto Op = dyn_cast<xilinx::AIEX::MultiDestOp>(ops)) {
    } else if (auto endswitchOp = dyn_cast<xilinx::AIE::EndOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a Multicast op");
    }
  }

  return success();
}

LogicalResult xilinx::AIEX::BroadcastPacketOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front()) {
    if (auto Op = dyn_cast<xilinx::AIEX::BPIDOp>(ops)) {
    } else if (auto endswitchOp = dyn_cast<xilinx::AIE::EndOp>(ops)) {
    } else {
      return ops.emitOpError("cannot be contained in a BroadcastPacket op");
    }
  }

  return success();
}

LogicalResult xilinx::AIEX::IpuDmaMemcpyNdOp::verify() {
  ::mlir::MemRefType buffer = getMemref().getType();
  if (!buffer.getElementType().isInteger(32))
    return emitOpError("must be used with memref type i32.");
  uint32_t strides[3]{};
  strides[2] = static_cast<uint32_t>(
      getStride3().getDefiningOp<arith::ConstantIntOp>().value());
  strides[1] = static_cast<uint32_t>(
      getStride2().getDefiningOp<arith::ConstantIntOp>().value());
  strides[0] = static_cast<uint32_t>(
      getStride1().getDefiningOp<arith::ConstantIntOp>().value());
  if (static_cast<uint32_t>(
          getLength3().getDefiningOp<arith::ConstantIntOp>().value()) > 64)
    return emitOpError("Length 3 exceeds the [1:64] range.");
  if (strides[1] &&
      static_cast<uint32_t>(
          getLength1().getDefiningOp<arith::ConstantIntOp>().value()) > 0x3FF)
    return emitOpError("Length 1 exceeds the [0:1023] range.");
  if (strides[0] &&
      static_cast<uint32_t>(
          getLength0().getDefiningOp<arith::ConstantIntOp>().value()) > 0x3FF)
    return emitOpError("Length 0 exceeds the [0:1023] range.");
  if (strides[2] > 0x100000)
    return emitOpError("Stride 3 exceeds the [1:1M] range.");
  if (strides[1] > 0x100000)
    return emitOpError("Stride 2 exceeds the [1:1M] range.");
  if (strides[0] > 0x100000)
    return emitOpError("Stride 1 exceeds the [1:1M] range.");
  return success();
}

LogicalResult xilinx::AIEX::IpuShimTilePushQueueOp::verify() {
  const auto &target_model = getTargetModel(*this);
  auto num_bds = target_model.getNumBDs(0, 0); // assume shim
  if (getBdId() > num_bds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getRepeatCount() > 255)
    return emitOpError("Repeat count exceeds the [0:255] range.");
  return success();
}

LogicalResult xilinx::AIEX::IpuWriteBdExShimTileOp::verify() {
  const auto &target_model = getTargetModel(*this);
  auto num_bds = target_model.getNumBDs(0, 0); // assume shim
  if (getBdId() > num_bds)
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