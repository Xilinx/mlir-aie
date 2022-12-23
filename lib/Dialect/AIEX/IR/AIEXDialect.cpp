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
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

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
