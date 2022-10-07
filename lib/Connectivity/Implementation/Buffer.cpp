//===- Buffer.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Buffer.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace xilinx::phy;
using namespace xilinx::phy::connectivity;

mlir::Operation *BufferImplementation::createOperation() {
  assert(queue && "a buffer must be associated with a queue");

  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  return builder.create<physical::BufferOp>(
      builder.getUnknownLoc(),
      queue.getQueue().getType().dyn_cast<spatial::QueueType>().getDatatype());
}

void BufferImplementation::translateUserOperation(mlir::Value value,
                                                  mlir::Operation *user) {

  if (auto emplace = dyn_cast<spatial::EmplaceOp>(user)) {
    emplace.getResult().replaceAllUsesWith(value);
  } else if (auto front = dyn_cast<spatial::FrontOp>(user)) {
    front.getResult().replaceAllUsesWith(value);
  }
}

void BufferImplementation::addSpatialOperation(mlir::Operation *spatial) {
  if (auto queue_op = dyn_cast<spatial::QueueOp>(spatial)) {
    assert(!queue && "a buffer can only hold a queue");
    queue = queue_op;
  } else {
    assert(false && "a buffer can only implement a queue");
  }
}

void BufferImplementation::addSpatialFlow(mlir::Operation *src,
                                          mlir::Operation *dest) {
  if (auto queue_op = dyn_cast<spatial::QueueOp>(src))
    addSpatialOperation(src);
  if (auto queue_op = dyn_cast<spatial::QueueOp>(dest))
    addSpatialOperation(dest);
}
