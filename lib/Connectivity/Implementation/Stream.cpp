//===- Stream.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Stream.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace xilinx::phy;
using namespace xilinx::phy::connectivity;
using namespace xilinx::phy::physical;

Operation *StreamImplementation::createOperation() {
  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  auto *mlir_context = builder.getContext();

  auto i32_type = builder.getI32Type();
  auto o_stream_type = OStreamType::get(mlir_context, i32_type);
  auto i_stream_type = IStreamType::get(mlir_context, i32_type);

  auto tags_attr = ArrayAttr();

  if (streamHasTags()) {
    llvm::SetVector<int64_t> tags;
    for (auto flow : flows)
      tags.insert(context.getFlowTag(flow));

    tags_attr = builder.getI64ArrayAttr(tags.getArrayRef());
  }

  return builder.create<StreamOp>(builder.getUnknownLoc(), o_stream_type,
                                  i_stream_type, tags_attr);
}

bool StreamImplementation::streamHasTags() {
  return has_broadcast_neighbor || (src_queues.size() > 1) ||
         (dest_queues.size() > 1);
}

void StreamImplementation::addSpatialFlow(Operation *src, Operation *dest) {
  flows.insert(std::make_pair(src, dest));

  if (auto queue_op = dyn_cast<spatial::QueueOp>(src))
    src_queues.insert(queue_op);
  if (auto queue_op = dyn_cast<spatial::QueueOp>(dest))
    dest_queues.push_back(queue_op);
}

void StreamImplementation::addNeighbor(std::weak_ptr<Implementation> neighbor) {
  if (neighbor.lock()->phy.metadata.count("impl") &&
      neighbor.lock()->phy.metadata["impl"] == "broadcast_packet")
    // TODO: check if the stream_hub has multiple flows connected instead
    has_broadcast_neighbor = true;
}

void StreamImplementation::addPredecessor(std::weak_ptr<Implementation> pred,
                                          Operation *src, Operation *dest) {
  addNeighbor(pred);
}

void StreamImplementation::addSuccessor(std::weak_ptr<Implementation> succ,
                                        Operation *src, Operation *dest) {
  addNeighbor(succ);
}
