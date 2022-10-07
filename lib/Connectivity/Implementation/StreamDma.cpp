//===- StreamDma.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/StreamDma.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace xilinx::phy;
using namespace xilinx::phy::connectivity;
using namespace xilinx::phy::physical;

Operation *StreamDmaImplementation::createOperation() {

  assert((istream.expired() || ostream.expired()) &&
         "a stream dma can only connect to one stream");
  assert(!(istream.expired() && ostream.expired()) &&
         "a stream dma must be connected to a stream");

  // Identify if it is an iutput stream dma or an output dma
  Operation *stream_op;
  int acquire, release, result_index;
  bool tagged = false;

  if (!ostream.expired()) {
    stream_op = ostream.lock()->getOperation();
    acquire = 1, release = 0, result_index = 0 /* ostream */;
    // ostream may be tagged
    tagged = dyn_cast<StreamOp>(stream_op).getTags().has_value();

  } else {
    stream_op = istream.lock()->getOperation();
    acquire = 0, release = 1, result_index = 1 /* istream */;
  }
  Value endpoint = stream_op->getResult(result_index);

  // physical.stream_dma(endpoint) {
  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  auto stream_dma =
      builder.create<StreamDmaOp>(builder.getUnknownLoc(), endpoint);

  auto *connection_block = &stream_dma.getConnections().emplaceBlock();
  auto dma_builder = OpBuilder::atBlockEnd(connection_block);
  StreamDmaConnectOp first_connect, previous_connect;

  for (auto flow_buffer : buffers) {
    auto flow = flow_buffer.first;
    auto tag = tagged ? dma_builder.getI64IntegerAttr(context.getFlowTag(flow))
                      : IntegerAttr();
    auto lock = locks[flow].lock()->getOperation()->getResult(0);
    auto buffer = buffers[flow].lock()->getOperation()->getResult(0);
    auto buffer_size = buffer.getType().dyn_cast<MemRefType>().getShape()[0];

    // connection = physical.stream_dma_connect<tag>(%lock[acquire -> release],
    //                  %buffer[0 : buffer_size] : memref<1024xi32>)
    auto connection = dma_builder.create<StreamDmaConnectOp>(
        builder.getUnknownLoc(),
        StreamDmaConnectType::get(builder.getContext()), tag, lock, acquire,
        release, buffer, 0, buffer_size, Value());

    // previous_connect = ..., connection)
    if (previous_connect)
      previous_connect.getNextMutable().assign(connection);

    if (!first_connect)
      first_connect = connection;
    previous_connect = connection;
  }

  // last_connect = ..., first_connect)
  previous_connect.getNextMutable().assign(first_connect);

  // }
  dma_builder.create<EndOp>(builder.getUnknownLoc());

  return stream_dma;
}

void StreamDmaImplementation::addPredecessor(std::weak_ptr<Implementation> pred,
                                             Operation *src, Operation *dest) {
  if (pred.lock()->phy.key == "stream") {
    assert((istream.expired() || istream.lock() == pred.lock()) &&
           "a stream dma can only connect to one istream");
    istream = pred;

  } else
    addStorage(pred, src, dest);
}

void StreamDmaImplementation::addSuccessor(std::weak_ptr<Implementation> succ,
                                           Operation *src, Operation *dest) {
  if (succ.lock()->phy.key == "stream") {
    assert((ostream.expired() || ostream.lock() == succ.lock()) &&
           "a stream dma can only connect to one ostream");
    ostream = succ;

  } else
    addStorage(succ, src, dest);
}

void StreamDmaImplementation::addStorage(std::weak_ptr<Implementation> storage,
                                         Operation *src, Operation *dest) {
  if (storage.lock()->phy.key == "buffer") {
    buffers[context.getFlowSignature(std::make_pair(src, dest))] = storage;

  } else if (storage.lock()->phy.key == "lock") {
    locks[context.getFlowSignature(std::make_pair(src, dest))] = storage;

  } else {
    assert(false && "a stream dma can only connect to buffers or locks");
  }
}
