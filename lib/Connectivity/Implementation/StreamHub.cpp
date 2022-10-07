//===- StreamHub.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/StreamHub.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace xilinx::phy;
using namespace xilinx::phy::connectivity;

Operation *StreamHubImplementation::createOperation() {
  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  auto *mlir_context = builder.getContext();

  auto i32_type = builder.getI32Type();
  auto stream_hub_type = physical::StreamHubType::get(mlir_context, i32_type);

  llvm::SetVector<mlir::Value> endpoints;
  for (auto pred : preds)
    endpoints.insert(pred.lock()->getOperation()->getResult(1 /* istream */));
  for (auto succ : succs)
    endpoints.insert(succ.lock()->getOperation()->getResult(0 /* ostream */));

  return builder.create<physical::StreamHubOp>(
      builder.getUnknownLoc(), stream_hub_type, endpoints.getArrayRef());
}

void StreamHubImplementation::addPredecessor(std::weak_ptr<Implementation> pred,
                                             Operation *src, Operation *dest) {
  preds.push_back(pred);
}

void StreamHubImplementation::addSuccessor(std::weak_ptr<Implementation> succ,
                                           Operation *src, Operation *dest) {
  succs.push_back(succ);
}
