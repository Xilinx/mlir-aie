//===- Implementation.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation.h"

#include "phy/Connectivity/Implementation/Buffer.h"
#include "phy/Connectivity/Implementation/Core.h"
#include "phy/Connectivity/Implementation/Lock.h"
#include "phy/Connectivity/Implementation/Stream.h"
#include "phy/Connectivity/Implementation/StreamDma.h"
#include "phy/Connectivity/Implementation/StreamHub.h"

#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace xilinx::phy::connectivity;

std::shared_ptr<Implementation>
xilinx::phy::connectivity::implementationFactory(
    PhysicalResource phy, ImplementationContext &context) {
  if (phy.key == "buffer") {
    return std::make_shared<BufferImplementation>(phy, context);
  }
  if (phy.key == "core") {
    return std::make_shared<CoreImplementation>(phy, context);
  }
  if (phy.key == "lock") {
    return std::make_shared<LockImplementation>(phy, context);
  }
  if (phy.key == "stream") {
    return std::make_shared<StreamImplementation>(phy, context);
  }
  if (phy.key == "stream_dma") {
    return std::make_shared<StreamDmaImplementation>(phy, context);
  }
  if (phy.key == "stream_hub") {
    return std::make_shared<StreamHubImplementation>(phy, context);
  }
  return nullptr;
}

void Implementation::attachMetadata() {
  auto builder = mlir::OpBuilder::atBlockEnd(context.module.getBody());

  for (auto metadata : phy.metadata) {
    std::string attr_name = context.device + "." + metadata.first;
    implemented_op->setAttr(attr_name, builder.getStringAttr(metadata.second));
  }
}

mlir::Operation *Implementation::getOperation() {
  if (!implemented_op) {
    implemented_op = this->createOperation();
    attachMetadata();
  }
  return implemented_op;
}

void ImplementationContext::place(Operation *spatial, ResourceList resources) {
  for (auto phy : resources.phys) {
    auto identifier = phy.toString();
    if (!impls.count(identifier))
      impls[identifier] = implementationFactory(phy, *this);

    if (impls[identifier]) {
      impls[identifier]->addSpatialOperation(spatial);
      placements[spatial].push_back(impls[identifier]);
    } else {
      spatial->emitWarning() << phy.key << " cannot be implemented.";
    }
  }
}

static void populateSibling(std::list<std::weak_ptr<Implementation>> &impls) {
  for (auto impl_1 : impls) {
    for (auto impl_2 : impls) {
      impl_1.lock()->addSibling(impl_2);
    }
  }
}

static void populatePair(std::list<std::weak_ptr<Implementation>> &pred_impls,
                         std::list<std::weak_ptr<Implementation>> &succ_impls,
                         Operation *src, Operation *dest) {
  for (auto pred_impl : pred_impls) {
    for (auto succ_impl : succ_impls) {
      succ_impl.lock()->addPredecessor(pred_impl, src, dest);
      pred_impl.lock()->addSuccessor(succ_impl, src, dest);
    }
  }
}

void ImplementationContext::route(Operation *src, Operation *dest,
                                  std::list<ResourceList> resources) {

  std::list<std::list<std::weak_ptr<Implementation>>> route_impls;

  // Implement the route
  route_impls.push_back(placements[src]);
  for (auto resource_list : resources) {
    route_impls.emplace_back();

    for (auto phy : resource_list.phys) {
      auto identifier = phy.toString();
      if (!impls.count(identifier))
        impls[identifier] = implementationFactory(phy, *this);

      if (impls[identifier]) {
        impls[identifier]->addSpatialFlow(src, dest);
        route_impls.back().push_back(impls[identifier]);
      } else {
        src->emitWarning() << phy.key << " cannot be implemented.";
        dest->emitRemark() << phy.key << " was used to connect to here.";
      }
    }
  }
  route_impls.push_back(placements[dest]);

  // Populate the neighbor information
  for (auto curr = route_impls.begin(), prev = route_impls.end();
       curr != route_impls.end(); prev = curr, curr++) {
    populateSibling(*curr);
    if (prev != route_impls.end())
      populatePair(*prev, *curr, src, dest);
  }
}

void ImplementationContext::implementAll() {
  // Making sure each is implemented as an operations
  for (auto impl : impls) {
    if (impl.second) {
      impl.second->getOperation();
    }
  }
}

std::pair<mlir::Operation *, mlir::Operation *>
ImplementationContext::getFlowSignature(
    std::pair<mlir::Operation *, mlir::Operation *> flow) {

  // Return if the flow is already signature
  if (!flow.first)
    return flow;
  if (!flow.second)
    return flow;

  // Broadcast detection
  if (llvm::isa<spatial::QueueOp>(flow.first) &&
      llvm::isa<spatial::NodeOp>(flow.second)) {
    // Flows to different nodes from the same queue is the same flow.
    flow.second = nullptr;
  }

  return flow;
}

long ImplementationContext::getFlowTag(
    std::pair<mlir::Operation *, mlir::Operation *> flow) {

  flow = getFlowSignature(flow);

  if (!flow_tags.count(flow)) {
    // Allocate a new tag if not available yet.
    flow_tags[flow] = getUniqueTag();
  }
  return flow_tags[flow];
}

StringAttr ImplementationContext::getUniqueSymbol(llvm::StringRef base,
                                                  Operation *op) {

  StringAttr symbol;
  do {
    unique_suffix[base]++;
    symbol = StringAttr::get(op->getContext(),
                             base.str() + std::to_string(unique_suffix[base]));
  } while (SymbolTable::lookupNearestSymbolFrom(op, symbol));

  return symbol;
}

long ImplementationContext::getUniqueTag() { return next_tag++; }
