//===- Stream.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_H

#include "phy/Connectivity/Implementation.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <list>
#include <map>
#include <set>
#include <utility>

namespace xilinx {
namespace phy {
namespace connectivity {

class StreamImplementation : public Implementation {

  // Overrides
protected:
  mlir::Operation *createOperation() override;

public:
  StreamImplementation(PhysicalResource phy, ImplementationContext &context)
      : Implementation(phy, context), has_broadcast_neighbor(false) {}
  ~StreamImplementation() override {}

  void addSpatialFlow(mlir::Operation *src, mlir::Operation *dest) override;
  void addPredecessor(std::weak_ptr<Implementation> pred, mlir::Operation *src,
                      mlir::Operation *dest) override;
  void addSuccessor(std::weak_ptr<Implementation> succ, mlir::Operation *src,
                    mlir::Operation *dest) override;

protected:
  // multiple connections from a source can be broadcast
  std::set<spatial::QueueOp> src_queues;
  // multiple connections to a destination shall be distinct with tag
  std::list<spatial::QueueOp> dest_queues;

  // keep track of all flows
  std::set<std::pair<mlir::Operation *, mlir::Operation *>> flows;
  // if a strem has broadcast as its neighbor, it must has tag
  bool has_broadcast_neighbor;

public:
  void addNeighbor(std::weak_ptr<Implementation> neighbor);
  bool streamHasTags();
};

} // namespace connectivity
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_H
