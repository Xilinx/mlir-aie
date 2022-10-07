//===- LayoutOps.cpp - Implement the Layout operations --------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::xilinx::phy::layout;
using namespace ::xilinx::phy::spatial;

LogicalResult RouteOp::verify() {
  Type src_type = getSrc().getType();
  Type dest_type = getDest().getType();

  if (src_type.isa<NodeType>() && dest_type.isa<NodeType>())
    return emitOpError("a node cannot be connected to a node using a flow");

  if (src_type.isa<QueueType>() && dest_type.isa<QueueType>())
    return emitOpError("a queue cannot be connected to a queue using a flow");

  Type datatype;
  if (auto src_queue = src_type.dyn_cast<QueueType>())
    datatype = src_queue.getDatatype();
  else if (auto dest_queue = dest_type.dyn_cast<QueueType>())
    datatype = dest_queue.getDatatype();
  else
    return emitOpError("one endpoint of the flow must be a queue");

  return success();
}
