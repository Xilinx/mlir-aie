//===- AIETraceOps.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Implementation of AIE trace operations
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// TraceOp
//===----------------------------------------------------------------------===//

void TraceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // No results for this operation
}

LogicalResult TraceOp::verify() {
  // Count trace events
  int eventCount = 0;
  for (auto &op : getBody().getOps()) {
    if (isa<TraceEventOp>(op)) {
      eventCount++;
    }
  }

  // Check max 8 events
  if (eventCount > 8) {
    return emitOpError("trace unit supports maximum 8 events, got ")
           << eventCount;
  }

  // Verify tile operand is a TileOp
  if (!getTile().getDefiningOp<TileOp>()) {
    return emitOpError("tile operand must be a TileOp");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TraceEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceEventOp::verify() {
  // Basic validation - event name should not be empty
  if (getEvent().getName().empty()) {
    return emitOpError("event name cannot be empty");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TracePacketOp
//===----------------------------------------------------------------------===//

LogicalResult TracePacketOp::verify() {
  // Packet ID range is already enforced by Confined constraint in TableGen
  // Just verify it's within valid range
  int32_t id = getId();
  if (id < 1 || id > 31) {
    return emitOpError("packet ID must be in range [1, 31], got ") << id;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TraceStartEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceStartEventOp::verify() {
  // Must have either broadcast or event, but not both
  bool hasBroadcast = getBroadcast().has_value();
  bool hasEvent = getEvent().has_value();

  if (!hasBroadcast && !hasEvent) {
    return emitOpError("must specify either broadcast or event");
  }

  if (hasBroadcast && hasEvent) {
    return emitOpError("cannot specify both broadcast and event");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TraceStopEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceStopEventOp::verify() {
  // Must have either broadcast or event, but not both
  bool hasBroadcast = getBroadcast().has_value();
  bool hasEvent = getEvent().has_value();

  if (!hasBroadcast && !hasEvent) {
    return emitOpError("must specify either broadcast or event");
  }

  if (hasBroadcast && hasEvent) {
    return emitOpError("cannot specify both broadcast and event");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TraceStartConfigOp
//===----------------------------------------------------------------------===//

LogicalResult TraceStartConfigOp::verify() {
  // Verify that the referenced symbol exists
  // This will be checked more thoroughly during lowering
  auto symbolName = getTraceConfig();
  if (symbolName.empty()) {
    return emitOpError("trace config symbol name cannot be empty");
  }

  return success();
}
