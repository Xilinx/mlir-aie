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

  // Track combo/edge slot usage within this trace
  llvm::DenseSet<uint32_t> comboSlots;
  llvm::DenseSet<uint32_t> edgeSlots;

  for (auto &op : getBody().getOps()) {
    if (auto comboOp = dyn_cast<TraceComboEventOp>(op)) {
      uint32_t slot = comboOp.getSlot();
      if (!comboSlots.insert(slot).second) {
        return comboOp.emitOpError("combo event slot ")
               << slot << " already in use in this trace";
      }
    } else if (auto edgeOp = dyn_cast<TraceEdgeEventOp>(op)) {
      uint32_t slot = edgeOp.getSlot();
      if (!edgeSlots.insert(slot).second) {
        return edgeOp.emitOpError("edge detection slot ")
               << slot << " already in use in this trace";
      }
    }
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
// TracePortOp
//===----------------------------------------------------------------------===//

LogicalResult TracePortOp::verify() {
  // Get parent trace and tile
  auto trace = (*this)->getParentOfType<TraceOp>();
  if (!trace) {
    return emitOpError("must be nested in aie.trace");
  }

  auto tileOp = dyn_cast<TileOp>(trace.getTile().getDefiningOp());
  if (!tileOp) {
    return emitOpError("trace tile must be a TileOp");
  }

  // Get target model
  auto device = trace->getParentOfType<DeviceOp>();
  const auto &targetModel = device.getTargetModel();

  // Convert DMAChannelDir to master flag: S2MM=master, MM2S=slave
  bool isMaster = (getDirection() == DMAChannelDir::S2MM);

  // Verify port is valid for this tile
  if (!targetModel.isValidStreamSwitchPort(tileOp.getCol(), tileOp.getRow(),
                                           getPort(), getChannel(),
                                           isMaster)) {
    return emitOpError("invalid stream switch port configuration for tile (")
           << tileOp.getCol() << ", " << tileOp.getRow() << ")";
  }

  // Check for duplicate slots within same trace
  for (auto &op : trace.getBody().getOps()) {
    if (auto otherPort = dyn_cast<TracePortOp>(op)) {
      if (otherPort != *this && otherPort.getSlot() == getSlot()) {
        return emitOpError("duplicate port slot ")
               << getSlot() << " in trace " << trace.getSymName();
      }
    }
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
// TraceComboEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceComboEventOp::verify() {
  uint32_t slot = getSlot();

  // Check slot is valid (0, 1, or 2)
  if (slot > 2) {
    return emitOpError("combo event slot must be 0, 1, or 2, got ") << slot;
  }

  // Validate event selection based on slot
  std::string eventAName = getEventA().getName().str();
  std::string eventBName = getEventB().getName().str();

  if (slot == 0) {
    // Combo 0: should not use eventC/D or combo results
    if (eventAName.find("COMBO_EVENT") != std::string::npos ||
        eventBName.find("COMBO_EVENT") != std::string::npos) {
      return emitOpError("combo slot 0 should use regular events, not "
                         "COMBO_EVENT_* (uses eventA/B)");
    }
  } else if (slot == 1) {
    // Combo 1: should not use combo results
    if (eventAName.find("COMBO_EVENT") != std::string::npos ||
        eventBName.find("COMBO_EVENT") != std::string::npos) {
      return emitOpError("combo slot 1 should use regular events, not "
                         "COMBO_EVENT_* (uses eventC/D)");
    }
  } else if (slot == 2) {
    // Combo 2 is hierarchical - must use COMBO_EVENT_0 and COMBO_EVENT_1
    if (eventAName != "COMBO_EVENT_0") {
      return emitOpError("combo slot 2 first event must be COMBO_EVENT_0 "
                         "(hierarchical), got ")
             << eventAName;
    }
    if (eventBName != "COMBO_EVENT_1") {
      return emitOpError("combo slot 2 second event must be COMBO_EVENT_1 "
                         "(hierarchical), got ")
             << eventBName;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TraceEdgeEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceEdgeEventOp::verify() {
  uint32_t slot = getSlot();

  // Check slot is valid (0 or 1)
  if (slot > 1) {
    return emitOpError("edge detection slot must be 0 or 1, got ") << slot;
  }

  // Edge events should not be other edge/combo events
  std::string eventName = getEvent().getName().str();
  if (eventName.find("EDGE_DETECTION_EVENT") != std::string::npos) {
    return emitOpError("edge detection source should be a regular event, not "
                       "another EDGE_DETECTION_EVENT");
  }
  if (eventName.find("COMBO_EVENT") != std::string::npos) {
    return emitOpError("edge detection source should be a regular event, not "
                       "a COMBO_EVENT (combo events can be used but may have "
                       "unexpected behavior)");
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
