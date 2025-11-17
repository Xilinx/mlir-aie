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
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// TraceEventAttr Helper Methods
//===----------------------------------------------------------------------===//

std::string TraceEventAttr::getEventName() const {
  if (auto strAttr = llvm::dyn_cast<StringAttr>(getValue())) {
    // If the string is fully qualified (contains '::'), extract just the name
    StringRef strValue = strAttr.getValue();
    size_t pos = strValue.find("::");
    if (pos != StringRef::npos) {
      return strValue.substr(pos + 2).str();
    }
    return strAttr.getValue().str();
  }

  // // Check for typed enum attributes and use stringify functions
  // Attribute value = getValue();
  // if (auto coreEvt = llvm::dyn_cast<CoreEventAIE2Attr>(value)) {
  //   return stringifyCoreEventAIE2(coreEvt.getValue()).str();
  // }
  // if (auto memEvt = llvm::dyn_cast<MemEventAIE2Attr>(value)) {
  //   return stringifyMemEventAIE2(memEvt.getValue()).str();
  // }
  // if (auto shimEvt = llvm::dyn_cast<ShimTileEventAIE2Attr>(value)) {
  //   return stringifyShimTileEventAIE2(shimEvt.getValue()).str();
  // }
  // if (auto memTileEvt = llvm::dyn_cast<MemTileEventAIE2Attr>(value)) {
  //   return stringifyMemTileEventAIE2(memTileEvt.getValue()).str();
  // }

  // Fallback: shouldn't reach here for well-formed IR
  return "42";
}

bool TraceEventAttr::isStringAttr() const {
  return llvm::isa<StringAttr>(getValue());
}

std::optional<int64_t> TraceEventAttr::getEnumValue() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getValue())) {
    return intAttr.getInt();
  }
  // String values don't have enum values - they're event names
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Tile Type Validation Helper
//===----------------------------------------------------------------------===//

static bool isValidEventForTile(TileOp tile, Attribute eventAttr) {
  int row = tile.getRow();

  // Determine tile type
  bool isShimTile = (row == 0);
  bool isMemTile = (row == 1);  // Mem tiles are at row 1
  bool isCoreTile = (row >= 2); // Core tiles are at row 2 and above

  // If eventAttr is a TraceEventAttr, extract the inner attribute
  Attribute innerAttr = eventAttr;
  if (auto traceEvent = llvm::dyn_cast<TraceEventAttr>(eventAttr)) {
    innerAttr = traceEvent.getValue();
  }

  // If it's a string, we'll validate later during lowering
  // (requires event database lookup)
  if (llvm::isa<StringAttr>(innerAttr)) {
    return true;
  }

  // For typed enums, check the attribute type directly
  if (isCoreTile) {
    // Core tiles can use CoreEventAIE, CoreEventAIE2, CoreEventAIE2P
    if (llvm::isa<CoreEventAIEAttr>(innerAttr) ||
        llvm::isa<CoreEventAIE2Attr>(innerAttr) ||
        llvm::isa<CoreEventAIE2PAttr>(innerAttr)) {
      return true;
    }
    // Also allow MemEvent for memory module of core tile
    if (llvm::isa<MemEventAIEAttr>(innerAttr) ||
        llvm::isa<MemEventAIE2Attr>(innerAttr) ||
        llvm::isa<MemEventAIE2PAttr>(innerAttr)) {
      return true;
    }
  } else if (isMemTile) {
    // Mem tiles use MemTileEventAIE2, MemTileEventAIE2P
    if (llvm::isa<MemTileEventAIE2Attr>(innerAttr) ||
        llvm::isa<MemTileEventAIE2PAttr>(innerAttr)) {
      return true;
    }
  } else if (isShimTile) {
    // Shim tiles use ShimTileEventAIE, ShimTileEventAIE2, ShimTileEventAIE2P
    if (llvm::isa<ShimTileEventAIEAttr>(innerAttr) ||
        llvm::isa<ShimTileEventAIE2Attr>(innerAttr) ||
        llvm::isa<ShimTileEventAIE2PAttr>(innerAttr)) {
      return true;
    }
  }

  return false;
}

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
// Helper function for parsing event values (string or typed enum)
// Made available in xilinx::AIE namespace for reuse in AIEDialect.cpp
//===----------------------------------------------------------------------===//

ParseResult xilinx::AIE::parseTraceEvent(AsmParser &parser, Attribute &result) {
  MLIRContext *ctx = parser.getContext();

  // Try to parse as string first
  std::string strValue;
  if (succeeded(parser.parseOptionalString(&strValue))) {
    result = StringAttr::get(ctx, strValue);
    return success();
  }

  // Try to parse as enum (EnumType::VALUE)
  StringRef enumTypeName;
  llvm::SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseKeyword(&enumTypeName))) {
    return parser.emitError(loc, "expected string or enum event");
  }

  if (failed(parser.parseColon()) || failed(parser.parseColon())) {
    return parser.emitError(loc, "expected '::' after enum type name");
  }

  StringRef caseName;
  if (failed(parser.parseKeyword(&caseName))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected enum case name");
  }
  // Validate the enum and convert to string to avoid ambiguity between
  // AIE2 and AIE2P enums which have identical integer values

  // Define a helper struct for enum validation
  struct EnumValidator {
    StringRef name;
    std::function<std::optional<uint32_t>(StringRef)> symbolizer;
  };

  // Table of supported enum types and their symbolizer functions
  static const EnumValidator validators[] = {
      // AIE2 event enums
      {"CoreEventAIE2",
       [](StringRef s) { return symbolizeCoreEventAIE2(s).has_value(); }},
      {"MemEventAIE2",
       [](StringRef s) {
         auto result = symbolizeMemEventAIE2(s);
         return result.has_value();
       }},
      {"MemTileEventAIE2",
       [](StringRef s) {
         auto result = symbolizeMemTileEventAIE2(s);
         return result.has_value();
       }},
      {"ShimTileEventAIE2",
       [](StringRef s) {
         auto result = symbolizeShimTileEventAIE2(s);
         return result.has_value();
       }},
      // AIE event enums
      {"CoreEventAIE",
       [](StringRef s) {
         auto result = symbolizeCoreEventAIE(s);
         return result.has_value();
       }},
      {"MemEventAIE",
       [](StringRef s) {
         auto result = symbolizeMemEventAIE(s);
         return result.has_value();
       }},
      {"ShimTileEventAIE",
       [](StringRef s) {
         auto result = symbolizeShimTileEventAIE(s);
         return result.has_value();
       }},
      // AIE2P event enums
      {"CoreEventAIE2P",
       [](StringRef s) {
         auto result = symbolizeCoreEventAIE2P(s);
         return result.has_value();
       }},
      {"MemEventAIE2P",
       [](StringRef s) {
         auto result = symbolizeMemEventAIE2P(s);
         return result.has_value();
       }},
      {"MemTileEventAIE2P",
       [](StringRef s) {
         auto result = symbolizeMemTileEventAIE2P(s);
         return result.has_value();
       }},
      {"ShimTileEventAIE2P",
       [](StringRef s) {
         auto result = symbolizeShimTileEventAIE2P(s);
         return result.has_value();
       }},
  };

  // Look up and validate the enum type
  for (const auto &validator : validators) {
    if (enumTypeName != validator.name)
      continue;
    if (!validator.symbolizer(caseName))
      return parser.emitError(loc, "unknown ")
             << enumTypeName << " value: " << caseName;
    result = StringAttr::get(ctx, enumTypeName + "::" + caseName);
    return success();
  }

  return parser.emitError(loc, "unknown event enum type: ") << enumTypeName;
}

void xilinx::AIE::printTraceEventEnum(AsmPrinter &printer, Attribute attr) {
  if (auto traceAttr = llvm::dyn_cast<TraceEventAttr>(attr)) {
    printTraceEventEnum(printer, traceAttr.getValue());
    return;
  }
  if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
    // If string contains "::" (enum format), print without quotes
    if (strAttr.getValue().contains("::")) {
      printer << strAttr.getValue();
      return;
    }
    printer << "\"" << strAttr.getValue() << "\"";
  }
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    printer << intAttr.getInt();
    return;
  }
  // // AIE2 event enums
  // else if (auto coreEvt = llvm::dyn_cast<CoreEventAIE2Attr>(attr)) {
  //   printer << "CoreEventAIE2::" <<
  //   stringifyCoreEventAIE2(coreEvt.getValue());
  // } else if (auto memEvt = llvm::dyn_cast<MemEventAIE2Attr>(attr)) {
  //   printer << "MemEventAIE2::" << stringifyMemEventAIE2(memEvt.getValue());
  // } else if (auto memTileEvt = llvm::dyn_cast<MemTileEventAIE2Attr>(attr)) {
  //   printer << "MemTileEventAIE2::"
  //           << stringifyMemTileEventAIE2(memTileEvt.getValue());
  // } else if (auto shimTileEvt = llvm::dyn_cast<ShimTileEventAIE2Attr>(attr))
  // {
  //   printer << "ShimTileEventAIE2::"
  //           << stringifyShimTileEventAIE2(shimTileEvt.getValue());
  // }
  // // AIE event enums
  // else if (auto coreEvt = llvm::dyn_cast<CoreEventAIEAttr>(attr)) {
  //   printer << "CoreEventAIE::" << stringifyCoreEventAIE(coreEvt.getValue());
  // } else if (auto memEvt = llvm::dyn_cast<MemEventAIEAttr>(attr)) {
  //   printer << "MemEventAIE::" << stringifyMemEventAIE(memEvt.getValue());
  // } else if (auto shimTileEvt = llvm::dyn_cast<ShimTileEventAIEAttr>(attr)) {
  //   printer << "ShimTileEventAIE::"
  //           << stringifyShimTileEventAIE(shimTileEvt.getValue());
  // }
  // // AIE2P event enums
  // else if (auto coreEvt = llvm::dyn_cast<CoreEventAIE2PAttr>(attr)) {
  //   printer << "CoreEventAIE2P::"
  //           << stringifyCoreEventAIE2P(coreEvt.getValue());
  // } else if (auto memEvt = llvm::dyn_cast<MemEventAIE2PAttr>(attr)) {
  //   printer << "MemEventAIE2P::" <<
  //   stringifyMemEventAIE2P(memEvt.getValue());
  // } else if (auto memTileEvt = llvm::dyn_cast<MemTileEventAIE2PAttr>(attr)) {
  //   printer << "MemTileEventAIE2P::"
  //           << stringifyMemTileEventAIE2P(memTileEvt.getValue());
  // } else if (auto shimTileEvt = llvm::dyn_cast<ShimTileEventAIE2PAttr>(attr))
  // {
  //   printer << "ShimTileEventAIE2P::"
  //           << stringifyShimTileEventAIE2P(shimTileEvt.getValue());
  // }
}

//===----------------------------------------------------------------------===//
// TraceEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceEventOp::verify() {
  // Get event name/value
  auto eventAttr = getEvent();

  // Basic validation - event should not be empty
  std::string eventName = eventAttr.getEventName();
  if (eventName.empty()) {
    return emitOpError("event name cannot be empty");
  }

  // Validate event type matches tile type
  auto trace = (*this)->getParentOfType<TraceOp>();
  if (!trace) {
    return emitOpError("must be nested in aie.trace");
  }

  auto tileOp = dyn_cast<TileOp>(trace.getTile().getDefiningOp());
  if (!tileOp) {
    return emitOpError("trace tile must be a TileOp");
  }

  if (!isValidEventForTile(tileOp, eventAttr.getValue())) {
    int row = tileOp.getRow();
    std::string tileTypeStr;
    if (row == 0)
      tileTypeStr = "shim tile";
    else if (row == 1)
      tileTypeStr = "mem tile";
    else
      tileTypeStr = "core tile";

    return emitOpError("event '")
           << eventName << "' is not valid for " << tileTypeStr << " at ("
           << tileOp.getCol() << ", " << row << ")";
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
                                           getPort(), getChannel(), isMaster)) {
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

// void TraceEventOp::print(OpAsmPrinter &p) {
//   p << "<";
//   printTraceEventEnum(p, getEvent().getValue());
//   p << ">";
//   if (auto label = getLabel()) {
//     p << " label = " << label;
//   }
//   p.printOptionalAttrDict((*this)->getAttrs(),
//                           /*elidedAttrs=*/{"event", "label"});
// }

// ParseResult TraceEventOp::parse(OpAsmParser &parser, OperationState &result)
// {
//   Attribute innerValue;

//   if (parser.parseLess())
//     return failure();

//   if (failed(parseTraceEvent(parser, innerValue)))
//     return failure();

//   // Wrap in TraceEventAttr
//   auto traceEvent = TraceEventAttr::get(parser.getContext(), innerValue);
//   result.attributes.set("event", traceEvent);

//   if (parser.parseGreater())
//     return failure();

//   // Parse optional label
//   if (succeeded(parser.parseOptionalKeyword("label"))) {
//     StringAttr label;
//     if (parser.parseEqual() ||
//         parser.parseAttribute(label, "label", result.attributes))
//       return failure();
//   }

//   if (parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   return success();
// }

//===----------------------------------------------------------------------===//
// TraceStartEventOp and TraceStopEventOp
//===----------------------------------------------------------------------===//

void TraceStartEventOp::print(OpAsmPrinter &p) {
  if (auto broadcast = getBroadcast()) {
    p << " broadcast = " << broadcast.value();
  }
  if (auto event = getEvent()) {
    p << " event = <";
    printTraceEventEnum(p, event->getValue());
    p << ">";
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"broadcast", "event"});
}

ParseResult TraceStartEventOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  // Parse optional broadcast
  if (succeeded(parser.parseOptionalKeyword("broadcast"))) {
    IntegerAttr broadcast;
    if (parser.parseEqual() ||
        parser.parseAttribute(broadcast, parser.getBuilder().getI32Type(),
                              "broadcast", result.attributes))
      return failure();
  }

  // Parse optional event
  if (succeeded(parser.parseOptionalKeyword("event"))) {
    if (parser.parseEqual() || parser.parseLess())
      return failure();

    Attribute innerValue;
    if (failed(parseTraceEvent(parser, innerValue)))
      return failure();

    auto traceEvent = TraceEventAttr::get(parser.getContext(), innerValue);
    result.attributes.set("event", traceEvent);

    if (parser.parseGreater())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void TraceStopEventOp::print(OpAsmPrinter &p) {
  if (auto broadcast = getBroadcast()) {
    p << " broadcast = " << broadcast.value();
  }
  if (auto event = getEvent()) {
    p << " event = <";
    printTraceEventEnum(p, event->getValue());
    p << ">";
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"broadcast", "event"});
}

ParseResult TraceStopEventOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse optional broadcast
  if (succeeded(parser.parseOptionalKeyword("broadcast"))) {
    IntegerAttr broadcast;
    if (parser.parseEqual() ||
        parser.parseAttribute(broadcast, parser.getBuilder().getI32Type(),
                              "broadcast", result.attributes))
      return failure();
  }

  // Parse optional event
  if (succeeded(parser.parseOptionalKeyword("event"))) {
    if (parser.parseEqual() || parser.parseLess())
      return failure();

    Attribute innerValue;
    if (failed(parseTraceEvent(parser, innerValue)))
      return failure();

    auto traceEvent = TraceEventAttr::get(parser.getContext(), innerValue);
    result.attributes.set("event", traceEvent);

    if (parser.parseGreater())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

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

  // Get parent trace and tile for validation
  auto trace = (*this)->getParentOfType<TraceOp>();
  if (!trace) {
    return emitOpError("must be nested in aie.trace");
  }

  auto tileOp = dyn_cast<TileOp>(trace.getTile().getDefiningOp());
  if (!tileOp) {
    return emitOpError("trace tile must be a TileOp");
  }

  // Validate event types match tile
  if (!isValidEventForTile(tileOp, getEventA().getValue())) {
    return emitOpError("eventA is not valid for this tile type");
  }
  if (!isValidEventForTile(tileOp, getEventB().getValue())) {
    return emitOpError("eventB is not valid for this tile type");
  }

  // Validate event selection based on slot
  std::string eventAName = getEventA().getEventName();
  std::string eventBName = getEventB().getEventName();

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

// void TraceComboEventOp::print(OpAsmPrinter &p) {
//   auto printEvent = [&](TraceEventAttr evt) {
//     printTraceEventEnum(p, evt.getValue());
//   };

//   p << "<" << getSlot() << "> <";
//   printEvent(getEventA());
//   p << "> " << getLogic() << " <";
//   printEvent(getEventB());
//   p << ">";
//   p.printOptionalAttrDict(
//       (*this)->getAttrs(),
//       /*elidedAttrs=*/{"slot", "eventA", "logic", "eventB"});
// }

// ParseResult TraceComboEventOp::parse(OpAsmParser &parser,
//                                      OperationState &result) {
//   IntegerAttr slot;
//   Attribute eventAValue, eventBValue;
//   ComboLogicAttr logic;

//   if (parser.parseLess() ||
//       parser.parseAttribute(slot, parser.getBuilder().getI32Type(), "slot",
//                             result.attributes) ||
//       parser.parseGreater() || parser.parseLess())
//     return failure();

//   if (failed(parseTraceEvent(parser, eventAValue)))
//     return failure();

//   auto eventA = TraceEventAttr::get(parser.getContext(), eventAValue);
//   result.attributes.set("eventA", eventA);

//   if (parser.parseGreater())
//     return failure();

//   // Parse logic as keyword (AND, OR, etc.)
//   StringRef logicStr;
//   if (failed(parser.parseKeyword(&logicStr)))
//     return failure();

//   auto logicEnum = symbolizeComboLogic(logicStr);
//   if (!logicEnum) {
//     return parser.emitError(parser.getCurrentLocation(),
//                             "unknown combo logic: ")
//            << logicStr;
//   }
//   logic = ComboLogicAttr::get(parser.getContext(), *logicEnum);
//   result.attributes.set("logic", logic);

//   if (parser.parseLess())
//     return failure();

//   if (failed(parseTraceEvent(parser, eventBValue)))
//     return failure();

//   auto eventB = TraceEventAttr::get(parser.getContext(), eventBValue);
//   result.attributes.set("eventB", eventB);

//   if (parser.parseGreater())
//     return failure();

//   if (parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   return success();
// }

//===----------------------------------------------------------------------===//
// TraceEdgeEventOp
//===----------------------------------------------------------------------===//

LogicalResult TraceEdgeEventOp::verify() {
  uint32_t slot = getSlot();

  // Check slot is valid (0 or 1)
  if (slot > 1) {
    return emitOpError("edge detection slot must be 0 or 1, got ") << slot;
  }

  // Get parent trace and tile for validation
  auto trace = (*this)->getParentOfType<TraceOp>();
  if (!trace) {
    return emitOpError("must be nested in aie.trace");
  }

  auto tileOp = dyn_cast<TileOp>(trace.getTile().getDefiningOp());
  if (!tileOp) {
    return emitOpError("trace tile must be a TileOp");
  }

  // Validate event type matches tile
  if (!isValidEventForTile(tileOp, getEvent().getValue())) {
    return emitOpError("event is not valid for this tile type");
  }

  // Edge events should not be other edge/combo events
  std::string eventName = getEvent().getEventName();
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

void TraceEdgeEventOp::print(OpAsmPrinter &p) {
  p << "<" << getSlot() << "> event = <";
  printTraceEventEnum(p, getEvent().getValue());
  p << "> trigger = " << getTrigger();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"slot", "event", "trigger"});
}

ParseResult TraceEdgeEventOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  IntegerAttr slot;
  Attribute eventValue;
  EdgeTriggerAttr trigger;

  if (parser.parseLess() ||
      parser.parseAttribute(slot, parser.getBuilder().getI32Type(), "slot",
                            result.attributes) ||
      parser.parseGreater() || parser.parseKeyword("event") ||
      parser.parseEqual() || parser.parseLess())
    return failure();

  // Parse event value (string or enum)
  if (failed(parseTraceEvent(parser, eventValue)))
    return failure();

  auto event = TraceEventAttr::get(parser.getContext(), eventValue);
  result.attributes.set("event", event);

  if (parser.parseGreater() || parser.parseKeyword("trigger") ||
      parser.parseEqual())
    return failure();

  // Parse trigger as keyword (RISING, FALLING, BOTH)
  StringRef triggerStr;
  if (failed(parser.parseKeyword(&triggerStr)))
    return failure();

  auto triggerEnum = symbolizeEdgeTrigger(triggerStr);
  if (!triggerEnum) {
    return parser.emitError(parser.getCurrentLocation(),
                            "unknown edge trigger: ")
           << triggerStr;
  }
  trigger = EdgeTriggerAttr::get(parser.getContext(), *triggerEnum);
  result.attributes.set("trigger", trigger);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

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
