//===- AIETraceToConfig.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Pass to lower aie.trace to aie.trace.config
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIERegisterDatabase.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Helper function to determine module name based on tile type
static StringRef getModuleForTile(TileOp tile) {
  if (tile.isShimTile()) {
    return "pl"; // Shim tiles use PL module events
  } else if (tile.isMemTile()) {
    return "mem_tile";
  } else {
    return "core";
  }
}

struct AIETraceToConfigPass : AIETraceToConfigBase<AIETraceToConfigPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device);

    // Load register database for event lookup
    auto regDB = RegisterDatabase::loadAIE2();
    if (!regDB) {
      device.emitError("Failed to load register database");
      return signalPassFailure();
    }

    // Collect all trace operations
    SmallVector<TraceOp> traces;
    device.walk([&](TraceOp trace) { traces.push_back(trace); });

    for (auto trace : traces) {
      // Create config symbol name
      std::string configName = (trace.getSymName().str() + "_config");
      auto tile = cast<TileOp>(trace.getTile().getDefiningOp());
      StringRef module = getModuleForTile(tile);

      // Insert trace.config after trace declaration
      builder.setInsertionPointAfter(trace);
      auto configOp = builder.create<TraceConfigOp>(
          trace.getLoc(), trace.getTile(), builder.getStringAttr(configName));

      // Build register writes inside config body
      Block *configBody = new Block();
      configOp.getBody().push_back(configBody);
      OpBuilder configBuilder = OpBuilder::atBlockEnd(configBody);

      // 1. Emit Trace_Control0 fields
      // Check for start/stop events
      for (auto &op : trace.getBody().getOps()) {
        if (auto startOp = dyn_cast<TraceStartEventOp>(op)) {
          uint32_t startEvent = 0;
          if (startOp.getBroadcast()) {
            startEvent = *startOp.getBroadcast();
          } else if (auto eventAttr = startOp.getEvent()) {
            // Look up event number from database
            StringRef eventName = eventAttr->getName();
            auto eventNum = regDB->lookupEvent(eventName, module);
            if (eventNum) {
              startEvent = *eventNum;
            } else {
              trace.emitWarning("Unknown event: ") << eventName;
              startEvent = 1; // Fallback to TRUE event
            }
          }

          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control0"),
              builder.getStringAttr("Trace_Start_Event"),
              builder.getI32IntegerAttr(startEvent),
              builder.getStringAttr("start event"));
        }

        if (auto stopOp = dyn_cast<TraceStopEventOp>(op)) {
          uint32_t stopEvent = 0;
          if (stopOp.getBroadcast()) {
            stopEvent = *stopOp.getBroadcast();
          } else if (auto eventAttr = stopOp.getEvent()) {
            // Look up event number from database
            StringRef eventName = eventAttr->getName();
            auto eventNum = regDB->lookupEvent(eventName, module);
            if (eventNum) {
              stopEvent = *eventNum;
            } else {
              trace.emitWarning("Unknown event: ") << eventName;
              stopEvent = 0; // Fallback to NONE event
            }
          }

          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control0"),
              builder.getStringAttr("Trace_Stop_Event"),
              builder.getI32IntegerAttr(stopEvent),
              builder.getStringAttr("stop event"));
        }

        // Emit mode if present
        if (auto modeOp = dyn_cast<TraceModeOp>(op)) {
          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control0"),
              builder.getStringAttr("Mode"),
              builder.getI32IntegerAttr(
                  static_cast<uint32_t>(modeOp.getMode())),
              builder.getStringAttr("trace mode"));
        }

        // Emit packet config if present
        if (auto packetOp = dyn_cast<TracePacketOp>(op)) {
          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control1"),
              builder.getStringAttr("ID"),
              builder.getI32IntegerAttr(packetOp.getId()),
              builder.getStringAttr("packet ID"));

          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control1"),
              builder.getStringAttr("Packet_Type"),
              builder.getI32IntegerAttr(
                  static_cast<uint32_t>(packetOp.getType())),
              builder.getStringAttr("packet type"));
        }
      }

      // 2. Emit event slots (Trace_Event0 / Trace_Event1)
      SmallVector<TraceEventOp> events;
      for (auto &op : trace.getBody().getOps()) {
        if (auto eventOp = dyn_cast<TraceEventOp>(op)) {
          events.push_back(eventOp);
        }
      }

      for (size_t i = 0; i < events.size() && i < 8; ++i) {
        // getName() returns const std::string&
        const std::string &eventName = events[i].getEvent().getName();

        // Look up event number from database
        auto eventNum = regDB->lookupEvent(eventName, module);
        if (!eventNum) {
          trace.emitWarning("Unknown event: ") << eventName;
          continue;
        }

        // Determine which register and field
        StringRef registerName = (i < 4) ? "Trace_Event0" : "Trace_Event1";
        std::string fieldName = "Trace_Event" + std::to_string(i % 4);

        // Emit register write with event number as integer
        configBuilder.create<TraceRegOp>(
            trace.getLoc(), builder.getStringAttr(registerName),
            builder.getStringAttr(fieldName),
            builder.getI32IntegerAttr(*eventNum),
            builder.getStringAttr("event slot " + std::to_string(i)));
      }

      // Add terminator
      configBuilder.create<EndOp>(trace.getLoc());

      // Update all trace.start_config references
      device.walk([&](TraceStartConfigOp startConfig) {
        if (startConfig.getTraceConfig() == trace.getSymName()) {
          startConfig.setTraceConfigAttr(
              SymbolRefAttr::get(builder.getContext(), configName));
        }
      });

      // Remove original trace op
      trace.erase();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIETraceToConfigPass() {
  return std::make_unique<AIETraceToConfigPass>();
}
