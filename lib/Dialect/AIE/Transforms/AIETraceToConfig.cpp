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
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIETraceToConfigPass : AIETraceToConfigBase<AIETraceToConfigPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device);
    const auto &targetModel = device.getTargetModel();

    // Collect all trace operations
    SmallVector<TraceOp> traces;
    device.walk([&](TraceOp trace) { traces.push_back(trace); });

    for (auto trace : traces) {
      // Create config symbol name
      std::string configName = (trace.getSymName().str() + "_config");
      auto tile = cast<TileOp>(trace.getTile().getDefiningOp());

      // Find packet type (if any)
      TracePacketType packetType = TracePacketType::Core; // default
      for (auto &op : trace.getBody().getOps()) {
        if (auto packetOp = dyn_cast<TracePacketOp>(op)) {
          packetType = packetOp.getType();
          break;
        }
      }

      // Insert trace.config after trace declaration
      builder.setInsertionPointAfter(trace);
      auto configOp = builder.create<TraceConfigOp>(
          trace.getLoc(), trace.getTile(), builder.getStringAttr(configName),
          TracePacketTypeAttr::get(builder.getContext(), packetType));

      // Build register writes inside config body
      Block *configBody = new Block();
      configOp.getBody().push_back(configBody);
      OpBuilder configBuilder = OpBuilder::atBlockEnd(configBody);

      bool isMem = (packetType == TracePacketType::Mem);

      // Process combo/edge events FIRST (before other trace config)
      // This ensures COMBO_EVENT_*/EDGE_DETECTION_EVENT_* are configured
      // before they can be referenced in trace.event operations

      // 0a. Emit combo event configurations
      for (auto &op : trace.getBody().getOps()) {
        if (auto comboOp = dyn_cast<TraceComboEventOp>(op)) {
          uint32_t slot = comboOp.getSlot();

          // Get input events - use getEventName() helper
          std::string eventAName = comboOp.getEventA().getEventName();
          std::string eventBName = comboOp.getEventB().getEventName();
          ComboLogic logic = comboOp.getLogic();

          // If enum, use enum value directly; otherwise lookup by name
          std::optional<uint32_t> eventANum, eventBNum;

          if (auto enumValA = comboOp.getEventA().getEnumValue()) {
            eventANum = static_cast<uint32_t>(*enumValA);
          } else {
            eventANum = targetModel.lookupEvent(eventAName, tile, isMem);
          }

          if (auto enumValB = comboOp.getEventB().getEnumValue()) {
            eventBNum = static_cast<uint32_t>(*enumValB);
          } else {
            eventBNum = targetModel.lookupEvent(eventBName, tile, isMem);
          }

          if (!eventANum) {
            comboOp.emitError("unknown event: ") << eventAName;
            return signalPassFailure();
          }
          if (!eventBNum) {
            comboOp.emitError("unknown event: ") << eventBName;
            return signalPassFailure();
          }

          // Map slot to input event fields
          StringRef eventAField, eventBField, controlField;
          if (slot == 0) {
            eventAField = "eventA";
            eventBField = "eventB";
            controlField = "combo0";
          } else if (slot == 1) {
            eventAField = "eventC";
            eventBField = "eventD";
            controlField = "combo1";
          } else if (slot == 2) {
            // Combo2 is hierarchical - reuses eventA/B fields but represents
            // combo0/combo1
            eventAField = "eventA";
            eventBField = "eventB";
            controlField = "combo2";
          }

          // Emit Combo_event_inputs register fields
          configBuilder.create<TraceRegOp>(
              comboOp.getLoc(), builder.getStringAttr("Combo_event_inputs"),
              builder.getStringAttr(eventAField), comboOp.getEventA(),
              /*mask=*/nullptr,
              builder.getStringAttr("combo" + std::to_string(slot) +
                                    " eventA"));

          configBuilder.create<TraceRegOp>(
              comboOp.getLoc(), builder.getStringAttr("Combo_event_inputs"),
              builder.getStringAttr(eventBField), comboOp.getEventB(),
              /*mask=*/nullptr,
              builder.getStringAttr("combo" + std::to_string(slot) +
                                    " eventB"));

          // Emit Combo_event_control register field
          configBuilder.create<TraceRegOp>(
              comboOp.getLoc(), builder.getStringAttr("Combo_event_control"),
              builder.getStringAttr(controlField),
              builder.getI32IntegerAttr(static_cast<uint32_t>(logic)),
              /*mask=*/nullptr,
              builder.getStringAttr("combo" + std::to_string(slot) + " logic"));
        }
      }

      // 0b. Emit edge detection configurations
      for (auto &op : trace.getBody().getOps()) {
        if (auto edgeOp = dyn_cast<TraceEdgeEventOp>(op)) {
          uint32_t slot = edgeOp.getSlot();
          std::string eventName = edgeOp.getEvent().getEventName();
          EdgeTrigger trigger = edgeOp.getTrigger();

          // If enum, use enum value directly; otherwise lookup by name
          std::optional<uint32_t> eventNum;

          if (auto enumVal = edgeOp.getEvent().getEnumValue()) {
            eventNum = static_cast<uint32_t>(*enumVal);
          } else {
            eventNum = targetModel.lookupEvent(eventName, tile, isMem);
          }

          if (!eventNum) {
            edgeOp.emitError("unknown event: ") << eventName;
            return signalPassFailure();
          }

          // Map slot to field names
          StringRef eventField =
              (slot == 0) ? "Edge_Detection_Event_0" : "Edge_Detection_Event_1";
          StringRef risingField = (slot == 0)
                                      ? "Edge_Detection_0_Trigger_Rising"
                                      : "Edge_Detection_1_Trigger_Rising";
          StringRef fallingField = (slot == 0)
                                       ? "Edge_Detection_0_Trigger_Falling"
                                       : "Edge_Detection_1_Trigger_Falling";

          // Source event
          configBuilder.create<TraceRegOp>(
              edgeOp.getLoc(),
              builder.getStringAttr("Edge_Detection_event_control"),
              builder.getStringAttr(eventField),
              builder.getStringAttr(eventName),
              /*mask=*/nullptr,
              builder.getStringAttr("edge" + std::to_string(slot) + " source"));

          // Trigger mode
          bool rising =
              (trigger == EdgeTrigger::RISING || trigger == EdgeTrigger::BOTH);
          bool falling =
              (trigger == EdgeTrigger::FALLING || trigger == EdgeTrigger::BOTH);

          configBuilder.create<TraceRegOp>(
              edgeOp.getLoc(),
              builder.getStringAttr("Edge_Detection_event_control"),
              builder.getStringAttr(risingField),
              builder.getI32IntegerAttr(rising ? 1 : 0),
              /*mask=*/nullptr,
              builder.getStringAttr("edge" + std::to_string(slot) + " rising"));

          configBuilder.create<TraceRegOp>(
              edgeOp.getLoc(),
              builder.getStringAttr("Edge_Detection_event_control"),
              builder.getStringAttr(fallingField),
              builder.getI32IntegerAttr(falling ? 1 : 0),
              /*mask=*/nullptr,
              builder.getStringAttr("edge" + std::to_string(slot) +
                                    " falling"));
        }
      }

      // 1. Emit Trace_Control0 fields
      // Check for start/stop events
      for (auto &op : trace.getBody().getOps()) {
        if (auto startOp = dyn_cast<TraceStartEventOp>(op)) {
          uint32_t startEvent = 0;
          if (startOp.getBroadcast()) {
            startEvent = *startOp.getBroadcast();
          } else if (auto eventAttr = startOp.getEvent()) {
            // Use getEventName() helper and check for enum
            std::string eventName = eventAttr->getEventName();

            std::optional<uint32_t> eventNum;
            if (auto enumVal = eventAttr->getEnumValue()) {
              eventNum = static_cast<uint32_t>(*enumVal);
            } else {
              eventNum = targetModel.lookupEvent(eventName, tile, isMem);
            }

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
              /*mask=*/nullptr, builder.getStringAttr("start event"));
        }

        if (auto stopOp = dyn_cast<TraceStopEventOp>(op)) {
          uint32_t stopEvent = 0;
          if (stopOp.getBroadcast()) {
            stopEvent = *stopOp.getBroadcast();
          } else if (auto eventAttr = stopOp.getEvent()) {
            // Use getEventName() helper and check for enum
            std::string eventName = eventAttr->getEventName();

            std::optional<uint32_t> eventNum;
            if (auto enumVal = eventAttr->getEnumValue()) {
              eventNum = static_cast<uint32_t>(*enumVal);
            } else {
              eventNum = targetModel.lookupEvent(eventName, tile, isMem);
            }

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
              /*mask=*/nullptr, builder.getStringAttr("stop event"));
        }

        // Emit mode if present
        if (auto modeOp = dyn_cast<TraceModeOp>(op)) {
          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control0"),
              builder.getStringAttr("Mode"),
              builder.getI32IntegerAttr(
                  static_cast<uint32_t>(modeOp.getMode())),
              /*mask=*/nullptr, builder.getStringAttr("trace mode"));
        }

        // Emit packet config if present
        if (auto packetOp = dyn_cast<TracePacketOp>(op)) {
          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control1"),
              builder.getStringAttr("ID"),
              builder.getI32IntegerAttr(packetOp.getId()),
              /*mask=*/nullptr, builder.getStringAttr("packet ID"));

          configBuilder.create<TraceRegOp>(
              trace.getLoc(), builder.getStringAttr("Trace_Control1"),
              builder.getStringAttr("Packet_Type"),
              builder.getI32IntegerAttr(
                  static_cast<uint32_t>(packetOp.getType())),
              /*mask=*/nullptr, builder.getStringAttr("packet type"));
        }
      }

      // 2. Emit port configurations (Stream_Switch_Event_Port_Selection_0/1)
      for (auto &op : trace.getBody().getOps()) {
        if (auto portOp = dyn_cast<TracePortOp>(op)) {
          uint32_t slot = portOp.getSlot();

          // Determine which register based on slot
          StringRef registerName = (slot < 4)
                                       ? "Stream_Switch_Event_Port_Selection_0"
                                       : "Stream_Switch_Event_Port_Selection_1";

          // Generate field names
          std::string idFieldName = "Port_" + std::to_string(slot) + "_ID";
          std::string masterSlaveFieldName =
              "Port_" + std::to_string(slot) + "_Master_Slave";

          // Generate port value string "PORT:CHANNEL"
          std::string portValue = stringifyWireBundle(portOp.getPort()).str() +
                                  ":" + std::to_string(portOp.getChannel());

          // Convert DMAChannelDir to master flag: S2MM=master(1), MM2S=slave(0)
          int masterSlaveValue =
              (portOp.getDirection() == DMAChannelDir::S2MM) ? 1 : 0;

          // Emit Port_N_ID field
          configBuilder.create<TraceRegOp>(
              portOp.getLoc(), builder.getStringAttr(registerName),
              builder.getStringAttr(idFieldName),
              builder.getStringAttr(portValue), // "NORTH:1" format
              /*mask=*/nullptr,
              builder.getStringAttr("port " + std::to_string(slot) + " ID"));

          // Emit Port_N_Master_Slave field
          configBuilder.create<TraceRegOp>(
              portOp.getLoc(), builder.getStringAttr(registerName),
              builder.getStringAttr(masterSlaveFieldName),
              builder.getI32IntegerAttr(masterSlaveValue),
              /*mask=*/nullptr,
              builder.getStringAttr("port " + std::to_string(slot) +
                                    " master/slave"));
        }
      }

      // 3. Emit event slots (Trace_Event0 / Trace_Event1)
      SmallVector<TraceEventOp> events;
      for (auto &op : trace.getBody().getOps()) {
        if (auto eventOp = dyn_cast<TraceEventOp>(op)) {
          events.push_back(eventOp);
        }
      }

      for (size_t i = 0; i < events.size() && i < 8; ++i) {
        std::string eventName = events[i].getEvent().getEventName();

        // If enum, use enum value directly; otherwise lookup by name
        std::optional<uint32_t> eventNum;

        if (auto enumVal = events[i].getEvent().getEnumValue()) {
          eventNum = static_cast<uint32_t>(*enumVal);
        } else {
          eventNum = targetModel.lookupEvent(eventName, tile, isMem);
        }

        if (!eventNum) {
          trace.emitWarning("Unknown event: ") << eventName;
          continue;
        }

        // Determine which register and field
        StringRef registerName = (i < 4) ? "Trace_Event0" : "Trace_Event1";
        std::string fieldName = "Trace_Event" + std::to_string(i);

        // Emit register write with event number as integer
        configBuilder.create<TraceRegOp>(
            trace.getLoc(), builder.getStringAttr(registerName),
            builder.getStringAttr(fieldName),
            events[i].getEvent(), // builder.getI32IntegerAttr(*eventNum),
            /*mask=*/nullptr, builder.getStringAttr(eventName));
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

//===----------------------------------------------------------------------===//
// AIETraceRegPackWritesPass - Pack multiple register field writes
//===----------------------------------------------------------------------===//

struct AIETraceRegPackWritesPass
    : AIETraceRegPackWritesBase<AIETraceRegPackWritesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();

    // Process each trace config
    device.walk([&](TraceConfigOp configOp) {
      // Determine module based on tile type
      auto tile = cast<TileOp>(configOp.getTile().getDefiningOp());

      // Phase 1: Convert field+value to mask+shifted_value
      SmallVector<TraceRegOp> regsToConvert;
      for (auto &op : configOp.getBody().front()) {
        if (auto regOp = dyn_cast<TraceRegOp>(op)) {
          if (regOp.getField() && !regOp.getMask()) {
            regsToConvert.push_back(regOp);
          }
        }
      }

      OpBuilder builder(&configOp.getBody().front(),
                        configOp.getBody().front().begin());

      for (auto regOp : regsToConvert) {
        // Look up register and field information
        const RegisterInfo *regInfo =
            targetModel.lookupRegister(regOp.getRegName(), tile);

        if (!regInfo) {
          regOp.emitError("Register not found in database: ")
              << regOp.getRegName();
          return signalPassFailure();
        }

        const BitFieldInfo *fieldInfo = regInfo->getField(*regOp.getField());
        if (!fieldInfo) {
          regOp.emitError("Field not found in register: ")
              << *regOp.getField() << " in " << regOp.getRegName();
          return signalPassFailure();
        }

        // Get the value - handle both integers and port strings
        uint32_t value = 0;
        Attribute valAttr = regOp.getValue();
        if (auto traceEventAttr = dyn_cast<TraceEventAttr>(valAttr)) {
          valAttr = traceEventAttr.getValue();
          // if it's a string, lookup the enum value
          if (auto strAttr = dyn_cast<StringAttr>(valAttr)) {
            std::string eventName = strAttr.getValue().str();
            std::optional<uint32_t> eventNum =
                targetModel.lookupEvent(eventName, tile, false);
            if (!eventNum) {
              regOp.emitError("Unknown event: ") << eventName;
              return signalPassFailure();
            }
            value = *eventNum;
            valAttr = builder.getI32IntegerAttr(value);
          }
        }
        if (auto intAttr = dyn_cast<IntegerAttr>(valAttr)) {
          // Integer value
          value = intAttr.getInt();
        } else if (auto strAttr = dyn_cast<StringAttr>(valAttr)) {
          // String value - check if it's a port specification
          StringRef valueStr = strAttr.getValue();

          // Determine master/slave from field name
          // If field name contains "Master_Slave", this is not a port ID field
          // Port ID fields are named "Port_N_ID"
          bool isMasterSlaveField =
              fieldInfo->name.find("Master_Slave") != std::string::npos;

          if (!isMasterSlaveField && valueStr.contains(':')) {
            // This looks like "PORT:CHANNEL" format
            // We need master/slave info - look for corresponding Master_Slave
            // field
            bool master = false; // Default to slave

            // Search for companion Master_Slave field in same register
            for (auto &siblingOp : configOp.getBody().front()) {
              if (auto siblingReg = dyn_cast<TraceRegOp>(siblingOp)) {
                if (siblingReg.getRegName() == regOp.getRegName() &&
                    siblingReg.getField() &&
                    siblingReg.getField()->contains("Master_Slave")) {
                  // Found companion field - extract master flag
                  if (auto siblingInt =
                          dyn_cast<IntegerAttr>(siblingReg.getValue())) {
                    master = (siblingInt.getInt() != 0);
                  }
                  break;
                }
              }
            }

            // Resolve port value
            auto portIndex =
                targetModel.resolvePortValue(valueStr, tile, master);
            if (!portIndex) {
              regOp.emitError("Failed to resolve port value: ") << valueStr;
              return signalPassFailure();
            }
            value = *portIndex;
          } else {
            regOp.emitError("Unsupported string value: ") << valueStr;
            return signalPassFailure();
          }
        } else {
          regOp.emitError("Unsupported value type in pack pass");
          return signalPassFailure();
        }

        // Compute mask and shifted value
        uint32_t mask = ((1u << fieldInfo->getWidth()) - 1)
                        << fieldInfo->bit_start;
        uint32_t shiftedValue = targetModel.encodeFieldValue(*fieldInfo, value);
        // Create new operation with mask
        builder.setInsertionPoint(regOp);
        builder.create<TraceRegOp>(regOp.getLoc(), regOp.getRegNameAttr(),
                                   nullptr, // no field
                                   builder.getI32IntegerAttr(shiftedValue),
                                   builder.getI32IntegerAttr(mask),
                                   regOp.getCommentAttr());

        // Remove old operation
        regOp.erase();
      }

      // Phase 2: Merge writes to same register with non-overlapping masks
      bool changed = true;
      while (changed) {
        changed = false;

        // Collect all register writes
        SmallVector<TraceRegOp> regWrites;
        for (TraceRegOp op : configOp.getBody().front().getOps<TraceRegOp>()) {
          if (op.getMask()) {
            regWrites.push_back(op);
          }
        }

        // Try to merge pairs
        for (size_t i = 0; i < regWrites.size() && !changed; ++i) {
          for (size_t j = i + 1; j < regWrites.size() && !changed; ++j) {
            TraceRegOp reg1 = regWrites[i];
            TraceRegOp reg2 = regWrites[j];

            // Must be same register
            if (reg1.getRegName() != reg2.getRegName())
              continue;

            auto mask1Attr = reg1.getMask();
            auto mask2Attr = reg2.getMask();
            if (!mask1Attr || !mask2Attr)
              continue;

            uint32_t mask1 = *mask1Attr;
            uint32_t mask2 = *mask2Attr;

            // Check for overlap
            if (mask1 & mask2) {
              reg1.emitError("Overlapping masks for register ")
                  << reg1.getRegName() << ": mask1=" << mask1
                  << " mask2=" << mask2;
              return signalPassFailure();
            }

            // Merge the two writes
            uint32_t value1 = dyn_cast<IntegerAttr>(reg1.getValue()).getInt();
            uint32_t value2 = dyn_cast<IntegerAttr>(reg2.getValue()).getInt();
            uint32_t mergedValue = value1 | value2;
            uint32_t mergedMask = mask1 | mask2;

            // Create merged operation
            builder.setInsertionPoint(reg1);
            std::string comment;
            if (reg1.getComment())
              comment += reg1.getComment()->str();
            if (reg2.getComment()) {
              if (!comment.empty())
                comment += " + ";
              comment += reg2.getComment()->str();
            }

            builder.create<TraceRegOp>(
                reg1.getLoc(), reg1.getRegNameAttr(), nullptr,
                builder.getI32IntegerAttr(mergedValue),
                builder.getI32IntegerAttr(mergedMask),
                comment.empty() ? nullptr : builder.getStringAttr(comment));

            // Remove both old operations
            reg1.erase();
            reg2.erase();
            changed = true;
          }
        }
      }
    });
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIETraceRegPackWritesPass() {
  return std::make_unique<AIETraceRegPackWritesPass>();
}
