//===- AIEInsertTraceFlows.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include <climits>
#include <map>
#include <set>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEINSERTTRACEFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct TraceInfo {
  TraceOp traceOp;
  TileOp tile;
  int packetId;
  TracePacketType packetType;
  WireBundle tracePort; // Trace:0 (core) or Trace:1 (mem)
  int traceChannel;     // Port number (0 for core, 1 for mem)
  std::optional<int>
      startBroadcast;               // Broadcast channel for timer sync (if any)
  std::optional<int> stopBroadcast; // Broadcast channel for trace stop (if any)
};

/// Per-channel DMA resource allocation.
struct ChannelDescriptor {
  int channel;      // S2MM channel number
  int bdId;         // Buffer descriptor ID
  int argIdx;       // Runtime sequence argument index
  int bufferOffset; // Byte offset within the shared trace buffer
};

struct ShimInfo {
  TileOp shimTile;
  int channel;      // S2MM channel
  int bdId;         // Buffer descriptor ID
  int argIdx;       // Runtime sequence argument index
  int bufferOffset; // Base byte offset for trace within the XRT buffer
  std::vector<TraceInfo> traceSources;     // All traces routed to this shim
  std::optional<int> startBroadcast;       // Broadcast to trigger for start
  std::optional<int> stopBroadcast;        // Broadcast to trigger for stop
  std::vector<ChannelDescriptor> channels; // Per-channel descriptors
  std::vector<int> traceChannelAssignment; // Per-trace index into channels
};

struct AIEInsertTraceFlowsPass
    : xilinx::AIE::impl::AIEInsertTraceFlowsBase<AIEInsertTraceFlowsPass> {

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device);
    const auto &targetModel = device.getTargetModel();

    // Verify no LogicalTileOps remain — placement must run before this pass
    bool hasLogicalTile = false;
    device.walk([&](LogicalTileOp op) {
      op.emitError() << "LogicalTileOp must be resolved to TileOp before "
                        "running -aie-insert-trace-flows (run -aie-place-tiles "
                        "first)";
      hasLogicalTile = true;
    });
    if (hasLogicalTile)
      return signalPassFailure();

    // Phase 1: Collect all trace operations
    SmallVector<TraceOp> traces;
    device.walk([&](TraceOp trace) { traces.push_back(trace); });

    if (traces.empty())
      return;

    // Phase 1b: Find runtime_sequence and trace.host_config within it
    RuntimeSequenceOp runtimeSeq = nullptr;
    TraceHostConfigOp hostConfig = nullptr;
    for (auto &op : device.getBody()->getOperations()) {
      if (auto seq = dyn_cast<RuntimeSequenceOp>(&op)) {
        runtimeSeq = seq;
        for (auto &subOp : seq.getBody().front().getOperations()) {
          if (auto hc = dyn_cast<TraceHostConfigOp>(&subOp)) {
            hostConfig = hc;
            break;
          }
        }
        break;
      }
    }

    // Require runtime_sequence when trace ops are present
    if (!runtimeSeq) {
      device.emitError()
          << "aie.trace ops found but no runtime_sequence defined";
      return signalPassFailure();
    }

    // Require trace.host_config in runtime_sequence
    if (!hostConfig) {
      runtimeSeq.emitError()
          << "runtime_sequence with traces requires aie.trace.host_config";
      return signalPassFailure();
    }

    // Get configuration from host_config
    int bufferSizeBytes = hostConfig.getBufferSize();
    int traceArgIdx = hostConfig.getArgIdx();
    auto routing = hostConfig.getRouting();

    // arg_idx=-1 means "append trace after last tensor"
    int traceBufferOffset = 0; // in bytes
    if (traceArgIdx == -1) {
      auto args = runtimeSeq.getBody().getArguments();
      assert(!args.empty() && "runtime_sequence must have args for arg_idx=-1");

      Value lastArg = args.back();
      traceArgIdx = args.size() - 1;

      auto memrefType = cast<MemRefType>(lastArg.getType());
      traceBufferOffset = memrefType.getNumElements() *
                          (memrefType.getElementTypeBitWidth() / 8);
    }

    // Remove host_config op
    hostConfig.erase();

    // Phase 2: Analyze traces and allocate resources
    std::vector<TraceInfo> traceInfos;
    int nextPacketId = clPacketIdStart;

    for (auto trace : traces) {
      auto tile = cast<TileOp>(trace.getTile().getDefiningOp());

      // Find packet ID and type from trace body
      std::optional<int> packetId;
      std::optional<TracePacketType> packetType;
      TracePacketOp existingPacketOp = nullptr;
      for (auto &op : trace.getBody().getOps()) {
        if (auto packetOp = dyn_cast<TracePacketOp>(op)) {
          existingPacketOp = packetOp;
          packetId = packetOp.getId();
          packetType = packetOp.getType();
          break;
        }
      }

      // Determine packet type from tile type if not specified
      if (!packetType) {
        if (tile.isShimTile()) {
          packetType = TracePacketType::ShimTile;
        } else if (tile.isMemTile()) {
          packetType = TracePacketType::MemTile;
        } else {
          // Core tile defaults to core type
          packetType = TracePacketType::Core;
        }
      }

      // Allocate packet ID if not specified
      if (!packetId) {
        packetId = nextPacketId++;
      }

      // If there was no explicit TracePacketOp, materialize one so that
      // downstream passes (e.g., -aie-trace-to-config) see consistent info.
      if (!existingPacketOp) {
        OpBuilder traceBuilder(&trace.getBody().front(),
                               trace.getBody().front().begin());
        TracePacketOp::create(traceBuilder, trace.getLoc(), *packetId,
                              *packetType);
      }

      // Determine trace port based on packet type
      WireBundle tracePort = WireBundle::Trace;
      int traceChannel = 0;
      if (*packetType == TracePacketType::Mem) {
        traceChannel = 1; // Mem trace uses port 1
      }

      // Find start/stop configuration (broadcast or event)
      std::optional<int> startBroadcast;
      std::optional<int> stopBroadcast;
      bool hasStartConfig = false;
      bool hasStopConfig = false;
      for (auto &op : trace.getBody().getOps()) {
        if (auto startOp = dyn_cast<TraceStartEventOp>(op)) {
          hasStartConfig = true;
          if (startOp.getBroadcast())
            startBroadcast = *startOp.getBroadcast();
        }
        if (auto stopOp = dyn_cast<TraceStopEventOp>(op)) {
          hasStopConfig = true;
          if (stopOp.getBroadcast())
            stopBroadcast = *stopOp.getBroadcast();
        }
      }

      // Require explicit start/stop configuration
      if (!hasStartConfig) {
        trace.emitError() << "trace is missing 'aie.trace.start'";
        return signalPassFailure();
      }
      if (!hasStopConfig) {
        trace.emitError() << "trace is missing 'aie.trace.stop'";
        return signalPassFailure();
      }

      TraceInfo info;
      info.traceOp = trace;
      info.tile = tile;
      info.packetId = *packetId;
      info.packetType = *packetType;
      info.tracePort = tracePort;
      info.traceChannel = traceChannel;
      info.startBroadcast = startBroadcast;
      info.stopBroadcast = stopBroadcast;
      traceInfos.push_back(info);
    }

    // Phase 2b: Select shim tiles based on routing strategy
    std::map<int, ShimInfo> shimInfos; // col -> ShimInfo

    if (routing == TraceShimRouting::Single) {
      // All traces route to column 0 shim
      int targetCol = 0;
      TileOp shimTile = nullptr;
      for (auto tile : device.getOps<TileOp>()) {
        if (tile.getCol() == targetCol && tile.getRow() == 0) {
          shimTile = tile;
          break;
        }
      }

      if (!shimTile) {
        builder.setInsertionPointToStart(&device.getRegion().front());
        shimTile = TileOp::create(builder, device.getLoc(), targetCol, 0);
      }

      ShimInfo shimInfo;
      shimInfo.shimTile = shimTile;
      shimInfo.channel = clShimChannel;
      shimInfo.bdId = clDefaultBdId;
      shimInfo.argIdx = traceArgIdx;
      shimInfo.bufferOffset = traceBufferOffset;
      shimInfo.traceSources = traceInfos;
      // Collect broadcast channels from traces that use them
      for (auto &trace : traceInfos) {
        if (trace.startBroadcast && !shimInfo.startBroadcast)
          shimInfo.startBroadcast = trace.startBroadcast;
        if (trace.stopBroadcast && !shimInfo.stopBroadcast)
          shimInfo.stopBroadcast = trace.stopBroadcast;
      }

      // Map all source columns to the single shim
      for (auto &info : traceInfos) {
        int col = info.tile.getCol();
        if (shimInfos.find(col) == shimInfos.end()) {
          shimInfos[col] = shimInfo;
        }
      }
      if (shimInfos.find(targetCol) == shimInfos.end()) {
        shimInfos[targetCol] = shimInfo;
      }
    }

    // Phase 2b-lateral: Optionally redirect shim targets to spare columns.
    // This is a post-processing step that works with ANY routing strategy.
    // When per-column routing returns, lateral routing will compose with it
    // (each column's shim redirects to its nearest spare).
    if (clLateralRouting) {
      std::set<int> activeColumns;
      device.walk([&](CoreOp core) {
        auto coreTile = cast<TileOp>(core.getTile().getDefiningOp());
        activeColumns.insert(coreTile.getCol());
      });

      // Collect all unique shim target columns and find redirects
      std::map<int, int> redirects; // old target col -> new target col
      for (auto &[col, shimInfo] : shimInfos) {
        int curTarget = shimInfo.shimTile.getCol();
        if (redirects.count(curTarget))
          continue; // already computed

        int spare = -1;
        if (clLateralTargetCol >= 0) {
          // Forced lateral target: always redirect unless it's a no-op.
          if (clLateralTargetCol == curTarget)
            continue;
          // Validate the forced column is in range and is a shim NOC tile.
          int numCols = targetModel.columns();
          if (clLateralTargetCol >= numCols ||
              !targetModel.isShimNOCTile(clLateralTargetCol, 0)) {
            device.emitError() << "lateral-target-col " << clLateralTargetCol
                               << " is not a valid shim NOC tile (device has "
                               << numCols << " columns)";
            return signalPassFailure();
          }
          spare = clLateralTargetCol;
        } else {
          // Auto-detect: only redirect from active columns to a spare.
          if (activeColumns.count(curTarget) == 0)
            continue; // already spare, no redirect needed
          spare = findNearestSpareColumn(curTarget, activeColumns, targetModel);
        }
        if (spare >= 0)
          redirects[curTarget] = spare;
      }

      // Apply redirects: rebuild shimInfos with new target columns
      if (!redirects.empty()) {
        std::map<int, ShimInfo> newShimInfos;
        for (auto &[col, shimInfo] : shimInfos) {
          int curTarget = shimInfo.shimTile.getCol();
          auto it = redirects.find(curTarget);
          if (it != redirects.end()) {
            int newTarget = it->second;
            shimInfo.shimTile = getOrCreateShim(device, builder, newTarget);
          }
          newShimInfos[col] = shimInfo;
        }
        shimInfos = std::move(newShimInfos);
      }
    }

    // Phase 2b-conflict: Check S2MM channel availability on target shims.
    // If channels are occupied, adjust: use free channel, redirect lateral,
    // or error.
    auto usedChannels = scanUsedS2MMChannels(device);

    for (auto &[col, shimInfo] : shimInfos) {
      int shimCol = shimInfo.shimTile.getCol();
      auto usedIt = usedChannels.find(shimCol);
      if (usedIt == usedChannels.end())
        continue; // No conflicts on this shim

      const auto &usedSet = usedIt->second;
      int freeCount = 2 - static_cast<int>(usedSet.size());

      if (freeCount <= 0) {
        // No channels free -- try lateral redirect
        if (clLateralRouting) {
          std::set<int> activeColumns;
          device.walk([&](CoreOp core) {
            auto coreTile = cast<TileOp>(core.getTile().getDefiningOp());
            activeColumns.insert(coreTile.getCol());
          });
          // Also treat columns with full shims as "active" for spare search
          for (auto &[fullCol, channels] : usedChannels) {
            if (static_cast<int>(channels.size()) >= 2)
              activeColumns.insert(fullCol);
          }
          int spare =
              findNearestSpareColumn(shimCol, activeColumns, targetModel);
          if (spare >= 0) {
            shimInfo.shimTile = getOrCreateShim(device, builder, spare);
            // Reset channel to default since spare shim is clean
            shimInfo.channel = clShimChannel;
            continue;
          }
        }
        // No lateral available -- emit error
        device.emitError()
            << "no S2MM channels available on shim tile at column " << shimCol
            << " (both channels in use by existing flows); enable "
               "lateral-routing to redirect to a spare column";
        return signalPassFailure();
      }

      if (freeCount == 1) {
        // One channel free -- use it
        int freeChannel = -1;
        for (int ch = 0; ch < 2; ch++) {
          if (usedSet.count(ch) == 0) {
            freeChannel = ch;
            break;
          }
        }
        shimInfo.channel = freeChannel;
      }
    }

    // Build channel descriptors and trace-to-channel assignments
    for (auto &[col, shimInfo] : shimInfos) {
      int shimCol = shimInfo.shimTile.getCol();
      std::set<int> available = {0, 1};
      auto usedIt = usedChannels.find(shimCol);
      if (usedIt != usedChannels.end()) {
        for (int ch : usedIt->second)
          available.erase(ch);
      }
      shimInfo.channels = buildChannelDescriptors(
          shimInfo.traceSources.size(), shimInfo.channel, shimInfo.bdId,
          shimInfo.argIdx, shimInfo.bufferOffset, bufferSizeBytes, available);
      // Round-robin assignment of traces to channels
      for (size_t i = 0; i < shimInfo.traceSources.size(); i++) {
        shimInfo.traceChannelAssignment.push_back(i % shimInfo.channels.size());
      }
    }

    // Phase 2c: Rewrite broadcast to USER_EVENT for destination shim tiles
    // (shim can't listen for its own broadcast)
    for (auto &info : traceInfos) {
      if (!info.tile.isShimTile())
        continue;

      int col = info.tile.getCol();
      auto shimIt = shimInfos.find(col);
      if (shimIt == shimInfos.end())
        continue;

      // Check if this traced shim tile IS the destination shim
      if (shimIt->second.shimTile.getTileID() != info.tile.getTileID())
        continue;

      // This shim is both traced and the destination - rewrite start/stop ops
      // to use USER_EVENT_1/0 instead of broadcast
      for (auto &op : info.traceOp.getBody().getOps()) {
        if (auto startOp = dyn_cast<TraceStartEventOp>(op)) {
          if (startOp.getBroadcast()) {
            // Replace broadcast with USER_EVENT_1
            startOp->removeAttr("broadcast");
            startOp->setAttr("event", TraceEventAttr::get(builder.getContext(),
                                                          builder.getStringAttr(
                                                              "USER_EVENT_1")));
          }
        }
        if (auto stopOp = dyn_cast<TraceStopEventOp>(op)) {
          if (stopOp.getBroadcast()) {
            // Replace broadcast with USER_EVENT_0
            stopOp->removeAttr("broadcast");
            stopOp->setAttr("event", TraceEventAttr::get(builder.getContext(),
                                                         builder.getStringAttr(
                                                             "USER_EVENT_0")));
          }
        }
      }

      info.startBroadcast = std::nullopt;
      info.stopBroadcast = std::nullopt;
    }

    // Phase 3: Insert packet flows
    // Insert before the device terminator
    Block &deviceBlock = device.getRegion().front();
    builder.setInsertionPoint(deviceBlock.getTerminator());

    for (auto &info : traceInfos) {
      // Find target shim for this trace
      int col = info.tile.getCol();
      ShimInfo &shimInfo = shimInfos[col];

      // Find this trace's channel assignment
      int chanIdx = 0;
      for (size_t i = 0; i < shimInfo.traceSources.size(); i++) {
        if (shimInfo.traceSources[i].packetId == info.packetId) {
          chanIdx = shimInfo.traceChannelAssignment[i];
          break;
        }
      }
      auto &chanDesc = shimInfo.channels[chanIdx];

      // Create packet flow
      auto packetFlowOp = PacketFlowOp::create(
          builder, device.getLoc(), builder.getI8IntegerAttr(info.packetId),
          nullptr, nullptr);

      Block *flowBody = new Block();
      packetFlowOp.getPorts().push_back(flowBody);
      OpBuilder flowBuilder = OpBuilder::atBlockEnd(flowBody);

      PacketSourceOp::create(flowBuilder, device.getLoc(),
                             Value(info.tile.getResult()), info.tracePort,
                             info.traceChannel);

      PacketDestOp::create(flowBuilder, device.getLoc(),
                           Value(shimInfo.shimTile.getResult()),
                           WireBundle::DMA, chanDesc.channel);

      EndOp::create(flowBuilder, device.getLoc());

      packetFlowOp->setAttr("keep_pkt_header", builder.getBoolAttr(true));
    }

    // Phase 4: Insert runtime sequence operations
    Block &seqBlock = runtimeSeq.getBody().front();

    // Find the last TraceStartConfigOp in the runtime sequence
    Operation *lastStartConfig = nullptr;
    for (auto &op : seqBlock.getOperations()) {
      if (isa<TraceStartConfigOp>(op)) {
        lastStartConfig = &op;
      }
    }

    // Insert after the last start_config op, or at start if none found
    if (lastStartConfig) {
      builder.setInsertionPointAfter(lastStartConfig);
    } else {
      builder.setInsertionPointToStart(&seqBlock);
    }

    // 4b. Configure timer sync for tiles using broadcast-based start
    std::set<std::tuple<int, int, bool>> processedTiles;
    for (auto &info : traceInfos) {
      if (!info.startBroadcast)
        continue;

      int col = info.tile.getCol();
      int row = info.tile.getRow();
      bool isMemTrace = info.packetType == TracePacketType::Mem;

      if (processedTiles.count({col, row, isMemTrace}))
        continue;
      processedTiles.insert({col, row, isMemTrace});

      // Skip destination shim tiles (handled in 4f)
      if (info.tile.isShimTile()) {
        auto shimIt = shimInfos.find(col);
        if (shimIt != shimInfos.end() &&
            shimIt->second.shimTile.getTileID() == info.tile.getTileID()) {
          continue;
        }
      }

      // Compute timer control address
      uint32_t timerCtrlAddr =
          computeTimerCtrlAddress(info.tile, targetModel, isMemTrace);

      std::string broadcastEventName;
      if (info.tile.isShimTile()) {
        broadcastEventName =
            "BROADCAST_A_" + std::to_string(*info.startBroadcast);
      } else {
        broadcastEventName =
            "BROADCAST_" + std::to_string(*info.startBroadcast);
      }

      auto broadcastEvent = targetModel.lookupEvent(
          broadcastEventName, info.tile.getTileID(), isMemTrace);
      if (!broadcastEvent) {
        info.traceOp.emitError() << "Failed to lookup broadcast event '"
                                 << broadcastEventName << "'";
        return signalPassFailure();
      }
      const RegisterInfo *timerReg = targetModel.lookupRegister(
          "Timer_Control", info.tile.getTileID(), isMemTrace);
      if (!timerReg)
        llvm::report_fatal_error("Failed to lookup Timer_Control register");
      const BitFieldInfo *resetField = timerReg->getField("Reset_Event");
      if (!resetField)
        llvm::report_fatal_error(
            "Failed to lookup Reset_Event field in Timer_Control");
      uint32_t timerCtrlValue =
          targetModel.encodeFieldValue(*resetField, *broadcastEvent);

      xilinx::AIEX::NpuWrite32Op::create(
          builder, runtimeSeq.getLoc(), timerCtrlAddr, timerCtrlValue, nullptr,
          builder.getI32IntegerAttr(col), builder.getI32IntegerAttr(row));
    }

    // 4c-4f. Insert per-shim configurations
    std::set<int> configuredShimCols;
    for (auto &[col, shimInfo] : shimInfos) {
      int shimCol = shimInfo.shimTile.getCol();
      if (!configuredShimCols.insert(shimCol).second)
        continue;

      for (auto &chanDesc : shimInfo.channels) {
        // Convert buffer size (bytes) to 32-bit words for buffer_length
        int bufferLengthWords = bufferSizeBytes / 4;

        // 4c. Write buffer descriptor
        xilinx::AIEX::NpuWriteBdOp::create(
            builder, runtimeSeq.getLoc(),
            shimCol,           // column
            chanDesc.bdId,     // bd_id
            bufferLengthWords, // buffer_length (in 32-bit words)
            0,                 // buffer_offset
            1,                 // enable_packet
            0,                 // out_of_order_id
            0,                 // packet_id (not used for reception)
            0,                 // packet_type (not used for reception)
            0, 0, 0, 0, 0,
            0, // d0_size, d0_stride, d1_size, d1_stride, d2_size, d2_stride
            0, 0, 0, // iteration_current, iteration_size, iteration_stride
            0,       // next_bd
            0,       // row
            0,       // use_next_bd
            1,       // valid_bd
            0, 0, 0, 0, 0,    // lock_rel_val, lock_rel_id, lock_acq_enable,
                              // lock_acq_val, lock_acq_id
            0, 0, 0, 0, 0, 0, // d0_zero_before, d1_zero_before, d2_zero_before,
                              // d0_zero_after, d1_zero_after, d2_zero_after
            clTraceBurstLength // burst_length
        );

        // 4d. Address patch -- each channel gets its own offset within the
        // shared trace buffer (the secondary channel starts at
        // baseOffset + bufferSizeBytes when distribute is active).
        uint32_t bdAddress = computeBDAddress(shimCol, chanDesc.bdId,
                                              shimInfo.shimTile, targetModel);
        xilinx::AIEX::NpuAddressPatchOp::create(builder, runtimeSeq.getLoc(),
                                                bdAddress, chanDesc.argIdx,
                                                chanDesc.bufferOffset);

        // 4e. DMA channel configuration — set Controller_ID from tile attribute
        uint32_t ctrlAddr =
            computeCtrlAddress(DMAChannelDir::S2MM, chanDesc.channel,
                               shimInfo.shimTile, targetModel);
        ensureControllerIdAttr(shimInfo.shimTile, targetModel);
        auto ctrlIdAttr =
            shimInfo.shimTile->getAttrOfType<PacketInfoAttr>("controller_id");
        std::string ctrlRegName =
            (chanDesc.channel == 0) ? "DMA_S2MM_0_Ctrl" : "DMA_S2MM_1_Ctrl";
        const RegisterInfo *ctrlReg = targetModel.lookupRegister(
            ctrlRegName, shimInfo.shimTile.getTileID());
        if (!ctrlReg)
          llvm::report_fatal_error(llvm::Twine("Failed to lookup ") +
                                   ctrlRegName);
        const BitFieldInfo *ctrlIdField = ctrlReg->getField("Controller_ID");
        if (!ctrlIdField)
          llvm::report_fatal_error("Failed to lookup Controller_ID field in " +
                                   llvm::Twine(ctrlRegName));
        uint32_t ctrlIdValue =
            targetModel.encodeFieldValue(*ctrlIdField, ctrlIdAttr.getPktId());
        auto ctrlIdMask = targetModel.getFieldMask(*ctrlIdField);
        if (!ctrlIdMask)
          llvm::report_fatal_error(
              "Controller_ID field does not fit in 32-bit register");
        xilinx::AIEX::NpuMaskWrite32Op::create(
            builder, runtimeSeq.getLoc(), ctrlAddr, ctrlIdValue, *ctrlIdMask,
            nullptr, builder.getI32IntegerAttr(shimCol),
            builder.getI32IntegerAttr(0));

        // Push BD to task queue
        std::string taskQueueRegName = (chanDesc.channel == 0)
                                           ? "DMA_S2MM_0_Task_Queue"
                                           : "DMA_S2MM_1_Task_Queue";
        const RegisterInfo *queueReg = targetModel.lookupRegister(
            taskQueueRegName, shimInfo.shimTile.getTileID());
        if (!queueReg)
          llvm::report_fatal_error(llvm::Twine("Failed to lookup ") +
                                   taskQueueRegName);
        const BitFieldInfo *tokenField =
            queueReg->getField("Enable_Token_Issue");
        const BitFieldInfo *bdIdField = queueReg->getField("Start_BD_ID");
        if (!tokenField || !bdIdField)
          llvm::report_fatal_error(
              "Failed to lookup Enable_Token_Issue or Start_BD_ID fields");
        uint32_t queueValue =
            targetModel.encodeFieldValue(*tokenField, 1) |
            targetModel.encodeFieldValue(*bdIdField, chanDesc.bdId);
        xilinx::AIEX::NpuWrite32Op::create(
            builder, runtimeSeq.getLoc(), queueReg->offset, queueValue, nullptr,
            builder.getI32IntegerAttr(shimCol), builder.getI32IntegerAttr(0));
      }

      // 4f. Shim timer and broadcast control (only if start broadcast is used)
      if (shimInfo.startBroadcast) {
        // Look up USER_EVENT_1 from target model (trigger event)
        auto userEvent1 = targetModel.lookupEvent(
            "USER_EVENT_1", shimInfo.shimTile.getTileID(), false);
        if (!userEvent1)
          llvm::report_fatal_error("Failed to lookup USER_EVENT_1 event");

        uint32_t shimTimerCtrlAddr =
            computeTimerCtrlAddress(shimInfo.shimTile, targetModel, false);
        const RegisterInfo *shimTimerReg = targetModel.lookupRegister(
            "Timer_Control", shimInfo.shimTile.getTileID(), false);
        if (!shimTimerReg)
          llvm::report_fatal_error("Failed to lookup shim Timer_Control");
        const BitFieldInfo *shimResetField =
            shimTimerReg->getField("Reset_Event");
        if (!shimResetField)
          llvm::report_fatal_error(
              "Failed to lookup Reset_Event in shim Timer_Control");
        uint32_t shimTimerCtrlValue =
            targetModel.encodeFieldValue(*shimResetField, *userEvent1);
        xilinx::AIEX::NpuWrite32Op::create(
            builder, runtimeSeq.getLoc(), shimTimerCtrlAddr, shimTimerCtrlValue,
            nullptr, builder.getI32IntegerAttr(shimCol),
            builder.getI32IntegerAttr(0));

        // Configure broadcast register with USER_EVENT_1
        std::string broadcastRegName =
            "Event_Broadcast" + std::to_string(*shimInfo.startBroadcast) + "_A";
        const RegisterInfo *broadcastReg = targetModel.lookupRegister(
            broadcastRegName, shimInfo.shimTile.getTileID());
        if (!broadcastReg)
          llvm::report_fatal_error(llvm::Twine("Failed to lookup ") +
                                   broadcastRegName);
        xilinx::AIEX::NpuWrite32Op::create(
            builder, runtimeSeq.getLoc(), broadcastReg->offset, *userEvent1,
            nullptr, builder.getI32IntegerAttr(shimCol),
            builder.getI32IntegerAttr(0));

        // Generate USER_EVENT_1 to trigger the broadcast
        const RegisterInfo *eventGenReg = targetModel.lookupRegister(
            "Event_Generate", shimInfo.shimTile.getTileID());
        if (!eventGenReg)
          llvm::report_fatal_error("Failed to lookup Event_Generate register");
        xilinx::AIEX::NpuWrite32Op::create(
            builder, runtimeSeq.getLoc(), eventGenReg->offset, *userEvent1,
            nullptr, builder.getI32IntegerAttr(shimCol),
            builder.getI32IntegerAttr(0));
      }
    }

    // Phase 4g: Insert trace stop at end of runtime sequence
    builder.setInsertionPointToEnd(&seqBlock);

    std::set<int> stoppedShimCols;
    for (auto &[col, shimInfo] : shimInfos) {
      if (!shimInfo.stopBroadcast)
        continue;

      int shimCol = shimInfo.shimTile.getCol();
      if (!stoppedShimCols.insert(shimCol).second)
        continue;

      auto userEvent0 = targetModel.lookupEvent(
          "USER_EVENT_0", shimInfo.shimTile.getTileID(), false);
      if (!userEvent0)
        llvm::report_fatal_error("Failed to lookup USER_EVENT_0 event");

      std::string broadcastRegName =
          "Event_Broadcast" + std::to_string(*shimInfo.stopBroadcast) + "_A";
      const RegisterInfo *broadcastReg = targetModel.lookupRegister(
          broadcastRegName, shimInfo.shimTile.getTileID());
      if (!broadcastReg)
        llvm::report_fatal_error(llvm::Twine("Failed to lookup ") +
                                 broadcastRegName);
      xilinx::AIEX::NpuWrite32Op::create(
          builder, runtimeSeq.getLoc(), broadcastReg->offset, *userEvent0,
          nullptr, builder.getI32IntegerAttr(shimCol),
          builder.getI32IntegerAttr(0));

      const RegisterInfo *stopEventGenReg = targetModel.lookupRegister(
          "Event_Generate", shimInfo.shimTile.getTileID());
      if (!stopEventGenReg)
        llvm::report_fatal_error("Failed to lookup Event_Generate register");
      xilinx::AIEX::NpuWrite32Op::create(
          builder, runtimeSeq.getLoc(), stopEventGenReg->offset, *userEvent0,
          nullptr, builder.getI32IntegerAttr(shimCol),
          builder.getI32IntegerAttr(0));
    }
  }

private:
  /// Ensure a tile has a controller_id attribute. If missing, assign one
  /// using the target model's mapping.
  void ensureControllerIdAttr(TileOp tile, const AIETargetModel &tm) {
    if (tile->hasAttr("controller_id"))
      return;
    auto idMap = tm.getTileToControllerIdMap();
    int pktId = idMap[{tile.colIndex(), tile.rowIndex()}];
    auto pktInfoAttr = PacketInfoAttr::get(tile->getContext(), 0, pktId);
    tile->setAttr("controller_id", pktInfoAttr);
  }

  uint32_t computeBDAddress(int col, int bdId, TileOp shimTile,
                            const AIETargetModel &tm) {
    int row = shimTile.getRow();
    return tm.getDmaBdAddress(col, row, bdId) +
           tm.getDmaBdAddressOffset(col, row);
  }

  // Compute DMA task queue address
  uint32_t computeTaskQueueAddress(DMAChannelDir dir, int channel,
                                   TileOp shimTile, const AIETargetModel &tm) {
    std::string regName;
    if (dir == DMAChannelDir::S2MM) {
      regName =
          (channel == 0) ? "DMA_S2MM_0_Task_Queue" : "DMA_S2MM_1_Task_Queue";
    } else { // MM2S
      regName =
          (channel == 0) ? "DMA_MM2S_0_Task_Queue" : "DMA_MM2S_1_Task_Queue";
    }
    const RegisterInfo *reg = tm.lookupRegister(regName, shimTile.getTileID());
    if (!reg)
      llvm::report_fatal_error(llvm::Twine("Failed to lookup ") + regName);
    return reg->offset;
  }

  // Compute DMA control register address
  uint32_t computeCtrlAddress(DMAChannelDir dir, int channel, TileOp shimTile,
                              const AIETargetModel &tm) {
    std::string regName;
    if (dir == DMAChannelDir::S2MM) {
      regName = (channel == 0) ? "DMA_S2MM_0_Ctrl" : "DMA_S2MM_1_Ctrl";
    } else { // MM2S
      regName = (channel == 0) ? "DMA_MM2S_0_Ctrl" : "DMA_MM2S_1_Ctrl";
    }
    const RegisterInfo *reg = tm.lookupRegister(regName, shimTile.getTileID());
    if (!reg)
      llvm::report_fatal_error(llvm::Twine("Failed to lookup ") + regName);
    return reg->offset;
  }

  // Compute timer control address based on tile type
  uint32_t computeTimerCtrlAddress(TileOp tile, const AIETargetModel &tm,
                                   bool isMemTrace) {
    // Use register database to lookup Timer_Control for the appropriate module
    const RegisterInfo *reg =
        tm.lookupRegister("Timer_Control", tile.getTileID(), isMemTrace);
    if (!reg)
      llvm::report_fatal_error("Failed to lookup Timer_Control register");
    return reg->offset;
  }

  /// Build channel descriptors. Always includes the primary channel.
  /// Adds a secondary channel when distribute-channels is enabled and there
  /// are multiple traces. AIE2 shim tiles have exactly 2 S2MM DMA channels.
  ///
  /// Both channels share the same arg_idx (XRT buffer). The buffer is split
  /// by offset: channel 0 starts at the base bufferOffset, channel 1 starts
  /// at bufferOffset + bufferSizeBytes. The host must allocate a trace buffer
  /// of 2 * bufferSizeBytes when distribute is active.
  std::vector<ChannelDescriptor>
  buildChannelDescriptors(size_t numTraces, int primaryChannel, int primaryBdId,
                          int primaryArgIdx, int baseBufferOffset,
                          int bufferSizeBytes,
                          const std::set<int> &availableChannels) {
    std::vector<ChannelDescriptor> chans;
    chans.push_back(
        {primaryChannel, primaryBdId, primaryArgIdx, baseBufferOffset});
    if (clDistributeChannels && numTraces > 1 && primaryBdId > 0) {
      int ch2 = (primaryChannel == 1) ? 0 : 1;
      // Only add secondary channel if it's available (not claimed by existing
      // flows)
      if (availableChannels.count(ch2)) {
        chans.push_back({ch2, primaryBdId - 1, primaryArgIdx,
                         baseBufferOffset + bufferSizeBytes});
      }
    }
    return chans;
  }

  /// Find or create a shim tile at the given column.
  TileOp getOrCreateShim(DeviceOp device, OpBuilder &builder, int col) {
    for (auto tile : device.getOps<TileOp>()) {
      if (tile.getCol() == col && tile.getRow() == 0)
        return tile;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&device.getRegion().front());
    return TileOp::create(builder, device.getLoc(), col, 0);
  }

  /// Scan the device for existing S2MM channel claims on shim tiles.
  /// Checks aie.flow destinations, aie.packet_flow destinations, and
  /// ShimDMAAllocationOp declarations.
  std::map<int, std::set<int>> scanUsedS2MMChannels(DeviceOp device) {
    std::map<int, std::set<int>> used; // shimCol -> set of used S2MM channels

    device.walk([&](FlowOp flow) {
      auto destTile = cast<TileOp>(flow.getDest().getDefiningOp());
      if (destTile.isShimTile() && flow.getDestBundle() == WireBundle::DMA) {
        used[destTile.getCol()].insert(flow.getDestChannel());
      }
    });

    device.walk([&](PacketFlowOp pktFlow) {
      for (auto &op : pktFlow.getPorts().front()) {
        if (auto dest = dyn_cast<PacketDestOp>(op)) {
          auto destTile = cast<TileOp>(dest.getTile().getDefiningOp());
          if (destTile.isShimTile() && dest.getBundle() == WireBundle::DMA) {
            used[destTile.getCol()].insert(dest.getChannel());
          }
        }
      }
    });

    device.walk([&](ShimDMAAllocationOp alloc) {
      if (alloc.getChannelDir() == DMAChannelDir::S2MM) {
        auto tile = alloc.getTileOp();
        used[tile.getCol()].insert(alloc.getChannelIndex());
      }
    });

    return used;
  }

  /// Find the nearest NOC shim column without active cores.
  /// Returns -1 if no spare column exists.
  int findNearestSpareColumn(int sourceCol, const std::set<int> &activeColumns,
                             const AIETargetModel &targetModel) {
    int bestCol = -1;
    int bestDist = INT_MAX;
    int numCols = targetModel.columns();
    for (int c = 0; c < numCols; c++) {
      if (activeColumns.count(c) == 0 && targetModel.isShimNOCTile(c, 0)) {
        int dist = std::abs(c - sourceCol);
        if (dist > 0 && dist < bestDist) {
          bestDist = dist;
          bestCol = c;
        }
      }
    }
    return bestCol;
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEInsertTraceFlowsPass() {
  return std::make_unique<AIEInsertTraceFlowsPass>();
}
