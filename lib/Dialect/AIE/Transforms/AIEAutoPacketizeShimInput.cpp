//===- AIEAutoPacketizeShimInput.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEAUTOPACKETIZESHIMINPUT
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

// The shim MM2S channels one device of the module uses in one column.
struct DeviceColumn {
  SmallVector<ObjectFifoCreateOp> inputs; // shim-produced objectFIFOs, by name
  llvm::SmallSet<int, 4> declared;        // channels flows and shim muxes name
  llvm::SmallSet<int, 4> circuit;         // ... of which circuit-switched
};

// One column across every device of the module.
struct Column {
  int numChans = 0;
  llvm::MapVector<DeviceOp, DeviceColumn> devices;
};

std::optional<int> shimColumn(Value tile) {
  auto tileOp = dyn_cast_or_null<TileOp>(tile.getDefiningOp());
  if (!tileOp || !tileOp.isShimNOCorPLTile())
    return std::nullopt;
  return tileOp.colIndex();
}

// Element bytes times producer depth.
int64_t trafficProxy(ObjectFifoCreateOp fifo) {
  auto fifoType = dyn_cast<AIEObjectFifoType>(fifo.getElemType());
  if (!fifoType)
    return 0;
  MemRefType memref = fifoType.getElementType();
  DataLayout layout(fifo->getParentOfType<ModuleOp>());
  uint64_t bits = layout.getTypeSizeInBits(memref.getElementType());
  int64_t depth = 1;
  if (auto depths = dyn_cast<ArrayAttr>(fifo.getElemNumber()))
    depth = depths.empty() ? 1 : cast<IntegerAttr>(depths[0]).getInt();
  else if (auto scalar = dyn_cast<IntegerAttr>(fifo.getElemNumber()))
    depth = scalar.getInt();
  return memref.getNumElements() * (int64_t)llvm::divideCeil(bits, 8) * depth;
}

// Moves the input's consumers behind a memory-tile relay, so its data is
// packet-switched only up to the memory tile. An input that already ends at a
// memory tile, or whose consumer-side layout the relay would have to
// redistribute, stays as is.
void relayThroughMemTile(DeviceOp device, ObjectFifoCreateOp input) {
  SmallVector<Value> consumers(input.getConsumerTiles());
  auto first = dyn_cast_or_null<TileOp>(consumers.front().getDefiningOp());
  if (!first || first.isMemTile() || isa<ArrayAttr>(input.getElemNumber()) ||
      input.getConsumerElemType() || input.getConsDmaChannels() ||
      input.getAieStream() || input.getRepeatCount() || input.getIterCount() ||
      input.getPadDimensions() || input.getVia_DMA() || input.getPlio())
    return;
  for (BDDimLayoutArrayAttr dims : input.getDimensionsFromStreamPerConsumer())
    if (!dims.empty())
      return;
  const AIETargetModel &target = device.getTargetModel();
  int col = first.colIndex();
  int memRow = -1;
  for (int row = 0; row < target.rows() && memRow < 0; ++row)
    if (target.isMemTile(col, row))
      memRow = row;
  if (memRow < 0)
    return;

  MLIRContext *ctx = input.getContext();
  OpBuilder builder(input);
  TileOp memTile = TileOp::getOrCreate(builder, device, col, memRow);
  std::string relayName = (input.getSymName() + "_relay").str();
  builder.setInsertionPointAfter(input);
  ObjectFifoCreateOp::create(builder, input.getLoc(),
                             builder.getStringAttr(relayName),
                             memTile.getResult(), consumers,
                             input.getElemNumberAttr(), input.getElemType());
  input.getConsumerTilesMutable().assign(memTile.getResult());
  ObjectFifoLinkOp::create(
      builder, input.getLoc(),
      builder.getArrayAttr({FlatSymbolRefAttr::get(input.getSymNameAttr())}),
      builder.getArrayAttr({FlatSymbolRefAttr::get(ctx, relayName)}),
      builder.getI64ArrayAttr({}), builder.getI64ArrayAttr({0}));

  auto relay = FlatSymbolRefAttr::get(ctx, relayName);
  device.walk([&](Operation *op) {
    if (auto acquire = dyn_cast<ObjectFifoAcquireOp>(op)) {
      if (acquire.getObjFifoName() == input.getSymName())
        acquire.setObjFifoNameAttr(relay);
    } else if (auto release = dyn_cast<ObjectFifoReleaseOp>(op)) {
      if (release.getObjFifoName() == input.getSymName())
        release.setObjFifoNameAttr(relay);
    }
  });
}

struct AIEAutoPacketizeShimInputPass
    : xilinx::AIE::impl::AIEAutoPacketizeShimInputBase<
          AIEAutoPacketizeShimInputPass> {
  // Gathers, per column and device, the objectFIFO inputs and the shim MM2S
  // channels the device's flows and shim muxes already use.
  static void collect(ModuleOp module, llvm::MapVector<int, Column> &columns) {
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      const AIETargetModel &target = device.getTargetModel();
      // aie-generate-column-control-overlay adds no overlay on AIE1.
      if (target.getTargetArch() == AIEArch::AIE1)
        continue;
      auto columnOf = [&](int col) -> DeviceColumn & {
        Column &column = columns[col];
        column.numChans =
            target.getNumSourceShimMuxConnections(col, 0, WireBundle::DMA);
        return column.devices[device];
      };
      for (auto fifo : device.getOps<ObjectFifoCreateOp>())
        if (auto col = shimColumn(fifo.getProducerTile()))
          columnOf(*col).inputs.push_back(fifo);
      for (auto flow : device.getOps<FlowOp>())
        if (flow.getSourceBundle() == WireBundle::DMA)
          if (auto col = shimColumn(flow.getSource())) {
            columnOf(*col).declared.insert(flow.getSourceChannel());
            columnOf(*col).circuit.insert(flow.getSourceChannel());
          }
      for (auto flow : device.getOps<PacketFlowOp>()) {
        if (llvm::all_of(flow.getOps<PacketDestOp>(), [](PacketDestOp dest) {
              return dest.getBundle() == WireBundle::TileControl;
            }))
          continue;
        for (auto src : flow.getOps<PacketSourceOp>())
          if (src.getBundle() == WireBundle::DMA)
            if (auto col = shimColumn(src.getTile()))
              columnOf(*col).declared.insert(src.channelIndex());
      }
      for (auto mux : device.getOps<ShimMuxOp>())
        for (auto connect : mux.getConnections().front().getOps<ConnectOp>())
          if (connect.getSourceBundle() == WireBundle::DMA) {
            DeviceColumn &dc = columnOf(mux.getTileOp().colIndex());
            dc.declared.insert(connect.sourceIndex());
            dc.circuit.insert(connect.sourceIndex());
          }
    }
    for (auto &[col, column] : columns)
      for (auto &[device, dc] : column.devices) {
        llvm::sort(dc.inputs, [](ObjectFifoCreateOp a, ObjectFifoCreateOp b) {
          return a.getSymName() < b.getSymName();
        });
        for (auto input : dc.inputs)
          if (auto pin = input.getProdDmaChannel())
            if (!input.getPacket())
              dc.circuit.insert(*pin);
      }
  }

  // As objectFIFO allocation gives them out: a pinned input keeps its pin, and
  // each other input takes the lowest channel not yet used.
  static llvm::SmallSet<int, 4> allocatedChannels(const Column &column) {
    llvm::SmallSet<int, 4> all;
    for (auto &[device, dc] : column.devices) {
      llvm::SmallSet<int, 4> used = dc.declared;
      for (auto input : dc.inputs)
        if (auto pin = input.getProdDmaChannel())
          used.insert(*pin);
      for (auto input : dc.inputs) {
        if (input.getProdDmaChannel())
          continue;
        for (int chan = 0; chan < column.numChans; ++chan)
          if (used.insert(chan).second)
            break;
      }
      all.insert(used.begin(), used.end());
    }
    return all;
  }

  // The control channels the control-channels option requests, by column.
  LogicalResult parseControlChannels(llvm::DenseMap<int, int> &requested) {
    for (StringRef entry : clControlChannels) {
      auto [colText, chanText] = entry.split(':');
      int col, chan;
      if (colText.trim().getAsInteger(10, col) ||
          chanText.trim().getAsInteger(10, chan))
        return getOperation()->emitError()
               << "control-channels entry '" << entry
               << "' is not <column>:<channel>";
      requested[col] = chan;
    }
    return success();
  }

  // Gives the device the overlay's `ctrlpkt_*` allocation of the column's
  // control channel, marked is_ctrl_pkt_overlay, which objectFIFO lowering and
  // the overlay then use.
  static void declareControlChannel(DeviceOp device, int col, int chan) {
    if (getCtrlPktShimAlloc(device, col))
      return;
    auto shim = llvm::find_if(device.getOps<TileOp>(), [&](TileOp tile) {
      return tile.colIndex() == col && tile.isShimNOCorPLTile();
    });
    if (shim == device.getOps<TileOp>().end())
      return;
    std::string name = getCtrlPktAllocName(col, /*isShimMM2S=*/true, chan);
    Operation *alloc = device.lookupSymbol(name);
    if (!alloc) {
      OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
      alloc = ShimDMAAllocationOp::create(builder, (*shim).getLoc(), name,
                                          (*shim).getResult(),
                                          DMAChannelDir::MM2S, chan,
                                          /*plio=*/false, nullptr);
    }
    AIEDialect::IsCtrlPktOverlayAttrHelper(device.getContext())
        .setAttr(alloc, UnitAttr::get(device.getContext()));
  }

  // Pins one device's inputs of the column: one shares the control channel,
  // made packet-switched if need be, and the others take the other channels.
  LogicalResult assignChannels(DeviceOp device, int col, const Column &column,
                               DeviceColumn &dc, int controlChan) {
    SmallVector<bool> taken(column.numChans, false);
    taken[controlChan] = true;
    SmallVector<ObjectFifoCreateOp> unpinned;
    bool controlChanHasInput = false;
    for (auto input : dc.inputs) {
      if (auto pin = input.getProdDmaChannel()) {
        controlChanHasInput |= (int)*pin == controlChan;
        if ((int)*pin < column.numChans)
          taken[*pin] = true;
      } else {
        unpinned.push_back(input);
      }
    }
    // The channels the given inputs take beside the pinned ones, in order; -1
    // for an input left without one.
    auto place = [&](ArrayRef<ObjectFifoCreateOp> inputs) {
      SmallVector<bool> on = taken;
      SmallVector<int> chans;
      for (auto input : inputs) {
        int chan = 0;
        while (chan < column.numChans &&
               (on[chan] ||
                (input.getPacket() ? dc.circuit : dc.declared).contains(chan)))
          ++chan;
        chans.push_back(chan < column.numChans ? chan : -1);
        if (chan < column.numChans)
          on[chan] = true;
      }
      return chans;
    };
    if (!controlChanHasInput && !unpinned.empty()) {
      auto packet = llvm::find_if(
          unpinned, [](ObjectFifoCreateOp input) { return input.getPacket(); });
      ObjectFifoCreateOp chosen;
      if (packet != unpinned.end()) {
        chosen = *packet;
      } else if (llvm::is_contained(place(unpinned), -1)) {
        // The fewest consumers, then the least traffic, then the last name.
        auto key = [](ObjectFifoCreateOp input) {
          return std::make_pair(input.getConsumerTiles().size(),
                                trafficProxy(input));
        };
        chosen = unpinned.front();
        for (auto input : unpinned)
          if (key(input) <= key(chosen))
            chosen = input;
        chosen.setPacket(true);
        chosen.emitRemark() << "auto-packetized for control on shim MM2S "
                            << "channel " << controlChan << " of column " << col
                            << ": the design's inputs leave no other channel "
                            << "free for it";
        relayThroughMemTile(device, chosen);
      }
      if (chosen) {
        chosen.setProdDmaChannel(controlChan);
        llvm::erase(unpinned, chosen);
      }
    }
    for (auto [input, chan] : llvm::zip_equal(unpinned, place(unpinned))) {
      if (chan < 0)
        return input.emitError()
               << "column " << col << " has no shim MM2S channel left for this "
               << "objectFIFO beside the control packets on channel "
               << controlChan;
      input.setProdDmaChannel(chan);
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::DenseMap<int, int> requested;
    if (failed(parseControlChannels(requested)))
      return signalPassFailure();
    llvm::MapVector<int, Column> columns;
    collect(module, columns);

    for (auto &[col, column] : columns) {
      bool hasInputs = llvm::any_of(column.devices, [](auto &entry) {
        return !entry.second.inputs.empty();
      });
      // A column with a channel no device uses gives control that channel.
      if (column.numChans < 2 || !hasInputs ||
          (int)allocatedChannels(column).size() < column.numChans) {
        if (!requested.count(col))
          continue;
        module.emitError() << "control-channels names column " << col
                           << ", which leaves a channel free for control";
        return signalPassFailure();
      }
      llvm::SmallSet<int, 4> circuit;
      for (auto &[device, dc] : column.devices)
        circuit.insert(dc.circuit.begin(), dc.circuit.end());
      int controlChan = -1;
      if (auto it = requested.find(col); it != requested.end()) {
        if (it->second >= 0 && it->second < column.numChans &&
            !circuit.contains(it->second))
          controlChan = it->second;
      } else {
        for (int chan = 0; chan < column.numChans && controlChan < 0; ++chan)
          if (!circuit.contains(chan))
            controlChan = chan;
      }
      if (controlChan < 0) {
        column.devices.front().first.emitError()
            << "column " << col << " has no shim MM2S channel control packets "
            << "can share: "
            << (requested.count(col) ? "the requested one is out of range or "
                                       "carries circuit-switched data"
                                     : "every channel carries circuit-switched "
                                       "data (a flow, a shim mux or a pinned "
                                       "circuit objectFIFO) in some device");
        return signalPassFailure();
      }
      for (auto &[device, dc] : column.devices) {
        if (failed(assignChannels(device, col, column, dc, controlChan)))
          return signalPassFailure();
        declareControlChannel(device, col, controlChan);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEAutoPacketizeShimInputPass() {
  return std::make_unique<AIEAutoPacketizeShimInputPass>();
}
