//===- AIEGenerateColumnControlOverlay.cpp ----------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include <map>
#include <set>
#include <utility>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEGENERATECOLUMNCONTROLOVERLAY
#define GEN_PASS_DEF_AIEASSIGNTILECTRLIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-generate-column-control-overlay"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Delegate to AIETargetModel::getTileToControllerIdMap.
DenseMap<AIE::TileID, int>
getTileToControllerIdMap(bool clColumnWiseUniqueIDs,
                         const AIETargetModel &targetModel) {
  return targetModel.getTileToControllerIdMap(clColumnWiseUniqueIDs);
}

// AIE arch-specific row id to shim dma mm2s channel mapping. All shim mm2s
// channels were assumed to be available for control packet flow routing (i.e.
// not reserved by any aie.flow circuit-switched routing).
DenseMap<int, int> getRowToShimChanMap(const AIETargetModel &targetModel,
                                       WireBundle bundle) {
  DenseMap<int, int> rowMap;
  SmallVector<int>
      thresholdsToNextShimChannel; // a list of thresholds on the number of
                                   // control ports that the ith shim channel
                                   // could connect to, before advancing to
                                   // the next shim channel in round robin
  TileID shimTile = {0, 0};
  while (!targetModel.isShimNOCTile(shimTile.col, shimTile.row)) {
    shimTile.col++;
    if (shimTile.col == targetModel.columns()) {
      shimTile.col = 0;
      shimTile.row++;
    }
    assert(shimTile.col != targetModel.columns() ||
           shimTile.row != targetModel.rows());
  }

  int numShimChans = targetModel.getNumSourceShimMuxConnections(
      shimTile.col, shimTile.row, AIE::WireBundle::DMA);
  for (int i = 1; i < numShimChans + 1; i++)
    thresholdsToNextShimChannel.push_back(targetModel.rows() / numShimChans *
                                          i);

  if (bundle == WireBundle::DMA) { // Ctrl packets
    int shimChanIdx = 0;
    for (int r = 0; r < targetModel.rows(); r++) {
      if (r >= thresholdsToNextShimChannel[shimChanIdx])
        shimChanIdx++;
      rowMap[r] = shimChanIdx;
    }
  } else if (bundle == WireBundle::South) { // TCT
    for (int r = 0; r < targetModel.rows(); r++)
      rowMap[r] = 0;
  }

  return rowMap;
}

struct AIEAssignTileCtrlIDsPass
    : xilinx::AIE::impl::AIEAssignTileCtrlIDsBase<AIEAssignTileCtrlIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    // Collect all TileOps in columns occupied by the design.
    llvm::MapVector<AIE::TileID, AIE::TileOp> tiles;
    llvm::SmallSet<int, 1> occupiedCols;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[{colIndex, rowIndex}] = tile;
      occupiedCols.insert(colIndex);
    }

    auto tileIDMap =
        getTileToControllerIdMap(clColumnWiseUniqueIDs, targetModel);
    for (int col : occupiedCols) {
      SmallVector<AIE::TileOp> tilesOnCol;
      for (auto &[tId, tOp] : tiles) {
        if (tId.col != col)
          continue;
        tilesOnCol.push_back(tOp);
      }

      for (auto tOp : tilesOnCol) {
        if (tOp->hasAttr("controller_id"))
          continue;
        auto pktInfoAttr = AIE::PacketInfoAttr::get(
            tOp->getContext(), /*pkt_type*/ 0,
            /*pkt_id*/ tileIDMap[{tOp.colIndex(), tOp.rowIndex()}]);
        tOp->setAttr("controller_id", pktInfoAttr);
      }
    }
  }
};

struct AIEGenerateColumnControlOverlayPass
    : xilinx::AIE::impl::AIEGenerateColumnControlOverlayBase<
          AIEGenerateColumnControlOverlayPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  // (col, MM2S chan) -> symbol name of the data aie.shim_dma_allocation that
  // already claims it, across every participating device in the module (not
  // just the device currently being processed). Rebuilt at the start of
  // every runOnOperation() call; see the comment where it's populated.
  std::map<std::pair<int, int>, StringRef> moduleDataAllocByColChan;

  // (col, MM2S chan) reserved by a circuit-switched aie.flow / aie.packet_flow
  // on ANY participating device. Control ingress must avoid these (a circuit
  // monopolizes the physical channel; it cannot time-share with control the
  // way a data shim_dma_allocation can). Built module-wide -- not per device --
  // so the standalone `@ctrl_pkt_overlay` device and every config device pick
  // the SAME channel for a controlled tile; a per-device scan would let a
  // config that carries the circuit relocate while a sibling that does not
  // keeps the mandated channel, so the resident overlay and config delivery
  // would target different channels and wedge. Rebuilt each runOnOperation().
  std::set<std::pair<int, int>> moduleCircuitOccupiedByColChan;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Gather source devices in module order. Skip a previously-generated
    // overlay device so the pass is idempotent on its own output.
    SmallVector<DeviceOp> sourceDevices;
    for (auto dev : module.getOps<DeviceOp>()) {
      if (dev.getSymName() == "ctrl_pkt_overlay")
        continue;
      sourceDevices.push_back(dev);
    }

    // Devices that receive the overlay: those that have not opted out via
    // `needs_ctrl_pkt_overlay = false`.
    SmallVector<DeviceOp> participating;
    for (auto dev : sourceDevices) {
      if (deviceOptedOut(dev)) {
        if (clEmitStandaloneOverlay) {
          dev->setAttr("has_ctrl_pkt_overlay", builder.getBoolAttr(false));
        }
        continue;
      }
      participating.push_back(dev);
    }

    // Module-wide index of shim MM2S channels already claimed by a design's
    // own data aie.shim_dma_allocation, keyed by (col, chan). A baked multi-
    // config overlay module (aiecc's unionConfigDesigns, used by
    // --reconfig-method) keeps the control-issuing host device (no
    // tiles yet at this point in the pipeline) separate from each per-config
    // data-plane device; a later pass hoists each config's data allocation
    // into the host device so its own dma_memcpy_nd's `metadata` resolves,
    // but that hoist runs AFTER this pass. So a per-device SSA scan alone
    // (getAvailableShimChans) cannot see a sibling device's claim on the
    // physical shim channel it's about to route control onto. Built ONCE,
    // here, from the ORIGINAL participating devices before any device is
    // touched, so freshly-created ctrlpkt_... allocations never pollute it.
    moduleDataAllocByColChan.clear();
    for (auto dev : participating)
      for (auto allocOp : dev.getOps<AIE::ShimDMAAllocationOp>())
        if (allocOp.getChannelDir() == AIE::DMAChannelDir::MM2S)
          moduleDataAllocByColChan[{allocOp.getTileOp().colIndex(),
                                    (int)allocOp.getChannelIndex()}] =
              allocOp.getSymName();

    // Module-wide index of shim MM2S channels reserved by circuit-switched
    // routing, unioned across all participating devices (see the member's
    // comment for why module-wide). A channel is circuit-occupied iff a
    // circuit aie.flow reserves it, regardless of a co-emitted data
    // aie.shim_dma_allocation (a circuit objectFifo shim input emits both;
    // the alloc must not mask the circuit reservation, else control lands on
    // a circuit-mode slave port -- invalid, single SlvPktEn bit).
    moduleCircuitOccupiedByColChan.clear();
    for (auto dev : participating) {
      const auto &tm = dev.getTargetModel();
      for (auto tile : dev.getOps<AIE::TileOp>()) {
        if (!tm.isShimNOCTile(tile.colIndex(), tile.rowIndex()))
          continue;
        auto avail = getAvailableShimChans(dev, tile, WireBundle::DMA,
                                           /*isShimMM2S=*/true);
        int numChans = tm.getNumSourceShimMuxConnections(
            tile.colIndex(), tile.rowIndex(), WireBundle::DMA);
        for (int c = 0; c < numChans; c++) {
          bool circuit = avail.occupiedByCircuit.count(c) > 0;
          if (circuit)
            moduleCircuitOccupiedByColChan.insert({tile.colIndex(), c});
        }
      }
    }

    // A standalone `@ctrl_pkt_overlay` device references a single overlay
    // shape, so every participating device must expose the same set of tiles
    // for that shape to be identical across them.
    if (clEmitStandaloneOverlay)
      shareTilesAcrossDevices(participating);

    // Apply the overlay in-place to participating devices.
    for (auto dev : participating) {
      if (failed(applyOverlayToDevice(dev)))
        return signalPassFailure();
      if (clEmitStandaloneOverlay)
        dev->setAttr("has_ctrl_pkt_overlay", builder.getBoolAttr(true));
    }

    // Emit standalone `@ctrl_pkt_overlay` device.
    if (clEmitStandaloneOverlay) {
      if (failed(createOverlayDevice(module, builder, participating))) {
        return signalPassFailure();
      }
    }
  }

  // Collect the union of tiles across `devices`, recording one prototype
  // TileOp per tile so its attributes can be copied when the tile is cloned.
  static void
  collectTileUnion(ArrayRef<DeviceOp> devices,
                   llvm::SmallSetVector<AIE::TileID, 8> &unionTiles,
                   llvm::DenseMap<AIE::TileID, AIE::TileOp> &prototypeTile) {
    for (auto dev : devices)
      for (auto tOp : dev.getOps<AIE::TileOp>()) {
        AIE::TileID id{tOp.colIndex(), tOp.rowIndex()};
        unionTiles.insert(id);
        if (!prototypeTile.contains(id))
          prototypeTile[id] = tOp;
      }
  }

  // Clone every tile in `unionTiles` not already present in `device`, copying
  // the prototype's attributes so downstream passes that compare attribute
  // dictionaries (e.g. AIEMaterializeRuntimeSequences) match.
  static void cloneMissingTiles(
      DeviceOp device, const llvm::SmallSetVector<AIE::TileID, 8> &unionTiles,
      const llvm::DenseMap<AIE::TileID, AIE::TileOp> &prototypeTile) {
    llvm::SmallSet<AIE::TileID, 8> existing;
    for (auto tOp : device.getOps<AIE::TileOp>())
      existing.insert({tOp.colIndex(), tOp.rowIndex()});
    OpBuilder b = OpBuilder::atBlockBegin(device.getBody());
    for (auto id : unionTiles) {
      if (existing.contains(id))
        continue;
      b.clone(*prototypeTile.lookup(id).getOperation());
    }
  }

  // Give every device in `devices` the union of their tiles, so an overlay
  // routed onto any of them has the same shape (routes, shim_dma_allocations).
  static void shareTilesAcrossDevices(ArrayRef<DeviceOp> devices) {
    llvm::SmallSetVector<AIE::TileID, 8> unionTiles;
    llvm::DenseMap<AIE::TileID, AIE::TileOp> prototypeTile;
    collectTileUnion(devices, unionTiles, prototypeTile);
    for (auto dev : devices)
      cloneMissingTiles(dev, unionTiles, prototypeTile);
  }

  // Emit a standalone `@ctrl_pkt_overlay` device holding only the overlay and
  // the union of tiles it references. Downstream consumers compile it on its
  // own to ship a reconfigure-only PDI.
  LogicalResult createOverlayDevice(ModuleOp module, OpBuilder &builder,
                                    ArrayRef<DeviceOp> participating) {
    if (participating.empty())
      return success();

    // All participating devices must share the same target.
    DeviceOp firstDev = participating.front();
    auto refDevice = firstDev.getDevice();
    for (auto dev : llvm::drop_begin(participating)) {
      if (dev.getDevice() != refDevice) {
        return dev->emitOpError(
            "cannot generate a single standalone ctrl_pkt_overlay device: "
            "participating devices have mismatched target architectures.");
      }
    }

    if (module.lookupSymbol("ctrl_pkt_overlay")) {
      return module.emitOpError(
          "a symbol named `ctrl_pkt_overlay` already exists in the module; "
          "cannot create a standalone ctrl_pkt_overlay device.");
    }

    builder.setInsertionPointToEnd(module.getBody());
    Location loc = firstDev.getLoc();
    auto overlayDevice = AIE::DeviceOp::create(
        builder, loc, refDevice, builder.getStringAttr("ctrl_pkt_overlay"));
    overlayDevice.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&overlayDevice.getRegion().front());
    AIE::EndOp::create(builder, loc);

    // Populate the overlay device with the union of tiles referenced across
    // participating devices, then route the overlay onto it.
    llvm::SmallSetVector<AIE::TileID, 8> unionTiles;
    llvm::DenseMap<AIE::TileID, AIE::TileOp> prototypeTile;
    collectTileUnion(participating, unionTiles, prototypeTile);
    cloneMissingTiles(overlayDevice, unionTiles, prototypeTile);

    // allowCrossDeviceDataShare=false: this standalone device is a bare
    // routing skeleton -- createOverlayDevice only clones TILES from
    // `participating` (see cloneMissingTiles), never their data
    // aie.shim_dma_allocations, so it can never locally carry the shared
    // symbol moduleDataAllocByColChan would point it at. It is also the
    // device baked as `main:init`'s resident PDI, which -- unlike the
    // control-issuing host device that later gets a config's data allocation
    // hoisted into it -- needs its OWN dedicated ctrlpkt_... allocation to
    // correctly bring up the shim DMA queue for a channel it routes control
    // over, even when that same physical channel is shared with data
    // elsewhere in the module.
    if (failed(applyOverlayToDevice(overlayDevice,
                                    /*allowCrossDeviceDataShare=*/false)))
      return failure();

    overlayDevice->setAttr("has_ctrl_pkt_overlay", builder.getBoolAttr(true));
    return success();
  }

  // Apply the column-control overlay to `device` in place. Returns failure on
  // a routing conflict. `allowCrossDeviceDataShare` gates whether a shim
  // channel already claimed by a data aie.shim_dma_allocation on a SIBLING
  // device (moduleDataAllocByColChan) may be shared with control on THIS
  // device; false for the standalone `@ctrl_pkt_overlay` device (see its call
  // site in createOverlayDevice).
  LogicalResult applyOverlayToDevice(DeviceOp device,
                                     bool allowCrossDeviceDataShare = true) {
    const auto &targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return success(); // Disable this pass for AIE1; AIE1 support NYI.

    // Idempotency: a device may already carry this overlay when the pass is
    // applied a second time -- the reconfigure flow pre-applies the overlay
    // with `aie-opt -aie-generate-column-control-overlay` and then runs aiecc,
    // whose input pipeline runs this pass again
    // (test/npu-xrt/ctrl_packet_reconfig, test/aiecc/cpp_ctrlpkt). Re-applying
    // would double the control flows and, for a whole-array column with no data
    // allocation to share, hit the channel-already-reserved check below. A
    // control route to a tile's TileControl port is unique to this overlay (no
    // other pass emits one before it), so its presence marks an
    // already-overlaid device: skip re-applying.
    for (auto pktFlow : device.getOps<AIE::PacketFlowOp>())
      for (auto destOp : pktFlow.getOps<AIE::PacketDestOp>())
        if (destOp.getBundle() == WireBundle::TileControl)
          return success();

    // Collect existing TileOps
    llvm::MapVector<AIE::TileID, AIE::TileOp> tiles;
    for (auto tile : device.getOps<AIE::TileOp>())
      tiles[{tile.colIndex(), tile.rowIndex()}] = tile;
    if (tiles.empty())
      return success();

    // Both widenings below are scoped to the control-packet configuration path
    // (`route-shim-to-tile-ctrl`); do not add tile declarations if the control
    // overlay was not requested, as to not congest routing needlessly.
    SmallVector<int> colsToCover;
    if (clRouteShimDmaToTileCTRL && clWholeArrayControlCoverage) {
      // Whole-array column coverage: cover every physical column of the device,
      // not just the occupied bounding box. The pathfinder routes config data
      // flows AFTER this pass runs, and it spills relay switchboxes into
      // columns outside the occupied span as a congestion detour (e.g. a
      // col0-only design whose vertical spine is full relays a flow out through
      // col1 and back). Those relays are reconfigured by control packets, so
      // they need a control route + a shim DMA allocation + a controller_id;
      // covering only minOccupiedCol..maxOccupiedCol left the spill column
      // bare, so its control ingress resolved to a dangling ctrlpkt_col<N>_...
      // symbol at AICtrlPacketToDma. This is the column analogue of the
      // whole-array ROW coverage below. A design that never routes into a
      // column simply leaves its overlay control routes idle. On npu2 every
      // column's row-0 tile is a ShimNOC tile, so each covered column can host
      // control ingress.
      for (int col = 0; col < targetModel.columns(); col++)
        colsToCover.push_back(col);
    } else {
      // Occupied-column coverage. The default path for a non-control build;
      // also the whole-array-control-coverage=false ablation arm, which
      // reproduces the relay-spill no-route when data detours into an uncovered
      // column.
      for (auto &[tId, tOp] : tiles)
        if (!llvm::is_contained(colsToCover, tId.col))
          colsToCover.push_back(tId.col);
    }

    auto tileIDMap = getTileToControllerIdMap(true, targetModel);
    for (int col : colsToCover) {
      builder.setInsertionPointToStart(device.getBody());
      AIE::TileOp shimTile = TileOp::getOrCreate(builder, device, col, 0);
      if (clRouteShimDmaToTileCTRL)
        tiles[{col, 0}] = shimTile;

      if (clRouteShimCTRLToTCT == "all-tiles" ||
          clRouteShimCTRLToTCT == "shim-only") {
        // Get all tile ops on column col
        SmallVector<AIE::TileOp> tilesOnCol;
        for (auto &[tId, tOp] : tiles) {
          if (tId.col != col)
            continue;
          if (clRouteShimCTRLToTCT == "shim-only" && !tOp.isShimNOCorPLTile())
            continue;
          tilesOnCol.push_back(tOp);
        }

        if (failed(generatePacketFlowsForControl(
                builder, device, shimTile, AIE::WireBundle::South, tilesOnCol,
                AIE::WireBundle::TileControl, 0, tileIDMap, false,
                allowCrossDeviceDataShare)))
          return failure();
      }
      if (clRouteShimDmaToTileCTRL) {
        // Cover every row up to the tallest occupied row, not just the rows the
        // design declared in this column. Pathfinding routes data flows THROUGH
        // shim-input and pass-through columns, spilling relay switchboxes into
        // their upper rows; those relays are reconfigured by control packets,
        // so they need control routes and controller_ids too. Covering only a
        // column's declared rows left a shim-input column (declared shim at row
        // 0) at row 0, so its relay tiles were fabricated bare when
        // control-packet headers were baked (no controller_id -> build
        // failure).
        //
        // The cap is the full physical height (rows()-1), not the tallest
        // occupied row: the router graph spans all physical rows, so a relay
        // CAN in principle land above the occupied span (an asymmetry with the
        // whole-array COLUMN coverage above), and control for a column now
        // rides a single consolidated packet channel (one shim MM2S trunk
        // per column, not one channel per covered row), so covering more
        // rows no longer claims additional shim DMA channels. Broadening to
        // full height was previously unsafe on the 2-channel shim before that
        // consolidation; it is safe now.
        int maxRow = device.getTargetModel().rows() - 1;
        SmallVector<AIE::TileOp> tilesOnCol;
        for (int row = 0; row <= maxRow; row++) {
          auto tOp = TileOp::getOrCreate(builder, device, col, row);
          tilesOnCol.push_back(tOp);
        }

        if (failed(generatePacketFlowsForControl(
                builder, device, shimTile, AIE::WireBundle::DMA, tilesOnCol,
                AIE::WireBundle::TileControl, 0, tileIDMap, true,
                allowCrossDeviceDataShare)))
          return failure();
      }
    }
    return success();
  }

  // Return true when the user has explicitly disabled overlay generation for
  // this device via `needs_ctrl_pkt_overlay = false`.
  static bool deviceOptedOut(DeviceOp device) {
    auto attr = device->getAttrOfType<BoolAttr>("needs_ctrl_pkt_overlay");
    return attr && !attr.getValue();
  }

  AIE::PacketFlowOp createPacketFlowOp(OpBuilder &builder, Location loc,
                                       int &flowID, Value source,
                                       xilinx::AIE::WireBundle sourceBundle,
                                       uint32_t sourceChannel, Value dest,
                                       xilinx::AIE::WireBundle destBundle,
                                       uint32_t destChannel,
                                       mlir::BoolAttr keep_pkt_header = nullptr,
                                       mlir::BoolAttr ctrl_pkt_flow = nullptr) {
    OpBuilder::InsertionGuard guard(builder);

    AIE::PacketFlowOp pktFlow = AIE::PacketFlowOp::create(
        builder, loc, flowID++, keep_pkt_header, ctrl_pkt_flow);
    Region &r_pktFlow = pktFlow.getPorts();
    Block *b_pktFlow = builder.createBlock(&r_pktFlow);
    builder.setInsertionPointToStart(b_pktFlow);
    AIE::PacketSourceOp::create(builder, loc, source, sourceBundle,
                                sourceChannel);
    AIE::PacketDestOp::create(builder, loc, dest, destBundle, destChannel);
    AIE::EndOp::create(builder, loc);
    return pktFlow;
  }

  // Result of a shim-channel occupancy scan: `availableShimChans` lists
  // channels free of any circuit-switched aie.flow or aie.packet_flow (these
  // remain hard-reserved -- sharing is not sound for them); `occupiedByData`
  // maps a channel already claimed by an existing aie.shim_dma_allocation
  // (typically a design's own data DMA) to that allocation's symbol, so
  // control-packet routing can reuse it instead of double-booking the
  // physical channel; `occupiedByCircuit` lists channels reserved by a
  // circuit-switched aie.flow specifically. A circuit objectFifo shim input
  // emits BOTH an aie.flow and an aie.shim_dma_allocation on the same
  // channel, so a channel can appear in both `occupiedByData` and
  // `occupiedByCircuit` -- the circuit reservation must not be masked by the
  // co-present data alloc, else control would land on a circuit-mode slave
  // port (invalid, single SlvPktEn bit).
  struct ShimChanAvailability {
    SmallVector<int> availableShimChans;
    DenseMap<int, StringRef> occupiedByData;
    DenseSet<int> occupiedByCircuit;
  };

  // Scan `shimTile`'s existing users to determine, per shim channel in
  // `shimWireBundle`/`isShimMM2S` direction, whether it is free, reserved by
  // a circuit-switched aie.flow or aie.packet_flow, or already claimed by an
  // aie.shim_dma_allocation that control packets may share.
  ShimChanAvailability getAvailableShimChans(DeviceOp device, TileOp shimTile,
                                             WireBundle shimWireBundle,
                                             bool isShimMM2S) {
    ShimChanAvailability result;
    DenseMap<int, Operation *> reservedChanUsers;
    const auto &targetModel = device.getTargetModel();
    AIE::DMAChannelDir wantDir =
        isShimMM2S ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;

    for (auto *user : shimTile.getResult().getUsers()) {
      if (auto fOp = dyn_cast<AIE::FlowOp>(user)) {
        if (isShimMM2S && fOp.getSource() == shimTile &&
            fOp.getSourceBundle() == shimWireBundle) {
          reservedChanUsers[fOp.getSourceChannel()] = fOp;
          result.occupiedByCircuit.insert(fOp.getSourceChannel());
        } else if (!isShimMM2S && fOp.getDest() == shimTile &&
                   fOp.getDestBundle() == shimWireBundle) {
          reservedChanUsers[fOp.getDestChannel()] = fOp;
          result.occupiedByCircuit.insert(fOp.getDestChannel());
        }
      } else if (auto srcOp = dyn_cast<AIE::PacketSourceOp>(user)) {
        if (isShimMM2S && srcOp.getBundle() == shimWireBundle)
          reservedChanUsers[srcOp.channelIndex()] = srcOp;
      } else if (auto destOp = dyn_cast<AIE::PacketDestOp>(user)) {
        if (!isShimMM2S && destOp.getBundle() == shimWireBundle)
          reservedChanUsers[destOp.channelIndex()] = destOp;
      } else if (auto allocOp = dyn_cast<AIE::ShimDMAAllocationOp>(user)) {
        if (allocOp.getChannelDir() == wantDir)
          result.occupiedByData[(int)allocOp.getChannelIndex()] =
              allocOp.getSymName();
      } else if (auto muxOp = dyn_cast<AIE::ShimMuxOp>(user)) {
        // A hand-written aie.shim_mux (raw manual routing, with no
        // aie.shim_dma_allocation / aie.flow / aie.packet_flow to declare its
        // shim channel) hard-reserves that channel: it is a circuit route
        // control packets cannot time-share, so it must not be shared like a
        // data allocation. A connect<DMA : c, North : x> occupies MM2S chan c;
        // a connect<North : x, DMA : c> occupies S2MM chan c. The pathfinder
        // creates its own shim_mux ops only after this pass, so every shim_mux
        // present here is a design's own manual routing.
        for (auto connectOp : muxOp.getOps<AIE::ConnectOp>()) {
          if (isShimMM2S && connectOp.getSourceBundle() == shimWireBundle) {
            reservedChanUsers[connectOp.sourceIndex()] = muxOp;
            result.occupiedByCircuit.insert(connectOp.sourceIndex());
          } else if (!isShimMM2S &&
                     connectOp.getDestBundle() == shimWireBundle) {
            reservedChanUsers[connectOp.destIndex()] = muxOp;
            result.occupiedByCircuit.insert(connectOp.destIndex());
          }
        }
      }
    }
    int numShimChans = 0;
    if (isShimMM2S)
      numShimChans = targetModel.getNumSourceShimMuxConnections(
          shimTile.colIndex(), shimTile.rowIndex(), shimWireBundle);
    else
      numShimChans = targetModel.getNumDestShimMuxConnections(
          shimTile.colIndex(), shimTile.rowIndex(), shimWireBundle);
    for (int i = 0; i < numShimChans; i++) {
      if (!reservedChanUsers.count(i))
        result.availableShimChans.push_back(i);
    }

    return result;
  }

  // Choose ONE shim MM2S channel to carry a whole column's control. One packet
  // channel suffices (MAX_PACKET_STREAM_CAPACITY=32 >> rows/col).
  // Least-disturbing: prefer a channel with no design data alloc; else the
  // lowest channel not held by a circuit flow. Returns -1 only when every
  // channel is circuit-occupied.
  int chooseCtrlShimChan(const AIETargetModel &tm, WireBundle shimWireBundle,
                         TileOp shimTile) {
    int col = shimTile.colIndex();
    int numChans = tm.getNumSourceShimMuxConnections(col, shimTile.rowIndex(),
                                                     shimWireBundle);
    // Prefer a fully-free channel (no circuit AND no data alloc).
    for (int c = 0; c < numChans; c++)
      if (!moduleCircuitOccupiedByColChan.count({col, c}) &&
          !moduleDataAllocByColChan.count({col, c}))
        return c;
    // Else the lowest channel not held by a circuit flow (may share a data
    // alloc).
    for (int c = 0; c < numChans; c++)
      if (!moduleCircuitOccupiedByColChan.count({col, c}))
        return c;
    return -1;
  }

  // Create packet flows per col which moves control packets to and from shim
  // dma
  LogicalResult generatePacketFlowsForControl(
      OpBuilder builder, DeviceOp device, TileOp shimTile,
      WireBundle shimWireBundle, const SmallVector<AIE::TileOp> &ctrlTiles,
      WireBundle ctrlWireBundle, int coreOrMemChanId,
      DenseMap<TileID, int> tileIDMap, bool isShimMM2S,
      bool allowCrossDeviceDataShare = true) {
    int ctrlPktFlowID = 0;
    auto rowToShimChanMap =
        getRowToShimChanMap(device.getTargetModel(), shimWireBundle);
    // Get all available shim channels (plus any already claimed by a data
    // aie.shim_dma_allocation that control may share), to verify that the
    // channel mandated for each row is usable.
    auto shimChanAvailability =
        getAvailableShimChans(device, shimTile, shimWireBundle, isShimMM2S);
    auto &availableShimChans = shimChanAvailability.availableShimChans;
    auto &occupiedByData = shimChanAvailability.occupiedByData;
    int col = shimTile.colIndex();

    // Is `chan` already claimed by a data aie.shim_dma_allocation, either on
    // this device (occupiedByData, an SSA-based scan of shimTile's users) or
    // -- when `allowCrossDeviceDataShare` (false for the standalone
    // `@ctrl_pkt_overlay` device, see applyOverlayToDevice) -- on a sibling
    // device sharing the same physical shim tile column
    // (moduleDataAllocByColChan; needed for a baked multi-config overlay
    // module, where the control-issuing host device doesn't yet have its own
    // copy of a sibling config's data allocation at the point this pass
    // runs; see where moduleDataAllocByColChan is populated). Only meaningful
    // for the shim's MM2S leg -- the module-wide index is MM2S-only, matching
    // the direction control ever claims on the shim side (S2MM sharing isn't
    // a case this pass handles).
    auto sharedWithData = [&](int chan) {
      return occupiedByData.count(chan) ||
             (allowCrossDeviceDataShare && isShimMM2S &&
              moduleDataAllocByColChan.count({col, chan}));
    };

    // Single-trunk selection for the MM2S/ingress leg: the whole column's
    // control rides ONE shim DMA channel, computed once (independent of
    // row) instead of per-tile. Computed unconditionally but only consulted
    // when isShimMM2S below; harmless to compute for the S2MM leg too since
    // chooseCtrlShimChan only reads column-wide circuit/data-alloc state.
    int trunkChan = -1;
    if (isShimMM2S) {
      // CONSUME Stage-1's stamp when present (spec 5.5). Stage-1's
      // AIEAutoPacketizeControlIngress stamps the union-chosen control trunk
      // channel K on this column's row-0 shim tile as `ctrl_pkt_trunk_chan`,
      // and conforms each config to pin control's leg to that K. Stage-1's
      // choice is AUTHORITATIVE: it unions the data-pin / shim-mux
      // reservations across ALL configs -- knowledge this per-device recompute
      // lacks -- and, crucially, its "shareable channel" criterion treats a
      // packet leg's channel as co-tenantable by control (it deliberately puts
      // control on the packet trunk), whereas chooseCtrlShimChan below treats
      // any data alloc (packet included) as occupied and would flee to a free
      // channel. Those criteria legitimately diverge (e.g. a single packet
      // ingress leg with a free sibling channel), so recomputing here and
      // asserting agreement wrongly rejects a correct build. Consume K instead.
      if (auto kAttr =
              shimTile->getAttrOfType<IntegerAttr>("ctrl_pkt_trunk_chan")) {
        trunkChan = (int)kAttr.getInt();
      } else {
        // No stamp: Stage-1 did not run (an isolated overlay unit test). Fall
        // back to the overlay's own channel choice, exactly as before.
        trunkChan = chooseCtrlShimChan(device.getTargetModel(), shimWireBundle,
                                       shimTile);
        if (trunkChan < 0) {
          device->emitOpError(
              "failed to generate column control overlay: all shim mm2s dma "
              "channels for column ")
              << col << " are reserved by circuit-switched flows, so control "
              << "packets cannot ingress to column " << col
              << "; free or packetize a shim ingress, or reduce the design's "
                 "shim circuit usage.";
          return failure();
        }
      }
    }

    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto tOp : ctrlTiles) {
      if (tOp->hasAttr("controller_id"))
        ctrlPktFlowID =
            (int)tOp->getAttrOfType<AIE::PacketInfoAttr>("controller_id")
                .getPktId();
      else {
        // Fall back to the target-model tile->controller-id map. A tile that is
        // neither annotated nor present in the map has no legal control-packet
        // id: baking pkt_id 0 would misroute silently, so fail loud instead.
        auto it = tileIDMap.find({tOp.colIndex(), tOp.rowIndex()});
        if (it == tileIDMap.end()) {
          tOp.emitOpError("control overlay: tile has no controller_id and is "
                          "absent from the tile-to-controller-id map; cannot "
                          "assign a control-packet flow id");
          return failure();
        }
        ctrlPktFlowID = it->second;
      }
      // Check shim channel availability. A channel already claimed by a data
      // aie.shim_dma_allocation is usable too -- control packets time-share
      // it with the data DMA (disjoint dispatches) instead of double-booking
      // a second allocation on the same physical channel. Only a channel
      // reserved by circuit-switched routing (aie.flow/aie.packet_flow), or
      // one that doesn't exist, is a hard failure.
      // Occupancy-aware channel choice for the MM2S/ingress leg: relocate off
      // a circuit-occupied mandated channel instead of hard-failing. The S2MM
      // leg keeps the fixed mandated channel and the original usability check.
      int chosenChan;
      if (isShimMM2S) {
        chosenChan = trunkChan;
        // Only when this row's control is relocated off its fixed mandated
        // channel onto the column trunk, record the chosen channel on the
        // controlled tile so AIECtrlPacketToDma delivers on the same channel
        // the overlay routed. When not relocated, AIECtrlPacketToDma's
        // fallback recomputes the same mandated channel, so no attribute is
        // needed -- keeping unrelocated IR (and existing tests) unperturbed.
        if (chosenChan != rowToShimChanMap[tOp.rowIndex()])
          tOp->setAttr("ctrl_pkt_shim_chan",
                       builder.getI32IntegerAttr(chosenChan));
      } else {
        chosenChan = rowToShimChanMap[tOp.rowIndex()];
        if (!llvm::is_contained(availableShimChans, chosenChan) &&
            !sharedWithData(chosenChan)) {
          device->emitOpError(
              "failed to generate column control overlay from shim dma to tile "
              "ctrl ports, because some shim mm2s dma channels were reserved "
              "from routing control packets.");
          return failure();
        }
      }

      // Snapshot this tile's control-flow id BEFORE createPacketFlowOp, which
      // post-increments ctrlPktFlowID (PacketFlowOp::create(..., flowID++,
      // ...)). The shim alloc below must carry the id of the flow it
      // represents, not the already-bumped next id.
      int allocPktId = ctrlPktFlowID;

      auto keep_pkt_header = builder.getBoolAttr(true);
      auto ctrl_pkt_flow = builder.getBoolAttr(true);
      if (isShimMM2S)
        (void)createPacketFlowOp(builder, tOp.getLoc(), ctrlPktFlowID, shimTile,
                                 shimWireBundle, chosenChan, tOp,
                                 ctrlWireBundle, coreOrMemChanId,
                                 keep_pkt_header, ctrl_pkt_flow);
      else
        (void)createPacketFlowOp(builder, tOp.getLoc(), ctrlPktFlowID, tOp,
                                 ctrlWireBundle, coreOrMemChanId, shimTile,
                                 shimWireBundle, chosenChan, keep_pkt_header,
                                 ctrl_pkt_flow);

      // Generate shim dma alloc ops as handle for runtime sequence to pickup,
      // when issuing control packets
      if (shimWireBundle != WireBundle::DMA)
        continue;

      AIE::DMAChannelDir dir =
          isShimMM2S ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;
      int chan = chosenChan;

      // This channel is already claimed by a data aie.shim_dma_allocation
      // (on this device, or a sibling device sharing the same physical shim
      // tile column): share it (control packets and data are temporally
      // disjoint -- config_N reconfigure dispatch vs. run) rather than
      // materializing a second allocation on the same physical (tile, dir,
      // chan), which the device silently rejects at PDI load.
      // AIECtrlPacketToDma resolves the shared symbol by (tile, dir, chan)
      // lookup, not by name -- and, for the cross-device case, by whatever a
      // later pass hoists onto this device (see moduleDataAllocByColChan).
      if (sharedWithData(chan))
        continue;

      std::string dma_name = "ctrlpkt";
      dma_name += "_col" + std::to_string(col);   // col
      dma_name += isShimMM2S ? "_mm2s" : "_s2mm"; // dir
      dma_name += "_chan" + std::to_string(chan); // chan

      // check to see if ShimDMAAllocationOp already exists
      if (device.lookupSymbol(dma_name))
        continue;

      // Mark control's own shim allocation as a packet occupant (control is
      // always a packet flow, id = allocPktId = this tile's control-flow id,
      // snapshotted above before the flow-id post-increment). This does NOT
      // feed DMAChannelAnalysis: that analysis is constructed once, in
      // AIEObjectFifoAllocate.cpp (assignChannels) via the
      // aie-objectfifo-allocate pass, which runs BEFORE this overlay pass
      // (see the pass order in tools/aiecc/IRTransforms.h), so it never
      // observes this alloc's $packet. It is also not read on control's own
      // issuance path: AIECtrlPacketToDma builds control's NpuDmaMemcpyNdOp
      // with an explicit null $packet of its own (control embeds its header
      // in the payload, so AIEDmaToNpu's enable_packet bit must stay off),
      // and that op's own attr -- not this alloc's -- is what AIEDmaToNpu
      // reads. Today this attribute is documentation: it records that the
      // channel carries a packet flow, consistent with every other
      // packet-class shim endpoint, and is the field
      // AIESubstituteShimDMAAllocations (DMAConfigureTaskForOp substitution)
      // would pick up if control were ever routed through that task-based
      // path instead.
      AIE::ShimDMAAllocationOp::create(
          builder, tOp.getLoc(), StringRef(dma_name), shimTile.getResult(), dir,
          chosenChan, /*plio=*/false,
          AIE::PacketInfoAttr::get(builder.getContext(), /*pkt_type=*/0,
                                   /*pkt_id=*/allocPktId));
    }
    return success();
  }

  // Get packet-flow op with the same source or destination
  AIE::PacketFlowOp getPktFlowWithSameSrcOrDst(DeviceOp device, TileOp srcTile,
                                               WireBundle srcBundle,
                                               int srcChan, TileOp destTile,
                                               WireBundle destBundle,
                                               int destChan) {
    AIE::PacketFlowOp result = nullptr;
    device.walk([&](AIE::PacketFlowOp fOp) {
      for (auto srcOp : fOp.getOps<AIE::PacketSourceOp>()) {
        if (srcOp.getTile() == srcTile && srcOp.getBundle() == srcBundle &&
            srcOp.channelIndex() == srcChan) {
          result = fOp;
          return;
        }
      }
      for (auto destOp : fOp.getOps<AIE::PacketDestOp>()) {
        if (destOp.getTile() == destTile && destOp.getBundle() == destBundle &&
            destOp.channelIndex() == destChan) {
          result = fOp;
          return;
        }
      }
    });
    return result;
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignTileCtrlIDsPass() {
  return std::make_unique<AIEAssignTileCtrlIDsPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIE::createAIEGenerateColumnControlOverlayPass() {
  return std::make_unique<AIEGenerateColumnControlOverlayPass>();
}

void populateAIEColumnControlOverlay(DeviceOp &device) {}
