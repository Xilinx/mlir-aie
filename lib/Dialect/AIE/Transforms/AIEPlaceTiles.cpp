//===- AIEPlaceTiles.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "aie/Dialect/AIE/Transforms/AIEStructuralRank.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"

#include <cmath>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEPLACETILES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-place-tiles"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct ConvertLogicalTileToTile : OpConversionPattern<LogicalTileOp> {
  ConvertLogicalTileToTile(MLIRContext *context, DeviceOp &d, Placer &p,
                           PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), placer(p) {}

  LogicalResult
  matchAndRewrite(LogicalTileOp logicalTile, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto placement = placer.getPlacement(logicalTile);
    if (!placement)
      return logicalTile.emitError("no placement found for logical tile");

    // Handle merging multiple logical tiles to same physical tile
    TileOp tileOp = TileOp::getOrCreate(rewriter, device, placement->col,
                                        placement->row, logicalTile.getLoc());
    rewriter.modifyOpInPlace(tileOp, [&] {
      tileOp->setLoc(
          rewriter.getFusedLoc({tileOp.getLoc(), logicalTile.getLoc()}));
    });

    if (auto controllerId = logicalTile->getAttr("controller_id"))
      tileOp->setAttr("controller_id", controllerId);

    rewriter.replaceOp(logicalTile, tileOp.getResult());
    return success();
  }

private:
  DeviceOp device;
  Placer &placer;
};

struct AIEPlaceTilesPass
    : xilinx::AIE::impl::AIEPlaceTilesBase<AIEPlaceTilesPass> {

  AIEPlaceTilesPass() = default;

  AIEPlaceTilesPass(const AIEPlaceTilesOptions &options)
      : AIEPlaceTilesBase(options) {}

  // Parses the SA's charges once, before any device is placed. Names are
  // symbol references, so any name can be written.
  LogicalResult initialize(MLIRContext *context) override {
    walked.clear();
    coreSizes.clear();
    auto parseSymbol = [&](StringRef text) {
      ScopedDiagnosticHandler quiet(context,
                                    [](Diagnostic &) { return success(); });
      return dyn_cast_or_null<SymbolRefAttr>(parseAttribute(text, context));
    };
    for (StringRef entry : clSAWalkedFifos) {
      SymbolRefAttr fifo = parseSymbol(entry);
      if (!fifo || fifo.getNestedReferences().size() != 1)
        return emitError(UnknownLoc::get(context), "sa-walked-fifos entry '")
               << entry << "' is not @<device>::@<fifo>";
      walked.push_back(
          {fifo.getRootReference().str(), fifo.getLeafReference().str()});
    }
    for (StringRef entry : clSACoreDataSizes) {
      auto [core, charge] = entry.rsplit('=');
      auto [name, ordinal] = core.rsplit(':');
      auto [bytes, alignment] = charge.split('@');
      auto device = dyn_cast_or_null<FlatSymbolRefAttr>(parseSymbol(name));
      unsigned index;
      int64_t size;
      int64_t align = 1;
      if (!device || ordinal.getAsInteger(10, index) ||
          bytes.getAsInteger(10, size) || size < 0 ||
          (charge.contains('@') && (alignment.getAsInteger(10, align) ||
                                    align <= 0 || !llvm::isPowerOf2_64(align))))
        return emitError(UnknownLoc::get(context), "sa-core-data-sizes entry '")
               << entry << "' is not @<device>:<core>=<bytes>[@<align>]";
      coreSizes.push_back({device.getValue().str(), index, size, align});
    }
    return success();
  }

  // One device after another, in structural order: a device's placement
  // may read the devices placed before it.
  void runOnOperation() override {
    SmallVector<DeviceOp> devices(getOperation().getOps<DeviceOp>());
    if (devices.size() > 1) {
      DenseMap<Operation *, unsigned> rank =
          StructuralGraph::of(getOperation()).rank();
      llvm::stable_sort(devices, [&](DeviceOp a, DeviceOp b) {
        return rank.lookup(a) < rank.lookup(b);
      });
    }
    residents.clear();
    placedCores.clear();
    programKeys.clear();
    // Configurations the host loads one right after the other.
    consecutive.clear();
    ModuleOp module = getOperation();
    module.walk([&](RuntimeSequenceOp sequence) {
      Operation *previous = nullptr;
      sequence.walk([&](Operation *op) {
        FlatSymbolRefAttr loaded;
        if (auto configure = dyn_cast<AIEX::ConfigureOp>(op))
          loaded = configure.getSymbolAttr();
        else if (auto load = dyn_cast<AIEX::NpuLoadPdiOp>(op))
          loaded = load.getDeviceRefAttr();
        Operation *device =
            loaded ? SymbolTable::lookupSymbolIn(module, loaded) : nullptr;
        if (!isa_and_nonnull<DeviceOp>(device))
          return;
        if (previous && previous != device) {
          consecutive[previous].insert(device);
          consecutive[device].insert(previous);
        }
        previous = device;
      });
    });
    for (DeviceOp device : devices)
      if (failed(pinResidents(device)) || failed(placeDevice(device)) ||
          failed(recordPlaced(device)))
        return signalPassFailure();
  }

  // Pins each buffer resident for a memref.global an earlier device placed
  // to the tile and address that device gave it, so the data stays put
  // across the switch.
  LogicalResult pinResidents(DeviceOp device) {
    for (BufferOp buffer : device.getOps<BufferOp>()) {
      FlatSymbolRefAttr global = buffer.getResidentAttr();
      auto first = global ? residents.find(global.getAttr()) : residents.end();
      if (first == residents.end())
        continue;
      auto [twin, at] = first->second;
      Operation *tile = buffer.getTile().getDefiningOp();
      auto like = cast<TileLike>(tile);
      AIETileType there = device.getTargetModel().getTileType(at.col, at.row);
      if (like.getTileType() != there) {
        InFlightDiagnostic diag =
            buffer.emitError("buffer resident for ")
            << global << " is on a " << stringifyAIETileType(like.getTileType())
            << ", but the earlier buffer resident for it is on a "
            << stringifyAIETileType(there) << " at (" << at.col << ", "
            << at.row << ")";
        diag.attachNote(twin.getLoc()) << "the earlier buffer";
        return diag;
      }
      std::optional<int> col = like.tryGetCol(), row = like.tryGetRow();
      if ((col && *col != at.col) || (row && *row != at.row)) {
        InFlightDiagnostic diag =
            buffer.emitError("buffer resident for ")
            << global << " is on a tile pinned away from (" << at.col << ", "
            << at.row << "), where the earlier buffer resident for it is";
        diag.attachNote(twin.getLoc()) << "the earlier buffer";
        return diag;
      }
      if (auto logical = dyn_cast<LogicalTileOp>(tile)) {
        logical.setCol(at.col);
        logical.setRow(at.row);
      }
      auto address = twin.getAddress(), own = buffer.getAddress();
      if (address && own && *own != *address) {
        InFlightDiagnostic diag =
            buffer.emitError("buffer resident for ")
            << global << " has address " << *own
            << ", but the earlier buffer resident for it has " << *address;
        diag.attachNote(twin.getLoc()) << "the earlier buffer";
        return diag;
      }
      if (address)
        buffer.setAddress(address);
      else if (!own)
        buffer.emitWarning("buffer resident for ")
            << global
            << " has no address, so its data may move between "
               "configurations; give it and its earlier twin one";
    }
    return success();
  }

  // Records where `device`, now placed, keeps its resident buffers and its
  // cores' programs.
  LogicalResult recordPlaced(DeviceOp device) {
    for (BufferOp buffer : device.getOps<BufferOp>())
      if (FlatSymbolRefAttr global = buffer.getResidentAttr())
        if (auto tile = dyn_cast<TileOp>(buffer.getTile().getDefiningOp()))
          residents.try_emplace(global.getAttr(), buffer, tile.getTileID());
    DenseMap<TileID, unsigned> &cores = placedCores[device];
    for (auto [tile, key] : programKeysOf(device))
      if (auto placed = dyn_cast<TileOp>(tile))
        cores[placed.getTileID()] = key;
    return success();
  }

  // Each tile op holding a core, keyed by its core's structure, names and
  // symbols left out, so equal programs key alike across devices.
  DenseMap<Operation *, unsigned> programKeysOf(DeviceOp device) {
    StructuralGraph graph = StructuralGraph::of(device);
    DenseMap<Operation *, unsigned> keys;
    for (CoreOp core : device.getOps<CoreOp>()) {
      auto [entry, added] = programKeys.try_emplace(graph.features.lookup(core),
                                                    programKeys.size() + 1);
      keys[core.getTile().getDefiningOp()] = entry->second;
    }
    return keys;
  }

  LogicalResult placeDevice(DeviceOp device) {

    // Create placer
    std::shared_ptr<Placer> placer;
    std::shared_ptr<SAPlacer> saPlacer;
    switch (clPlacerType) {
    case PlacerType::SequentialPlacer: {
      std::optional<int> coresPerCol = std::nullopt;
      if (clCoresPerCol >= 0)
        coresPerCol = clCoresPerCol;
      placer = std::make_shared<SequentialPlacer>(
          coresPerCol, clMergeLogicalTiles, clSpreadUnanchoredTiles);
      break;
    }
    case PlacerType::SAPlacer: {
      double effort = clSAEffort.getValue();
      if (!std::isfinite(effort) || effort <= 0) {
        device.emitError("sa-effort must be positive and finite, got ")
            << effort;
        return failure();
      }
      int64_t budget = clPlacementBudget;
      if (budget <= 0) {
        device.emitError("placement-budget must be positive, got ") << budget;
        return failure();
      }
      SAConfig config;
      config.effort = effort;
      config.placementBudget = budget;
      // A device another device's entry names is reported once, by the
      // module's first device; each device checks the entries naming it.
      auto module = device->getParentOfType<ModuleOp>();
      if (module && *module.getOps<DeviceOp>().begin() == device) {
        llvm::StringSet<> devices;
        for (DeviceOp other : module.getOps<DeviceOp>())
          devices.insert(other.getSymName());
        llvm::SmallVector<StringRef> named;
        for (auto &[name, fifo] : walked)
          named.push_back(name);
        for (auto &[name, index, size, align] : coreSizes)
          named.push_back(name);
        for (StringRef name : named)
          if (!devices.contains(name)) {
            device.emitError(
                "sa-walked-fifos or sa-core-data-sizes names no device @")
                << name;
            return failure();
          }
      }
      for (auto &[name, fifo] : walked) {
        if (name != device.getSymName())
          continue;
        if (!isa_and_nonnull<ObjectFifoCreateOp>(
                SymbolTable::lookupSymbolIn(device, fifo))) {
          device.emitError("sa-walked-fifos names no objectFIFO @")
              << fifo << " in @" << name;
          return failure();
        }
        config.walkedFifos.push_back(fifo);
      }
      size_t cores = llvm::range_size(device.getOps<CoreOp>());
      for (auto &[name, index, size, align] : coreSizes) {
        if (name != device.getSymName())
          continue;
        if (index >= cores) {
          device.emitError("sa-core-data-sizes names core ")
              << index << " of @" << name << ", which has " << cores
              << " cores";
          return failure();
        }
        config.coreDataCharges.insert_or_assign(
            index, SAConfig::CoreDataCharge(size, align));
      }
      placer = saPlacer = std::make_shared<SAPlacer>(clSASeed, config);
      break;
    }
    }

    placer->initialize(
        device, clCtrlPktOverlay && getsColumnControlOverlay(device, true),
        clCtrlPktOverlayAdaptive, clPacketSwObjectFifos);
    SmallVector<DenseMap<TileID, unsigned>> neighbours;
    if (auto adjacent = consecutive.find(device); adjacent != consecutive.end())
      for (Operation *other : adjacent->second)
        if (auto cores = placedCores.find(other); cores != placedCores.end())
          neighbours.push_back(cores->second);
    if (!neighbours.empty())
      placer->setNeighbourCores(programKeysOf(device), std::move(neighbours));
    if (failed(placer->place(device)))
      return failure();
    if (saPlacer) {
      saFinalCost += saPlacer->getFinalCost();
      saLockGroups += saPlacer->getFinalLockGroups();
      saLockPenalty += saPlacer->getFinalLockPenalty();
      saLockStopped += saPlacer->getFinalLockStopped();
    }

    ConversionTarget target(getContext());
    target.addLegalOp<TileOp>();
    target.addIllegalOp<LogicalTileOp>();
    target.addLegalDialect<AIEDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLogicalTileToTile>(device.getContext(), device,
                                           *placer);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      return failure();

    // Generate objectfifo.allocate ops right after their respective objectfifo
    auto &allocs = placer->getAllocates();
    if (!allocs.empty()) {
      // Deduplicate: only emit one allocate per fifo
      llvm::DenseSet<Operation *> emitted;
      for (auto &[fifoOp, delegateTileID] : allocs) {
        if (emitted.count(fifoOp))
          continue;
        emitted.insert(fifoOp);

        auto ofOp = dyn_cast<ObjectFifoCreateOp>(fifoOp);
        if (!ofOp)
          continue;

        // Insert right after the objectfifo op
        OpBuilder builder(ofOp->getContext());
        builder.setInsertionPointAfter(ofOp);

        TileOp delegateTile =
            TileOp::getOrCreate(builder, device, delegateTileID.col,
                                delegateTileID.row, ofOp.getLoc());
        ObjectFifoAllocateOp::create(
            builder, ofOp.getLoc(),
            SymbolRefAttr::get(builder.getContext(), ofOp.getSymName()),
            delegateTile.getResult());
      }
    }
    return success();
  }

private:
  // The first buffer placed resident for each memref.global, and its tile.
  DenseMap<StringAttr, std::pair<BufferOp, TileID>> residents;
  // Each placed device's cores, by position, and the program key each runs.
  DenseMap<Operation *, DenseMap<TileID, unsigned>> placedCores;
  llvm::StringMap<unsigned> programKeys;
  DenseMap<Operation *, llvm::SetVector<Operation *>> consecutive;
  // sa-walked-fifos, as (device, fifo).
  llvm::SmallVector<std::pair<std::string, std::string>> walked;
  // sa-core-data-sizes, as (device, core, bytes, alignment).
  llvm::SmallVector<std::tuple<std::string, unsigned, int64_t, int64_t>>
      coreSizes;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
AIE::createAIEPlaceTilesPass(const AIEPlaceTilesOptions &options) {
  return std::make_unique<AIEPlaceTilesPass>(options);
}
