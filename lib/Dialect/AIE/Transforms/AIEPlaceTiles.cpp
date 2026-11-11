//===- AIEPlaceTiles.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

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

  // Parses the SA's charges once, before any device is placed.
  LogicalResult initialize(MLIRContext *context) override {
    walked.clear();
    coreSizes.clear();
    for (StringRef entry : clSAWalkedFifos) {
      auto [name, fifo] = entry.rsplit(':');
      if (name.empty() || fifo.empty() || name == entry)
        return emitError(UnknownLoc::get(context), "sa-walked-fifos entry '")
               << entry << "' is not <device>:<fifo>";
      walked.push_back({name.str(), fifo.str()});
    }
    for (StringRef entry : clSACoreDataSizes) {
      auto [core, charge] = entry.rsplit('=');
      auto [name, ordinal] = core.rsplit(':');
      auto [bytes, alignment] = charge.split('@');
      unsigned index;
      int64_t size;
      int64_t align = 1;
      if (name.empty() || ordinal.getAsInteger(10, index) ||
          bytes.getAsInteger(10, size) || size < 0 ||
          (charge.contains('@') && (alignment.getAsInteger(10, align) ||
                                    align <= 0 || !llvm::isPowerOf2_64(align))))
        return emitError(UnknownLoc::get(context), "sa-core-data-sizes entry '")
               << entry << "' is not <device>:<core>=<bytes>[@<align>]";
      coreSizes.push_back({name.str(), index, size, align});
    }
    return success();
  }

  // One device after another, in module order: a device's placement may
  // read the devices placed before it.
  void runOnOperation() override {
    for (DeviceOp device :
         llvm::make_early_inc_range(getOperation().getOps<DeviceOp>()))
      if (failed(placeDevice(device)))
        return signalPassFailure();
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
            device.emitError("SA placer charges name no device @") << name;
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
