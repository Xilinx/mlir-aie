//===- AIECreateFlows.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-create-flows"
static llvm::cl::opt<bool>
    debugRoute("debug-route",
               llvm::cl::desc("Enable Debugging of routing process"),
               llvm::cl::init(false));

typedef llvm::Optional<std::pair<Operation *, Port>> PortConnection;

class TileAnalysis {
  ModuleOp &module;
  int maxcol, maxrow;
  DenseMap<std::pair<int, int>, TileOp> coordToTile;
  DenseMap<std::pair<int, int>, SwitchboxOp> coordToSwitchbox;
  DenseMap<std::pair<int, int>, ShimMuxOp> coordToShimMux;
  // DenseMap<int, ShimSwitchboxOp> coordToShimSwitchbox;
  DenseMap<int, PLIOOp> coordToPLIO;

public:
  int getMaxCol() { return maxcol; }
  int getMaxRow() { return maxrow; }
  int getConstantInt(Value val) { return 0; }
  TileAnalysis(ModuleOp &m) : module(m) {
    maxcol = 0;
    maxrow = 0;
    for (auto tileOp : module.getOps<TileOp>()) {
      int col, row;
      col = tileOp.colIndex();
      row = tileOp.rowIndex();
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      assert(coordToTile.count(std::make_pair(col, row)) == 0);
      coordToTile[std::make_pair(col, row)] = tileOp;
    }
    for (auto switchboxOp : module.getOps<SwitchboxOp>()) {
      int col, row;
      col = switchboxOp.colIndex();
      row = switchboxOp.rowIndex();
      assert(coordToSwitchbox.count(std::make_pair(col, row)) == 0);
      coordToSwitchbox[std::make_pair(col, row)] = switchboxOp;
    }
    for (auto switchboxOp : module.getOps<ShimMuxOp>()) {
      int col, row;
      col = switchboxOp.colIndex();
      row = switchboxOp.rowIndex();
      assert(coordToShimMux.count(std::make_pair(col, row)) == 0);
      coordToShimMux[std::make_pair(col, row)] = switchboxOp;
    }
  }

  TileOp getTile(OpBuilder &builder, int col, int row) {
    if (coordToTile.count(std::make_pair(col, row))) {
      return coordToTile[std::make_pair(col, row)];
    } else {
      TileOp tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
      coordToTile[std::make_pair(col, row)] = tileOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return tileOp;
    }
  }
  SwitchboxOp getSwitchbox(OpBuilder &builder, int col, int row) {
    assert(col >= 0);
    assert(row >= 0);
    if (coordToSwitchbox.count(std::make_pair(col, row))) {
      return coordToSwitchbox[std::make_pair(col, row)];
    } else {
      SwitchboxOp switchboxOp = builder.create<SwitchboxOp>(
          builder.getUnknownLoc(), getTile(builder, col, row));
      // coordToTile[std::make_pair(col, row)]);
      switchboxOp.ensureTerminator(switchboxOp.connections(), builder,
                                   builder.getUnknownLoc());
      coordToSwitchbox[std::make_pair(col, row)] = switchboxOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return switchboxOp;
    }
  }
  ShimMuxOp getShimMux(OpBuilder &builder, int col) {
    assert(col >= 0);
    int row = 0;
    if (coordToShimMux.count(std::make_pair(col, row))) {
      return coordToShimMux[std::make_pair(col, row)];
    } else {
      assert(getTile(builder, col, row).isShimNOCTile());
      ShimMuxOp switchboxOp = builder.create<ShimMuxOp>(
          builder.getUnknownLoc(), getTile(builder, col, row));
      switchboxOp.ensureTerminator(switchboxOp.connections(), builder,
                                   builder.getUnknownLoc());
      coordToShimMux[std::make_pair(col, row)] = switchboxOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return switchboxOp;
    }
  }
  PLIOOp getPLIO(OpBuilder &builder, int col) {
    if (coordToPLIO.count(col)) {
      return coordToPLIO[col];
    } else {
      IntegerType i32 = builder.getIntegerType(32);
      PLIOOp op = builder.create<PLIOOp>(builder.getUnknownLoc(),
                                         builder.getIndexType(),
                                         IntegerAttr::get(i32, (int32_t)col));
      coordToPLIO[col] = op;
      maxcol = std::max(maxcol, col);
      return op;
    }
  }
};

struct RouteFlows : public OpConversionPattern<AIE::FlowOp> {
  using OpConversionPattern<AIE::FlowOp>::OpConversionPattern;
  ModuleOp &module;
  TileAnalysis &analysis;
  RouteFlows(MLIRContext *context, ModuleOp &m, TileAnalysis &a,
             PatternBenefit benefit = 1)
      : OpConversionPattern<FlowOp>(context, benefit), module(m), analysis(a) {}

  LogicalResult match(AIE::FlowOp op) const override { return success(); }

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int &outIndex) const {

    LLVM_DEBUG(llvm::dbgs()
               << " - addConnection (" << stringifyWireBundle(inBundle) << " : "
               << inIndex << ") -> (" << stringifyWireBundle(outBundle) << " : "
               << outIndex << ")\n");

    Region &r = op.connections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    std::vector<int> validIndices;

    // TODO: may need to reserve index for DMA connections so that PLIO doesn't
    // use them

    // if this interconnect is a shim-mux then..
    if (isa<ShimMuxOp>(op)) {
      // (enforce) in-bound connections from 'South' must route straight
      // through. (enforce) in-bound connections from north to South must route
      // straight through
      if ((inBundle == WireBundle::South && outBundle == WireBundle::North) ||
          (inBundle == WireBundle::North && outBundle == WireBundle::South)) {
        outIndex = inIndex;
      }
      // (enforce) out-bound connections from 'DMA' on index 0 go out on
      // north-bound 3 (enforce) out-bound connections from 'DMA' on index 1 go
      // out on north-bound 7
      else if (inBundle == WireBundle::DMA && outBundle == WireBundle::North) {
        if (inIndex == 0)
          outIndex = 3;
        else
          outIndex = 7;
      }
    } else { // not a shimmux
      // if shim switch to shimmux connection...
      // (enforce) in-bound connections from north to DMA channel 0 must come in
      // on index 2 (enforce) in-bound connections from north to DMA channel 1
      // must come in on index 3
      if (op.rowIndex() == 0 && outBundle == WireBundle::South &&
          flowOp.destBundle() == WireBundle::DMA) {
        if (flowOp.destChannel() == 0)
          outIndex = 2;
        else
          outIndex = 3;
      } else if (outBundle == WireBundle::North)
        validIndices = {0, 1, 2, 3, 4, 5};
      else
        validIndices = {0, 1, 2,
                        3}; // most connections have 4 valid outIndex options
    }

    // If no index has been dictated, find an index that is bigger than any
    // existing index.
    if (outIndex == -1) {
      uint8_t choice = 0;
      outIndex = validIndices[choice];

      for (auto connectOp : b.getOps<ConnectOp>()) {
        if (connectOp.destBundle() == outBundle)
          while (connectOp.destIndex() >= outIndex) {
            outIndex = validIndices[++choice];
          }
      }
      for (auto masterOp : b.getOps<MasterSetOp>()) {
        if (masterOp.destBundle() == outBundle)
          while (masterOp.destIndex() >= outIndex) {
            outIndex = validIndices[++choice];
          }
      }
      if (choice >= validIndices.size())
        op.emitOpError("\nAIECreateFlows: Illegal routing channel detected!\n")
            << "Too many routes in tile (" << op.colIndex() << ", "
            << op.rowIndex() << ")\n";
    }

    // This might fail if an outIndex was exactly specified.
    rewriter.template create<ConnectOp>(rewriter.getUnknownLoc(), inBundle,
                                        inIndex, outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    if (debugRoute) {
      int col, row;
      col = op.colIndex();
      row = op.rowIndex();
      llvm::dbgs() << "\t\tRoute@(" << col << "," << row
                   << "): " << stringifyWireBundle(inBundle) << ":" << inIndex
                   << "->" << stringifyWireBundle(outBundle) << ":" << outIndex
                   << "\n";
    }
  }

  void rewrite(AIE::FlowOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    WireBundle sourceBundle = op.sourceBundle();
    int sourceIndex = op.sourceIndex();
    WireBundle destBundle = op.destBundle();
    int destIndex = op.destIndex();

    int col, row;
    if (FlowEndPoint source =
            dyn_cast<FlowEndPoint>(op.source().getDefiningOp())) {
      col = source.colIndex();
      row = source.rowIndex();
    } else
      llvm_unreachable("Unimplemented case");

    int destcol, destrow;
    if (FlowEndPoint dest = dyn_cast<FlowEndPoint>(op.dest().getDefiningOp())) {
      destcol = dest.colIndex();
      destrow = dest.rowIndex();
    } else
      llvm_unreachable("Unimplemented case");

    if (debugRoute)
      llvm::dbgs() << "Route: " << col << "," << row << "->" << destcol << ","
                   << destrow << "\n";

    WireBundle bundle = sourceBundle;
    int index = sourceIndex;

    // sourcing from the shim tile
    // If intent is to route from PLIO - "south" is specified  (Attach ShimMux
    // to to PLIO) if intent is to route from DMA - "DMA" is specificed ( Attach
    // ShimMux to DMA)
    if (row == 0) {
      // The Shim row of tiles needs some extra connectivity
      // FIXME: is this correct in a ShimPLTile?
      LLVM_DEBUG(llvm::dbgs() << "\tInitial Extra shim connectivity\t");
      ShimMuxOp shimMuxOp = analysis.getShimMux(rewriter, col);
      int internalIndex = -1;
      addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()), op,
                    bundle, index, WireBundle::North, internalIndex);
      bundle = WireBundle::South;
      index = internalIndex;
      // does not update row... instead connection made in `while' loop
    }

    int nextcol = col, nextrow = row;
    WireBundle nextBundle;
    int done = false;
    while (!done) {
      // Travel vertically to the destination row, then horizontally to the
      // destination column

      // Create a connection inside this switchbox.
      WireBundle outBundle;
      int outIndex = -1; // pick connection.
      if (row > destrow) {
        outBundle = WireBundle::South;
        nextBundle = WireBundle::North;
        nextrow = row - 1;
      } else if (row < destrow) {
        outBundle = WireBundle::North;
        nextBundle = WireBundle::South;
        nextrow = row + 1;
      } else if (col > destcol) {
        outBundle = WireBundle::West;
        nextBundle = WireBundle::East;
        nextcol = col - 1;
      } else if (col < destcol) {
        outBundle = WireBundle::East;
        nextBundle = WireBundle::West;
        nextcol = col + 1;
      } else {
        assert(row == destrow && col == destcol);
        // In the destination tile streamswitch,  done, so connect to the
        // correct target bundle.
        outBundle = destBundle;
        outIndex = destIndex;
        done = true;
      }
      if (nextrow < 0) {
        assert(false);
      } else {

        if (row == 0 && done) {
          // we reached our destination, and that destination is in the shim,
          // we're terminating in either the DMA or the PLIO
          // The Shim row of tiles needs some extra connectivity
          // FIXME: is this correct in a ShimPLTile?
          SwitchboxOp swOp = analysis.getSwitchbox(rewriter, col, row);
          ShimMuxOp shimMuxOp = analysis.getShimMux(rewriter, col);
          int internalIndex = -1;
          LLVM_DEBUG(llvm::dbgs() << "\tExtra shim switch connectivity\t");
          addConnection(rewriter, cast<Interconnect>(swOp.getOperation()), op,
                        bundle, index, WireBundle::South, internalIndex);

          LLVM_DEBUG(llvm::dbgs() << "\tExtra shim DMA connectivity\t");
          addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                        op, WireBundle::North, internalIndex, outBundle,
                        outIndex);

        } else {
          // Most tiles are simple and just go through a switchbox.
          LLVM_DEBUG(llvm::dbgs() << "\tRegular switch connectivity\t");
          SwitchboxOp swOp = analysis.getSwitchbox(rewriter, col, row);
          addConnection(rewriter, cast<Interconnect>(swOp.getOperation()), op,
                        bundle, index, outBundle, outIndex);
        }
      }
      if (done)
        break;
      col = nextcol;
      row = nextrow;
      bundle = nextBundle;
      index = outIndex;
    }

    rewriter.eraseOp(Op);
  }
};

struct AIERouteFlowsPass : public AIERouteFlowsBase<AIERouteFlowsPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    TileAnalysis analysis(m);
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    // Populate tiles and switchboxes.
    for (int col = 0; col <= analysis.getMaxCol(); col++) {
      for (int row = 0; row <= analysis.getMaxRow(); row++) {
        analysis.getTile(builder, col, row);
      }
    }
    for (int col = 0; col <= analysis.getMaxCol(); col++) {
      for (int row = 0; row <= analysis.getMaxRow(); row++) {
        analysis.getSwitchbox(builder, col, row);
      }
    }
    for (int col = 0; col <= analysis.getMaxCol(); col++) {
      analysis.getPLIO(builder, col);
    }
    // Populate wires betweeh switchboxes and tiles.
    for (int col = 0; col <= analysis.getMaxCol(); col++) {
      for (int row = 0; row <= analysis.getMaxRow(); row++) {
        auto tile = analysis.getTile(builder, col, row);
        auto sw = analysis.getSwitchbox(builder, col, row);
        if (col > 0) {
          // connections east-west between stream switches
          auto westsw = analysis.getSwitchbox(builder, col - 1, row);
          builder.create<WireOp>(builder.getUnknownLoc(), westsw,
                                 WireBundle::East, sw, WireBundle::West);
        }
        if (row > 0) {
          // connections between abstract 'core' of tile
          builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                 WireBundle::Core, sw, WireBundle::Core);
          // connections between abstract 'dma' of tile
          builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::DMA,
                                 sw, WireBundle::DMA);
          // connections north-south inside array ( including connection to shim
          // row)
          auto southsw = analysis.getSwitchbox(builder, col, row - 1);
          builder.create<WireOp>(builder.getUnknownLoc(), southsw,
                                 WireBundle::North, sw, WireBundle::South);
        } else if (row == 0) {
          if (tile.isShimNOCTile()) {
            auto shimsw = analysis.getShimMux(builder, col);
            builder.create<WireOp>(
                builder.getUnknownLoc(), shimsw,
                WireBundle::North, // Changed to connect into the north
                sw, WireBundle::South);
            // PLIO is attached to shim mux
            auto plio = analysis.getPLIO(builder, col);
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, shimsw,
                                   WireBundle::South);

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                   WireBundle::DMA, shimsw, WireBundle::DMA);
          } else if (tile.isShimPLTile()) {
            // PLIO is attached directly to switch
            auto plio = analysis.getPLIO(builder, col);
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }
    ConversionTarget target(getContext());
    target.addLegalOp<ConnectOp>();
    target.addLegalOp<SwitchboxOp>();
    target.addLegalOp<ShimMuxOp>();
    target.addLegalOp<EndOp>();

    OwningRewritePatternList patterns(&getContext());
    patterns.insert<RouteFlows>(m.getContext(), m, analysis);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    return;
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIERouteFlowsPass() {
  return std::make_unique<AIERouteFlowsPass>();
}
