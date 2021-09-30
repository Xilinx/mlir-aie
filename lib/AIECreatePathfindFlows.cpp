//===- AIECreatePathfindFlows.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "aie/AIEDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include <aie/AIEPathfinder.h>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-create-pathfinder-flows"
static llvm::cl::opt<bool> debugRoute("debug-pathfinder",
                                      llvm::cl::desc("Enable Debugging of Pathfinder routing process"),
                                      llvm::cl::init(false));

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const & e){
// boost expects this exception to be defined.
// do nothing
}

std::string stringifyDirs(std::set<Port> dirs) {
  unsigned int count = 0;
  std::string out = "{";
  for(Port dir : dirs) {
    switch(dir.first) {
      case WireBundle::Core: out += "Core"; break;
      case WireBundle::DMA: out += "DMA"; break;
      case WireBundle::North: out += "N"; break;
      case WireBundle::East: out += "E"; break; 
      case WireBundle::South: out += "S"; break;
      case WireBundle::West: out += "W"; break;
      default: out += "X";
    }
    out += std::to_string(dir.second);
    if(++count < dirs.size())
      out += ",";
  }
  out += "}";
  return out;
}

std::string stringifyDir(Port dir) {
  return stringifyDirs(std::set<Port>({dir}));
}
std::string stringifySwitchSettings(SwitchSettings settings) {
  std::string out = "\tSwitchSettings: ";
  for(auto iter = settings.begin(); iter != settings.end(); iter++) {
      out += (std::string)"(" + std::to_string((*iter).first->col) + ", " +
        std::to_string((*iter).first->row) + ") " + 
        stringifyDir((*iter).second.first) + " -> " + 
        stringifyDirs((*iter).second.second) + " | ";
  }
  return out + "\n";
}

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR environment.
// It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings 
// It then converts these settings to MLIR operations
class DynamicTileAnalysis { 
public:
  ModuleOp &module;
  int maxcol, maxrow;
  Pathfinder pathfinder;
  std::map< PathEndPoint, SwitchSettings > flow_solutions;
  std::map< PathEndPoint, bool > processed_flows;

  DenseMap<Coord, TileOp> coordToTile;
  DenseMap<Coord, SwitchboxOp> coordToSwitchbox;
  DenseMap<Coord, ShimMuxOp> coordToShimMux;
  DenseMap<int, PLIOOp> coordToPLIO;

  const int MAX_ITERATIONS = 1000; // how long until declared unroutable

  DynamicTileAnalysis(ModuleOp &m) : module(m) {
    LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
    // find the maxcol and maxrow
    maxcol = 0;
    maxrow = 0;
    for(TileOp tileOp : m.getOps<TileOp>()) {
      maxcol = std::max(maxcol, tileOp.colIndex());
      maxrow = std::max(maxrow, tileOp.rowIndex());
    }

    pathfinder = Pathfinder(maxcol, maxrow);

    // for each flow in the module, add it to pathfinder
    // each source can map to multiple different destinations (fanout)
    for(FlowOp flowOp : module.getOps<FlowOp>()) {
      TileOp srcTile = cast<TileOp>(flowOp.source().getDefiningOp());
      TileOp dstTile = cast<TileOp>(flowOp.dest().getDefiningOp());
      Coord srcCoords = std::make_pair(srcTile.colIndex(), srcTile.rowIndex());
      Coord dstCoords = std::make_pair(dstTile.colIndex(), dstTile.rowIndex());
      Port srcPort = std::make_pair(flowOp.sourceBundle(), flowOp.sourceChannel());
      Port dstPort = std::make_pair(flowOp.destBundle(), flowOp.destChannel());
      LLVM_DEBUG(llvm::dbgs() << "\tAdding Flow: (" << srcCoords.first << ", " << srcCoords.second << ")" <<
          stringifyWireBundle(srcPort.first) << (int)srcPort.second <<
          " -> (" << dstCoords.first << ", " << dstCoords.second << ")" <<
          stringifyWireBundle(dstPort.first) << (int)dstPort.second << "\n");
      pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort);
    }

    // add existing connections so Pathfinder knows which resources are available
    // search all existing SwitchBoxOps for exising connections
    for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
      for(ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
        Coord existing_coord = std::make_pair(switchboxOp.colIndex(), switchboxOp.rowIndex());
        Port existing_port = std::make_pair(connectOp.destBundle(), connectOp.destChannel());
        pathfinder.addFixedConnection(existing_coord, existing_port);
      }
    }

    //all flows are now populated, call the congestion-aware pathfinder algorithm
    flow_solutions = pathfinder.findPaths(MAX_ITERATIONS);

    //initialize all flows as unprocessed to prep for rewrite
    for(auto iter = flow_solutions.begin(); iter != flow_solutions.end(); iter++) {
      processed_flows[(*iter).first] = false;
      LLVM_DEBUG(llvm::dbgs() << "Flow starting at (" << (*iter).first.first->col << "," << (*iter).first.first->row << "):\t");
      LLVM_DEBUG(llvm::dbgs() << stringifySwitchSettings((*iter).second));
    }

    // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
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
    for (auto shimmuxOp : module.getOps<ShimMuxOp>()) {
      int col, row;
      col = shimmuxOp.colIndex();
      row = shimmuxOp.rowIndex();
      assert(coordToShimMux.count(std::make_pair(col, row)) == 0);
      coordToShimMux[std::make_pair(col, row)] = shimmuxOp;
    }

    LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
  }

  int getMaxCol() { return maxcol; }
  int getMaxRow() { return maxrow; }

  TileOp getTile(OpBuilder &builder, int col, int row) {
    if(coordToTile.count(std::make_pair(col, row))) {
      return coordToTile[std::make_pair(col, row)];
    } else {
      TileOp tileOp =
        builder.create<TileOp>(builder.getUnknownLoc(), col, row);
      coordToTile[std::make_pair(col, row)] = tileOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return tileOp;
    }
  }
  SwitchboxOp getSwitchbox(OpBuilder &builder, int col, int row) {
    assert(col >= 0);
    assert(row >= 0);
    if(coordToSwitchbox.count(std::make_pair(col, row))) {
      return coordToSwitchbox[std::make_pair(col, row)];
    } else {
      SwitchboxOp switchboxOp =
        builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                    getTile(builder, col, row));
                                    //coordToTile[std::make_pair(col, row)]);
      switchboxOp.ensureTerminator(switchboxOp.connections(),
                                   builder,
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
    if(coordToShimMux.count(std::make_pair(col, row))) {
      return coordToShimMux[std::make_pair(col, row)];
    } else {
      ShimMuxOp switchboxOp =
        builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                    getTile(builder, col, row));
      switchboxOp.ensureTerminator(switchboxOp.connections(),
                                   builder,
                                   builder.getUnknownLoc());
      coordToShimMux[std::make_pair(col, row)] = switchboxOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      // check if the shim tile has noc connection
      if(!getTile(builder, col, row).isShimNOCTile())
        switchboxOp.emitError("Attempting to route to or from ShimTile (") << col << ", " << row << "), which has no NOC connection";
      return switchboxOp;
    }
  }
  PLIOOp getPLIO(OpBuilder &builder, int col) {
    if(coordToPLIO.count(col)) {
      return coordToPLIO[col];
    } else {
      IntegerType i32 = builder.getIntegerType(32);
      PLIOOp op =
        builder.create<PLIOOp>(builder.getUnknownLoc(),
                               builder.getIndexType(),
                               IntegerAttr::get(i32, (int32_t)col));
      coordToPLIO[col] = op;
      maxcol = std::max(maxcol, col);
      return op;
    }
  }
};

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : public OpConversionPattern<AIE::FlowOp> {
  using OpConversionPattern<AIE::FlowOp>::OpConversionPattern;
  ModuleOp &module;
  DynamicTileAnalysis &analyzer;
ConvertFlowsToInterconnect(
    MLIRContext * context, 
    ModuleOp &m, 
    DynamicTileAnalysis &a,  
    PatternBenefit benefit = 1) 
    : OpConversionPattern<AIE::FlowOp>(context, benefit), 
    module(m), analyzer(a) {}

  LogicalResult match(Operation *op) const override {
    return success();
  }

  void addConnection(ConversionPatternRewriter &rewriter,
                    // could be a shim-mux or a switchbox. 
                     Interconnect op,
                     FlowOp flowOp,
                     WireBundle inBundle,
                     int inIndex,
                     WireBundle outBundle,
                     int outIndex) const {

    Region &r = op.connections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    rewriter.template create<ConnectOp>(rewriter.getUnknownLoc(),
      inBundle, inIndex, outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs() << "\t\taddConnection() (" << op.colIndex() << "," << op.rowIndex() << ") " <<
      stringifyWireBundle(inBundle) << inIndex << " -> " << stringifyWireBundle(outBundle) << outIndex << "\n");
  }

  void rewrite(AIE::FlowOp flowOp, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();

    TileOp srcTile = cast<TileOp>(flowOp.source().getDefiningOp());
    TileOp dstTile = cast<TileOp>(flowOp.dest().getDefiningOp());
    TileID srcCoords = std::make_pair(srcTile.colIndex(), srcTile.rowIndex());
    TileID dstCoords = std::make_pair(dstTile.colIndex(), dstTile.rowIndex());
    auto srcBundle = flowOp.sourceBundle();
    auto srcChannel= flowOp.sourceChannel();
    auto dstBundle = flowOp.destBundle();
    auto dstChannel= flowOp.destChannel();
    Port srcPort = std::make_pair(srcBundle, srcChannel);
    // Port dstPort = std::make_pair(dstBundle, dstChannel);
    LLVM_DEBUG(llvm::dbgs() << "\n\t---Begin rewrite() for flowOp: (" << 
        srcCoords.first << ", " << srcCoords.second << ")" <<
        stringifyWireBundle(srcBundle) << (int)srcChannel<<
        " -> (" << dstCoords.first << ", " << dstCoords.second << ")" <<
        stringifyWireBundle(dstBundle) << (int)dstChannel << "\n\t");

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    Switchbox* srcSB = analyzer.pathfinder.getSwitchbox(srcCoords);
    PathEndPoint srcPoint = std::make_pair(srcSB, srcPort);
    if(analyzer.processed_flows[srcPoint] == false) {
      SwitchSettings settings = analyzer.flow_solutions[srcPoint];
      // add connections for all of the Switchboxes in SwitchSettings
      for(auto map_iter = settings.begin(); map_iter != settings.end(); map_iter++) {
        Switchbox* curr = (*map_iter).first;
        SwitchSetting s = (*map_iter).second;
        SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, curr->col, curr->row);
        int shim_ch = srcChannel;
        //TODO: must reserve N3, N7, S2, S3 for DMA connections
        if(curr == srcSB && srcSB->row == 0) {
          // shim DMAs at start of flows
          ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB->col);
          if(srcBundle == WireBundle::DMA)
            shim_ch = (srcChannel == 0 ? 3 : 7); // must be either DMA0 -> N3 or DMA1 -> N7
          else if(srcChannel >= 3) shim_ch = srcChannel + 1;
          addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
            srcBundle, srcChannel, WireBundle::North, shim_ch);
        }
        for(auto it = s.second.begin(); it != s.second.end(); it++) {
          WireBundle bundle = (*it).first;
          int channel = (*it).second; 
          // handle special shim connectivity
          if(curr == srcSB && srcSB->row == 0) {
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()), flowOp,
              WireBundle::South, shim_ch, bundle, channel);
          } else if ( curr->row == 0 && 
                    (bundle == WireBundle::DMA ||
                     bundle == WireBundle::PLIO) ) {
            // shim DMAs at end of flows
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr->col);
            if(bundle == WireBundle::DMA)
              shim_ch = (channel == 0 ? 2 : 3); // must be either N2 -> DMA0 or N3 -> DMA1
            else if(srcChannel >= 3) shim_ch = srcChannel + 1;
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()), flowOp,
              s.first.first, s.first.second, WireBundle::South, shim_ch);
            addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
              WireBundle::North, shim_ch, bundle, channel);
          } else {
            // otherwise, regular switchbox connection
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()), flowOp,
            s.first.first, s.first.second, bundle, channel);
          }
        }

          LLVM_DEBUG(llvm::dbgs() << " (" << curr->col << "," << curr->row << ") " << 
            stringifyDir(s.first) << " -> " << stringifyDirs(s.second) << " | ");
      }

      LLVM_DEBUG(llvm::dbgs() << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");
      analyzer.processed_flows[srcPoint] = true;
    }
    else LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");

    rewriter.eraseOp(Op);
  }
};


struct AIEPathfinderPass : public AIERoutePathfinderFlowsBase<AIEPathfinderPass> {
 
   /* Overall Flow
    rewrite switchboxes to assign unassigned connections, ensure this can be done concurrently ( by different threads)

    // Goal is to rewrite all flows in the module into switchboxes + shim-mux

    // multiple passes of the rewrite pattern rewriting streamswitch configurations to routes

    // rewrite flows to stream-switches using 'weights' from analysis pass. 

    // check a region is legal

    // rewrite stream-switches (within a bounding box) back to flows */

  void runOnOperation() override {

    // create analysis pass with routing graph for entire device
    LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

    ModuleOp m = getOperation();
    DynamicTileAnalysis analyzer(m);
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    // Populate tiles and switchboxes.
    for(int col = 0; col <= analyzer.getMaxCol(); col++) {
      for(int row = 0; row <= analyzer.getMaxRow(); row++) {
        analyzer.getTile(builder, col, row);
      }
    }
    for(int col = 0; col <= analyzer.getMaxCol(); col++) {
      for(int row = 0; row <= analyzer.getMaxRow(); row++) {
        analyzer.getSwitchbox(builder, col, row);
      }
    }
    for(int col = 0; col <= analyzer.getMaxCol(); col++) {
      analyzer.getPLIO(builder, col);
    }

    // Populate wires between switchboxes and tiles.
    for(int col = 0; col <= analyzer.getMaxCol(); col++) {
      for(int row = 0; row <= analyzer.getMaxRow(); row++) {
        auto tile = analyzer.getTile(builder, col, row);
        auto sw = analyzer.getSwitchbox(builder, col, row);
        if(col > 0) {
          // connections east-west between stream switches
          auto westsw = analyzer.getSwitchbox(builder, col-1, row);
          builder.create<WireOp>(builder.getUnknownLoc(),
                                  westsw,
                                  WireBundle::East,
                                  sw,
                                  WireBundle::West);
        }
        if(row > 0) {
          // connections between abstract 'core' of tile 
          builder.create<WireOp>(builder.getUnknownLoc(),
                                  tile,
                                  WireBundle::Core,
                                  sw,
                                  WireBundle::Core);
          // connections between abstract 'dma' of tile                        
          builder.create<WireOp>(builder.getUnknownLoc(),
                                  tile,
                                  WireBundle::DMA,
                                  sw,
                                  WireBundle::DMA);
          // connections north-south inside array ( including connection to shim row)                        
          auto southsw = analyzer.getSwitchbox(builder, col, row-1);
          builder.create<WireOp>(builder.getUnknownLoc(),
                                  southsw,
                                  WireBundle::North,
                                  sw,
                                  WireBundle::South);
        } else if(row == 0) {
          if (tile.isShimNOCTile()) {
            auto shimsw = analyzer.getShimMux(builder, col);
            builder.create<WireOp>(
                builder.getUnknownLoc(), shimsw,
                WireBundle::North, // Changed to connect into the north
                sw, WireBundle::South);
            // PLIO is attached to shim mux
            auto plio = analyzer.getPLIO(builder, col);
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, shimsw,
                                   WireBundle::South);

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                   WireBundle::DMA, shimsw, WireBundle::DMA);
          } else if (tile.isShimPLTile()) {
            // PLIO is attached directly to switch
            auto plio = analyzer.getPLIO(builder, col);
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }

    // Apply rewrite rule to switchboxes to add assignments to every 'connect' operation inside
    ConversionTarget target(getContext());
    target.addLegalOp<ConnectOp>();
    target.addLegalOp<SwitchboxOp>();
    target.addLegalOp<ShimMuxOp>();
    target.addLegalOp<EndOp>();

    OwningRewritePatternList patterns(&getContext());
    patterns.insert<ConvertFlowsToInterconnect>(m.getContext(), m, analyzer);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    return;
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}