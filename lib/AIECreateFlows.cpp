// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "AIEDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static llvm::cl::opt<bool> debugRoute("debug-route",
                                      llvm::cl::desc("Enable Debugging of routing process"),
                                      llvm::cl::init(false));

typedef llvm::Optional<std::pair<Operation *, Port>> PortConnection;

class TileAnalysis {
  ModuleOp &module;
  int maxcol, maxrow;
  DenseMap<std::pair<int, int>, TileOp> coordToTile;
  DenseMap<std::pair<int, int>, SwitchboxOp> coordToSwitchbox;
  DenseMap<int, ShimSwitchboxOp> coordToShimSwitchbox;
  DenseMap<int, PLIOOp> coordToPLIO;
public:
  int getMaxCol() {
    return maxcol;
  }
  int getMaxRow() {
    return maxrow;
  }
  int getConstantInt(Value val) {
    return 0;
  }
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
  }

  TileOp getTile(OpBuilder &builder, int col, int row) {
    if(coordToTile.count(std::make_pair(col, row))) {
      return coordToTile[std::make_pair(col, row)];
    } else {
      IntegerType i32 = builder.getIntegerType(32);
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
      IntegerType i32 = builder.getIntegerType(32);
      SwitchboxOp switchboxOp =
        builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                    builder.getIndexType(),
                                    col, row);
      switchboxOp.ensureTerminator(switchboxOp.connections(),
                                   builder,
                                   builder.getUnknownLoc());
      coordToSwitchbox[std::make_pair(col, row)] = switchboxOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return switchboxOp;
    }
  }
  ShimSwitchboxOp getShimSwitchbox(OpBuilder &builder, int col) {
    assert(col >= 0);
    if(coordToShimSwitchbox.count(col)) {
      return coordToShimSwitchbox[col];
    } else {
      IntegerType i32 = builder.getIntegerType(32);
      ShimSwitchboxOp switchboxOp =
        builder.create<ShimSwitchboxOp>(builder.getUnknownLoc(),
                                    builder.getIndexType(),
                                    IntegerAttr::get(i32, (int)col));
      switchboxOp.ensureTerminator(switchboxOp.connections(),
                                   builder,
                                   builder.getUnknownLoc());
      coordToShimSwitchbox[col] = switchboxOp;
      maxcol = std::max(maxcol, col);
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
                               IntegerAttr::get(i32, (int)col));
      coordToPLIO[col] = op;
      maxcol = std::max(maxcol, col);
      return op;
    }
  }
};

struct RouteFlows : public OpConversionPattern<AIE::FlowOp> {
  using OpConversionPattern<AIE::FlowOp>::OpConversionPattern;
  TileAnalysis &analysis;
  ModuleOp &module;
  RouteFlows(MLIRContext *context, ModuleOp &m, TileAnalysis &a,
             PatternBenefit benefit = 1)
    : OpConversionPattern<FlowOp>(context, benefit),
    module(m), analysis(a) {}

  LogicalResult match(Operation *op) const override {
    return success();
  }

  void addConnection(ConversionPatternRewriter &rewriter,
                     Region &r,
                     WireBundle inBundle,
                     int inIndex,
                     WireBundle outBundle,
                     int &outIndex) const {
    Block &b = r.front();
    rewriter.setInsertionPoint(b.getTerminator());
    if(outIndex == -1) {
      // Find an index that is bigger than any existing index.
      outIndex = 0;
      for (auto connectOp : b.getOps<ConnectOp>()) {
        if(connectOp.destBundle() == outBundle &&
           connectOp.destIndex() >= outIndex) {
          outIndex = connectOp.destIndex()+1;
        }
      }
    }

    // This might fail if an outIndex was exactly specified.
    ConnectOp connectOp =
      rewriter.template create<ConnectOp>(rewriter.getUnknownLoc(),
                                          inBundle,
                                          inIndex,
                                          outBundle,
                                          outIndex);
  }
  void rewrite(AIE::FlowOp op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    //    Operation *newOp = rewriter.clone(*Op);
    // Operation *newOp = rewriter.create<FlowOp>(Op->getLoc(),
    //                                            Op->getResultTypes(),
    //                                            Op->getOperands(),
    //                                            Op->getAttrs());
    //    newOp->setOperands(Op->getOperands());
    //newOp->setAttr("HasSwitchbox", BoolAttr::get(true, rewriter.getContext()));
    WireBundle sourceBundle = op.sourceBundle();
    int sourceIndex = op.sourceIndex();
    WireBundle destBundle = op.destBundle();
    int destIndex = op.destIndex();

    int col, row;
    if(TileOp source = dyn_cast_or_null<TileOp>(op.source().getDefiningOp())) {
      col = source.colIndex();
      row = source.rowIndex();
    } else if(PLIOOp source = dyn_cast_or_null<PLIOOp>(op.source().getDefiningOp())) {
      col = source.colIndex();
      row = -2;
    } else llvm_unreachable("Unimplemented case");

    int destcol, destrow;
    if(TileOp dest = dyn_cast_or_null<TileOp>(op.dest().getDefiningOp())) {
      destcol = dest.colIndex();
      destrow = dest.rowIndex();
    } else if(PLIOOp dest = dyn_cast_or_null<PLIOOp>(op.dest().getDefiningOp())) {
      destcol = dest.colIndex();
      destrow = -2;
    } else llvm_unreachable("Unimplemented case");

    if(debugRoute)
      llvm::dbgs() << "Route: " << col << "," << row << "->"
                   << destcol << "," << destrow << "\n";

    WireBundle bundle = sourceBundle;
    int index = sourceIndex;
    int nextcol = col, nextrow = row;
    WireBundle nextBundle;
    int done = false;
    while(!done) {
      // Create a connection inside this switchbox.
      WireBundle outBundle;
      int outIndex = -1; // pick connection.
      if(row > destrow) {
        outBundle = WireBundle::South;
        nextBundle = WireBundle::North;
        nextrow = row-1;
      } else if(row < destrow) {
        outBundle = WireBundle::North;
        nextBundle = WireBundle::South;
        nextrow = row+1;
      } else if(col > destcol) {
        outBundle = WireBundle::West;
        nextBundle = WireBundle::East;
        nextcol = col-1;
      } else if(col < destcol) {
        outBundle = WireBundle::East;
        nextBundle = WireBundle::West;
        nextcol = col+1;
      } else {
        assert(row == destrow && col == destcol);
        // done, so connect to the correct target bundle.
        outBundle = destBundle;
        outIndex = destIndex;
        done = true;
      }
      if(nextrow < 0) {
        ShimSwitchboxOp swOp = analysis.getShimSwitchbox(rewriter, col);
        Region &r = swOp.connections();
        addConnection(rewriter, r, bundle, index, outBundle, outIndex);
      } else {
        SwitchboxOp swOp = analysis.getSwitchbox(rewriter, col, row);
        int col, row;
        col = swOp.colIndex();
        row = swOp.rowIndex();
        Region &r = swOp.connections();
        addConnection(rewriter, r, bundle, index, outBundle, outIndex);
        if(debugRoute)
          llvm::dbgs() << "Route@(" << col << "," << row << "): " << stringifyWireBundle(bundle) << ":"
                       << index << "->" << stringifyWireBundle(outBundle) << ":" << outIndex
                       << "\n";
      }
      if(done) break;
      col = nextcol;
      row = nextrow;
      bundle = nextBundle;
      index = outIndex;
    }

    rewriter.eraseOp(Op);
  }
};


struct AIECreateSwitchboxPass : public PassWrapper<AIECreateSwitchboxPass,
                                                   OperationPass<ModuleOp>> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    TileAnalysis analysis(m);
    IntegerType i32 = IntegerType::get(32, m.getContext());

    OpBuilder builder(m.getBody()->getTerminator());

    // Populate tiles and switchboxes.
    for(int col = 0; col <= analysis.getMaxCol(); col++) {
      for(int row = 0; row <= analysis.getMaxRow(); row++) {
        analysis.getTile(builder, col, row);
      }
    }
    for(int col = 0; col <= analysis.getMaxCol(); col++) {
      for(int row = 0; row <= analysis.getMaxRow(); row++) {
        analysis.getSwitchbox(builder, col, row);
      }
    }
    for(int col = 0; col <= analysis.getMaxCol(); col++) {
      analysis.getShimSwitchbox(builder, col);
    }
    for(int col = 0; col <= analysis.getMaxCol(); col++) {
      analysis.getPLIO(builder, col);
    }
    // Populate wires betweeh switchboxes and tiles.
    for(int col = 0; col <= analysis.getMaxCol(); col++) {
      for(int row = 0; row <= analysis.getMaxRow(); row++) {
        auto tile = analysis.getTile(builder, col, row);
        auto sw = analysis.getSwitchbox(builder, col, row);
        WireOp meWireOp =
          builder.create<WireOp>(builder.getUnknownLoc(),
                                 tile,
                                 WireBundle::ME,
                                 sw,
                                 WireBundle::ME);
        WireOp dmaWireOp =
          builder.create<WireOp>(builder.getUnknownLoc(),
                                 tile,
                                 WireBundle::DMA,
                                 sw,
                                 WireBundle::DMA);
        if(col > 0) {
          auto westsw = analysis.getSwitchbox(builder, col-1, row);
          WireOp switchboxOp =
            builder.create<WireOp>(builder.getUnknownLoc(),
                                   westsw,
                                   WireBundle::East,
                                   sw,
                                   WireBundle::West);
        }
        if(row > 0) {
          auto southsw = analysis.getSwitchbox(builder, col, row-1);
          WireOp switchboxOp =
            builder.create<WireOp>(builder.getUnknownLoc(),
                                   southsw,
                                   WireBundle::North,
                                   sw,
                                   WireBundle::South);
        } else if(row == 0) {
          auto southsw = analysis.getShimSwitchbox(builder, col);
          WireOp switchboxOp =
            builder.create<WireOp>(builder.getUnknownLoc(),
                                   southsw,
                                   WireBundle::North,
                                   sw,
                                   WireBundle::South);
          if(col > 0) {
            auto westsw = analysis.getShimSwitchbox(builder, col-1);
            WireOp switchboxOp =
              builder.create<WireOp>(builder.getUnknownLoc(),
                                     westsw,
                                     WireBundle::East,
                                     southsw,
                                     WireBundle::West);
          }
          auto plio = analysis.getPLIO(builder, col);
          WireOp PLIOOp =
            builder.create<WireOp>(builder.getUnknownLoc(),
                                   plio,
                                   WireBundle::North,
                                   southsw,
                                   WireBundle::South);
        }
      }
    }
    ConversionTarget target(getContext());
    target.addLegalOp<ConnectOp>();
    //target.addDynamicallyLegalOp<FlowOp>([](FlowOp op) { return (bool)op.getOperation()->getAttrOfType<BoolAttr>("IsRouted"); });

    OwningRewritePatternList patterns;
    patterns.insert<RouteFlows>(m.getContext(), m, analysis);
    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
    return;
  }
};

void xilinx::AIE::registerAIECreateFlowsPass() {
    PassRegistration<AIECreateSwitchboxPass>(
      "aie-create-flows",
      "Extract flows from a placed and routed design");
}
