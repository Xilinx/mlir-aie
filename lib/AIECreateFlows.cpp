// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "AIEDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aie;

typedef llvm::Optional<std::pair<Operation *, Port>> PortConnection;

class CoreAnalysis {
  ModuleOp &module;
  int maxcol, maxrow;
  DenseMap<std::pair<int, int>, CoreOp> coordToCore;
  DenseMap<std::pair<int, int>, SwitchboxOp> coordToSwitchbox;
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
  CoreAnalysis(ModuleOp &m) : module(m) {
    maxcol = 0;
    maxrow = 0;
    for (auto coreOp : module.getOps<CoreOp>()) {
      int col, row;
      col = coreOp.col().getZExtValue();
      row = coreOp.row().getZExtValue();
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      assert(coordToCore.count(std::make_pair(col, row)) == 0);
      coordToCore[std::make_pair(col, row)] = coreOp;
    }
    for (auto switchboxOp : module.getOps<SwitchboxOp>()) {
      int col, row;
      col = switchboxOp.col().getZExtValue();
      row = switchboxOp.row().getZExtValue();
      assert(coordToSwitchbox.count(std::make_pair(col, row)) == 0);
      coordToSwitchbox[std::make_pair(col, row)] = switchboxOp;
    }
  }

  CoreOp getCore(OpBuilder &builder, int col, int row) {
    if(coordToCore.count(std::make_pair(col, row))) {
      return coordToCore[std::make_pair(col, row)];
    } else {
      IntegerType i32 = builder.getIntegerType(32);
      CoreOp coreOp =
        builder.create<CoreOp>(builder.getUnknownLoc(),
                               builder.getIndexType(),
                               IntegerAttr::get(i32, (int)col),
                               IntegerAttr::get(i32, (int)row));
      coordToCore[std::make_pair(col, row)] = coreOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return coreOp;
    }
  }
  SwitchboxOp getSwitchbox(OpBuilder &builder, int col, int row) {
    if(coordToSwitchbox.count(std::make_pair(col, row))) {
      return coordToSwitchbox[std::make_pair(col, row)];
    } else {
      IntegerType i32 = builder.getIntegerType(32);
      SwitchboxOp switchboxOp =
        builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                    builder.getIndexType(),
                                    IntegerAttr::get(i32, (int)col),
                                    IntegerAttr::get(i32, (int)row));
      switchboxOp.ensureTerminator(switchboxOp.connections(),
                                   builder,
                                   builder.getUnknownLoc());
      coordToSwitchbox[std::make_pair(col, row)] = switchboxOp;
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      return switchboxOp;
    }
  }
};

struct RouteFlows : public OpConversionPattern<aie::FlowOp> {
  using OpConversionPattern<aie::FlowOp>::OpConversionPattern;
  CoreAnalysis &analysis;
  ModuleOp &module;
  RouteFlows(MLIRContext *context, ModuleOp &m, CoreAnalysis &a,
             PatternBenefit benefit = 1)
    : OpConversionPattern<FlowOp>(context, benefit),
    module(m), analysis(a) {}

  PatternMatchResult match(Operation *op) const override {
    return matchSuccess();
  }

  int addConnection(ConversionPatternRewriter &rewriter,
                    SwitchboxOp switchboxOp,
                    WireBundle inBundle,
                    int inIndex,
                    WireBundle outBundle) const {
    int col, row;
    col = switchboxOp.col().getZExtValue();
    row = switchboxOp.row().getZExtValue();
    llvm::dbgs() << "Route: " << stringifyWireBundle(inBundle) << ":"
                 << inIndex << "->" << stringifyWireBundle(outBundle)
                 << "@(" << col << "," << row << ")\n";
    rewriter.setInsertionPoint(switchboxOp.connections().front().getTerminator());
    int outIndex = 0;
    // Find an index that is bigger than any existing index.
    Region &r = switchboxOp.connections();
    Block &b = r.front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if(connectOp.destBundle() == outBundle &&
         connectOp.destIndex() >= outIndex) {
        outIndex = connectOp.destIndex()+1;
      }
    }

    ConnectOp connectOp =
      rewriter.template create<ConnectOp>(rewriter.getUnknownLoc(),
                                          inBundle,
                                          APInt(32, inIndex),
                                          outBundle,
                                          APInt(32, outIndex));
    return outIndex;
  }
  void rewrite(aie::FlowOp op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    //    Operation *newOp = rewriter.clone(*Op);
    // Operation *newOp = rewriter.create<FlowOp>(Op->getLoc(),
    //                                            Op->getResultTypes(),
    //                                            Op->getOperands(),
    //                                            Op->getAttrs());
    //    newOp->setOperands(Op->getOperands());
    //newOp->setAttr("HasSwitchbox", BoolAttr::get(true, rewriter.getContext()));
    CoreOp source = dyn_cast_or_null<CoreOp>(op.source().getDefiningOp());
    WireBundle sourceBundle = op.sourceBundle();
    int sourceIndex = op.sourceIndex();
    CoreOp dest = dyn_cast_or_null<CoreOp>(op.dest().getDefiningOp());
    WireBundle destBundle = op.destBundle();
    int destIndex = op.destIndex();

    int col, row;
    col = source.col().getZExtValue();
    row = source.row().getZExtValue();
    int destcol, destrow;
    destcol = dest.col().getZExtValue();
    destrow = dest.row().getZExtValue();

    WireBundle bundle = sourceBundle;
    int index = sourceIndex;
    int nextcol = col, nextrow = row;
    WireBundle nextBundle;
    int done = false;
    while(!done) {
      // Create a connection inside this switchbox.
      SwitchboxOp swOp = analysis.getSwitchbox(rewriter, nextcol, nextrow);
      WireBundle outBundle;
      if(col > destcol) {
        outBundle = WireBundle::West;
        nextBundle = WireBundle::East;
        nextcol = col-1;
      } else if(col < destcol) {
        outBundle = WireBundle::East;
        nextBundle = WireBundle::West;
        nextcol = col+1;
      } else if(row > destrow) {
        outBundle = WireBundle::South;
        nextBundle = WireBundle::North;
        nextrow = row-1;
      } else if(row < destrow) {
        outBundle = WireBundle::North;
        nextBundle = WireBundle::South;
        nextrow = row+1;
      } else {
        assert(row == destrow && col == destcol);
        // done, so connect to the correct target bundle.
        outBundle = destBundle;
        done = true;
      }
      index = addConnection(rewriter, swOp, bundle, index, outBundle);
      if(done) break;
      col = nextcol;
      row = nextrow;
      bundle = nextBundle;
    }

    rewriter.eraseOp(Op);
    //rewriter.replaceOp(Op, newOp->getOpResults());

    //rewriter.setInsertionPoint(Op->getBlock()->getTerminator());
    // // Corresponds to ME0 and ME1
    // for(int i = 0; i < 2; i++) {
    //   PortConnection t = analysis.getConnectedCore(op,
    //                                                std::make_pair(WireBundle(0), i));
    //   if(t.hasValue()) {
    //     Operation *destOp = t.getValue().first;
    //     Port destPort = t.getValue().second;
    //     IntegerType i32 = IntegerType::get(32, rewriter.getContext());
    //     Operation *flowOp = rewriter.create<FlowOp>(Op->getLoc(),
    //                                                 newOp->getResult(0),
    //                                                 IntegerAttr::get(i32, (int)WireBundle(0)),
    //                                                 IntegerAttr::get(i32, i),
    //                                                 destOp->getResult(0),
    //                                                 IntegerAttr::get(i32, (int)destPort.first),
    //                                                 IntegerAttr::get(i32, (int)destPort.second));
    //   }
    // }
    // updateRootInPlace(op, [&] {
    //   });
  }
};


struct AIECreateSwitchboxPass : public ModulePass<AIECreateSwitchboxPass> {
  void runOnModule() override {

    ModuleOp m = getModule();
    CoreAnalysis analysis(m);
    IntegerType i32 = IntegerType::get(32, m.getContext());

    OpBuilder builder(m.getBody()->getTerminator());

    // Populate cores and switchboxes.
    for(int col = 1; col <= analysis.getMaxCol(); col++) {
      for(int row = 1; row <= analysis.getMaxRow(); row++) {
        analysis.getCore(builder, col, row);
      }
    }
    for(int col = 1; col <= analysis.getMaxCol(); col++) {
      for(int row = 1; row <= analysis.getMaxRow(); row++) {
        analysis.getSwitchbox(builder, col, row);
      }
    }
    // Populate wires betweeh switchboxes and cores.
    for(int col = 1; col <= analysis.getMaxCol(); col++) {
      for(int row = 1; row <= analysis.getMaxRow(); row++) {
        auto core = analysis.getCore(builder, col, row);
        auto sw = analysis.getSwitchbox(builder, col, row);
        WireOp switchboxOp =
          builder.create<WireOp>(builder.getUnknownLoc(),
                                 core,
                                 IntegerAttr::get(i32, (int)WireBundle::ME),
                                 sw,
                                 IntegerAttr::get(i32, (int)WireBundle::ME));
        if(col > 1) {
          auto westsw = analysis.getSwitchbox(builder, col-1, row);
          WireOp switchboxOp =
            builder.create<WireOp>(builder.getUnknownLoc(),
                                   westsw,
                                   IntegerAttr::get(i32, (int)WireBundle::East),
                                   sw,
                                   IntegerAttr::get(i32, (int)WireBundle::West));
        }
        if(row > 1) {
          auto southsw = analysis.getSwitchbox(builder, col, row-1);
          WireOp switchboxOp =
            builder.create<WireOp>(builder.getUnknownLoc(),
                                   southsw,
                                   IntegerAttr::get(i32, (int)WireBundle::North),
                                   sw,
                                   IntegerAttr::get(i32, (int)WireBundle::South));
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

void xilinx::aie::registerAIECreateFlowsPass() {
    PassRegistration<AIECreateSwitchboxPass>(
      "aie-create-flows",
      "Extract flows from a placed and routed design");
}
