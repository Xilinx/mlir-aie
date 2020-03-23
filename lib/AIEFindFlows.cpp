// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "AIEDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aie;

typedef llvm::Optional<std::pair<Operation *, SlavePortEnum>> PortConnection;
class ConnectivityAnalysis {
  ModuleOp &module;

public:
  ConnectivityAnalysis(ModuleOp &m) : module(m) {}

private:
  PortConnection
  getConnectionThroughWire(Operation *op,
                           MasterPortEnum masterPort) const {
    for (auto wireOp : module.getOps<WireOp>()) {
      if(wireOp.source().getDefiningOp() == op &&
         wireOp.sourcePort() == masterPort) {
        Operation *other = wireOp.dest().getDefiningOp();
        SlavePortEnum otherPort = wireOp.destPort();
        return std::make_pair(other, otherPort);
      }
      // if(wireOp.dest().getDefiningOp() == op &&
      //    wireOp.destPort() == masterPort) {
      //   Operation *other = wireOp.source().getDefiningOp();
      //   SlavePortEnum otherPort = wireOp.sourcePort();
      //   return std::make_pair(other, otherPort);
      // }
    }
    return None;
  }

  llvm::Optional<MasterPortEnum>
  getConnectionThroughSwitchbox(SwitchboxOp op,
                                SlavePortEnum sourcePort) const {
    Region &r = op.connections();
    Block &b = r.front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if(connectOp.sourcePort() == sourcePort) {
        return connectOp.destPort();
      }
    }
    return llvm::None;
  }

public:
  PortConnection
  getConnectedCore(CoreOp coreOp,
                   MasterPortEnum port) const {
    Operation *next = coreOp.getOperation();
    llvm::Optional<MasterPortEnum> nextPort = port;
    PortConnection t = getConnectionThroughWire(next, nextPort.getValue());
    assert(t.hasValue());

    bool valid = false;
    while(true) {
      Operation *other = t.getValue().first;
      SlavePortEnum otherPort = t.getValue().second;
      if(auto coreOp = dyn_cast_or_null<CoreOp>(other)) {
        break;
      } else if(auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        nextPort = getConnectionThroughSwitchbox(switchOp, otherPort);
        next = switchOp;
      }
      if(!nextPort.hasValue()) {
        break;
      }
      t = getConnectionThroughWire(next, nextPort.getValue());
      assert(t.hasValue());
      // other = t.getValue().first;
      // otherPort = t.getValue().second;
    }
    if(auto destCoreOp = dyn_cast_or_null<CoreOp>(t.getValue().first)) {
      return t;
    } else {
      return None;
    }
  }
};

struct StartFlow : public OpConversionPattern<aie::CoreOp> {
  using OpConversionPattern<aie::CoreOp>::OpConversionPattern;
  ConnectivityAnalysis analysis;
  ModuleOp &module;
  StartFlow(MLIRContext *context, ModuleOp &m, ConnectivityAnalysis a,
            PatternBenefit benefit = 1)
      : OpConversionPattern<CoreOp>(context, benefit),
    module(m), analysis(a) {}

  PatternMatchResult match(Operation *op) const override {
    return matchSuccess();
  }

  void rewrite(aie::CoreOp op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    Operation *newOp = rewriter.clone(*Op);
    // Operation *newOp = rewriter.create<CoreOp>(Op->getLoc(),
    //                                            Op->getResultTypes(),
    //                                            Op->getOperands(),
    //                                            Op->getAttrs());
    //    newOp->setOperands(Op->getOperands());
    newOp->setAttr("HasFlow", BoolAttr::get(true, rewriter.getContext()));
    //rewriter.eraseOp(Op);
    rewriter.replaceOp(Op, newOp->getOpResults());

    rewriter.setInsertionPoint(Op->getBlock()->getTerminator());
    // Corresponds to ME0 and ME1
    for(int i = 0; i < 2; i++) {
      PortConnection t = analysis.getConnectedCore(op, MasterPortEnum(i));
      if(t.hasValue()) {
        Operation *destOp = t.getValue().first;
        SlavePortEnum destPort = t.getValue().second;

        SmallVector<Type, 4> voidType;
        SmallVector<Value, 4> flowOperands;
        SmallVector<NamedAttribute, 4> flowAttrs;
        flowOperands.push_back(newOp->getResult(0));
        flowOperands.push_back(destOp->getResult(0));
        IntegerType i32 = IntegerType::get(32, rewriter.getContext());
        flowAttrs.push_back(std::make_pair(Identifier::get("sourcePort", rewriter.getContext()),
                                         IntegerAttr::get(i32, i)));
        flowAttrs.push_back(std::make_pair(Identifier::get("destPort", rewriter.getContext()),
                                           IntegerAttr::get(i32, (int)destPort)));
        //flowAttrs.push_back(SlavePortEnum(0));
        Operation *flowOp = rewriter.create<FlowOp>(Op->getLoc(),
                                                    voidType,
                                                    flowOperands,
                                                    flowAttrs);
      }
    }
    // updateRootInPlace(op, [&] {
    //   });
  }
};


struct AIEFindFlowsPass : public ModulePass<AIEFindFlowsPass> {
  void runOnModule() override {
    //    OwningRewritePatternList patterns;
    // populateWithGenerated(&getContext(), &patterns);

    // RewritePatternMatcher matcher(patterns);
    // matcher.matchAndRewrite(getOperation(), *this);
    //    applyPatternsGreedily(getModule(), patterns);
    // ConversionTarget target(getContext());
    // target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    // (void)applyPartialConversion(getModule(), target, patterns);

    ModuleOp m = getModule();
    ConnectivityAnalysis analysis(m);

    for (auto coreOp : m.getOps<CoreOp>()) {
      PortConnection t = analysis.getConnectedCore(coreOp, MasterPortEnum(0));
      //      coreOp.dump();
      if(t.hasValue()) {
        coreOp.getOperation()->print(llvm::dbgs());
        llvm::dbgs() << " -> \n";
        t.getValue().first->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
        // while(nextPort.hasValue()) {
      // for (auto wireOp : m.getOps<WireOp>()) {
      //   if(wireOp.source() == coreOp) {
      //     auto other = wireOp.dest();
      //     if(auto switchOp = cast<SwitchboxOp>(other.getDefiningOp())) {
      //       nextPort = getConnectionThroughSwitchbox(switchOp, wireOp.destPort());
      //       next = switchOp;
      //     }
      //   }
      //   if(wireOp.dest() == coreOp) {
      //     wireOp.dump();
      //   }
      // }
    }

    // Region &r = m.getRegion(0);
    // Block &b = r.getBlock(0);
    // for(auto &op : b) {
    //   op.dump();
    // }
    ConversionTarget target(getContext());
    target.addLegalOp<FlowOp>();
    target.addDynamicallyLegalOp<CoreOp>([](CoreOp op) { return (bool)op.getOperation()->getAttrOfType<BoolAttr>("HasFlow"); });
    //   target.addDynamicallyLegalDialect<AIEDialect>();

    OwningRewritePatternList patterns;
    patterns.insert<StartFlow>(m.getContext(), m, analysis);
    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
    return;
  }
};

void xilinx::aie::registerAIEFindFlowsPass() {
    PassRegistration<AIEFindFlowsPass>(
      "aie-find-flows",
      "Extract flows from a placed and routed design");
}
