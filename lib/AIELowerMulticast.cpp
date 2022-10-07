#include "aie/AIEDialect.h"
#include "aie/AIENetlistAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-lower-multicast"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIELowerMulticastPass : public AIEMulticastBase<AIELowerMulticastPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    for (auto multicast : m.getOps<MulticastOp>()) {
      Region &r = multicast.getPorts();
      Block &b = r.front();
      Port sourcePort = multicast.port();
      TileOp srcTile = dyn_cast<TileOp>(multicast.getTile().getDefiningOp());
      for (Operation &Op : b.getOperations()) {
        if (MultiDestOp multiDest = dyn_cast<MultiDestOp>(Op)) {
          TileOp destTile =
              dyn_cast<TileOp>(multiDest.getTile().getDefiningOp());
          Port destPort = multiDest.port();
          builder.create<FlowOp>(builder.getUnknownLoc(), srcTile,
                                 sourcePort.first, sourcePort.second, destTile,
                                 destPort.first, destPort.second);
        }
      }
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<AIEOpRemoval<MulticastOp>>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIELowerMulticastPass() {
  return std::make_unique<AIELowerMulticastPass>();
}