#include "aie/AIEDialect.h"
#include "aie/AIENetlistAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-create-lower-packet"

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

struct AIEBroadcastPacketPass
    : public AIEBroadcastPacketBase<AIEBroadcastPacketPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    for (auto broadcastpacket : m.getOps<BroadcastPacketOp>()) {
      Region &r = broadcastpacket.ports();
      Block &b = r.front();
      Port sourcePort = broadcastpacket.port();
      TileOp srcTile = dyn_cast<TileOp>(broadcastpacket.tile().getDefiningOp());

      for (Operation &Op : b.getOperations()) {
        if (BPIDOp bpid = dyn_cast<BPIDOp>(Op)) {
          Region &r_bpid = bpid.ports();
          Block &b_bpid = r_bpid.front();
          int flowID = bpid.IDInt();
          builder.setInsertionPointAfter(broadcastpacket);
          PacketFlowOp pkFlow =
              builder.create<PacketFlowOp>(builder.getUnknownLoc(), flowID);
          Region &r_pkFlow = pkFlow.ports();
          Block *b_pkFlow = builder.createBlock(&r_pkFlow);
          builder.setInsertionPointToStart(b_pkFlow);
          builder.create<PacketSourceOp>(builder.getUnknownLoc(), srcTile,
                                         sourcePort.first, sourcePort.second);
          for (Operation &op : b_bpid.getOperations()) {
            if (BPDestOp bpdest = dyn_cast<BPDestOp>(op)) {
              TileOp destTile = dyn_cast<TileOp>(bpdest.tile().getDefiningOp());
              Port destPort = bpdest.port();
              builder.setInsertionPointToEnd(b_pkFlow);
              builder.create<PacketDestOp>(builder.getUnknownLoc(), destTile,
                                           destPort.first, destPort.second);
            }
          }
          builder.setInsertionPointToEnd(b_pkFlow);
          builder.create<EndOp>(builder.getUnknownLoc());
        }
      }
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<AIEOpRemoval<BroadcastPacketOp>>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEBroadcastPacketPass() {
  return std::make_unique<AIEBroadcastPacketPass>();
}