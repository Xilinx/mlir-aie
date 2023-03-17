#include "aie/Dialect/AIE/AIENetlistAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
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
using namespace xilinx::AIEX;

template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

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

    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    for (auto broadcastpacket : device.getOps<BroadcastPacketOp>()) {
      Region &r = broadcastpacket.getPorts();
      Block &b = r.front();
      Port sourcePort = broadcastpacket.port();
      TileOp srcTile =
          dyn_cast<TileOp>(broadcastpacket.getTile().getDefiningOp());

      for (Operation &Op : b.getOperations()) {
        if (BPIDOp bpid = dyn_cast<BPIDOp>(Op)) {
          Region &r_bpid = bpid.getPorts();
          Block &b_bpid = r_bpid.front();
          int flowID = bpid.IDInt();
          builder.setInsertionPointAfter(broadcastpacket);
          PacketFlowOp pkFlow =
              builder.create<PacketFlowOp>(builder.getUnknownLoc(), flowID);
          Region &r_pkFlow = pkFlow.getPorts();
          Block *b_pkFlow = builder.createBlock(&r_pkFlow);
          builder.setInsertionPointToStart(b_pkFlow);
          builder.create<PacketSourceOp>(builder.getUnknownLoc(), srcTile,
                                         sourcePort.first, sourcePort.second);
          for (Operation &op : b_bpid.getOperations()) {
            if (BPDestOp bpdest = dyn_cast<BPDestOp>(op)) {
              TileOp destTile =
                  dyn_cast<TileOp>(bpdest.getTile().getDefiningOp());
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

    patterns.add<AIEOpRemoval<BroadcastPacketOp>>(device.getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIEBroadcastPacketPass() {
  return std::make_unique<AIEBroadcastPacketPass>();
}