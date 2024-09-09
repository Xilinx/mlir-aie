
#include "../PassDetail.h"

#include "aie/Conversion/AIEToTransaction/AIEToTransaction.h"

using namespace mlir;

namespace {

struct ConvertAIEToTransactionPass
    : ConvertAIEToTransactionBase<ConvertAIEToTransactionPass> {
  void runOnOperation() override {
    auto device = getOperation();
    device.dump();
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createConvertAIEToTransactionPass() {
  return std::make_unique<ConvertAIEToTransactionPass>();
}
