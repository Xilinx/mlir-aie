//===- AIETxnToControlPacket.cpp --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-txn-to-control"

using namespace mlir;
using namespace xilinx;

namespace {

/// Pattern to convert transaction operations to control operations
struct BlockWriteToControlPacketPattern
    : public OpRewritePattern<AIEX::NpuBlockWriteOp> {
  using OpRewritePattern<AIEX::NpuBlockWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::NpuBlockWriteOp op,
                                PatternRewriter &rewriter) const override {

    Value memref = op.getData();
    int64_t width = cast<MemRefType>(memref.getType()).getElementTypeBitWidth();
    if (width != 32) {
      return op.emitWarning("Only 32-bit data type is supported for now");
    }

    memref::GetGlobalOp getGlobal = memref.getDefiningOp<memref::GetGlobalOp>();
    if (!getGlobal) {
      return op.emitError("Only MemRefs from memref.get_global are supported");
    }

    auto global = dyn_cast_if_present<memref::GlobalOp>(
        op->getParentOfType<AIE::DeviceOp>().lookupSymbol(getGlobal.getName()));
    if (!global) {
      return op.emitError("Global symbol not found");
    }

    auto initVal = global.getInitialValue();
    if (!initVal) {
      return op.emitError("Global symbol has no initial value");
    }

    auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
    if (!data) {
      return op.emitError(
          "Global symbol initial value is not a dense int array");
    }
    std::vector<int32_t> dataVec(data.value_begin<int32_t>(),
                                 data.value_end<int32_t>());
    rewriter.create<AIEX::NpuControlPacketOp>(
        op->getLoc(), op.getAddressAttr(), nullptr,
        /*opcode*/ rewriter.getI32IntegerAttr(0),
        /*stream_id*/ rewriter.getI32IntegerAttr(0),
        DenseI32ArrayAttr::get(op->getContext(), dataVec));
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {

struct AIETxnToControlPacketPass
    : public AIEX::AIETxnToControlPacketBase<AIETxnToControlPacketPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addIllegalOp<AIEX::NpuBlockWriteOp>();
    target.addLegalOp<AIEX::NpuControlPacketOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<BlockWriteToControlPacketPattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIETxnToControlPacketPass() {
  return std::make_unique<AIETxnToControlPacketPass>();
}
