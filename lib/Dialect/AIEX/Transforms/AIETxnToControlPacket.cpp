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

#include <algorithm>

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

/// Pattern to split control packets into smaller control packets
struct ControlPacketSplitPattern
    : public OpRewritePattern<AIEX::NpuControlPacketOp> {
  using OpRewritePattern<AIEX::NpuControlPacketOp>::OpRewritePattern;

  ControlPacketSplitPattern(MLIRContext *ctx, uint32_t max_payload_size)
      : OpRewritePattern(ctx), max_payload_size(max_payload_size) {}

  LogicalResult matchAndRewrite(AIEX::NpuControlPacketOp op,
                                PatternRewriter &rewriter) const override {
    auto data = op.getData();
    if (!data)
      return failure();

    uint32_t numElements = op.getDataAttr().size();

    if (numElements <= max_payload_size) {
      return failure(); // No splitting needed
    }

    auto chunks = llvm::divideCeil(numElements, max_payload_size);
    auto context = op.getContext();
    auto loc = op.getLoc();
    for (unsigned i = 0; i < chunks; ++i) {
      uint32_t startIdx = i * max_payload_size;
      uint32_t endIdx = std::min(startIdx + max_payload_size, numElements);

      SmallVector<int32_t, 4> chunkData;
      for (auto it = data->begin() + startIdx; it != data->begin() + endIdx;
           ++it) {
        chunkData.push_back(*it);
      }

      // Increment the address for each chunk
      auto incrementedAddress = rewriter.getUI32IntegerAttr(
          op.getAddress() + (i * 4 * sizeof(uint32_t)));

      rewriter.create<AIEX::NpuControlPacketOp>(
          loc, incrementedAddress, nullptr, op.getOpcodeAttr(),
          op.getStreamIdAttr(), DenseI32ArrayAttr::get(context, chunkData));
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  uint32_t max_payload_size;
};

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

struct AIELegalizeControlPacketPass
    : public AIEX::AIELegalizeControlPacketBase<AIELegalizeControlPacketPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<AIEX::NpuControlPacketOp>([](Operation *op) {
      auto packetOp = cast<AIEX::NpuControlPacketOp>(op);
      // Check the data size
      return packetOp.getDataAttr().size() <= 4;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<ControlPacketSplitPattern>(&getContext(), 4);

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

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIELegalizeControlPacketPass() {
  return std::make_unique<AIELegalizeControlPacketPass>();
}