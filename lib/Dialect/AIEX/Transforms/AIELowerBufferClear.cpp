//===- AIELowerBufferClear.cpp ------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERBUFFERCLEAR
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-lower-buffer-clear"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

struct BufferClearToBlockWritePattern : OpConversionPattern<BufferClearOp> {
  using OpConversionPattern<BufferClearOp>::OpConversionPattern;

  // Memo and next free name index for the zero-data globals, owned by the
  // pass. Mirrors WriteBdToBlockWritePattern in AIEDmaToNpu.cpp.
  llvm::DenseMap<Attribute, memref::GlobalOp> *dedupCache;
  unsigned *nextId;

  BufferClearToBlockWritePattern(
      MLIRContext *context,
      llvm::DenseMap<Attribute, memref::GlobalOp> *dedupCache, unsigned *nextId)
      : OpConversionPattern(context), dedupCache(dedupCache), nextId(nextId) {}

  LogicalResult
  matchAndRewrite(BufferClearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::TileOp tile = op.getTileOp();
    int col = tile.getCol();
    int row = tile.getRow();
    uint32_t length = op.getLength();

    // The emitted instruction grows with `length`, unlike the fixed-size
    // register pulse aiex.core_reset lowers to. See AIEX.td.
    std::vector<uint32_t> words(length, 0);

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    memref::GlobalOp global;
    {
      // Insert the zero-data global at device scope, ahead of the runtime
      // sequence, mirroring WriteBdToBlockWritePattern's insertion point.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op->getParentOfType<AIE::RuntimeSequenceOp>());
      global = getOrCreateDataMemref(rewriter, dev, op.getLoc(), words,
                                     dedupCache, nextId);
    }
    auto memref = memref::GetGlobalOp::create(
        rewriter, op.getLoc(), global.getType(), global.getName());

    IntegerAttr addrAttr = rewriter.getUI32IntegerAttr(op.getAddress());
    IntegerAttr colAttr = rewriter.getI32IntegerAttr(col);
    IntegerAttr rowAttr = rewriter.getI32IntegerAttr(row);
    rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(
        op, addrAttr, memref.getResult(), nullptr, colAttr, rowAttr);

    return success();
  };
};

struct AIELowerBufferClearPass
    : public xilinx::AIEX::impl::AIELowerBufferClearBase<
          AIELowerBufferClearPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuBlockWriteOp>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<BufferClearOp>();

    // Seed from existing globals so same-length clears share one, as
    // AIEDmaToNpuPass does for BD data.
    llvm::DenseMap<Attribute, memref::GlobalOp> dedupCache;
    unsigned nextId = 0;
    for (auto g : device.getOps<memref::GlobalOp>()) {
      if (auto initVal = g.getInitialValue())
        dedupCache.try_emplace(*initVal, g);
      StringRef suffix = g.getSymName();
      if (suffix.consume_front("blockwrite_data_")) {
        unsigned idx;
        if (!suffix.getAsInteger(10, idx) && idx >= nextId)
          nextId = idx + 1;
      }
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<BufferClearToBlockWritePattern>(&getContext(), &dedupCache,
                                                 &nextId);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerBufferClearPass() {
  return std::make_unique<AIELowerBufferClearPass>();
}
