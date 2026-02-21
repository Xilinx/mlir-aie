//===- AIEXToEmitC.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion from dynamic AIEX operations to EmitC dialect
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Create a emitc.call operation that calls a helper function
emitc::CallOp createEmitCCall(OpBuilder &builder, Location loc,
                              StringRef funcName, TypeRange resultTypes,
                              ValueRange operands) {
  return builder.create<emitc::CallOp>(loc, funcName, resultTypes, operands);
}

/// Create a emitc.call operation without result
emitc::CallOp createEmitCCallVoid(OpBuilder &builder, Location loc,
                                  StringRef funcName, ValueRange operands) {
  return createEmitCCall(builder, loc, funcName, TypeRange{}, operands);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert aiex.npu.dyn_write32 to EmitC function call
struct NpuDynWrite32ToEmitCPattern
    : public OpRewritePattern<NpuDynWrite32Op> {
  using OpRewritePattern<NpuDynWrite32Op>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(NpuDynWrite32Op op,
                  PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Generate call to: append_npu_write32(address, value)
    // The transaction vector will be managed at function level

    SmallVector<Value> callOperands = {op.getAddress(), op.getValue()};
    createEmitCCallVoid(rewriter, loc, "append_npu_write32", callOperands);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert aiex.npu.dyn_maskwrite32 to EmitC function call
struct NpuDynMaskWrite32ToEmitCPattern
    : public OpRewritePattern<NpuDynMaskWrite32Op> {
  using OpRewritePattern<NpuDynMaskWrite32Op>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(NpuDynMaskWrite32Op op,
                  PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> callOperands = {op.getAddress(),
                                       op.getValue(), op.getMask()};
    createEmitCCallVoid(rewriter, loc, "append_npu_maskwrite32", callOperands);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert aiex.npu.dyn_sync to EmitC function call
struct NpuDynSyncToEmitCPattern
    : public OpRewritePattern<NpuDynSyncOp> {
  using OpRewritePattern<NpuDynSyncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(NpuDynSyncOp op,
                  PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> callOperands = {
        op.getColumn(), op.getRow(),
        op.getDirection(), op.getChannel(),
        op.getColumnNum(), op.getRowNum()};
    createEmitCCallVoid(rewriter, loc, "append_npu_sync", callOperands);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert aiex.npu.dyn_dma_memcpy_nd to EmitC function call
struct NpuDynDmaMemcpyNdToEmitCPattern
    : public OpRewritePattern<NpuDynDmaMemcpyNdOp> {
  using OpRewritePattern<NpuDynDmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(NpuDynDmaMemcpyNdOp op,
                  PatternRewriter &rewriter) const override {
    // This is more complex - needs to encode BD configuration
    // For now, emit a comment placeholder
    // TODO: Implement full BD encoding

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertAIEXToEmitCPass
    : public ConvertAIEXToEmitCBase<ConvertAIEXToEmitCPass> {
  void runOnOperation() override {
    auto runtimeSeq = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // Walk all operations in the runtime sequence body and convert
    SmallVector<Operation*> opsToConvert;
    runtimeSeq.walk([&](Operation *op) {
      if (isa<NpuDynWrite32Op, NpuDynMaskWrite32Op, NpuDynSyncOp,
              NpuDynDmaMemcpyNdOp>(op)) {
        opsToConvert.push_back(op);
      }
    });

    for (Operation *op : opsToConvert) {
      builder.setInsertionPoint(op);

      if (auto writeOp = dyn_cast<NpuDynWrite32Op>(op)) {
        SmallVector<Value> callOperands = {writeOp.getAddress(), writeOp.getValue()};
        createEmitCCallVoid(builder, op->getLoc(), "append_npu_write32", callOperands);
        op->erase();
      } else if (auto maskWriteOp = dyn_cast<NpuDynMaskWrite32Op>(op)) {
        SmallVector<Value> callOperands = {maskWriteOp.getAddress(),
                                           maskWriteOp.getValue(), maskWriteOp.getMask()};
        createEmitCCallVoid(builder, op->getLoc(), "append_npu_maskwrite32", callOperands);
        op->erase();
      } else if (auto syncOp = dyn_cast<NpuDynSyncOp>(op)) {
        SmallVector<Value> callOperands = {
            syncOp.getColumn(), syncOp.getRow(),
            syncOp.getDirection(), syncOp.getChannel(),
            syncOp.getColumnNum(), syncOp.getRowNum()};
        createEmitCCallVoid(builder, op->getLoc(), "append_npu_sync", callOperands);
        op->erase();
      } else if (auto dmaOp = dyn_cast<NpuDynDmaMemcpyNdOp>(op)) {
        // TODO: Implement DMA conversion
        op->erase();
      }
    }
  }
};

} // namespace

namespace xilinx {
namespace AIEX {

std::unique_ptr<OperationPass<AIE::RuntimeSequenceOp>>
createConvertAIEXToEmitCPass() {
  return std::make_unique<ConvertAIEXToEmitCPass>();
}

} // namespace AIEX
} // namespace xilinx
