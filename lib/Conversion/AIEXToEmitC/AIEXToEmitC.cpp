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
#include "mlir/Transforms/DialectConversion.h"

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
    : public OpConversionPattern<NpuDynWrite32Op> {
  using OpConversionPattern<NpuDynWrite32Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NpuDynWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Generate call to: append_npu_write32(address, value)
    // The transaction vector will be managed at function level

    SmallVector<Value> callOperands = {adaptor.getAddress(),
                                       adaptor.getValue()};
    createEmitCCallVoid(rewriter, loc, "append_npu_write32", callOperands);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert aiex.npu.dyn_maskwrite32 to EmitC function call
struct NpuDynMaskWrite32ToEmitCPattern
    : public OpConversionPattern<NpuDynMaskWrite32Op> {
  using OpConversionPattern<NpuDynMaskWrite32Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NpuDynMaskWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> callOperands = {adaptor.getAddress(),
                                       adaptor.getValue(), adaptor.getMask()};
    createEmitCCallVoid(rewriter, loc, "append_npu_maskwrite32", callOperands);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert aiex.npu.dyn_sync to EmitC function call
struct NpuDynSyncToEmitCPattern
    : public OpConversionPattern<NpuDynSyncOp> {
  using OpConversionPattern<NpuDynSyncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NpuDynSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> callOperands = {
        adaptor.getColumn(), adaptor.getRow(),
        adaptor.getDirection(), adaptor.getChannel(),
        adaptor.getColumnNum(), adaptor.getRowNum()};
    createEmitCCallVoid(rewriter, loc, "append_npu_sync", callOperands);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert aiex.npu.dyn_dma_memcpy_nd to EmitC function call
struct NpuDynDmaMemcpyNdToEmitCPattern
    : public OpConversionPattern<NpuDynDmaMemcpyNdOp> {
  using OpConversionPattern<NpuDynDmaMemcpyNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NpuDynDmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

    ConversionTarget target(*context);
    target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect,
                           scf::SCFDialect, arith::ArithDialect>();
    target.addIllegalOp<NpuDynWrite32Op, NpuDynMaskWrite32Op, NpuDynSyncOp,
                        NpuDynDmaMemcpyNdOp>();

    RewritePatternSet patterns(context);
    patterns.add<NpuDynWrite32ToEmitCPattern, NpuDynMaskWrite32ToEmitCPattern,
                 NpuDynSyncToEmitCPattern, NpuDynDmaMemcpyNdToEmitCPattern>(
        context);

    if (failed(applyPartialConversion(runtimeSeq, target,
                                      std::move(patterns)))) {
      signalPassFailure();
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
