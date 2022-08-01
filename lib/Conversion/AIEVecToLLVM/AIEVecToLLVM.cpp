//===- AIEVecToLLVM.cpp - AIEVec to LLVM dialect conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"

#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

namespace xilinx {
namespace aievec {

class SRSOpConversion : public mlir::ConvertOpToLLVMPattern<xilinx::aievec::SRSOp> {
  public:
    using ConvertOpToLLVMPattern<xilinx::aievec::SRSOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(xilinx::aievec::SRSOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // If the intrinsic declaration doesn't exist, create it
      std::string intrinsicName = "__builtin_aie_bsrs_v16i8";
      auto module = op->getParentOfType<ModuleOp>();
      MLIRContext *context = rewriter.getContext();
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));
      auto shiftType = IntegerType::get(context, 8);
      auto shiftVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), shiftType, rewriter.getI8IntegerAttr(op.shift()));

      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = rewriter.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), intrinsicName,
            LLVM::LLVMFunctionType::get(op.result().getType(),
                                        {op.source().getType(),
                                         shiftType})
                                       );
        rewriter.setInsertionPoint(op);
      }

      // Create a constant for the shift value
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, ValueRange{op.source(), shiftVal});
      return success();
    }
};

class MulOpConversion : public mlir::ConvertOpToLLVMPattern<xilinx::aievec::MulOp> {
  public:
    using ConvertOpToLLVMPattern<xilinx::aievec::MulOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(xilinx::aievec::MulOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // If the intrinsic declaration doesn't exist, create it
      //std::string intrinsicName = "__builtin_aie_mul";
      std::string intrinsicName = "mul16";
      auto module = op->getParentOfType<ModuleOp>();
      MLIRContext *context = rewriter.getContext();
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

      // Set up the function declaration
      auto startType = IntegerType::get(context, 32);
      auto offsetsType = IntegerType::get(context, 32);
      auto stepType = IntegerType::get(context, 32);
      auto squareType = IntegerType::get(context, 32);
      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = rewriter.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), intrinsicName,
            LLVM::LLVMFunctionType::get(op.result().getType(),
                                        {op.lhs().getType(),
                                         startType,
                                         offsetsType,
                                         stepType,
                                         squareType,
                                         op.rhs().getType(),
                                         startType,
                                         offsetsType,
                                         stepType,
                                         squareType})
                                       );
        rewriter.setInsertionPoint(op);
      }

      // Create a constant for the shift value
      int xstart, xoffsets, xstep, xsquare, zstart, zoffsets, zstep, zsquare;
      op.xstart().getAsInteger(0, xstart);
      op.xoffsets().getAsInteger(0, xoffsets);
      op.xstep().getAsInteger(0, xstep);
      op.xsquare().getAsInteger(0, xsquare);
      op.zstart().getAsInteger(0, zstart);
      op.zoffsets().getAsInteger(0, zoffsets);
      op.zstep().getAsInteger(0, zstep);
      op.zsquare().getAsInteger(0, zsquare);
      auto xstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xstart));
      auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xoffsets));
      auto xstepVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xstep));
      auto xsquareVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xsquare));
      auto zstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zstart));
      auto zoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zoffsets));
      auto zstepVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zstep));
      auto zsquareVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zsquare));

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, ValueRange{op.lhs(), xstartVal, xoffsetsVal, xstepVal, xsquareVal, op.rhs(), zstartVal, zoffsetsVal, zstepVal, zsquareVal});
      return success();
    }
};

class FMAOpConversion : public mlir::ConvertOpToLLVMPattern<xilinx::aievec::FMAOp> {
  public:
    using ConvertOpToLLVMPattern<xilinx::aievec::FMAOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(xilinx::aievec::FMAOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // If the intrinsic declaration doesn't exist, create it
      //std::string intrinsicName = "__builtin_aie_mac";
      std::string intrinsicName = "mac16";
      auto module = op->getParentOfType<ModuleOp>();
      MLIRContext *context = rewriter.getContext();
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

      // Set up the function declaration
      auto startType = IntegerType::get(context, 32);
      auto offsetsType = IntegerType::get(context, 32);
      auto stepType = IntegerType::get(context, 32);
      auto squareType = IntegerType::get(context, 32);
      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = rewriter.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), intrinsicName,
            LLVM::LLVMFunctionType::get(op.result().getType(),
                                        {op.acc().getType(),
                                         op.lhs().getType(),
                                         startType,
                                         offsetsType,
                                         stepType,
                                         squareType,
                                         op.rhs().getType(),
                                         startType,
                                         offsetsType,
                                         stepType,
                                         squareType})
                                       );
        rewriter.setInsertionPoint(op);
      }

      // Create a constant for the shift value
      int xstart, xoffsets, xstep, xsquare, zstart, zoffsets, zstep, zsquare;
      op.xstart().getAsInteger(0, xstart);
      op.xoffsets().getAsInteger(0, xoffsets);
      op.xstep().getAsInteger(0, xstep);
      op.xsquare().getAsInteger(0, xsquare);
      op.zstart().getAsInteger(0, zstart);
      op.zoffsets().getAsInteger(0, zoffsets);
      op.zstep().getAsInteger(0, zstep);
      op.zsquare().getAsInteger(0, zsquare);
      auto xstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xstart));
      auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xoffsets));
      auto xstepVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xstep));
      auto xsquareVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xsquare));
      auto zstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zstart));
      auto zoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zoffsets));
      auto zstepVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zstep));
      auto zsquareVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zsquare));

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, ValueRange{op.acc(), op.lhs(), xstartVal, xoffsetsVal, xstepVal, xsquareVal, op.rhs(), zstartVal, zoffsetsVal, zstepVal, zsquareVal});
      return success();

    }
};

void populateAIEVecToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns) {
  patterns.add<xilinx::aievec::SRSOpConversion>(converter);
  patterns.add<xilinx::aievec::MulOpConversion>(converter);
  patterns.add<xilinx::aievec::FMAOpConversion>(converter);
}

struct ConvertAIEVecToLLVMPass
    : public ConvertAIEVecToLLVMBase<ConvertAIEVecToLLVMPass> {
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMTypeConverter converter(&getContext());
    populateAIEVecToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> xilinx::aievec::createConvertAIEVecToLLVMPass() {
  return std::make_unique<ConvertAIEVecToLLVMPass>();
}

} // xilinx
} // aievec
