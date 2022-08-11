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

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include <sstream>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

namespace xilinx {
namespace aievec {

std::string getVectorTypeString(VectorType type) {
  std::stringstream ss;
  auto size = getVectorLaneSize(type);
  ss << "v" << size;
  if (auto intType  = type.getElementType().dyn_cast<IntegerType>()) {
    ss << "int" << intType.getWidth();
  } else if (auto floatType = type.getElementType().dyn_cast<FloatType>()) {
    ss << "float";
  }
  return ss.str();
}

bool accIsDouble(VectorType type) {
  auto lanes = getVectorLaneSize(type);
  auto accWidth = type.getElementType().cast<IntegerType>().getWidth();
  if (accWidth == 80) accWidth = 96;
  return (lanes * accWidth) > 384;
}

std::string getMulOrFMABuiltinName(Operation *op) {
  std::string baseName;
  Value lhs, rhs, result;
  if (auto mulOp = dyn_cast<xilinx::aievec::MulOp>(op)) {
    baseName = "mul";
    lhs = mulOp.lhs();
    rhs = mulOp.rhs();
    result = mulOp.result();
  } else if (auto fmaOp = dyn_cast<xilinx::aievec::FMAOp>(op)) {
    baseName = "mac";
    lhs = fmaOp.lhs();
    rhs = fmaOp.rhs();
    result = fmaOp.result();
  }
  VectorType resultType = result.getType().cast<VectorType>();
  int resultSize = getVectorLaneSize(resultType);
  std::stringstream ss;
  ss << "__builtin_aie_";
  if (auto intType = resultType.getElementType().dyn_cast<IntegerType>()) {
    ss << baseName;
    ss << resultSize << "_"
       << getVectorTypeString(lhs.getType().cast<VectorType>()) << "_"
       << getVectorTypeString(rhs.getType().cast<VectorType>()) << "_"
       << (accIsDouble(resultType) ? "bm" : "am")
       << "_sw" << intType.getWidth();
  } else if (resultType.getElementType().dyn_cast<FloatType>()) {
    ss << "vfp" << baseName;
  }
  return ss.str();
}

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
      auto module = op->getParentOfType<ModuleOp>();
      MLIRContext *context = rewriter.getContext();

      auto startType = IntegerType::get(context, 32);
      auto offsetsType = VectorType::get({2}, IntegerType::get(context, 32));
      auto confType = VectorType::get({2}, IntegerType::get(context, 32));

      // If the intrinsic declaration doesn't exist, create it
      std::string builtinName = getMulOrFMABuiltinName(op);
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, builtinName));

      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = rewriter.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), builtinName,
            LLVM::LLVMFunctionType::get(op.result().getType(),
                                        {op.lhs().getType(),
                                         op.rhs().getType(),
                                         startType, /* xstart */
                                         startType, /* ystart */
                                         startType, /* zstart */
                                         offsetsType, /* xoffsets */
                                         offsetsType, /* zoffsets */
                                         confType})
                                       );
        rewriter.setInsertionPoint(op);
      }

      // Get the attribute values
      int32_t xstart, xoffsets, xoffsets_hi, xstep, xsquare, ystart, zstart, zoffsets, zoffsets_hi, zstep, zsquare;
      ystart = 0;
      op.xstart().getAsInteger(0, xstart);
      op.xoffsets().getAsInteger(0, xoffsets);
      op.xoffsets_hi().getAsInteger(0, xoffsets_hi);
      op.xstep().getAsInteger(0, xstep);
      op.xsquare().getAsInteger(0, xsquare);
      op.zstart().getAsInteger(0, zstart);
      op.zoffsets().getAsInteger(0, zoffsets);
      op.zoffsets_hi().getAsInteger(0, zoffsets_hi);
      op.zstep().getAsInteger(0, zstep);
      op.zsquare().getAsInteger(0, zsquare);

      // Encode the configuration register
      int32_t conf[2] = {0,0};
      conf[0] |= ((xstep & 0x3F) << 0) | ((zstep & 0x3F) << 8);
      conf[1] |= ((xsquare & 0xFF) << 0) | ((zsquare & 0xFF) << 8);

      // Create the constants and replace the op
      auto xstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xstart));
      auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), offsetsType, rewriter.getI32VectorAttr({xoffsets, xoffsets_hi}));
      auto ystartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(0));
      auto zstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zstart));
      auto zoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), offsetsType, rewriter.getI32VectorAttr({zoffsets, zoffsets_hi}));
      auto confVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), confType, rewriter.getI32VectorAttr({conf[0], conf[1]}));
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, ValueRange{op.lhs(), op.rhs(), xstartVal, ystartVal, zstartVal, xoffsetsVal, zoffsetsVal, confVal});
      return success();
    }
};

class FMAOpConversion : public mlir::ConvertOpToLLVMPattern<xilinx::aievec::FMAOp> {
  public:
    using ConvertOpToLLVMPattern<xilinx::aievec::FMAOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(xilinx::aievec::FMAOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      auto module = op->getParentOfType<ModuleOp>();
      MLIRContext *context = rewriter.getContext();

      auto startType = IntegerType::get(context, 32);
      auto offsetsType = VectorType::get({2}, IntegerType::get(context, 32));
      auto confType = VectorType::get({2}, IntegerType::get(context, 32));

      // If the intrinsic declaration doesn't exist, create it
      std::string builtinName = getMulOrFMABuiltinName(op);
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, builtinName));

      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = rewriter.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), builtinName,
            LLVM::LLVMFunctionType::get(op.result().getType(),
                                        {op.lhs().getType(),
                                         op.rhs().getType(),
                                         op.acc().getType(),
                                         startType, /* xstart */
                                         startType, /* ystart */
                                         startType, /* zstart */
                                         offsetsType, /* xoffsets */
                                         offsetsType, /* zoffsets */
                                         confType})
                                       );
        rewriter.setInsertionPoint(op);
      }

      // Get the attribute values
      int32_t xstart, xoffsets, xoffsets_hi, xstep, xsquare, ystart, zstart, zoffsets, zoffsets_hi, zstep, zsquare;
      ystart = 0;
      op.xstart().getAsInteger(0, xstart);
      op.xoffsets().getAsInteger(0, xoffsets);
      op.xoffsets_hi().getAsInteger(0, xoffsets_hi);
      op.xstep().getAsInteger(0, xstep);
      op.xsquare().getAsInteger(0, xsquare);
      op.zstart().getAsInteger(0, zstart);
      op.zoffsets().getAsInteger(0, zoffsets);
      op.zoffsets_hi().getAsInteger(0, zoffsets_hi);
      op.zstep().getAsInteger(0, zstep);
      op.zsquare().getAsInteger(0, zsquare);

      // Encode the configuration register
      int32_t conf[2] = {0,0};
      conf[0] |= ((xstep & 0x3F) << 0) | ((zstep & 0x3F) << 8);
      conf[1] |= ((xsquare & 0xFF) << 0) | ((zsquare & 0xFF) << 8);

      // Create the constants and replace the op
      auto xstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(xstart));
      auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), offsetsType, rewriter.getI32VectorAttr({xoffsets, xoffsets_hi}));
      auto ystartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(0));
      auto zstartVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), startType, rewriter.getI32IntegerAttr(zstart));
      auto zoffsetsVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), offsetsType, rewriter.getI32VectorAttr({zoffsets, zoffsets_hi}));
      auto confVal = rewriter.create<LLVM::ConstantOp>(op->getLoc(), confType, rewriter.getI32VectorAttr({conf[0], conf[1]}));
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, ValueRange{op.lhs(), op.rhs(), op.acc(), xstartVal, ystartVal, zstartVal, xoffsetsVal, zoffsetsVal, confVal});
      return success();
    }
};

// Generates the buffer address offset given the current index values
Value createLinearizedAccess(Value source, SmallVector<Value, 4> indices) {
  auto memRefType = source.getType().dyn_cast<MemRefType>();
  assert(memRefType &&
         "cannot creating linearized expression for non-memref type");
  ArrayRef<int64_t> stride = memRefType.getShape();

  Value *cur = nullptr;
  for (int dim = memRefType.getRank() - 1; dim >= 0; --dim) {
    // k * z_dim + j * y_dim + k
    // add the innermost , multiply each index by the dimension
  }
  return source;

  /*
  MemRefType memRefType = source.getType().dyn_cast<MemRefType>();
  assert(memRefType &&
         "cannot creating linearized expression for non-memref type");
  ArrayRef<int64_t> stride = memRefType.getShape();

  // The stride and indices size must match
  if (stride.size() != indices.size() ||
      (int)stride.size() != memRefType.getRank())
    return failure();

  // A stride contains two parts:
  int64_t numPart = 1;   // for static shaped dims
  std::string paramPart; // for dynamic shaped dims

  SmallVector<std::string, 4> accessVec;
  for (int dim = memRefType.getRank() - 1; dim >= 0; --dim) {
    // All the indices in the access expression must already be emitted
    if (!emitter.hasValueInScope(indices[dim]))
      return failure();

    // Form the access string for this dimension
    std::string cur;
    if (!paramPart.empty())
      cur = paramPart + "*";
    if (numPart > 1)
      cur += std::to_string(numPart) + "*";
    cur += emitter.getOrCreateName(indices[dim]);
    accessVec.push_back(cur);

    // Now update the numPart and paramPart to form the stride for the next
    // dimension
    if (memRefType.isDynamicDim(dim)) {
      StringRef param = emitter.getMemRefDimParam(source, dim);
      paramPart = param.str() + (paramPart.empty() ? "" : "*" + paramPart);
    } else
      numPart *= stride[dim];
  }
  // All the strides are in accessVec. Compose them
  while (!accessVec.empty()) {
    access += (access.empty() ? "" : "+") + accessVec.back();
    accessVec.pop_back();
  }
  // If the access is empty, make '0' as default access
  if (access.empty())
    access = "0";

  return success();
  */
}
/*
static Value castDataPtr(ConversionPatternRewriter &rewriter, Location loc,
                         Value ptr, MemRefType memRefType, Type vt) {
  auto pType = LLVM::LLVMPointerType::get(vt, memRefType.getMemorySpaceAsInt());
  return rewriter.create<LLVM::BitcastOp>(loc, pType, ptr);
}
*/

class UPDOpConversion : public mlir::ConvertOpToLLVMPattern<xilinx::aievec::UPDOp> {
  public:
    using ConvertOpToLLVMPattern<xilinx::aievec::UPDOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(xilinx::aievec::UPDOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // A bit more complicated: load the vector, then update result vector
      // AIE1 is capable of 128-bit on one bank and 256-bit loads on even-odd banks
      // Identify size of update
      int vecSizeInBits = getVectorSizeInBits(op.result().getType().cast<VectorType>());

      if (vecSizeInBits <= 256) {
        // total <=256-bit updates are much simpler:
        // we can do a direct load into the vector register
        // look at the indices to calculate the address
        // thanks vector for telling me about this function :)
        auto ptr = this->getStridedElementPtr(op->getLoc(),
          op.source().getType().cast<MemRefType>(),
          adaptor.source(),
          op.indices(),
          rewriter);
        auto vectorPtrType = LLVM::LLVMPointerType::get(
          op.result().getType().cast<VectorType>(),
          op.source().getType().cast<MemRefType>().getMemorySpaceAsInt());
        auto castedPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(),
          vectorPtrType,
          ptr);
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, castedPtr, 1);
      } else {
        // total >256-bit updates will require upd ops to fill the whole vector
        // each UDP op represents one of these 256-bit loads and updates
        return failure();
      }

      return success();

    }
};


void populateAIEVecToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns) {
  patterns.add<xilinx::aievec::SRSOpConversion>(converter);
  patterns.add<xilinx::aievec::MulOpConversion>(converter);
  patterns.add<xilinx::aievec::FMAOpConversion>(converter);
  //patterns.add<xilinx::aievec::UPDOpConversion>(converter);
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
