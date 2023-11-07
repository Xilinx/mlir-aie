//===- AIEVecToLLVM.cpp - AIEVec to LLVM dialect conversion ---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"

#include <sstream>

using namespace mlir;

namespace xilinx {
namespace aievec {

struct BufferParams {
  uint32_t start;
  uint32_t offsets;
  uint32_t offsets_hi;
  uint32_t step;
  uint32_t square;
};

std::string getVectorTypeString(VectorType type, bool abbrev = false,
                                bool acc = false) {
  std::stringstream ss;
  auto size = getVectorLaneSize(type);
  ss << "v" << size;
  if (auto intType = dyn_cast<IntegerType>(type.getElementType())) {
    ss << (acc ? "acc" : abbrev ? "i" : "int") << intType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type.getElementType())) {
    ss << (abbrev ? "f" : "float");
  }
  return ss.str();
}

std::string getMulOrFMAIntrinsicName(Operation *op) {
  std::string baseName;
  Value lhs, rhs, result;
  if (auto mulOp = dyn_cast<aievec::MulOp>(op)) {
    baseName = "mul";
    lhs = mulOp.getLhs();
    rhs = mulOp.getRhs();
    result = mulOp.getResult();
  } else if (auto fmaOp = dyn_cast<aievec::FMAOp>(op)) {
    baseName = "mac";
    lhs = fmaOp.getLhs();
    rhs = fmaOp.getRhs();
    result = fmaOp.getResult();
  }
  VectorType resultType = cast<VectorType>(result.getType());
  int resultSize = getVectorLaneSize(resultType);
  std::stringstream ss;
  ss << "llvm.aie.";
  if (auto intType = dyn_cast<IntegerType>(resultType.getElementType())) {
    ss << baseName;
    ss << resultSize << "."
       << getVectorTypeString(cast<VectorType>(lhs.getType()));
  } else if (dyn_cast<FloatType>(resultType.getElementType())) {
    ss << "vfp" << baseName;
  }
  return ss.str();
}

// Squashes the easy-to-read 16-bit square encoding into
// the 8-bit encoding the configuration register uses
uint32_t encodeSquare(uint32_t square) {
  uint32_t out = 0;
  out |= ((square >> 0) & 0x3) << 0;
  out |= ((square >> 4) & 0x3) << 2;
  out |= ((square >> 8) & 0x3) << 4;
  out |= ((square >> 12) & 0x3) << 6;
  return out & 0xFF;
}

// Encode the configuration register with buffer parameters and options
// TODO: struct to handle this?
void encodeConf(uint32_t conf[2], const BufferParams &x, const BufferParams &z,
                bool sub) {
  conf[0] |= ((x.step & 0x3F) << 0) | ((z.step & 0x3F) << 8);
  conf[1] |= (encodeSquare(x.square) << 0) | (encodeSquare(z.square) << 8);
  conf[1] |= sub << 17;
}

class AddOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::AddOp> {
public:
  using ConvertOpToLLVMPattern<aievec::AddOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.add conversion is not implemented\n";
    return failure();
  }
};

class SubOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::SubOp> {
public:
  using ConvertOpToLLVMPattern<aievec::SubOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.sub conversion is not implemented\n";
    return failure();
  }
};

class FMAOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::FMAOp> {
public:
  using ConvertOpToLLVMPattern<aievec::FMAOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::FMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    auto startType = IntegerType::get(context, 32);
    auto offsetsType = VectorType::get({2}, IntegerType::get(context, 32));
    auto confType = VectorType::get({2}, IntegerType::get(context, 32));

    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getMulOrFMAIntrinsicName(op);
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(
              op.getResult().getType(),
              {op.getLhs().getType(), op.getRhs().getType(),
               op.getAcc().getType(), startType, /* xstart */
               startType,                        /* ystart */
               startType,                        /* zstart */
               offsetsType,                      /* xoffsets */
               offsetsType,                      /* zoffsets */
               confType}));
    }

    // Parse the string attribute values
    BufferParams x = {};
    BufferParams z = {};
    op.getXstart().getAsInteger(0, x.start);
    op.getXoffsets().getAsInteger(0, x.offsets);
    op.getXoffsetsHi().getAsInteger(0, x.offsets_hi);
    op.getXstep().getAsInteger(0, x.step);
    op.getXsquare().getAsInteger(0, x.square);
    op.getZstart().getAsInteger(0, z.start);
    op.getZoffsets().getAsInteger(0, z.offsets);
    op.getZoffsetsHi().getAsInteger(0, z.offsets_hi);
    op.getZstep().getAsInteger(0, z.step);
    op.getZsquare().getAsInteger(0, z.square);

    // Encode the configuration register
    uint32_t conf[2] = {0, 0};
    encodeConf(conf, x, z, op.getFmsub());

    // Create the constants and replace the op
    auto xstartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(x.start));
    auto ystartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(0));
    auto zstartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(z.start));
    auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)x.offsets, (int32_t)x.offsets_hi}));
    auto zoffsetsVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)z.offsets, (int32_t)z.offsets_hi}));
    auto confVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), confType,
        rewriter.getI32VectorAttr({(int32_t)conf[0], (int32_t)conf[1]}));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, func,
        ValueRange{op.getLhs(), op.getRhs(), op.getAcc(), xstartVal, ystartVal,
                   zstartVal, xoffsetsVal, zoffsetsVal, confVal});
    return success();
  }
};

class MulOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::MulOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    auto startType = IntegerType::get(context, 32);
    auto offsetsType = VectorType::get({2}, IntegerType::get(context, 32));
    auto confType = VectorType::get({2}, IntegerType::get(context, 32));

    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getMulOrFMAIntrinsicName(op);
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(op.getResult().getType(),
                                      {op.getLhs().getType(),
                                       op.getRhs().getType(),
                                       startType,   /* xstart */
                                       startType,   /* ystart */
                                       startType,   /* zstart */
                                       offsetsType, /* xoffsets */
                                       offsetsType, /* zoffsets */
                                       confType}));
    }

    // Parse the string attribute values
    BufferParams x = {};
    BufferParams z = {};
    op.getXstart().getAsInteger(0, x.start);
    op.getXoffsets().getAsInteger(0, x.offsets);
    op.getXoffsetsHi().getAsInteger(0, x.offsets_hi);
    op.getXstep().getAsInteger(0, x.step);
    op.getXsquare().getAsInteger(0, x.square);
    op.getZstart().getAsInteger(0, z.start);
    op.getZoffsets().getAsInteger(0, z.offsets);
    op.getZoffsetsHi().getAsInteger(0, z.offsets_hi);
    op.getZstep().getAsInteger(0, z.step);
    op.getZsquare().getAsInteger(0, z.square);

    // Encode the configuration register
    uint32_t conf[2] = {0, 0};
    encodeConf(conf, x, z, false);

    // Create the constants and replace the op
    auto xstartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(x.start));
    auto ystartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(0));
    auto zstartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(z.start));
    auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)x.offsets, (int32_t)x.offsets_hi}));
    auto zoffsetsVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)z.offsets, (int32_t)z.offsets_hi}));
    auto confVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), confType,
        rewriter.getI32VectorAttr({(int32_t)conf[0], (int32_t)conf[1]}));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, func,
        ValueRange{op.getLhs(), op.getRhs(), xstartVal, ystartVal, zstartVal,
                   xoffsetsVal, zoffsetsVal, confVal});
    return success();
  }
};

class UPSOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::UPSOp> {
public:
  using ConvertOpToLLVMPattern<aievec::UPSOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::UPSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.ups conversion is not implemented\n";
    return failure();
  }
};

class SRSOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::SRSOp> {
public:
  using ConvertOpToLLVMPattern<aievec::SRSOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::SRSOp op) {
    std::stringstream ss;
    ss << "llvm.aie.";

    // Determine the prefix
    auto sourceType = cast<VectorType>(op.getSource().getType());
    auto resultType = cast<VectorType>(op.getResult().getType());
    auto sourceElType = cast<IntegerType>(sourceType.getElementType());
    auto resultElType = cast<IntegerType>(resultType.getElementType());

    auto sourceElWidth = sourceElType.getWidth();
    auto resultElWidth = resultElType.getWidth();

    if (sourceElWidth == 48 && resultElWidth == 8) {
      ss << (resultElType.getSignedness() == IntegerType::Unsigned ? 'u' : 'b');
    } else if ((sourceElWidth == 48 && resultElWidth == 32) ||
               (sourceElWidth == 80 && resultElWidth == 64)) {
      ss << 'l';
    }
    ss << "srs." << getVectorTypeString(resultType, true) << "."
       << getVectorTypeString(sourceType, false, true);

    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::SRSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getIntrinsicName(op);
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));
    auto shiftType = IntegerType::get(context, 32);

    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(op.getResult().getType(),
                                      {op.getSource().getType(), shiftType}));
    }

    // Create a constant for the shift value
    auto shiftVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), shiftType, rewriter.getI32IntegerAttr(op.getShift()));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, func, ValueRange{op.getSource(), shiftVal});
    return success();
  }
};

class UPDOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::UPDOp> {
public:
  using ConvertOpToLLVMPattern<aievec::UPDOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::UPDOp op, int loadSize) {
    auto resultType = cast<VectorType>(op.getResult().getType());
    std::stringstream ss;
    ss << "llvm.aie.upd.";
    ss << (loadSize == 128 ? 'v' : loadSize == 256 ? 'w' : 'x') << ".";
    ss << getVectorTypeString(resultType) << ".";
    // The index affects which intrinsic to call
    ss << (op.getIndex() == 0 ? "lo" : "hi");
    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::UPDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    // A bit more complicated: load the vector, then update result vector
    // AIE1 is capable of 128-bit on one bank and 256-bit loads on even-odd
    // banks Identify size of update
    int vecSizeInBits =
        getVectorSizeInBits(cast<VectorType>(op.getResult().getType()));

    auto ptr = this->getStridedElementPtr(
        op->getLoc(), cast<MemRefType>(op.getSource().getType()),
        adaptor.getSource(), adaptor.getIndices(), rewriter);

    // TODO: handle the offset field

    if (vecSizeInBits <= 256) {
      // Total <=256-bit updates are much simpler:
      // we can do a direct load into the vector register
      // look at the indices to calculate the address
      auto vectorPtrType = LLVM::LLVMPointerType::get(
          getContext(),
          cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt());
      auto castedPtr =
          rewriter.create<LLVM::BitcastOp>(op->getLoc(), vectorPtrType, ptr);
      auto vecType = cast<VectorType>(op.getResult().getType());
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, vecType, castedPtr, 1);
    } else {
      // Total >256-bit updates will require upd ops to fill the whole vector
      // each UDP op represents one of these 256-bit loads and updates

      // Determine the load size
      // TODO: no examples of 1024-bit output vectors: doesn't feel right
      // to attempt a 512-bit load to do an update like this
      int loadSize = vecSizeInBits == 256   ? 128
                     : vecSizeInBits == 512 ? 256
                                            : 512;

      // Create a vectorType for the load proper
      // Load half of the final result vector
      auto resultType = cast<VectorType>(op.getResult().getType());
      int lanes = getVectorLaneSize(resultType);
      auto loadType =
          VectorType::get({(int64_t)lanes / 2}, resultType.getElementType());

      // Load the vector
      auto vectorPtrType = LLVM::LLVMPointerType::get(
          getContext(),
          cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt());
      auto castedPtr =
          rewriter.create<LLVM::BitcastOp>(op->getLoc(), vectorPtrType, ptr);
      auto loadValue =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), loadType, castedPtr, 1);

      // Get set up for the intrinsic
      std::string intrinsicName = getIntrinsicName(op, loadSize);

      // If the intrinsic declaration doesn't exist, create it
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
          StringAttr::get(context, intrinsicName));

      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = rewriter.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), intrinsicName,
            LLVM::LLVMFunctionType::get(resultType, {resultType, loadType}));
      }

      // Determine what the destination is
      Value destValue;
      if (adaptor.getVector()) {
        // This UPD is using an existing destination vector
        destValue = adaptor.getVector();
      } else {
        // If this UPD is not working off of an existing destination vector,
        // create an undefined vector as the destination

        // TODO: determine if the undef intrinsic is needed or if an LLVM undef
        // suffices
        // destValue = rewriter.create<LLVM::UndefOp>(op->getLoc(), resultType);

        std::stringstream ss;
        ss << "llvm.aie." << getVectorTypeString(resultType) << ".undef";
        std::string intrinsicName = ss.str();

        auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
            StringAttr::get(rewriter.getContext(), intrinsicName));

        if (!func) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(module.getBody());
          func = rewriter.create<LLVM::LLVMFuncOp>(
              rewriter.getUnknownLoc(), intrinsicName,
              LLVM::LLVMFunctionType::get(resultType, {}));
        }
        destValue =
            rewriter.create<LLVM::CallOp>(op->getLoc(), func, ValueRange{})
                ->getOpResult(0);
      }

      // Create our call
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, func, ValueRange{destValue, loadValue});
    }

    return success();
  }
};

class ConcatOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ConcatOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ConcatOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::ConcatOp op) {
    auto sourceType = cast<VectorType>(op.getSources()[0].getType());
    std::stringstream ss;
    ss << "llvm.aie.concat.";
    ss << getVectorTypeString(sourceType, true);
    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getIntrinsicName(op);
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

    // TODO: support for more than 2 vector concat
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(
              op.getResult().getType(),
              {op.getSources()[0].getType(), op.getSources()[1].getType()}));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, func, ValueRange{op.getSources()[0], op.getSources()[1]});
    return success();
  }
};

class ExtOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::ExtOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExtOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::ExtOp op) {
    auto sourceType = cast<VectorType>(op.getSource().getType());
    auto resultType = cast<VectorType>(op.getResult().getType());
    int resultSize = getVectorSizeInBits(resultType);
    std::stringstream ss;
    ss << "llvm.aie.ext.";
    ss << (resultSize == 128 ? 'v' : resultSize == 256 ? 'w' : 'x') << ".";
    ss << getVectorTypeString(sourceType, true) << ".";
    // The index actually affects which intrinsic to call
    ss << (op.getIndex() == 0 ? "lo" : "hi");
    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::ExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getIntrinsicName(op);
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(op.getResult().getType(),
                                      {op.getSource().getType()}));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func,
                                              ValueRange{op.getSource()});
    return success();
  }
};

class SelectOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::SelectOp> {
public:
  using ConvertOpToLLVMPattern<aievec::SelectOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::SelectOp op) {
    auto xbuffType = cast<VectorType>(op.getXbuff().getType());
    std::stringstream ss;
    ss << "llvm.aie.prim." << getVectorTypeString(xbuffType) << ".select";
    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    auto selectType = IntegerType::get(context, 32);
    auto startType = IntegerType::get(context, 32);
    auto offsetsType = VectorType::get({2}, IntegerType::get(context, 32));
    auto confType = VectorType::get({2}, IntegerType::get(context, 32));

    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getIntrinsicName(op);
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(op.getResult().getType(),
                                      {op.getXbuff().getType(), selectType,
                                       startType,   /* xstart */
                                       startType,   /* ystart */
                                       offsetsType, /* xoffsets */
                                       offsetsType, /* yoffsets */
                                       confType}));
    }

    // Parse the string attribute values
    uint32_t select = 0;
    BufferParams x = {};
    BufferParams y = {};
    BufferParams z = {};

    op.getSelect().getAsInteger(0, select);
    op.getXstart().getAsInteger(0, x.start);
    op.getXoffsets().getAsInteger(0, x.offsets);
    op.getXoffsetsHi().getAsInteger(0, x.offsets_hi);
    op.getXsquare().getAsInteger(0, x.square);
    op.getYstart().getAsInteger(0, y.start);
    op.getYoffsets().getAsInteger(0, y.offsets);
    op.getYoffsetsHi().getAsInteger(0, y.offsets_hi);
    op.getYsquare().getAsInteger(0, y.square);

    // Encode the configuration register
    uint32_t conf[2] = {0, 0};
    encodeConf(conf, x, z, false);
    conf[1] |= encodeSquare(y.square) << 21;

    // Create the constants and replace the op
    auto selectVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), selectType, rewriter.getI32IntegerAttr(select));
    auto xstartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(x.start));
    auto ystartVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), startType, rewriter.getI32IntegerAttr(y.start));
    auto xoffsetsVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)x.offsets, (int32_t)x.offsets_hi}));
    auto yoffsetsVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)y.offsets, (int32_t)y.offsets_hi}));
    auto confVal = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), confType,
        rewriter.getI32VectorAttr({(int32_t)conf[0], (int32_t)conf[1]}));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, func,
        ValueRange{op.getXbuff(), selectVal, xstartVal, ystartVal, xoffsetsVal,
                   yoffsetsVal, confVal});
    return success();
  }
};

class PackOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::PackOp> {
public:
  using ConvertOpToLLVMPattern<aievec::PackOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::PackOp op) {
    auto sourceType = cast<VectorType>(op.getSource().getType());
    std::stringstream ss;
    ss << "llvm.aie.pack." << getVectorTypeString(sourceType);
    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    // If the intrinsic declaration doesn't exist, create it
    std::string intrinsicName = getIntrinsicName(op);
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
        StringAttr::get(context, intrinsicName));

    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), intrinsicName,
          LLVM::LLVMFunctionType::get(op.getResult().getType(),
                                      {op.getSource().getType()}));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func,
                                              ValueRange{op.getSource()});
    return success();
  }
};

class UnpackOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::UnpackOp> {
public:
  using ConvertOpToLLVMPattern<aievec::UnpackOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.unpack conversion is not implemented\n";
    return failure();
  }
};

class BroadcastOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::BroadcastOp> {
public:
  using ConvertOpToLLVMPattern<aievec::BroadcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.broadcast conversion is not implemented\n";
    return failure();
  }
};

class FMAElemOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::FMAElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::FMAElemOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::FMAElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.mac_elem conversion is not implemented\n";
    return failure();
  }
};

void populateAIEVecToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<AddOpConversion,
               SubOpConversion,
               FMAOpConversion,
               MulOpConversion,
               UPSOpConversion,
               SRSOpConversion,
               UPDOpConversion,
               ConcatOpConversion,
               ExtOpConversion,
               SelectOpConversion,
               PackOpConversion,
               UnpackOpConversion,
               BroadcastOpConversion,
               FMAElemOpConversion>(converter);
  // clang-format on
}

struct ConvertAIEVecToLLVMPass
    : public ConvertAIEVecToLLVMBase<ConvertAIEVecToLLVMPass> {
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMTypeConverter converter(&getContext());
    populateAIEVecToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass() {
  return std::make_unique<ConvertAIEVecToLLVMPass>();
}

} // namespace aievec
} // namespace xilinx
