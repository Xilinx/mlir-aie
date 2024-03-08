//===- AIEVecToLLVM.cpp - AIEVec to LLVM dialect conversion ---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include <numeric>
#include <sstream>

using namespace mlir;

namespace xilinx::aievec {

inline static Value bitcastValueToType(OpBuilder &builder, Location loc,
                                       Value val, Type dstTy) {
  return builder.create<LLVM::BitcastOp>(loc, dstTy, val).getResult();
}

// This function emits the instructions required to widen a 128b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<4xi32> to respect the intrinsic signature.
inline static Value widen128bVectorValueTo512b(OpBuilder &builder, Location loc,
                                               Value val) {
  return builder
      .create<xllvm::VectorSetI512I128IntrOp>(
          loc, VectorType::get({16}, builder.getI32Type()),
          bitcastValueToType(builder, loc, val,
                             VectorType::get({4}, builder.getI32Type())))
      .getResult();
}

// This function emits the instructions required to widen a 256b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<8xi32> to respect the intrinsic signature. It will also materialize
// a constant 0, used as an insertion index.
inline static Value widen256bVectorValueTo512b(OpBuilder &builder, Location loc,
                                               Value val) {
  auto cst0 =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), (int32_t)0);
  return builder
      .create<xllvm::VectorSetI512I256IntrOp>(
          loc, VectorType::get({16}, builder.getI32Type()),
          bitcastValueToType(builder, loc, val,
                             VectorType::get({8}, builder.getI32Type())),
          cst0)
      .getResult();
}

// This function emits the sequence of operations that forces a value into a
// specific type. This may include widening vectors to match a specific bit
// length.
static Value forceCastValueToType(OpBuilder &builder, Location loc, Value val,
                                  Type type) {
  auto valTy = val.getType();
  if (valTy == type)
    return val;
  auto srcVecTy = dyn_cast<VectorType>(valTy);
  if (srcVecTy) {
    auto dstVecTy = dyn_cast<VectorType>(type);
    assert(dstVecTy && "vector values cannot be forced into a non-vector type");
    assert(srcVecTy.getRank() == 1 && dstVecTy.getRank() == 1 &&
           "only flat 1D vectors can be force casted");
    int64_t dstVecLength =
        dstVecTy.getElementTypeBitWidth() * dstVecTy.getShape()[0];
    int64_t srcVecLength =
        srcVecTy.getElementTypeBitWidth() * srcVecTy.getShape()[0];
    if (srcVecLength != dstVecLength) {
      assert(srcVecLength < dstVecLength &&
             "only widening forced casts are supported");
      assert(dstVecLength == 512 &&
             (srcVecLength == 128 || srcVecLength == 256) &&
             "only 128b to 512b and 256b to 512b forced casts are supported");
      if (srcVecLength == 128)
        val = widen128bVectorValueTo512b(builder, loc, val);
      else
        val = widen256bVectorValueTo512b(builder, loc, val);
    }
  }
  return bitcastValueToType(builder, loc, val, type);
}

// This function emits the sequence of operations that forces a range of values
// to match the signature specified by the TypeRange. It can be used to convert
// the parameters of an op being converted to the types accepted by an
// intrinsic with a fixed signature that treats its inputs as "bags of bits".
static SmallVector<Value> forceCastOperandsToSignature(OpBuilder &builder,
                                                       Location loc,
                                                       ValueRange operands,
                                                       TypeRange signature) {
  return llvm::to_vector(llvm::map_range(
      llvm::zip_equal(operands, signature), [&](auto &&vt) -> Value {
        return forceCastValueToType(builder, loc, std::get<0>(vt),
                                    std::get<1>(vt));
      }));
}

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
  } else if (dyn_cast<FloatType>(type.getElementType())) {
    ss << (abbrev ? "f" : "float");
  }
  return ss.str();
}

std::string getMulOrFMAIntrinsicName(Operation *op) {
  std::string baseName;
  Value lhs, result;
  if (auto mulOp = dyn_cast<aievec::MulOp>(op)) {
    baseName = "mul";
    lhs = mulOp.getLhs();
    result = mulOp.getResult();
  } else if (auto fmaOp = dyn_cast<aievec::FMAOp>(op)) {
    baseName = "mac";
    lhs = fmaOp.getLhs();
    result = fmaOp.getResult();
  }
  VectorType resultType = cast<VectorType>(result.getType());
  int resultSize = getVectorLaneSize(resultType);
  std::stringstream ss;
  ss << "llvm.aie.";
  if (dyn_cast<IntegerType>(resultType.getElementType())) {
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
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, func, ValueRange{op.getSource(), op.getShift()});
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

class MatMulOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MatMulOp> {
  using ConvertOpToLLVMPattern<aievec::MatMulOp>::ConvertOpToLLVMPattern;

  struct DecodedMatMulOp {
    typedef enum { I32, I64, BF16 } Kind;

    Kind kind;
    Value lhs;
    Value rhs;
    Value acc;
    int conf;
  };

  static DecodedMatMulOp decodeMatMulOp(OpAdaptor op) {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value acc = op.getAcc();
    auto accVecTy = cast<VectorType>(acc.getType());
    if (isa<Float32Type>(accVecTy.getElementType()))
      // <4x8xbf16> x <8x4xbf16> + <4x4xf32>
      return {DecodedMatMulOp::Kind::BF16, lhs, rhs, acc, 28};

    int signConf = 0;
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
    if (auto extSIOp = lhs.getDefiningOp<arith::ExtSIOp>()) {
      lhs = extSIOp.getIn();
      lhsVecTy = cast<VectorType>(lhs.getType());
      lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
      signConf |= (1 << 9);
    } else if (auto extUIOp = lhs.getDefiningOp<arith::ExtUIOp>()) {
      lhs = extUIOp.getIn();
      lhsVecTy = cast<VectorType>(lhs.getType());
      lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
    } else {
      // NOTE: We're choosing 'signed' by default
      if (!lhsScaTy.isUnsigned())
        signConf |= (1 << 9);
    }
    auto lhsShape = lhsVecTy.getShape();

    auto rhsVecTy = cast<VectorType>(rhs.getType());
    auto rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
    if (auto extSIOp = rhs.getDefiningOp<arith::ExtSIOp>()) {
      rhs = extSIOp.getIn();
      rhsVecTy = cast<VectorType>(rhs.getType());
      rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
      signConf |= (1 << 8);
    } else if (auto extUIOp = rhs.getDefiningOp<arith::ExtUIOp>()) {
      rhs = extUIOp.getIn();
      rhsVecTy = cast<VectorType>(rhs.getType());
      rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
    } else {
      // NOTE: We're choosing 'signed' by default
      if (!rhsScaTy.isUnsigned())
        signConf |= (1 << 8);
    }

    unsigned lhsBitWidth = lhsScaTy.getWidth();
    unsigned rhsBitWidth = rhsScaTy.getWidth();
    auto accScaTy = cast<IntegerType>(accVecTy.getElementType());
    unsigned accBitWidth = accScaTy.getWidth();
    if (accBitWidth == 32) {
      if (lhsBitWidth == 8) {
        if (rhsBitWidth == 4) {
          // <4x16xi8> x <16x8xi4> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc, signConf};
        } else {
          // <4x8xi8> x <8x8xi8> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc, signConf | 8};
        }
      } else {
        if (rhsBitWidth == 8) {
          // <4x4xi16> x <4x8xi8> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc, signConf | 16};
        } else {
          // <4x2xi16> x <2x8xi16> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc, signConf | 2};
        }
      }
    }

    if (lhsBitWidth == 16) {
      if (rhsBitWidth == 8) {
        if (lhsShape == ArrayRef<int64_t>({2, 8})) {
          // <2x8xi16> x <8x8xi8> + <2x8xi64>
          return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc, signConf | 18};
        }
        // <4x8xi16> x <8x4xi8> + <4x4xi64>
        return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc, signConf | 50};
      }
      if (lhsShape == ArrayRef<int64_t>({2, 4})) {
        // <2x4xi16> x <4x8xi16> + <2x8xi64>
        return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc, signConf | 26};
      }
      // <4x4xi16> x <4x4xi16> + <4x4xi64>
      return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc, signConf | 58};
    }
    // <4x2xi32> x <2x4xi16> + <4x4xi64>
    return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc, signConf | 2};
  }

  static VectorType getFlattenedVectorType(VectorType vecTy) {
    if (vecTy.getRank() == 1)
      return vecTy;
    auto shape = vecTy.getShape();
    return VectorType::get(
        {std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())},
        vecTy.getElementType());
  }

  LogicalResult
  matchAndRewrite(aievec::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto decodedMatMulOp = decodeMatMulOp(adaptor);

    Location loc = op.getLoc();
    // Flatten the inputs
    auto lhsFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.lhs.getType()));
    decodedMatMulOp.lhs = rewriter.create<vector::ShapeCastOp>(
        loc, lhsFlattenedVecTy, decodedMatMulOp.lhs);
    auto rhsFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.rhs.getType()));
    decodedMatMulOp.rhs = rewriter.create<vector::ShapeCastOp>(
        loc, rhsFlattenedVecTy, decodedMatMulOp.rhs);
    auto accFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.acc.getType()));
    decodedMatMulOp.acc = rewriter.create<vector::ShapeCastOp>(
        loc, accFlattenedVecTy, decodedMatMulOp.acc);

    Type i32ty = rewriter.getI32Type();
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty, rewriter.getI32IntegerAttr(decodedMatMulOp.conf));
    SmallVector<Value> operands({decodedMatMulOp.lhs, decodedMatMulOp.rhs,
                                 decodedMatMulOp.acc, confCst});
    Value matMulResVal;
    if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::BF16)
      matMulResVal =
          rewriter
              .create<xllvm::MacConfBF16IntrOp>(
                  loc, VectorType::get({8}, rewriter.getI64Type()),
                  forceCastOperandsToSignature(
                      rewriter, loc, operands,
                      {VectorType::get({32}, rewriter.getBF16Type()),
                       VectorType::get({32}, rewriter.getBF16Type()),
                       VectorType::get({8}, rewriter.getI64Type()), i32ty}))
              .getResult();
    else {
      SmallVector<Type> intrFuncSig(
          {VectorType::get({64}, rewriter.getI8Type()),
           VectorType::get({16}, i32ty),
           VectorType::get({16}, rewriter.getI64Type()), i32ty});
      VectorType v16xi64ty = VectorType::get({16}, rewriter.getI64Type());
      if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::I32)
        matMulResVal = rewriter
                           .create<xllvm::MacConfAcc32IntrOp>(
                               loc, v16xi64ty,
                               forceCastOperandsToSignature(
                                   rewriter, loc, operands, intrFuncSig))
                           .getResult();
      else
        matMulResVal = rewriter
                           .create<xllvm::MacConfAcc64IntrOp>(
                               loc, v16xi64ty,
                               forceCastOperandsToSignature(
                                   rewriter, loc, operands, intrFuncSig))
                           .getResult();
    }

    auto castFromAcc =
        bitcastValueToType(rewriter, loc, matMulResVal, accFlattenedVecTy);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(),
                                                     castFromAcc);

    return success();
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
               FMAElemOpConversion,
               MatMulOpConversion>(converter);
  // clang-format on
}

struct ConvertAIEVecToLLVMPass
    : ConvertAIEVecToLLVMBase<ConvertAIEVecToLLVMPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());

    // Don't convert vector types, we want to handle multi-dimensional
    // vector on our own.
    converter.addConversion(
        [&](VectorType type) -> std::optional<Type> { return type; });

    populateAIEVecToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<AIEVecDialect>();
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect,
                           xilinx::xllvm::XLLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass() {
  return std::make_unique<ConvertAIEVecToLLVMPass>();
}

} // namespace xilinx::aievec
