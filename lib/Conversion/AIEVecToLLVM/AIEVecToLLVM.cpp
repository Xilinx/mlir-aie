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

class MulElemOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MulElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MulElemOp>::ConvertOpToLLVMPattern;

  struct DecodedMulElemOp {
    enum class Kind {
      // DtIn0_DtIn1_DtRes_CxMxKxN
      I8_I8_I32_32x1x2x1,
      I16_I16_I32_32x1x1x1,
      I32_I32_I64_32x1x2x1,
      BF16_BF16_FP32_16x1x2x1,
      FP32_FP32_FP32_16x1x1x1,
      UNSUPPORTED
      // TODO: I16_I16_I64_16x1x2x1
    };

    Kind kind;
    int conf;
  };

  // sgn_x: Sign mask of matrix X. If it is one matrix X is interpreted as
  // signed, else it treated as unsigned.
  // sgn_y: Sign mask of matrix Y. If it is one matrix Y is interpreted as
  // signed, else it treated as unsigned.
  // amode/bmode/variant: config acc width, mul precision, and mul mode
  // zero_acc: Zeroing of acc1. If it is one then acc1 is zeroed.
  // shift16: Shift mask of acc1. If a bit is set the <<16 operation will be
  // executed on acc1.
  // sub_mul: Negation mask of the matrix multiplication result. If it is
  // one the result of the operation will be negated.
  // sub_acc1: Negation mask of acc1. If it is one acc1 will be negated.
  // sub_acc2: Negation mask of acc2. If it is one acc2 will be negated.
  // sub_mask: Negation mask of complex multiplications. Negates a term of a
  // complex multiplication.
  static int aiev2_mul_mac_compute_control(int sgn_x, int sgn_y, int amode,
                                           int bmode, int variant, int zero_acc,
                                           int shift16, int sub_mul,
                                           int sub_acc1, int sub_acc2,
                                           int sub_mask) {
    return ((unsigned)sub_mask << 16) | ((unsigned)shift16 << 10) |
           ((unsigned)sub_mul << 11) | ((unsigned)sub_acc1 << 12) |
           ((unsigned)sub_acc2 << 13) | ((unsigned)amode << 1) |
           ((unsigned)bmode << 3) | ((unsigned)variant << 5) |
           (((unsigned)sgn_x << 9) | ((unsigned)sgn_y << 8)) |
           ((unsigned)zero_acc << 0);
  }

  static DecodedMulElemOp decodeMulElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (lhsScaTy.isa<IntegerType>()) {
      if (lhsBitWidth == 8) {
        return {DecodedMulElemOp::Kind::I8_I8_I32_32x1x2x1,
                aiev2_mul_mac_compute_control(
                    /*sgn_x=*/1, /*sgn_y=*/1, /*amode=*/0, /*bmode=*/1,
                    /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      } else if (lhsBitWidth == 16) {
        return {DecodedMulElemOp::Kind::I16_I16_I32_32x1x1x1,
                aiev2_mul_mac_compute_control(
                    /*sgn_x=*/1, /*sgn_y=*/1, /*amode=*/0, /*bmode=*/3,
                    /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      } else if (lhsBitWidth == 32) {
        // emulated I32 mul_elem
        return {DecodedMulElemOp::Kind::I32_I32_I64_32x1x2x1, -1};
      }
    } else {
      // Float types
      if (lhsBitWidth == 16) {
        return {DecodedMulElemOp::Kind::BF16_BF16_FP32_16x1x2x1,
                aiev2_mul_mac_compute_control(
                    /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
                    /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      } else if (lhsBitWidth == 32) {
        // emulated FP32 mul_elem
        return {DecodedMulElemOp::Kind::FP32_FP32_FP32_16x1x1x1, -1};
      }
    }

    return {DecodedMulElemOp::Kind::UNSUPPORTED, -1};
  }

  // This conversion pattern implements the below CPP emulated I32 mul_elem.
  // INTRINSIC(v16acc64)
  // mul_elem_16_2(v16int32 a0, v16int32 a1, v16int32 b0, v16int32 b1) {
  //   v32uint16 a_lo = (v32uint16)shuffle(a0, a1, 2);
  //   v32int16 a_hi = (v32int16)shuffle(a0, a1, 3);
  //   v32uint16 b_lo = (v32uint16)shuffle(b0, b1, 2);
  //   v32int16 b_hi = (v32int16)shuffle(b0, b1, 3);
  //   v16acc64 acc = ::mul_elem_16_2(a_hi, b_hi);
  //   acc = mac_elem_16_2_conf(a_hi, 1, b_lo, false, acc, 0, 1, 0, 0);
  //   acc = mac_elem_16_2_conf(a_lo, false, b_hi, 1, acc, 0, 0, 0, 0);
  //   acc = mac_elem_16_2_conf(a_lo, false, b_lo, false, acc, 0, 1, 0, 0);
  //   return acc;
  // }
  // Caller to the above CPP intrinsic:
  // v16int32 v1 = LHS();
  // v16int32 v2 = RHS();
  // v16acc64 v3 = mul_elem_16_2(v1, broadcast_zero_s32(), v2,
  // undef_v16int32());
  LogicalResult
  convertToEmulatedI32MulElem(aievec::MulElemOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {

    Location loc = op.getLoc();
    auto zeroCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto a0 = adaptor.getLhs();
    auto a1 = rewriter.create<xllvm::VectorBroadcast32I512IntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()), zeroCst);
    auto b0 = adaptor.getRhs();
    auto b1 = rewriter.create<xllvm::UndefV16I32IntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()));

    // 4* Shuffle
    auto a_lo = rewriter.create<xllvm::VectorShuffleIntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()), a0, a1,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(2)));
    auto a_hi = rewriter.create<xllvm::VectorShuffleIntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()), a0, a1,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(3)));
    auto b_lo = rewriter.create<xllvm::VectorShuffleIntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()), b0, b1,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(2)));
    auto b_hi = rewriter.create<xllvm::VectorShuffleIntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()), b0, b1,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(3)));
    // MUL + 3 * MAC
    auto mulConfCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(aiev2_mul_mac_compute_control(
            /*sgn_x=*/1, /*sgn_y=*/1, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0)));
    auto mulConfOp = rewriter.create<xllvm::MulConfAcc64IntrOp>(
        loc, VectorType::get({16}, rewriter.getI64Type()),
        forceCastOperandsToSignature(
            rewriter, loc,
            /*operands=*/{a_hi, b_hi, mulConfCst},
            /*signature=*/
            {VectorType::get({64}, rewriter.getI8Type()),
             VectorType::get({16}, rewriter.getI32Type()),
             rewriter.getI32Type()}));

    auto createMacConfOp = [&](SmallVector<Value> operands,
                               int macConf) -> Value {
      operands.push_back(rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(macConf)));
      return rewriter
          .create<xllvm::MacConfAcc64IntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc,
                  /*operands=*/operands,
                  /*signature=*/
                  {VectorType::get({64}, rewriter.getI8Type()),
                   VectorType::get({16}, rewriter.getI32Type()),
                   VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type()}))
          .getResult();
    };
    auto acc64Val = mulConfOp.getResult();
    acc64Val = createMacConfOp(
        SmallVector<Value>{a_hi, b_lo, acc64Val},
        aiev2_mul_mac_compute_control(
            /*sgn_x=*/1, /*sgn_y=*/0, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/1,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0));
    acc64Val = createMacConfOp(
        SmallVector<Value>{a_lo, b_hi, acc64Val},
        aiev2_mul_mac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/1, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0));
    acc64Val = createMacConfOp(
        SmallVector<Value>{a_lo, b_lo, acc64Val},
        aiev2_mul_mac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/1,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0));

    // create bitcast for result
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                 acc64Val);
    return success();
  }

  // This conversion pattern implements the below CPP emulated FP32 mul_elem.
  // inline v16accfloat mul_elem_16_accuracy_safe(v16float v1, v16float v2) {
  //     v32bfloat16 a = broadcast_zero_to_v32bfloat16();
  //     v32bfloat16 b = broadcast_zero_to_v32bfloat16();
  //     v32bfloat16 c = broadcast_zero_to_v32bfloat16();
  //     v32bfloat16 d = broadcast_zero_to_v32bfloat16();
  //     v32bfloat16 e = broadcast_zero_to_v32bfloat16();
  //     v32bfloat16 f = broadcast_zero_to_v32bfloat16();
  //     v32bfloat16 dummy0 = broadcast_one_to_v32bfloat16();
  //     a = insert(a,0,to_v16bfloat16((v16accfloat)v1));
  //     v16accfloat acc0 = msc_elem_16_2(a, dummy0, (v16accfloat)v1);
  //     b = insert(b,0,to_v16bfloat16(acc0));
  //     c = insert(c,0,to_v16bfloat16(msc_elem_16_2(b, dummy0, acc0)));
  //     d = insert(d,0,to_v16bfloat16((v16accfloat)v2));
  //     v16accfloat acc1 = msc_elem_16_2(d, dummy0, (v16accfloat)v2);
  //     e = insert(e,0,to_v16bfloat16(acc1));
  //     f = insert(f,0,to_v16bfloat16(msc_elem_16_2(e, dummy0, acc1)));
  //     return
  //     mac_elem_16_2(a,d,mac_elem_16_2(a,e,mac_elem_16_2(b,d,mac_elem_16_2(
  //        d,c,mac_elem_16_2(b,e,mac_elem_16_2(a,f,mac_elem_16_2(
  //           b,f,mac_elem_16_2(c,e,mul_elem_16_2(c,f)))))))));
  // }
  // Caller example when handling the elementwise mul of two v16float vectors
  // v16float v1 = LHS(); v16float v2 = RHS();
  // v16accfloat v3 = mul_elem_16(v1, v2);
  LogicalResult
  convertToEmulatedFP32MulElem(aievec::MulElemOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto zeroCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getBF16Type(),
        rewriter.getZeroAttr(rewriter.getBF16Type()));
    auto aZeros = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto bZeros = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto cZeros = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto dZeros = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto eZeros = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto fZeros = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto oneCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getBF16Type(),
        rewriter.getOneAttr(rewriter.getBF16Type()));
    auto dummy0 = rewriter.create<xllvm::VectorBroadcast16BF512IntrOp>(
        loc, VectorType::get({32}, rewriter.getBF16Type()), oneCst);
    auto zeroCstI32 = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto mscMacMulConfCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(aiev2_mul_mac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
            /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0)));

    auto extractV16FP32ToThreeV16BF16 =
        [&](Value inputV16FP32, Value aZeros, Value bZeros,
            Value cZeros) -> std::tuple<Value, Value, Value> {
      // a = insert(a,0,to_v16bfloat16((v16accfloat)v1));
      auto inputBitCasted =
          forceCastValueToType(rewriter, loc, inputV16FP32,
                               VectorType::get({8}, rewriter.getI64Type()));
      auto v1ToBF16 = rewriter.create<xllvm::Vector16AccFloatToV16BF16IntrOp>(
          loc, VectorType::get({16}, rewriter.getBF16Type()), inputBitCasted);
      auto a = rewriter.create<xllvm::UpdBF512BF256IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()), aZeros, v1ToBF16,
          zeroCstI32);

      // v16accfloat acc0 = msc_elem_16_2(a, dummy0, (v16accfloat)v1);
      auto acc0 = rewriter.create<xllvm::MscConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), a, dummy0,
          inputBitCasted, mscMacMulConfCst);

      // b = insert(b,0,to_v16bfloat16(acc0));
      auto acc0ToBF16 = rewriter.create<xllvm::Vector16AccFloatToV16BF16IntrOp>(
          loc, VectorType::get({16}, rewriter.getBF16Type()), acc0);
      auto b = rewriter.create<xllvm::UpdBF512BF256IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()), bZeros,
          acc0ToBF16, zeroCstI32);

      // c = insert(c,0,to_v16bfloat16(msc_elem_16_2(b, dummy0, acc0)));
      auto acc0Mscb = rewriter.create<xllvm::MscConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), b, dummy0, acc0,
          mscMacMulConfCst);
      auto acc0MscbToBF16 =
          rewriter.create<xllvm::Vector16AccFloatToV16BF16IntrOp>(
              loc, VectorType::get({16}, rewriter.getBF16Type()), acc0Mscb);
      auto c = rewriter.create<xllvm::UpdBF512BF256IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()), cZeros,
          acc0MscbToBF16, zeroCstI32);
      return std::make_tuple(a.getResult(), b.getResult(), c.getResult());
    };

    // Get v16vfloat16 a, b, c for representing v16float v1
    auto [a, b, c] =
        extractV16FP32ToThreeV16BF16(adaptor.getLhs(), aZeros, bZeros, cZeros);
    // Get v16vfloat16 d, e, f for representing v16float v2
    auto [d, e, f] =
        extractV16FP32ToThreeV16BF16(adaptor.getRhs(), dZeros, eZeros, fZeros);

    // 1 MUL + 8 * MACs
    auto cfMul = rewriter.create<xllvm::MulConfBF16IntrOp>(
        loc, VectorType::get({8}, rewriter.getI64Type()), c, f,
        mscMacMulConfCst);
    auto createMacOps = [&](Value lhs, Value rhs, Value acc) -> Value {
      return rewriter
          .create<xllvm::MacConfBF16IntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()), lhs, rhs, acc,
              mscMacMulConfCst)
          .getResult();
    };
    auto adMac = createMacOps(
        a, d,
        createMacOps(
            a, e,
            createMacOps(
                b, d,
                createMacOps(
                    d, c,
                    createMacOps(
                        b, e,
                        createMacOps(
                            a, f,
                            createMacOps(b, f, createMacOps(c, e, cfMul))))))));

    // create bitcast for result
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                 adMac);
    return success();
  }

  LogicalResult
  matchAndRewrite(aievec::MulElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto decodedMulElemOp = decodeMulElemOp(adaptor);

    if (decodedMulElemOp.kind == DecodedMulElemOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.mul_elem conversion is not supported.\n";
      return failure();
    }

    // Handle the emulated I32/FP32 mul_elem
    if (decodedMulElemOp.kind == DecodedMulElemOp::Kind::I32_I32_I64_32x1x2x1) {
      return convertToEmulatedI32MulElem(op, adaptor, rewriter);
    } else if (decodedMulElemOp.kind ==
               DecodedMulElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      return convertToEmulatedFP32MulElem(op, adaptor, rewriter);
    }

    // create constant for config
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(decodedMulElemOp.conf));
    Value mulElemOp = nullptr;
    SmallVector<Value> operands({adaptor.getLhs(), adaptor.getRhs(), confCst});

    // create xllvm intrinsic
    if (decodedMulElemOp.kind == DecodedMulElemOp::Kind::I16_I16_I32_32x1x1x1 ||
        decodedMulElemOp.kind == DecodedMulElemOp::Kind::I8_I8_I32_32x1x2x1) {
      mulElemOp = rewriter.create<xllvm::MulConfAcc32IntrOp>(
          loc, VectorType::get({16}, rewriter.getI64Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({64}, rewriter.getI8Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (decodedMulElemOp.kind ==
               DecodedMulElemOp::Kind::BF16_BF16_FP32_16x1x2x1) {
      mulElemOp = rewriter.create<xllvm::MulConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getBF16Type()),
               VectorType::get({32}, rewriter.getBF16Type()),
               rewriter.getI32Type()}));
    }

    // create bitcast for result
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                 mulElemOp);
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

  LogicalResult
  matchAndRewrite(aievec::SRSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // Integer types
    if (resultScaTy.isa<IntegerType>()) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

      // create xllvm intrinsic
      SmallVector<Value> operands(
          {adaptor.getSource(), adaptor.getShift(), signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 16) {
          rewriter.replaceOpWithNewOp<xllvm::I512V32Acc32SrsIntrOp>(
              op, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32) {
          rewriter.replaceOpWithNewOp<xllvm::I512V16Acc64SrsIntrOp>(
              op, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 256) {
        rewriter.replaceOpWithNewOp<xllvm::I256V32Acc32SrsIntrOp>(
            op, VectorType::get({32}, rewriter.getI8Type()),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({16}, rewriter.getI64Type()),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
      } else {
        op.emitWarning() << "aievec.srs with result vector size = "
                         << resultVectorSize << " is not supported.\n";
        return failure();
      }
    } else {
      // Float types
      if (resultVectorSize == 256) {
        rewriter.replaceOpWithNewOp<xllvm::Vector16AccFloatToV16BF16IntrOp>(
            op, VectorType::get({16}, rewriter.getBF16Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getSource()},
                {VectorType::get({8}, rewriter.getI64Type())}));
      } else {
        op.emitWarning() << "aievec.srs with result vector size = "
                         << resultVectorSize << " is not supported.\n";
        return failure();
      }
    }

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

        // TODO: determine if the undef intrinsic is needed or if an LLVM
        // undef suffices destValue =
        // rewriter.create<LLVM::UndefOp>(op->getLoc(), resultType);

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

  LogicalResult
  matchAndRewrite(aievec::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // create xllvm intrinsic
    auto concatOp = rewriter.create<xllvm::ConcatI512I256IntrOp>(
        loc, VectorType::get({16}, rewriter.getI32Type()),
        forceCastOperandsToSignature(
            rewriter, loc, adaptor.getSources(),
            {VectorType::get({8}, rewriter.getI32Type()),
             VectorType::get({8}, rewriter.getI32Type())}));

    // create bitcast for result
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                 concatOp);

    return success();
  }
};

class ExtOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::ExtOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // create constant for index
    auto indexCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    if (op.getIndex() == 1) {
      indexCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    }

    // create xllvm intrinsic
    SmallVector<Value> operands({adaptor.getSource(), indexCst});
    auto extOp = rewriter.create<xllvm::ExtI256I512IntrOp>(
        loc, VectorType::get({8}, rewriter.getI32Type()),
        forceCastOperandsToSignature(
            rewriter, loc, operands,
            {VectorType::get({16}, rewriter.getI32Type()),
             rewriter.getI32Type()}));

    // create bitcast for result
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                 extOp);

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

class BroadcastScalarOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::BroadcastScalarOp> {
public:
  using ConvertOpToLLVMPattern<
      aievec::BroadcastScalarOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::BroadcastScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (resultScaTy.isa<IntegerType>()) {
      Value src = adaptor.getSource();
      Type srcType = src.getType();
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();

      if (srcBitWidth < 32) {
        src = rewriter.create<LLVM::SExtOp>(loc, rewriter.getI32Type(),
                                            adaptor.getSource());
      } else if (srcBitWidth > 32) {
        src = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(),
                                             adaptor.getSource());
      }

      if (resultBitWidth == 8) {
        rewriter.replaceOpWithNewOp<xllvm::VectorBroadcast8I512IntrOp>(
            op, VectorType::get({64}, rewriter.getI8Type()), src);
      } else if (resultBitWidth == 32) {
        rewriter.replaceOpWithNewOp<xllvm::VectorBroadcast32I512IntrOp>(
            op, VectorType::get({16}, rewriter.getI32Type()), src);
      } else {
        op.emitWarning()
            << "aievec.broadcast_scalar conversion with result bitwidth "
            << resultBitWidth << " is not implemented.\n";
        return failure();
      }
    } else {
      // Float types
      if (resultBitWidth == 16) {
        rewriter.replaceOpWithNewOp<xllvm::VectorBroadcast16BF512IntrOp>(
            op, VectorType::get({32}, rewriter.getBF16Type()),
            adaptor.getSource());
      } else {
        op.emitWarning()
            << "aievec.broadcast_scalar conversion with result bitwidth "
            << resultBitWidth << " is not implemented.\n";
        return failure();
      }
    }

    return success();
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

// This pattern folds aievec.cast op. For AIE-ML, the accumulators are in 32/64
// bits, and the vectors are in 4/8/16/32 bits. Hence, we don't have to
// explicitly express the casting between accumulators and vectors at the LLVM
// dialect level. The backend LLVM compiler will decide the correct accumulator
// or vector registers given the ops and intrinsics.
class FoldAIECastOps : public mlir::ConvertOpToLLVMPattern<aievec::CastOp> {
  using ConvertOpToLLVMPattern<aievec::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(castOp, adaptor.getSource());
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
               BroadcastScalarOpConversion,
               FMAElemOpConversion,
               MulElemOpConversion,
               MatMulOpConversion,
               FoldAIECastOps>(converter);
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
