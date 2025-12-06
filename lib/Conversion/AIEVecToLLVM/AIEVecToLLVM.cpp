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
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Utils/Utils.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/TypeUtilities.h"
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
  auto dstVecTy = dyn_cast<VectorType>(type);

  if (srcVecTy) {
    assert(dstVecTy && "vector values cannot be forced into a non-vector type");

    // Flatten source vector if it's not rank-1
    auto flatSrcVecTy = getFlattenedVectorType(srcVecTy);
    if (srcVecTy != flatSrcVecTy)
      val = builder.create<vector::ShapeCastOp>(loc, flatSrcVecTy, val);

    // Flatten destination type if it's not rank-1
    auto flatDstVecTy = getFlattenedVectorType(dstVecTy);

    int64_t dstVecLength =
        flatDstVecTy.getElementTypeBitWidth() * flatDstVecTy.getShape()[0];
    int64_t srcVecLength =
        flatSrcVecTy.getElementTypeBitWidth() * flatSrcVecTy.getShape()[0];
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

    // Bitcast to flat destination type (bitcast only supports flat vectors)
    val = bitcastValueToType(builder, loc, val, flatDstVecTy);

    // Reshape back to original destination shape if needed
    if (flatDstVecTy != dstVecTy)
      val = builder.create<vector::ShapeCastOp>(loc, dstVecTy, val);

    return val;
  }

  // Non-vector types can be bitcast directly
  assert(!dstVecTy && "cannot force cast scalar to vector type");
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
static inline int aiev2_vmac_compute_control(int sgn_x, int sgn_y, int amode,
                                             int bmode, int variant,
                                             int zero_acc, int shift16,
                                             int sub_mul, int sub_acc1,
                                             int sub_acc2, int sub_mask) {
  return ((unsigned)sub_mask << 16) | ((unsigned)shift16 << 10) |
         ((unsigned)sub_mul << 11) | ((unsigned)sub_acc1 << 12) |
         ((unsigned)sub_acc2 << 13) | ((unsigned)amode << 1) |
         ((unsigned)bmode << 3) | ((unsigned)variant << 5) |
         (((unsigned)sgn_x << 9) | ((unsigned)sgn_y << 8)) |
         ((unsigned)zero_acc << 0);
}

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
  if (auto mulOp = dyn_cast<aievec::aie1::MulOp>(op)) {
    baseName = "mul";
    lhs = mulOp.getLhs();
    result = mulOp.getResult();
  } else if (auto fmaOp = dyn_cast<aievec::aie1::FMAOp>(op)) {
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

class AddOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::aie1::AddOp> {
public:
  using ConvertOpToLLVMPattern<aievec::aie1::AddOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.add conversion is not implemented\n";
    return failure();
  }
};

// AIE2 version of AddElemOp conversion
class AddElemOpAIE2Conversion
    : public mlir::ConvertOpToLLVMPattern<aievec::AddElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::AddElemOp>::ConvertOpToLLVMPattern;

  struct DecodedAddElemOp {
    enum class Kind { FP32_FP32_FP32_16x1x1x1, UNSUPPORTED };
    Kind kind;
    int conf;
  };

  static DecodedAddElemOp decodeAddElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedAddElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 32) {
        // FP32 add_elem
        return {DecodedAddElemOp::Kind::FP32_FP32_FP32_16x1x1x1, /*conf*/ 0};
      }
    }
    return {DecodedAddElemOp::Kind::UNSUPPORTED, -1};
  }

  LogicalResult
  matchAndRewrite(aievec::AddElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto decodedAddElemOp = decodeAddElemOp(adaptor);

    if (decodedAddElemOp.kind == DecodedAddElemOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.add_elem conversion is not supported.\n";
      return failure();
    }

    // Handle the FP32 add_elem for AIE2 - uses packed I64 representation
    if (decodedAddElemOp.kind ==
        DecodedAddElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      auto confCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(decodedAddElemOp.conf));
      SmallVector<Value> operands(
          {adaptor.getLhs(), adaptor.getRhs(), confCst});

      auto addElemOp = rewriter.create<xllvm::AddAccFloatAIE2IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({8}, rewriter.getI64Type()),
               VectorType::get({8}, rewriter.getI64Type()),
               rewriter.getI32Type()}));

      // create bitcast/shape_cast for result
      auto resultVal = forceCastValueToType(rewriter, loc, addElemOp,
                                            op.getResult().getType());
      rewriter.replaceOp(op, resultVal);
      return success();
    }

    op.emitWarning() << "aievec.add_elem conversion is not supported.\n";
    return failure();
  }
};

// AIE2 version of SubElemOp conversion
class SubElemOpAIE2Conversion
    : public mlir::ConvertOpToLLVMPattern<aievec::SubElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::SubElemOp>::ConvertOpToLLVMPattern;

  struct DecodedSubElemOp {
    enum class Kind { FP32_FP32_FP32_16x1x1x1, UNSUPPORTED };
    Kind kind;
    int conf;
  };

  static DecodedSubElemOp decodeSubElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedSubElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 32) {
        // FP32 sub_elem
        return {DecodedSubElemOp::Kind::FP32_FP32_FP32_16x1x1x1, /*conf*/ 0};
      }
    }
    return {DecodedSubElemOp::Kind::UNSUPPORTED, -1};
  }

  LogicalResult
  matchAndRewrite(aievec::SubElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto decodedSubElemOp = decodeSubElemOp(adaptor);

    if (decodedSubElemOp.kind == DecodedSubElemOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.sub_elem conversion is not supported.\n";
      return failure();
    }

    // Handle the FP32 sub_elem for AIE2 - uses packed I64 representation
    if (decodedSubElemOp.kind ==
        DecodedSubElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      auto confCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(decodedSubElemOp.conf));
      SmallVector<Value> operands(
          {adaptor.getLhs(), adaptor.getRhs(), confCst});

      auto subElemOp = rewriter.create<xllvm::SubAccFloatAIE2IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({8}, rewriter.getI64Type()),
               VectorType::get({8}, rewriter.getI64Type()),
               rewriter.getI32Type()}));

      // create bitcast/shape_cast for result
      auto resultVal = forceCastValueToType(rewriter, loc, subElemOp,
                                            op.getResult().getType());
      rewriter.replaceOp(op, resultVal);
      return success();
    }

    op.emitWarning() << "aievec.sub_elem conversion is not supported.\n";
    return failure();
  }
};

// AIE2p version of AddElemOp conversion
class AddElemOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::AddElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::AddElemOp>::ConvertOpToLLVMPattern;

  struct DecodedAddElemOp {
    enum class Kind { FP32_FP32_FP32_16x1x1x1, UNSUPPORTED };
    Kind kind;
    int conf;
  };

  static DecodedAddElemOp decodeAddElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedAddElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 32) {
        // FP32 add_elem
        return {DecodedAddElemOp::Kind::FP32_FP32_FP32_16x1x1x1, /*conf*/ 60};
      }
    }
    return {DecodedAddElemOp::Kind::UNSUPPORTED, -1};
  }

  LogicalResult
  matchAndRewrite(aievec::AddElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto decodedAddElemOp = decodeAddElemOp(adaptor);

    if (decodedAddElemOp.kind == DecodedAddElemOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.add_elem conversion is not supported.\n";
      return failure();
    }

    // Handle the FP32 add_elem for AIE2p
    // We need to expand <16xf32> to <64xf32> for the ACC2048 intrinsic
    if (decodedAddElemOp.kind ==
        DecodedAddElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      // Step 1: Bitcast <16 x float> to <8 x i64>
      auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
      auto lhsI64 =
          rewriter.create<LLVM::BitcastOp>(loc, v8i64Ty, adaptor.getLhs());
      auto rhsI64 =
          rewriter.create<LLVM::BitcastOp>(loc, v8i64Ty, adaptor.getRhs());

      // Step 2: Shuffle <8 x i64> to <32 x i64> (expand with poison values)
      auto v32i64Ty = VectorType::get({32}, rewriter.getI64Type());
      SmallVector<int64_t> expandMask = {0, 1, 2, 3, 4, 5, 6, 7};
      for (int i = 8; i < 32; ++i)
        expandMask.push_back(-1); // poison values

      auto lhsExpanded =
          rewriter.create<vector::ShuffleOp>(loc, lhsI64, lhsI64, expandMask);
      auto rhsExpanded =
          rewriter.create<vector::ShuffleOp>(loc, rhsI64, rhsI64, expandMask);

      // Step 3: Bitcast <32 x i64> to <64 x float>
      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());
      auto lhsF32 =
          rewriter.create<LLVM::BitcastOp>(loc, v64f32Ty, lhsExpanded);
      auto rhsF32 =
          rewriter.create<LLVM::BitcastOp>(loc, v64f32Ty, rhsExpanded);

      // Step 4: Call the ACC2048 intrinsic with conf=60
      auto confCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(60));

      // Create the intrinsic call
      auto addResult = rewriter.create<xllvm::AddACC2048AccFloatAIE2pIntrOp>(
          loc, v64f32Ty, lhsF32, rhsF32, confCst);

      // Step 5: Bitcast <64 x float> back to <32 x i64>
      auto resultI64 =
          rewriter.create<LLVM::BitcastOp>(loc, v32i64Ty, addResult);

      // Step 6: Shuffle to extract first 8 elements <32 x i64> -> <8 x i64>
      SmallVector<int64_t> extractMask = {0, 1, 2, 3, 4, 5, 6, 7};
      auto resultExtracted = rewriter.create<vector::ShuffleOp>(
          loc, resultI64, resultI64, extractMask);

      // Step 7: Bitcast <8 x i64> back to <16 x float>
      auto v16f32Ty = VectorType::get({16}, rewriter.getF32Type());
      auto finalResult =
          rewriter.create<LLVM::BitcastOp>(loc, v16f32Ty, resultExtracted);

      rewriter.replaceOp(op, finalResult);
      return success();
    }

    op.emitWarning() << "aievec.add_elem conversion is not supported.\n";
    return failure();
  }
};

// AIE2p version of SubElemOp conversion
class SubElemOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::SubElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::SubElemOp>::ConvertOpToLLVMPattern;

  struct DecodedSubElemOp {
    enum class Kind { FP32_FP32_FP32_16x1x1x1, UNSUPPORTED };
    Kind kind;
    int conf;
  };

  static DecodedSubElemOp decodeSubElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedSubElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 32) {
        // FP32 sub_elem
        return {DecodedSubElemOp::Kind::FP32_FP32_FP32_16x1x1x1, /*conf*/ 60};
      }
    }
    return {DecodedSubElemOp::Kind::UNSUPPORTED, -1};
  }

  LogicalResult
  matchAndRewrite(aievec::SubElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto decodedSubElemOp = decodeSubElemOp(adaptor);

    if (decodedSubElemOp.kind == DecodedSubElemOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.sub_elem conversion is not supported.\n";
      return failure();
    }

    // Handle the FP32 sub_elem for AIE2p
    // We need to expand <16xf32> to <64xf32> for the ACC2048 intrinsic
    if (decodedSubElemOp.kind ==
        DecodedSubElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      // Step 1: Bitcast <16 x float> to <8 x i64>
      auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
      auto lhsI64 =
          rewriter.create<LLVM::BitcastOp>(loc, v8i64Ty, adaptor.getLhs());
      auto rhsI64 =
          rewriter.create<LLVM::BitcastOp>(loc, v8i64Ty, adaptor.getRhs());

      // Step 2: Shuffle <8 x i64> to <32 x i64> (expand with poison values)
      auto v32i64Ty = VectorType::get({32}, rewriter.getI64Type());
      SmallVector<int64_t> expandMask = {0, 1, 2, 3, 4, 5, 6, 7};
      for (int i = 8; i < 32; ++i)
        expandMask.push_back(-1); // poison values

      auto lhsExpanded =
          rewriter.create<vector::ShuffleOp>(loc, lhsI64, lhsI64, expandMask);
      auto rhsExpanded =
          rewriter.create<vector::ShuffleOp>(loc, rhsI64, rhsI64, expandMask);

      // Step 3: Bitcast <32 x i64> to <64 x float>
      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());
      auto lhsF32 =
          rewriter.create<LLVM::BitcastOp>(loc, v64f32Ty, lhsExpanded);
      auto rhsF32 =
          rewriter.create<LLVM::BitcastOp>(loc, v64f32Ty, rhsExpanded);

      // Step 4: Call the ACC2048 intrinsic with conf=60
      auto confCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(60));

      // Create the intrinsic call
      auto subResult = rewriter.create<xllvm::SubACC2048AccFloatAIE2pIntrOp>(
          loc, v64f32Ty, lhsF32, rhsF32, confCst);

      // Step 5: Bitcast <64 x float> back to <32 x i64>
      auto resultI64 =
          rewriter.create<LLVM::BitcastOp>(loc, v32i64Ty, subResult);

      // Step 6: Shuffle to extract first 8 elements <32 x i64> -> <8 x i64>
      SmallVector<int64_t> extractMask = {0, 1, 2, 3, 4, 5, 6, 7};
      auto resultExtracted = rewriter.create<vector::ShuffleOp>(
          loc, resultI64, resultI64, extractMask);

      // Step 7: Bitcast <8 x i64> back to <16 x float>
      auto v16f32Ty = VectorType::get({16}, rewriter.getF32Type());
      auto finalResult =
          rewriter.create<LLVM::BitcastOp>(loc, v16f32Ty, resultExtracted);

      rewriter.replaceOp(op, finalResult);
      return success();
    }

    op.emitWarning() << "aievec.sub_elem conversion is not supported.\n";
    return failure();
  }
};

class SubOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::aie1::SubOp> {
public:
  using ConvertOpToLLVMPattern<aievec::aie1::SubOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "aie.sub conversion is not implemented\n";
    return failure();
  }
};

class FMAOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::aie1::FMAOp> {
public:
  using ConvertOpToLLVMPattern<aievec::aie1::FMAOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::FMAOp op, OpAdaptor adaptor,
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

class MulOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::aie1::MulOp> {
public:
  using ConvertOpToLLVMPattern<aievec::aie1::MulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::MulOp op, OpAdaptor adaptor,
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

  MulElemOpConversion(const LLVMTypeConverter &typeConverter,
                      Aie2Fp32Emulation aie2Fp32EmulationOption)
      : ConvertOpToLLVMPattern(typeConverter),
        aie2Fp32EmulationOption(aie2Fp32EmulationOption) {}

  Aie2Fp32Emulation aie2Fp32EmulationOption;

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

  static DecodedMulElemOp decodeMulElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      if (lhsBitWidth == 8) {
        return {DecodedMulElemOp::Kind::I8_I8_I32_32x1x2x1,
                aiev2_vmac_compute_control(
                    /*sgn_x=*/1, /*sgn_y=*/1, /*amode=*/0, /*bmode=*/1,
                    /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      } else if (lhsBitWidth == 16) {
        return {DecodedMulElemOp::Kind::I16_I16_I32_32x1x1x1,
                aiev2_vmac_compute_control(
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
                aiev2_vmac_compute_control(
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
  // Caller example when handling the elementwise mul of two v16int32 vectors.
  //   v16int32 v1 = LHS();
  //   v16int32 v2 = RHS();
  //   v16acc64 v3 = mul_elem_16_2(v1, broadcast_zero_s32(), v2,
  //   undef_v16int32());
  // Explantion:
  // a_lo = low_part(a0[0]--a0[15], a1[0]--a1[15])
  // a_hi = high_part(a0[0]--a0[15], a1[0]--a1[15])
  // b_lo = low_part(b0[0]--b0[15], b1[0]--b1[15])
  // b_hi = high_part(b0[0]--b0[15], b1[0]--b1[15])
  // The firt `acc` is from mul_elem_16_2(a_hi, b_hi), which performs 16 channel
  // of 1x2x1 matmul, acc[0] = a_hi[0]*b_hi[0]+a_hi[16]*b_hi[16], ... , acc[15]
  // = a_hi[15]*b_hi[15]+a_hi[31]*b_hi[31]. Then, the first MAC performs `acc`
  // left shift 16bit, and then 16 channel of 1x2x1 matmul (a_hi, b_lo)
  // accumulating to `acc`. The second MAC performs 16 channel of 1x2x1 matmul
  // (a_lo, b_hi) accumulating to `acc`. Finally, the third MAC performs 16
  // channel of 1x2x1 matmul (a_lo, b_hi) accumulating to `acc`.
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
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
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
        aiev2_vmac_compute_control(
            /*sgn_x=*/1, /*sgn_y=*/0, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/1,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0));
    acc64Val = createMacConfOp(
        SmallVector<Value>{a_lo, b_hi, acc64Val},
        aiev2_vmac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/1, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0));
    acc64Val = createMacConfOp(
        SmallVector<Value>{a_lo, b_lo, acc64Val},
        aiev2_vmac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/1,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0));

    // create bitcast/shape_cast for result
    auto resultVal =
        forceCastValueToType(rewriter, loc, acc64Val, op.getResult().getType());
    rewriter.replaceOp(op, resultVal);
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
  // Caller example when handling the elementwise mul of two v16float vectors.
  //   v16float v1 = LHS(); v16float v2 = RHS();
  //   v16accfloat v3 = mul_elem_16(v1, v2);
  // Explantion: For v32bfloat16 `a`, the first half v16bf16 contains `most
  // significant 7 bits of mantissa` from v1, and the second half v16bf16 are
  // zeros. For v16accfloat `acc0`, the MSC equals to "(original `v1` with 23
  // bits of mantissa) - (`a` with MSB 7 bits of mantissa from v1)". For
  // v32bfloat16 `b`, the first half v16bf16 contains `[7:13] bits of mantissa
  // from v1` from v1, and the second half v16bf16 are zeros. For v32bfloat16
  // `c`, the first half v16bf16 contains `[14:20] bits of mantissa from v1`
  // from v1, and the second half v16bf16 are zeros. Hence, we can represent
  // v16float in three v32bfloat16 and then perform 9 MUL/MAC in v32bfloat16 to
  // get the final elementwise multiplication result.

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
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
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
      auto v1ToBF16 =
          rewriter.create<xllvm::Vector16AccFloatToV16BF16AIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getBF16Type()),
              inputBitCasted);
      auto a = rewriter.create<xllvm::UpdBF512BF256IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()), aZeros, v1ToBF16,
          zeroCstI32);

      // v16accfloat acc0 = msc_elem_16_2(a, dummy0, (v16accfloat)v1);
      auto acc0 = rewriter.create<xllvm::MscConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), a, dummy0,
          inputBitCasted, mscMacMulConfCst);

      // b = insert(b,0,to_v16bfloat16(acc0));
      auto acc0ToBF16 =
          rewriter.create<xllvm::Vector16AccFloatToV16BF16AIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getBF16Type()), acc0);
      auto b = rewriter.create<xllvm::UpdBF512BF256IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()), bZeros,
          acc0ToBF16, zeroCstI32);

      // c = insert(c,0,to_v16bfloat16(msc_elem_16_2(b, dummy0, acc0)));
      auto acc0Mscb = rewriter.create<xllvm::MscConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), b, dummy0, acc0,
          mscMacMulConfCst);
      auto acc0MscbToBF16 =
          rewriter.create<xllvm::Vector16AccFloatToV16BF16AIE2IntrOp>(
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

    // Create 1 MUL and 2/5/8 MACs depending on the Aie2Fp32EmulationOption
    auto createMacOps = [&](Value lhs, Value rhs, Value acc) -> Value {
      return rewriter
          .create<xllvm::MacConfBF16IntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()), lhs, rhs, acc,
              mscMacMulConfCst)
          .getResult();
    };

    Value finalMacVal;
    if (aie2Fp32EmulationOption == Aie2Fp32Emulation::AccuracyFast) {
      // Fast and Accurate option. float a*b would require 6 mac operations.
      // Input fp32 number is split in to 3 bfloat16 numbers to extract all the
      // bits of the mantissa. float a,b; both a and b are split in to 3
      // bfloat16 numbers each. Hence there would be 9 mac operations in
      // multiplication of a and b. In the 9 mac operations to emulate fp32 mul,
      // mac operations with LSBs are ignored. (3 last terms). This helps
      // improve cycle count of mul and has least impact on accuracy of result.
      // This is the default option to the aiecompiler
      auto afMul = rewriter.create<xllvm::MulConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), a, f,
          mscMacMulConfCst);
      finalMacVal = createMacOps(
          a, d,
          createMacOps(
              a, e,
              createMacOps(b, d,
                           createMacOps(d, c, createMacOps(b, e, afMul)))));
    } else if (aie2Fp32EmulationOption == Aie2Fp32Emulation::AccuracyLow) {
      // Fast and least accurate option. float a*b would require 3 mac
      // operations.
      // Input fp32 number is split in to 2 bfloat16 numbers. Hence not all the
      // bits from mantissa can be used. float a,b; Both a and b are split in to
      // 2 bfloat16 numbers each. Hence there would be 4 mac operations in
      // multiplication of a and b. In the 4 mac operations to emulate fp32 mul,
      // mac operations with LSBs are ignored. (1 last term). This helps improve
      // cycle count of mul float a, b;
      auto bdMul = rewriter.create<xllvm::MulConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), b, d,
          mscMacMulConfCst);
      finalMacVal = createMacOps(a, d, createMacOps(a, e, bdMul));
    } else {
      // aie2Fp32EmulationOption == Aie2Fp32Emulation::AccuracySafe
      // Most accurate option since input fp32 number is split in to 3 bfloat16
      // numbers to extract all the bits of the mantissa. float a*b would
      // require 9 mac operations due to 3 bfloat16 splits each.
      auto cfMul = rewriter.create<xllvm::MulConfBF16IntrOp>(
          loc, VectorType::get({8}, rewriter.getI64Type()), c, f,
          mscMacMulConfCst);
      finalMacVal = createMacOps(
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
                              createMacOps(b, f,
                                           createMacOps(c, e, cfMul))))))));
    }

    // create bitcast/shape_cast for result
    auto resultVal = forceCastValueToType(rewriter, loc, finalMacVal,
                                          op.getResult().getType());
    rewriter.replaceOp(op, resultVal);
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

    // create bitcast/shape_cast for result
    auto resultVal = forceCastValueToType(rewriter, loc, mulElemOp,
                                          op.getResult().getType());
    rewriter.replaceOp(op, resultVal);
    return success();
  }
};

// AIE2p version of MulElemOp conversion
class MulElemOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MulElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MulElemOp>::ConvertOpToLLVMPattern;

  struct DecodedMulElemOp {
    enum class Kind {
      BF16_BF16_FP32_16x1x1x1, // 16-lane bf16 -> 16-lane f32
      BF16_BF16_FP32_32x1x2x1, // 32-lane bf16 -> 32-lane f32
      BF16_BF16_FP32_64x1x2x1, // 64-lane bf16 -> 64-lane f32
      UNSUPPORTED
    };
    Kind kind;
    int conf;
  };

  static DecodedMulElemOp decodeMulElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();
    int lhsLanes = getVectorLaneSize(lhsVecTy);

    // Integer types - not supported for AIE2p elementwise mul
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedMulElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 16) {
        // BF16 mul_elem
        if (lhsLanes == 16) {
          // 16-lane bfloat16 uses I512.I512.ACC512 intrinsic
          return {DecodedMulElemOp::Kind::BF16_BF16_FP32_16x1x1x1, /*conf*/ 60};
        } else if (lhsLanes == 32) {
          // 32-lane bfloat16 uses I512.I512.ACC1024 intrinsic
          return {DecodedMulElemOp::Kind::BF16_BF16_FP32_32x1x2x1, /*conf*/ 60};
        } else if (lhsLanes == 64) {
          // 64-lane bfloat16 uses I1024.I1024.ACC2048 intrinsic
          return {DecodedMulElemOp::Kind::BF16_BF16_FP32_64x1x2x1, /*conf*/ 60};
        }
      }
    }
    return {DecodedMulElemOp::Kind::UNSUPPORTED, -1};
  }

  LogicalResult
  matchAndRewrite(aievec::MulElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto decodedMulElemOp = decodeMulElemOp(adaptor);

    if (decodedMulElemOp.kind == DecodedMulElemOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.mul_elem conversion is not supported for "
                          "AIE2p.\n";
      return failure();
    }

    // Create constant for config
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(decodedMulElemOp.conf));

    Value mulElemOp = nullptr;

    // Handle BF16 mul_elem for AIE2p
    if (decodedMulElemOp.kind ==
        DecodedMulElemOp::Kind::BF16_BF16_FP32_16x1x1x1) {
      // 16-lane bfloat16: <16 x bfloat> x <16 x bfloat> -> <16 x float>
      // The intrinsic requires <32 x bfloat> inputs, so we need to pad

      // Pad LHS from 16 to 32 bfloat16 using shuffle
      SmallVector<int64_t> padMask;
      for (int i = 0; i < 16; ++i)
        padMask.push_back(i);
      for (int i = 16; i < 32; ++i)
        padMask.push_back(-1); // poison/undef

      auto lhsPadded = rewriter.create<vector::ShuffleOp>(
          loc, adaptor.getLhs(), adaptor.getLhs(), padMask);
      auto rhsPadded = rewriter.create<vector::ShuffleOp>(
          loc, adaptor.getRhs(), adaptor.getRhs(), padMask);

      SmallVector<Value> operands({lhsPadded, rhsPadded, confCst});

      // Call I512.I512.ACC512 intrinsic
      mulElemOp = rewriter.create<xllvm::MulConfBF16I512ACC512AIE2pIntrOp>(
          loc, VectorType::get({16}, rewriter.getF32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getBF16Type()),
               VectorType::get({32}, rewriter.getBF16Type()),
               rewriter.getI32Type()}));
    } else if (decodedMulElemOp.kind ==
               DecodedMulElemOp::Kind::BF16_BF16_FP32_32x1x2x1) {
      // 32-lane bfloat16: <32 x bfloat> x <32 x bfloat> -> <32 x float>
      SmallVector<Value> operands(
          {adaptor.getLhs(), adaptor.getRhs(), confCst});
      mulElemOp = rewriter.create<xllvm::MulConfBF16I512ACC1024AIE2pIntrOp>(
          loc, VectorType::get({32}, rewriter.getF32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getBF16Type()),
               VectorType::get({32}, rewriter.getBF16Type()),
               rewriter.getI32Type()}));
    } else if (decodedMulElemOp.kind ==
               DecodedMulElemOp::Kind::BF16_BF16_FP32_64x1x2x1) {
      // 64-lane bfloat16: <64 x bfloat> x <64 x bfloat> -> <64 x float>
      SmallVector<Value> operands(
          {adaptor.getLhs(), adaptor.getRhs(), confCst});
      mulElemOp = rewriter.create<xllvm::MulConfBF16I1024ACC2048AIE2pIntrOp>(
          loc, VectorType::get({64}, rewriter.getF32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({64}, rewriter.getBF16Type()),
               VectorType::get({64}, rewriter.getBF16Type()),
               rewriter.getI32Type()}));
    }

    // create bitcast/shape_cast for result
    auto resultVal = forceCastValueToType(rewriter, loc, mulElemOp,
                                          op.getResult().getType());
    rewriter.replaceOp(op, resultVal);
    return success();
  }
};

// Enum to represent different AIE target architectures
enum class AIEArch {
  AIE2,
  AIE2p,
};

class UPSOpAIE2Conversion : public mlir::ConvertOpToLLVMPattern<aievec::UPSOp> {
public:
  using ConvertOpToLLVMPattern<aievec::UPSOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::UPSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    VectorType flatResTy = getFlattenedVectorType(resultType);
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    Value opSrcVal = adaptor.getSource();
    auto srcVecTy = cast<VectorType>(opSrcVal.getType());
    auto fltSrcVecTy = getFlattenedVectorType(srcVecTy);
    if (srcVecTy != fltSrcVecTy)
      opSrcVal =
          rewriter
              .create<vector::ShapeCastOp>(op.getLoc(), fltSrcVecTy, opSrcVal)
              .getResult();

    // create xllvm intrinsic
    // Integer types
    Value upsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      auto shiftCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getShift()));

      SmallVector<Value> operands({opSrcVal, shiftCst, signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 32) {
          // v16int16 -> v16acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V16I256UpsAIE2IntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64) {
          // v8int32 -> v8acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V8I256UpsAIE2IntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 1024) {
        Value src = opSrcVal;
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 32 && srcBitWidth == 16) {
          // v32int16 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V32I512UpsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 32) {
          // v16int32 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V16I512UpsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16) {
          // v16int16 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V16I256UpsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 8) {
          // v32int8 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V32I256UpsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI8Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      }
    } else {
      // Float types
      // AIE2p uses native F32 types, AIE2 uses packed I64 types
      if (resultVectorSize == 512) {
        // v16bfloat16 -> v16accfloat
        upsIntrOp = rewriter.create<xllvm::Vector16BF16ToV16AccFloatAIE2IntrOp>(
            loc, VectorType::get({8}, rewriter.getI64Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {opSrcVal},
                {VectorType::get({16}, rewriter.getBF16Type())}));
      } else if (resultVectorSize == 1024) {
        // v32bfloat16 -> v32accfloat
        // The CPP example of the implementation is below:
        //   INTRINSIC(v32accfloat) ups_to_v32accfloat(v32bfloat16 a) {
        //     v16accfloat x0 = ups_to_v16accfloat(extract_v16bfloat16(a, 0));
        //     v16accfloat x1 = ups_to_v16accfloat(extract_v16bfloat16(a, 1));
        //     return concat(x0, x1);
        //   }
        auto indexZeroCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto indexOneCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto extractUps = [&](Value source, Value index) -> Value {
          auto extOp = rewriter.create<xllvm::ExtI256I512IntrOp>(
              loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return rewriter.create<xllvm::Vector16BF16ToV16AccFloatAIE2IntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({16}, rewriter.getBF16Type())}));
        };
        auto resLo = extractUps(opSrcVal, indexZeroCst);
        auto resHi = extractUps(opSrcVal, indexOneCst);
        // Concat the two 512-bit vector to a 1024-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        upsIntrOp = rewriter.create<xllvm::ConcatI1024I512IntrOp>(
            loc, VectorType::get({32}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {resLo, resHi},
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type())}));
      }
    }

    if (!upsIntrOp) {
      op.emitWarning() << "aievec.ups is not supported.\n";
      return failure();
    }

    // create bitcast for result if needed
    if (flatResTy != upsIntrOp.getType())
      upsIntrOp = rewriter.create<LLVM::BitcastOp>(loc, flatResTy, upsIntrOp);

    if (flatResTy != resultType)
      upsIntrOp =
          rewriter.create<vector::ShapeCastOp>(loc, resultType, upsIntrOp);

    rewriter.replaceOp(op, upsIntrOp);

    return success();
  }
};

// TODO: Split the op at AIEVec dialect level
class UPSOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::UPSOp> {
public:
  using ConvertOpToLLVMPattern<aievec::UPSOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::UPSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    VectorType flatResTy = getFlattenedVectorType(resultType);
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    Value opSrcVal = adaptor.getSource();
    auto srcVecTy = cast<VectorType>(opSrcVal.getType());
    auto fltSrcVecTy = getFlattenedVectorType(srcVecTy);
    if (srcVecTy != fltSrcVecTy)
      opSrcVal =
          rewriter
              .create<vector::ShapeCastOp>(op.getLoc(), fltSrcVecTy, opSrcVal)
              .getResult();

    // create xllvm intrinsic
    // Integer types
    Value upsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      auto shiftCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getShift()));

      SmallVector<Value> operands({opSrcVal, shiftCst, signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 32) {
          // v16int16 -> v16acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V16I256UpsAIE2pIntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64) {
          // v8int32 -> v8acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V8I256UpsAIE2pIntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 1024) {
        Value src = opSrcVal;
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();
        int srcLanes = getVectorLaneSize(srcType);
        int srcVectorSize = srcBitWidth * srcLanes;

        if (resultBitWidth == 32 && srcBitWidth == 16 && srcVectorSize == 512) {
          // v32int16 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V32I512UpsAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 32 &&
                   srcVectorSize == 512) {
          // v16int32 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V16I512UpsAIE2pIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16 &&
                   srcVectorSize == 256) {
          // v16int16 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V16I256UpsAIE2pIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 8 &&
                   srcVectorSize == 256) {
          // v32int8 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V32I256UpsAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI8Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 2048) {
        Value src = opSrcVal;
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();
        int srcLanes = getVectorLaneSize(srcType);
        int srcVectorSize = srcBitWidth * srcLanes;

        if (resultBitWidth == 32 && srcBitWidth == 8 && srcVectorSize == 512) {
          // v64int8 -> v64acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V64I512UpsAIE2pIntrOp>(
              loc, VectorType::get({64}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({64}, rewriter.getI8Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16 &&
                   srcVectorSize == 512) {
          // v32int16 -> v32acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V32I512UpsAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 16 &&
                   srcVectorSize == 1024) {
          // v64int16 -> v64acc32
          // Extract 2 chunks of v32int16 and convert each to v32acc32
          auto index0Cst = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
          auto index1Cst = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

          auto extractUps2048 = [&](Value source, Value index, Value shiftCst,
                                    Value signCst) -> Value {
            // Use vector::ShuffleOp to extract 512-bit from 1024-bit
            // Cast source to v32xi32 for shuffling
            auto v32i32Source = forceCastValueToType(
                rewriter, loc, source,
                VectorType::get({32}, rewriter.getI32Type()));

            // Determine shuffle mask based on index
            // index 0: elements [0-15]
            // index 1: elements [16-31]
            SmallVector<int64_t> shuffleMask;
            if (auto constIndex = index.getDefiningOp<LLVM::ConstantOp>()) {
              auto indexAttr = cast<IntegerAttr>(constIndex.getValue());
              int64_t idxVal = indexAttr.getInt();
              int startIdx = idxVal * 16;
              for (int i = 0; i < 16; ++i) {
                shuffleMask.push_back(startIdx + i);
              }
            } else {
              // Default to index 0 if not constant
              for (int i = 0; i < 16; ++i) {
                shuffleMask.push_back(i);
              }
            }

            auto extOp = rewriter.create<vector::ShuffleOp>(
                loc, v32i32Source, v32i32Source, shuffleMask);

            return rewriter.create<xllvm::Acc32V32I512UpsAIE2pIntrOp>(
                loc, VectorType::get({32}, rewriter.getI32Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {extOp, shiftCst, signCst},
                    {VectorType::get({32}, rewriter.getI16Type()),
                     rewriter.getI32Type(), rewriter.getI32Type()}));
          };

          auto res0 = extractUps2048(opSrcVal, index0Cst, shiftCst, signCst);
          auto res1 = extractUps2048(opSrcVal, index1Cst, shiftCst, signCst);

          // Concat two 1024-bit vectors to a 2048-bit vector using
          // vector::ShuffleOp
          SmallVector<int64_t> concatMask;
          for (int i = 0; i < 64; ++i) {
            concatMask.push_back(i);
          }
          upsIntrOp =
              rewriter.create<vector::ShuffleOp>(loc, res0, res1, concatMask);
        }
      }
    } else {
      // Float types
      // AIE2p uses native F32 types, AIE2 uses packed I64 types
      if (resultVectorSize == 512) {
        // v16bfloat16 -> v16accfloat
        upsIntrOp =
            rewriter.create<xllvm::Vector16BF16ToV16AccFloatAIE2pIntrOp>(
                loc, VectorType::get({16}, rewriter.getF32Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {opSrcVal},
                    {VectorType::get({16}, rewriter.getBF16Type())}));
      } else if (resultVectorSize == 1024) {
        // v32bfloat16 -> v32accfloat
        upsIntrOp =
            rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
                loc, VectorType::get({32}, rewriter.getF32Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {opSrcVal},
                    {VectorType::get({32}, rewriter.getBF16Type())}));
      } else if (resultVectorSize == 2048) {
        // v64bfloat16 -> v64accfloat
        // Extract 2 chunks of v32bfloat16 and convert each to v32accfloat
        auto index0Cst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto index1Cst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

        auto extractUps2048 = [&](Value source, Value index) -> Value {
          // Use vector::ShuffleOp to extract 512-bit from 1024-bit
          // Cast source to v32xi32 for shuffling
          auto v32i32Source = forceCastValueToType(
              rewriter, loc, source,
              VectorType::get({32}, rewriter.getI32Type()));

          // Determine shuffle mask based on index
          // index 0: elements [0-15]
          // index 1: elements [16-31]
          SmallVector<int64_t> shuffleMask;
          if (auto constIndex = index.getDefiningOp<LLVM::ConstantOp>()) {
            auto indexAttr = cast<IntegerAttr>(constIndex.getValue());
            int64_t idxVal = indexAttr.getInt();
            int startIdx = idxVal * 16;
            for (int i = 0; i < 16; ++i) {
              shuffleMask.push_back(startIdx + i);
            }
          } else {
            // Default to index 0 if not constant
            for (int i = 0; i < 16; ++i) {
              shuffleMask.push_back(i);
            }
          }

          auto extOp = rewriter.create<vector::ShuffleOp>(
              loc, v32i32Source, v32i32Source, shuffleMask);

          return rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getF32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({32}, rewriter.getBF16Type())}));
        };

        auto res0 = extractUps2048(opSrcVal, index0Cst);
        auto res1 = extractUps2048(opSrcVal, index1Cst);

        // Concat two 1024-bit vectors to a 2048-bit vector using
        // vector::ShuffleOp
        auto v32i32Res0 = forceCastValueToType(
            rewriter, loc, res0, VectorType::get({32}, rewriter.getI32Type()));
        auto v32i32Res1 = forceCastValueToType(
            rewriter, loc, res1, VectorType::get({32}, rewriter.getI32Type()));

        SmallVector<int64_t> concatMask;
        for (int i = 0; i < 64; ++i) {
          concatMask.push_back(i);
        }
        upsIntrOp = rewriter.create<vector::ShuffleOp>(loc, v32i32Res0,
                                                       v32i32Res1, concatMask);
      }
    }

    if (!upsIntrOp) {
      op.emitWarning() << "aievec.ups is not supported.\n";
      return failure();
    }

    // create bitcast for result if needed
    if (flatResTy != upsIntrOp.getType())
      upsIntrOp = rewriter.create<LLVM::BitcastOp>(loc, flatResTy, upsIntrOp);

    if (flatResTy != resultType)
      upsIntrOp =
          rewriter.create<vector::ShapeCastOp>(loc, resultType, upsIntrOp);

    rewriter.replaceOp(op, upsIntrOp);

    return success();
  }
};

class SRSOpAIE2Conversion : public mlir::ConvertOpToLLVMPattern<aievec::SRSOp> {
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
    Value srsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

      // create xllvm intrinsic
      SmallVector<Value> operands(
          {adaptor.getSource(), adaptor.getShift(), signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 16) {
          srsIntrOp = rewriter.create<xllvm::I512V32Acc32SrsAIE2IntrOp>(
              loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32) {
          srsIntrOp = rewriter.create<xllvm::I512V16Acc64SrsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 256) {
        Value src = adaptor.getSource();
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 16 && srcBitWidth == 32) {
          srsIntrOp = rewriter.create<xllvm::I256V16Acc32SrsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          srsIntrOp = rewriter.create<xllvm::I256V32Acc32SrsAIE2IntrOp>(
              loc, VectorType::get({32}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          srsIntrOp = rewriter.create<xllvm::I256V16Acc64SrsAIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          srsIntrOp = rewriter.create<xllvm::I256V8Acc64SrsAIE2IntrOp>(
              loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      }
    } else {
      // Float types
      if (resultVectorSize == 256) {
        srsIntrOp = rewriter.create<xllvm::Vector16AccFloatToV16BF16AIE2IntrOp>(
            loc, VectorType::get({16}, rewriter.getBF16Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getSource()},
                {VectorType::get({8}, rewriter.getI64Type())}));
      } else if (resultVectorSize == 512) {
        // v32accfloat -> v32bfloat16
        // The CPP example of the implementation is below:
        //   v32bfloat16 to_v32bfloat16(v32accfloat acc) {
        //     v16bfloat16 x0 = to_v16bfloat16(extract_v16accfloat(acc, 0));
        //     v16bfloat16 x1 = to_v16bfloat16(extract_v16accfloat(acc, 1));
        //     return concat(x0, x1);
        //   }
        auto indexZeroCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto indexOneCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto extractSrs = [&](Value source, Value index) -> Value {
          auto extOp = rewriter.create<xllvm::ExtI512I1024IntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return rewriter.create<xllvm::Vector16AccFloatToV16BF16AIE2IntrOp>(
              loc, VectorType::get({16}, rewriter.getBF16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({8}, rewriter.getI64Type())}));
        };
        auto resLo = extractSrs(adaptor.getSource(), indexZeroCst);
        auto resHi = extractSrs(adaptor.getSource(), indexOneCst);
        // Concat the two 256-bit vector to a 512-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        srsIntrOp = rewriter.create<xllvm::ConcatI512I256IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {resLo, resHi},
                {VectorType::get({8}, rewriter.getI32Type()),
                 VectorType::get({8}, rewriter.getI32Type())}));
      }
    }

    if (!srsIntrOp) {
      op.emitWarning() << "aievec.srs is not supported.\n";
      return failure();
    }

    // create bitcast/shape_cast for result if needed
    auto resultVal = forceCastValueToType(rewriter, loc, srsIntrOp,
                                          op.getResult().getType());
    rewriter.replaceOp(op, resultVal);

    return success();
  }
};

// TODO: Split the op at AIEVec dialect level
class SRSOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::SRSOp> {
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
    Value srsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

      // create xllvm intrinsic
      SmallVector<Value> operands(
          {adaptor.getSource(), adaptor.getShift(), signCst});
      if (resultVectorSize == 512) {
        Value src = adaptor.getSource();
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 16 && srcBitWidth == 32) {
          // v32acc32 -> v32int16
          srsIntrOp = rewriter.create<xllvm::I512V32Acc32SrsAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          // v32acc64 -> v32int16
          srsIntrOp = rewriter.create<xllvm::I512V32Acc64SrsAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          // v16acc64 -> v16int32
          srsIntrOp = rewriter.create<xllvm::I512V16Acc64SrsAIE2pIntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          // v64acc32 -> v64int8
          srsIntrOp = rewriter.create<xllvm::I512V64Acc32SrsAIE2pIntrOp>(
              loc, VectorType::get({64}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({64}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 256) {
        Value src = adaptor.getSource();
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 16 && srcBitWidth == 32) {
          // v16acc32 -> v16int16
          srsIntrOp = rewriter.create<xllvm::I256V16Acc32SrsAIE2pIntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          // v32acc32 -> v32int8
          srsIntrOp = rewriter.create<xllvm::I256V32Acc32SrsAIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          // v16acc64 -> v16int16
          srsIntrOp = rewriter.create<xllvm::I256V16Acc64SrsAIE2pIntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          // v8acc64 -> v8int32
          srsIntrOp = rewriter.create<xllvm::I256V8Acc64SrsAIE2pIntrOp>(
              loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 1024) {
        Value src = adaptor.getSource();
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 16 && srcBitWidth == 32) {
          // v64acc32 -> v64int16
          // Extract 2 chunks of v32acc32 and convert each to v32int16
          auto index0Cst = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
          auto index1Cst = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

          auto extractSrs1024 = [&](Value source, Value index, Value shiftCst,
                                    Value signCst) -> Value {
            // Use vector::ShuffleOp to extract 1024-bit from 2048-bit
            // Cast source to v64xi32 for shuffling
            auto v64i32Source = forceCastValueToType(
                rewriter, loc, source,
                VectorType::get({64}, rewriter.getI32Type()));

            // Determine shuffle mask based on index
            // index 0: elements [0-31]
            // index 1: elements [32-63]
            SmallVector<int64_t> shuffleMask;
            if (auto constIndex = index.getDefiningOp<LLVM::ConstantOp>()) {
              auto indexAttr = cast<IntegerAttr>(constIndex.getValue());
              int64_t idxVal = indexAttr.getInt();
              int startIdx = idxVal * 32;
              for (int i = 0; i < 32; ++i) {
                shuffleMask.push_back(startIdx + i);
              }
            } else {
              // Default to index 0 if not constant
              for (int i = 0; i < 32; ++i) {
                shuffleMask.push_back(i);
              }
            }

            auto extOp = rewriter.create<vector::ShuffleOp>(
                loc, v64i32Source, v64i32Source, shuffleMask);

            return rewriter.create<xllvm::I512V32Acc32SrsAIE2pIntrOp>(
                loc, VectorType::get({32}, rewriter.getI16Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {extOp, shiftCst, signCst},
                    {VectorType::get({32}, rewriter.getI32Type()),
                     rewriter.getI32Type(), rewriter.getI32Type()}));
          };

          auto res0 =
              extractSrs1024(src, index0Cst, adaptor.getShift(), signCst);
          auto res1 =
              extractSrs1024(src, index1Cst, adaptor.getShift(), signCst);

          // Concat two 512-bit vectors to a 1024-bit vector using
          // vector::ShuffleOp
          auto v16i32Res0 = forceCastValueToType(
              rewriter, loc, res0,
              VectorType::get({16}, rewriter.getI32Type()));
          auto v16i32Res1 = forceCastValueToType(
              rewriter, loc, res1,
              VectorType::get({16}, rewriter.getI32Type()));

          SmallVector<int64_t> concatMask;
          for (int i = 0; i < 32; ++i) {
            concatMask.push_back(i);
          }
          srsIntrOp = rewriter.create<vector::ShuffleOp>(
              loc, v16i32Res0, v16i32Res1, concatMask);
        }
      }
    } else {
      // Float types
      // AIE2p uses native F32 types, AIE2 uses packed I64 types
      if (resultVectorSize == 256) {
        // v16accfloat -> v16bfloat16
        srsIntrOp =
            rewriter.create<xllvm::Vector16AccFloatToV16BF16AIE2pIntrOp>(
                loc, VectorType::get({16}, rewriter.getBF16Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {adaptor.getSource()},
                    {VectorType::get({16}, rewriter.getF32Type())}));
      } else if (resultVectorSize == 512) {
        // v32accfloat -> v32bfloat16
        srsIntrOp =
            rewriter.create<xllvm::Vector32AccFloatToV32BF16AIE2pIntrOp>(
                loc, VectorType::get({32}, rewriter.getBF16Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {adaptor.getSource()},
                    {VectorType::get({32}, rewriter.getF32Type())}));
      } else if (resultVectorSize == 1024) {
        // v64accfloat -> v64bfloat16
        // Extract 2 chunks of v32accfloat and convert each to v32bfloat16
        auto index0Cst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto index1Cst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

        auto extractSrs1024 = [&](Value source, Value index) -> Value {
          // Use vector::ShuffleOp to extract 1024-bit from 2048-bit
          // Cast source to v64xi32 for shuffling
          auto v64i32Source = forceCastValueToType(
              rewriter, loc, source,
              VectorType::get({64}, rewriter.getI32Type()));

          // Determine shuffle mask based on index
          // index 0: elements [0-31]
          // index 1: elements [32-63]
          SmallVector<int64_t> shuffleMask;
          if (auto constIndex = index.getDefiningOp<LLVM::ConstantOp>()) {
            auto indexAttr = cast<IntegerAttr>(constIndex.getValue());
            int64_t idxVal = indexAttr.getInt();
            int startIdx = idxVal * 32;
            for (int i = 0; i < 32; ++i) {
              shuffleMask.push_back(startIdx + i);
            }
          } else {
            // Default to index 0 if not constant
            for (int i = 0; i < 32; ++i) {
              shuffleMask.push_back(i);
            }
          }

          auto extOp = rewriter.create<vector::ShuffleOp>(
              loc, v64i32Source, v64i32Source, shuffleMask);

          return rewriter.create<xllvm::Vector32AccFloatToV32BF16AIE2pIntrOp>(
              loc, VectorType::get({32}, rewriter.getBF16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({32}, rewriter.getF32Type())}));
        };

        auto res0 = extractSrs1024(adaptor.getSource(), index0Cst);
        auto res1 = extractSrs1024(adaptor.getSource(), index1Cst);

        // Concat two 512-bit vectors to a 1024-bit vector using
        // vector::ShuffleOp
        auto v16i32Res0 = forceCastValueToType(
            rewriter, loc, res0, VectorType::get({16}, rewriter.getI32Type()));
        auto v16i32Res1 = forceCastValueToType(
            rewriter, loc, res1, VectorType::get({16}, rewriter.getI32Type()));

        SmallVector<int64_t> concatMask;
        for (int i = 0; i < 32; ++i) {
          concatMask.push_back(i);
        }
        srsIntrOp = rewriter.create<vector::ShuffleOp>(loc, v16i32Res0,
                                                       v16i32Res1, concatMask);
      }
    }

    if (!srsIntrOp) {
      op.emitWarning() << "aievec.srs is not supported.\n";
      return failure();
    }

    // create bitcast/shape_cast for result if needed
    auto resultVal = forceCastValueToType(rewriter, loc, srsIntrOp,
                                          op.getResult().getType());
    rewriter.replaceOp(op, resultVal);

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
        rewriter, op->getLoc(), cast<MemRefType>(op.getSource().getType()),
        adaptor.getSource(), adaptor.getIndices());

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

    SmallVector<Value> sources = adaptor.getSources();
    Value src = sources.front();
    VectorType srcType = cast<VectorType>(src.getType());
    Type srcScalarType = srcType.getElementType();
    unsigned srcBitWidth = srcScalarType.getIntOrFloatBitWidth();
    int srcLanes = getVectorLaneSize(srcType);
    int srcVectorSize = srcBitWidth * srcLanes;

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    if (sources.size() != 2 && sources.size() != 4) {
      op.emitWarning() << "aievec.concat with " << sources.size()
                       << " operands is not supported.\n";
      return failure();
    }

    // create xllvm intrinsic
    Value concatOp = nullptr;
    if (srcVectorSize == 256 && resultVectorSize == 512) {
      concatOp = rewriter.create<xllvm::ConcatI512I256IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, adaptor.getSources(),
              {VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type())}));
    } else if (srcVectorSize == 256 && resultVectorSize == 1024) {
      concatOp = rewriter.create<xllvm::ConcatI1024I256IntrOp>(
          loc, VectorType::get({32}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, adaptor.getSources(),
              {VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type())}));
    } else if (srcVectorSize == 512 && resultVectorSize == 1024) {
      concatOp = rewriter.create<xllvm::ConcatI1024I512IntrOp>(
          loc, VectorType::get({32}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, adaptor.getSources(),
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type())}));
    } else {
      op.emitWarning() << "aievec.concat with " << srcVectorSize
                       << "-bit operands, and " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create bitcast/shape_cast for result
    auto resultVal =
        forceCastValueToType(rewriter, loc, concatOp, op.getResult().getType());
    rewriter.replaceOp(op, resultVal);

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

    Value src = adaptor.getSource();
    VectorType srcType = cast<VectorType>(src.getType());
    Type srcScalarType = srcType.getElementType();
    unsigned srcBitWidth = srcScalarType.getIntOrFloatBitWidth();
    int srcLanes = getVectorLaneSize(srcType);
    int srcVectorSize = srcBitWidth * srcLanes;

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // create constant for index
    auto indexCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(op.getIndex()));

    // create xllvm intrinsic
    SmallVector<Value> operands({adaptor.getSource(), indexCst});
    Value extOp = nullptr;
    // Integer types
    if (resultVectorSize == 256 && srcVectorSize == 512) {
      extOp = rewriter.create<xllvm::ExtI256I512IntrOp>(
          loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 512 && srcVectorSize == 1024) {
      extOp = rewriter.create<xllvm::ExtI512I1024IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 256 && srcVectorSize == 1024) {
      extOp = rewriter.create<xllvm::ExtI256I1024IntrOp>(
          loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 128 && srcVectorSize == 512) {
      auto shiftOp = adaptor.getSource();
      if (op.getIndex() > 0) {
        auto undefOp = rewriter.create<xllvm::UndefV16I32IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()));
        auto stepCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto shiftCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(op.getIndex() * 16));
        SmallVector<Value> shiftOperands{adaptor.getSource(), undefOp, stepCst,
                                         shiftCst};
        // Right shift the source vector in index * 16 bytes (i.e. in index *
        // 128 bits). The integer index is expected to be 0 to 3.
        shiftOp = rewriter.create<xllvm::VectorShiftI512I512IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, shiftOperands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
      }
      // The underlying intrinsic takes a source vector and extract the lowest
      // 128-bit. i.e. it always extracts the input vector with index = 0.
      extOp = rewriter.create<xllvm::ExtI128I512IntrOp>(
          loc, VectorType::get({4}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, /*operands=*/{shiftOp},
              {VectorType::get({16}, rewriter.getI32Type())}));
    } else {
      op.emitWarning() << "aievec.ext with " << srcVectorSize
                       << "-bit source, and " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create bitcast/shape_cast for result
    auto resultVal =
        forceCastValueToType(rewriter, loc, extOp, op.getResult().getType());
    rewriter.replaceOp(op, resultVal);

    return success();
  }
};

class SelectOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::aie1::SelectOp> {
public:
  using ConvertOpToLLVMPattern<aievec::aie1::SelectOp>::ConvertOpToLLVMPattern;

  static std::string getIntrinsicName(aievec::aie1::SelectOp op) {
    auto xbuffType = cast<VectorType>(op.getXbuff().getType());
    std::stringstream ss;
    ss << "llvm.aie.prim." << getVectorTypeString(xbuffType) << ".select";
    return ss.str();
  }

  LogicalResult
  matchAndRewrite(aievec::aie1::SelectOp op, OpAdaptor adaptor,
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

class MaxOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::MaxOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MaxOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    VectorType resultType = cast<VectorType>(op.getResult().getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // aievec.max op has the AllTypesMatch constraint on lhs/rhs/res
    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.max conversion with " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create xllvm intrinsic
    Value maxOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for third operand `cmp`
      // Note: `cmp` is implicitly treated as `sign` to the vmax intrinsic
      auto cmpCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        maxOp = rewriter.create<xllvm::VectorMaxLt8IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({2}, rewriter.getI32Type())}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({64}, rewriter.getI8Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 16) {
        maxOp = rewriter.create<xllvm::VectorMaxLt16IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({32}, rewriter.getI16Type()),
                 VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 32) {
        maxOp = rewriter.create<xllvm::VectorMaxLt32IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}));
      }
    } else {
      if (resultBitWidth == 16) {
        maxOp = rewriter.create<xllvm::VectorMaxLtBf16IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getBF16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getLhs(), adaptor.getRhs()},
                {VectorType::get({32}, rewriter.getBF16Type()),
                 VectorType::get({32}, rewriter.getBF16Type())}));
      }
    }

    if (!maxOp) {
      // We have checked the lhs/rhs/res to be 512-bit vectors. Hence, a
      // possible failure here is due to unsupported element datatype.
      op.emitWarning() << "aievec.max conversion fails due to unsupported "
                          "element data type.\n";
      return failure();
    }

    // create llvm.extractvalue for the first element in the LLVMStruct
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, maxOp,
                                                      /*position=*/0);

    return success();
  }
};

class MinOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::MinOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MinOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    VectorType resultType = cast<VectorType>(op.getResult().getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // aievec.min op has the AllTypesMatch constraint on lhs/rhs/res
    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.min conversion with " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create xllvm intrinsic
    Value minOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for third operand `cmp`
      // Note: `cmp` is implicitly treated as `sign` to the vmin intrinsic
      auto cmpCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        minOp = rewriter.create<xllvm::VectorMinGe8IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({2}, rewriter.getI32Type())}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({64}, rewriter.getI8Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 16) {
        minOp = rewriter.create<xllvm::VectorMinGe16IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({32}, rewriter.getI16Type()),
                 VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 32) {
        minOp = rewriter.create<xllvm::VectorMinGe32IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}));
      }
    } else {
      if (resultBitWidth == 16) {
        minOp = rewriter.create<xllvm::VectorMinGeBf16IntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getBF16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getLhs(), adaptor.getRhs()},
                {VectorType::get({32}, rewriter.getBF16Type()),
                 VectorType::get({32}, rewriter.getBF16Type())}));
      }
    }

    if (!minOp) {
      // We have checked the lhs/rhs/res to be 512-bit vectors. Hence, a
      // possible failure here is due to unsupported element datatype.
      op.emitWarning() << "aievec.min conversion fails due to unsupported "
                          "element data type.\n";
      return failure();
    }

    // create llvm.extractvalue for the first element in the LLVMStruct
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, minOp,
                                                      /*position=*/0);

    return success();
  }
};

// AIE2p version of MaxOp conversion
class MaxOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MaxOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MaxOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    VectorType resultType = cast<VectorType>(op.getResult().getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // aievec.max op has the AllTypesMatch constraint on lhs/rhs/res
    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.max conversion with " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create xllvm intrinsic
    Value maxOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for third operand `cmp`
      // Note: `cmp` is implicitly treated as `sign` to the vmax intrinsic
      auto cmpCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        maxOp = rewriter.create<xllvm::VectorMaxLt8AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({2}, rewriter.getI32Type())}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({64}, rewriter.getI8Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 16) {
        maxOp = rewriter.create<xllvm::VectorMaxLt16AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({32}, rewriter.getI16Type()),
                 VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 32) {
        maxOp = rewriter.create<xllvm::VectorMaxLt32AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}));
      }
    } else {
      if (resultBitWidth == 16) {
        maxOp = rewriter.create<xllvm::VectorMaxLtBf16AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getBF16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getLhs(), adaptor.getRhs()},
                {VectorType::get({32}, rewriter.getBF16Type()),
                 VectorType::get({32}, rewriter.getBF16Type())}));
      }
    }

    if (!maxOp) {
      // We have checked the lhs/rhs/res to be 512-bit vectors. Hence, a
      // possible failure here is due to unsupported element datatype.
      op.emitWarning() << "aievec.max conversion fails due to unsupported "
                          "element data type.\n";
      return failure();
    }

    // create llvm.extractvalue for the first element in the LLVMStruct
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, maxOp,
                                                      /*position=*/0);

    return success();
  }
};

// AIE2p version of MinOp conversion
class MinOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MinOp> {
public:
  using ConvertOpToLLVMPattern<aievec::MinOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    VectorType resultType = cast<VectorType>(op.getResult().getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // aievec.min op has the AllTypesMatch constraint on lhs/rhs/res
    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.min conversion with " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create xllvm intrinsic
    Value minOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for third operand `cmp`
      // Note: `cmp` is implicitly treated as `sign` to the vmin intrinsic
      auto cmpCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        minOp = rewriter.create<xllvm::VectorMinGe8AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({2}, rewriter.getI32Type())}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({64}, rewriter.getI8Type()),
                 VectorType::get({64}, rewriter.getI8Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 16) {
        minOp = rewriter.create<xllvm::VectorMinGe16AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({32}, rewriter.getI16Type()),
                 VectorType::get({32}, rewriter.getI16Type()),
                 rewriter.getI32Type()}));
      } else if (resultBitWidth == 32) {
        minOp = rewriter.create<xllvm::VectorMinGe32AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, operands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type()}));
      }
    } else {
      if (resultBitWidth == 16) {
        minOp = rewriter.create<xllvm::VectorMinGeBf16AIE2pIntrOp>(
            loc,
            mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                {VectorType::get({32}, rewriter.getBF16Type()),
                 rewriter.getI32Type()}),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getLhs(), adaptor.getRhs()},
                {VectorType::get({32}, rewriter.getBF16Type()),
                 VectorType::get({32}, rewriter.getBF16Type())}));
      }
    }

    if (!minOp) {
      // We have checked the lhs/rhs/res to be 512-bit vectors. Hence, a
      // possible failure here is due to unsupported element datatype.
      op.emitWarning() << "aievec.min conversion fails due to unsupported "
                          "element data type.\n";
      return failure();
    }

    // create llvm.extractvalue for the first element in the LLVMStruct
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, minOp,
                                                      /*position=*/0);

    return success();
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
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    if (resultVectorSize != 512) {
      op.emitWarning()
          << "aievec.broadcast_scalar conversion with result vector size "
          << resultVectorSize << " is not implemented.\n";
      return failure();
    }

    // Integer types
    if (llvm::isa<IntegerType>(resultScaTy)) {
      Value src = adaptor.getSource();
      Type srcType = src.getType();
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();

      if (srcBitWidth < 32) {
        src = rewriter.create<LLVM::SExtOp>(loc, rewriter.getI32Type(), src);
      }

      if (resultBitWidth == 8) {
        rewriter.replaceOpWithNewOp<xllvm::VectorBroadcast8I512IntrOp>(
            op, VectorType::get({64}, rewriter.getI8Type()), src);
      } else if (resultBitWidth == 16) {
        rewriter.replaceOpWithNewOp<xllvm::VectorBroadcast16I512IntrOp>(
            op, VectorType::get({32}, rewriter.getI16Type()), src);
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
      } else if (resultBitWidth == 32) {
        rewriter.replaceOpWithNewOp<xllvm::VectorBroadcastfloatI512IntrOp>(
            op, VectorType::get({16}, rewriter.getF32Type()),
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

// AIE2p version of BroadcastScalarOp conversion using insertelement +
// shufflevector
class BroadcastScalarOpAIE2pConversion
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
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // Support both 256-bit and 512-bit vectors for AIE2p
    if (resultVectorSize != 256 && resultVectorSize != 512) {
      op.emitWarning()
          << "aievec.broadcast_scalar conversion with result vector size "
          << resultVectorSize << " is not implemented for AIE2p.\n";
      return failure();
    }

    Value src = adaptor.getSource();
    Type srcType = src.getType();

    // For integer types, extend or truncate to match result element type
    if (llvm::isa<IntegerType>(resultScaTy)) {
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();
      if (srcBitWidth < resultBitWidth) {
        src = rewriter.create<LLVM::SExtOp>(loc, resultScaTy, src);
      } else if (srcBitWidth > resultBitWidth) {
        src = rewriter.create<LLVM::TruncOp>(loc, resultScaTy, src);
      }
    }

    // Create poison vector of the result type
    auto poisonVec = rewriter.create<LLVM::PoisonOp>(loc, resultType);

    // Insert scalar at position 0
    auto idx0 = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    auto insertedVec = rewriter.create<LLVM::InsertElementOp>(
        loc, resultType, poisonVec, src, idx0);

    // Create shufflevector mask with all zeros (broadcast position 0 to all
    // lanes)
    SmallVector<int64_t> broadcastMask(resultLanes, 0);
    auto broadcastVec = rewriter.create<vector::ShuffleOp>(
        loc, insertedVec, insertedVec, broadcastMask);

    rewriter.replaceOp(op, broadcastVec);
    return success();
  }
};

class ShiftOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::ShiftOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ShiftOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ShiftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.shift conversion with result vector size "
                       << resultVectorSize << " is not implemented.\n";
      return failure();
    }

    // assume step is always zero
    auto stepCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // create xllvm intrinsic
    Value shiftOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getLhs(), adaptor.getRhs(), stepCst, adaptor.getShift()});
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // Integer types
      shiftOp = rewriter.create<xllvm::VectorShiftI512I512IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      // Float types
      shiftOp = rewriter.create<xllvm::VectorShiftBF512BF512IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getBF16Type()),
               VectorType::get({32}, rewriter.getBF16Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    }

    // create bitcast/shape_cast for result
    auto resultVal =
        forceCastValueToType(rewriter, loc, shiftOp, op.getResult().getType());
    rewriter.replaceOp(op, resultVal);

    return success();
  }
};

// AIE2p version of ShiftOp conversion
class ShiftOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ShiftOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ShiftOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ShiftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.shift conversion with result vector size "
                       << resultVectorSize << " is not implemented.\n";
      return failure();
    }

    // assume step is always zero
    auto stepCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // create xllvm intrinsic
    Value shiftOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getLhs(), adaptor.getRhs(), stepCst, adaptor.getShift()});
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // Integer types - use AIE2p intrinsic
      shiftOp = rewriter.create<xllvm::VectorShiftI512I512AIE2pIntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      // Float types - use AIE2p intrinsic
      shiftOp = rewriter.create<xllvm::VectorShiftBF512BF512AIE2pIntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getBF16Type()),
               VectorType::get({32}, rewriter.getBF16Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    }

    // create bitcast/shape_cast for result
    auto resultVal =
        forceCastValueToType(rewriter, loc, shiftOp, op.getResult().getType());
    rewriter.replaceOp(op, resultVal);

    return success();
  }
};

class ExtractElemOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ExtElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExtElemOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ExtElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type resultType = op.getResult().getType();
    unsigned resultBitWidth = resultType.getIntOrFloatBitWidth();

    Value src = adaptor.getSource();
    VectorType srcType = cast<VectorType>(src.getType());
    Type srcScalarType = srcType.getElementType();
    unsigned srcBitWidth = srcScalarType.getIntOrFloatBitWidth();
    int srcLanes = getVectorLaneSize(srcType);
    int srcVectorSize = srcBitWidth * srcLanes;

    if (srcVectorSize != 512) {
      op.emitWarning() << "aievec.ext_elem conversion with source vector size "
                       << srcVectorSize << " is not supported.\n";
      return failure();
    }

    // create constant for sign
    auto signCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    // create xllvm intrinsic
    Value extElemOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getSource(), adaptor.getIndex(), signCst});
    if (resultBitWidth == 8) {
      extElemOp = rewriter.create<xllvm::VectorExtractElem8I512IntrOp>(
          loc, rewriter.getI32Type(),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({64}, rewriter.getI8Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else if (resultBitWidth == 16) {
      extElemOp = rewriter.create<xllvm::VectorExtractElem16I512IntrOp>(
          loc, rewriter.getI32Type(),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI16Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else if (resultBitWidth == 32) {
      extElemOp = rewriter.create<xllvm::VectorExtractElem32I512IntrOp>(
          loc, rewriter.getI32Type(),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      op.emitWarning() << "aievec.ext_elem conversion with result bit width "
                       << resultBitWidth << " is not implemented.\n";
      return failure();
    }

    // create truncation op (and bitcast op)
    if (llvm::isa<IntegerType>(resultType)) {
      if (resultBitWidth < 32) {
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultType, extElemOp);
      } else {
        rewriter.replaceOp(op, extElemOp);
      }
    } else {
      // Float types
      if (resultBitWidth == 16) {
        extElemOp = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI16Type(),
                                                   extElemOp);
      }
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType, extElemOp);
    }

    return success();
  }
};

class FMAElemOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::FMAElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::FMAElemOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::FMAElemOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = fmaOp.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto acc = adaptor.getAcc();
    auto lhsTy = cast<VectorType>(lhs.getType());
    auto rhsTy = cast<VectorType>(rhs.getType());
    auto accTy = cast<VectorType>(acc.getType());
    auto flatLhsTy = getFlattenedVectorType(lhsTy);
    auto flatRhsTy = getFlattenedVectorType(rhsTy);
    auto flatAccTy = getFlattenedVectorType(accTy);

    // Flatten operands, if needed
    if (lhsTy != flatLhsTy)
      lhs = rewriter.create<vector::ShapeCastOp>(loc, flatLhsTy, lhs);
    if (rhsTy != flatRhsTy)
      rhs = rewriter.create<vector::ShapeCastOp>(loc, flatRhsTy, rhs);
    if (accTy != flatAccTy)
      acc = rewriter.create<vector::ShapeCastOp>(loc, flatAccTy, acc);

    // Build vmac configuration constant
    Type i32ty = rewriter.getI32Type();
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty,
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
            /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
            /*sub_mask=*/0)));

    // Insert vmac intrinsic
    auto v32bf16Ty = VectorType::get({32}, rewriter.getBF16Type());
    auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
    auto macIntrOp = rewriter.create<xllvm::MacConfBF16IntrOp>(
        loc, v8i64Ty,
        forceCastOperandsToSignature(rewriter, loc, {lhs, rhs, acc, confCst},
                                     {v32bf16Ty, v32bf16Ty, v8i64Ty, i32ty}));

    // Recast/Reshape result
    auto resVal =
        forceCastValueToType(rewriter, loc, macIntrOp.getResult(), flatAccTy);
    if (flatAccTy != accTy)
      resVal = rewriter.create<vector::ShapeCastOp>(loc, accTy, resVal);

    rewriter.replaceOp(fmaOp, resVal);
    return success();
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
      return {DecodedMatMulOp::Kind::BF16, lhs, rhs, acc,
              aiev2_vmac_compute_control(
                  /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
                  /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                  /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                  /*sub_mask=*/0)};

    int signX = 0, signY = 0;
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
    if (auto extSIOp = lhs.getDefiningOp<arith::ExtSIOp>()) {
      lhs = extSIOp.getIn();
      lhsVecTy = cast<VectorType>(lhs.getType());
      lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
      signX = 1;
    } else if (auto extUIOp = lhs.getDefiningOp<arith::ExtUIOp>()) {
      lhs = extUIOp.getIn();
      lhsVecTy = cast<VectorType>(lhs.getType());
      lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
    } else {
      // NOTE: We're choosing 'signed' by default
      if (!lhsScaTy.isUnsigned())
        signX = 1;
    }
    auto lhsShape = lhsVecTy.getShape();

    auto rhsVecTy = cast<VectorType>(rhs.getType());
    auto rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
    if (auto extSIOp = rhs.getDefiningOp<arith::ExtSIOp>()) {
      rhs = extSIOp.getIn();
      rhsVecTy = cast<VectorType>(rhs.getType());
      rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
      signY = 1;
    } else if (auto extUIOp = rhs.getDefiningOp<arith::ExtUIOp>()) {
      rhs = extUIOp.getIn();
      rhsVecTy = cast<VectorType>(rhs.getType());
      rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
    } else {
      // NOTE: We're choosing 'signed' by default
      if (!rhsScaTy.isUnsigned())
        signY = 1;
    }

    unsigned lhsBitWidth = lhsScaTy.getWidth();
    unsigned rhsBitWidth = rhsScaTy.getWidth();
    auto accScaTy = cast<IntegerType>(accVecTy.getElementType());
    unsigned accBitWidth = accScaTy.getWidth();
    if (accBitWidth == 32) {
      if (lhsBitWidth == 8) {
        if (rhsBitWidth == 4) {
          // <4x16xi8> x <16x8xi4> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/0,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        } else {
          // <4x8xi8> x <8x8xi8> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/1,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        }
      } else {
        if (rhsBitWidth == 8) {
          // <4x4xi16> x <4x8xi8> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/2,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        } else {
          // <4x2xi16> x <2x8xi16> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/3,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        }
      }
    }

    if (lhsBitWidth == 16) {
      if (rhsBitWidth == 8) {
        if (lhsShape == ArrayRef<int64_t>({2, 8})) {
          // <2x8xi16> x <8x8xi8> + <2x8xi64>
          return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1,
                      /*bmode=*/2,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        }
        // <4x8xi16> x <8x4xi8> + <4x4xi64>
        return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
                aiev2_vmac_compute_control(
                    /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/2,
                    /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      }
      if (lhsShape == ArrayRef<int64_t>({2, 4})) {
        // <2x4xi16> x <4x8xi16> + <2x8xi64>
        return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
                aiev2_vmac_compute_control(
                    /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/3,
                    /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      }
      // <4x4xi16> x <4x4xi16> + <4x4xi64>
      return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
              aiev2_vmac_compute_control(
                  /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/3,
                  /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                  /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                  /*sub_mask=*/0)};
    }
    // <4x2xi32> x <2x4xi16> + <4x4xi64>
    return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
            aiev2_vmac_compute_control(
                /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/0,
                /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                /*sub_mask=*/0)};
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

// Helper function to transpose an 8×8 matrix using vshuffle intrinsics
// Input: v64f32 representing 8×8 matrix
// Output: v64f32 transposed matrix
static Value transpose8x8WithVShuffle(OpBuilder &rewriter, Location loc,
                                      Type i32ty, Value matrix64f32) {
  // Bitcast <64 x f32> to <64 x i32>
  auto matrix64i32 = rewriter.create<LLVM::BitcastOp>(
      loc, VectorType::get({64}, i32ty), matrix64f32);

  // Extract two <16 x i32> chunks from lower half [0-31]
  SmallVector<int64_t> chunk0Mask, chunk1Mask;
  for (int i = 0; i < 16; ++i) {
    chunk0Mask.push_back(i);
    chunk1Mask.push_back(16 + i);
  }
  auto matrix16i32_0 = rewriter.create<vector::ShuffleOp>(
      loc, matrix64i32, matrix64i32, chunk0Mask);
  auto matrix16i32_1 = rewriter.create<vector::ShuffleOp>(
      loc, matrix64i32, matrix64i32, chunk1Mask);

  // Apply vshuffle with modes 52 and 53
  auto shuffleMode52 = rewriter.create<LLVM::ConstantOp>(
      loc, i32ty, rewriter.getI32IntegerAttr(52));
  auto shuffleMode53 = rewriter.create<LLVM::ConstantOp>(
      loc, i32ty, rewriter.getI32IntegerAttr(53));

  auto shuffled52_lo = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
      loc, VectorType::get({16}, i32ty), matrix16i32_0, matrix16i32_1,
      shuffleMode52);
  auto shuffled53_lo = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
      loc, VectorType::get({16}, i32ty), matrix16i32_0, matrix16i32_1,
      shuffleMode53);

  // Concatenate to get transposed lower 32 elements
  SmallVector<int64_t> concatLowerMask;
  for (int i = 0; i < 32; ++i)
    concatLowerMask.push_back(i);
  auto transposedLower32 = rewriter.create<vector::ShuffleOp>(
      loc, shuffled52_lo, shuffled53_lo, concatLowerMask);

  // Extract two <16 x i32> chunks from upper half [32-63]
  SmallVector<int64_t> chunk2Mask, chunk3Mask;
  for (int i = 0; i < 16; ++i) {
    chunk2Mask.push_back(32 + i);
    chunk3Mask.push_back(48 + i);
  }
  auto matrix16i32_2 = rewriter.create<vector::ShuffleOp>(
      loc, matrix64i32, matrix64i32, chunk2Mask);
  auto matrix16i32_3 = rewriter.create<vector::ShuffleOp>(
      loc, matrix64i32, matrix64i32, chunk3Mask);

  // Apply vshuffle with modes 52 and 53 to upper half
  auto shuffled52_hi = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
      loc, VectorType::get({16}, i32ty), matrix16i32_2, matrix16i32_3,
      shuffleMode52);
  auto shuffled53_hi = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
      loc, VectorType::get({16}, i32ty), matrix16i32_2, matrix16i32_3,
      shuffleMode53);

  // Concatenate to get transposed upper 32 elements
  SmallVector<int64_t> concatUpperMask;
  for (int i = 0; i < 32; ++i)
    concatUpperMask.push_back(i);
  auto transposedUpper32 = rewriter.create<vector::ShuffleOp>(
      loc, shuffled52_hi, shuffled53_hi, concatUpperMask);

  // Concatenate lower and upper to get full <64 x i32> transposed result
  SmallVector<int64_t> finalConcatMask;
  for (int i = 0; i < 64; ++i)
    finalConcatMask.push_back(i);
  auto transposed64i32 = rewriter.create<vector::ShuffleOp>(
      loc, transposedLower32, transposedUpper32, finalConcatMask);

  // Cast back to <64 x f32>
  return rewriter.create<LLVM::BitcastOp>(
      loc, VectorType::get({64}, rewriter.getF32Type()), transposed64i32);
}

// Helper function to perform BFP16-based 8×8 matmul via mac_8x8_8x8T_conf
// LHS: v64accfloat, RHS: v64accfloat (transposed 8×8), ACC: v64i32
// Returns: v64i32 result
static Value performBFP16_8x8MatMul(OpBuilder &rewriter, Location loc,
                                    Type i32ty, Value lhs64f32,
                                    Value rhs64f32Transposed, Value acc64i32,
                                    Value confCst) {
  auto v64i32Ty = VectorType::get({64}, rewriter.getI32Type());

  // Convert both to BFP16 format
  auto bfpStructTy = mlir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {VectorType::get({64}, rewriter.getI8Type()),
                              VectorType::get({8}, rewriter.getI8Type())});

  auto lhsBFP =
      rewriter.create<xllvm::Vector64AccFloatToV64BFP16EBS8AIE2pIntrOp>(
          loc, bfpStructTy, lhs64f32);
  auto rhsBFP =
      rewriter.create<xllvm::Vector64AccFloatToV64BFP16EBS8AIE2pIntrOp>(
          loc, bfpStructTy, rhs64f32Transposed);

  // Extract mantissa and exponent
  auto lhsData = rewriter.create<LLVM::ExtractValueOp>(loc, lhsBFP, 0);
  auto lhsExp = rewriter.create<LLVM::ExtractValueOp>(loc, lhsBFP, 1);
  auto rhsData = rewriter.create<LLVM::ExtractValueOp>(loc, rhsBFP, 0);
  auto rhsExp = rewriter.create<LLVM::ExtractValueOp>(loc, rhsBFP, 1);

  // Perform BFP16 matmul
  return rewriter.create<xllvm::MacConfBFP576ACC2048AIE2pIntrOp>(
      loc, v64i32Ty, lhsData, lhsExp, rhsData, rhsExp, acc64i32, confCst);
}

// Helper function to perform 8×8×4 BF16 matmul following mac_8x8_8x4_bf16
// LHS: 64 bfloat16 (8×8 matrix), RHS: 32 bfloat16 (8×4 matrix)
// ACC: 32 float (8×4 result)
static Value perform8x8x4MatMul(OpBuilder &rewriter, Location loc, Type i32ty,
                                Value lhs64bf16, Value rhs32bf16,
                                Value acc32f32) {
  auto v32bf16Ty = VectorType::get({32}, rewriter.getBF16Type());
  auto v32f32Ty = VectorType::get({32}, rewriter.getF32Type());

  // Extract lower and upper halves of LHS (64 bfloat16 -> 2x 32 bfloat16)
  SmallVector<int64_t> lowerMask, upperMask;
  for (int i = 0; i < 32; ++i) {
    lowerMask.push_back(i);
    upperMask.push_back(32 + i);
  }

  auto xl =
      rewriter.create<vector::ShuffleOp>(loc, lhs64bf16, lhs64bf16, lowerMask);
  auto xh =
      rewriter.create<vector::ShuffleOp>(loc, lhs64bf16, lhs64bf16, upperMask);

  // Cast to v16xi32 for shuffle intrinsic
  auto xlI32 =
      forceCastValueToType(rewriter, loc, xl, VectorType::get({16}, i32ty));
  auto xhI32 =
      forceCastValueToType(rewriter, loc, xh, VectorType::get({16}, i32ty));

  // Shuffle with T16_8x8_lo (mode 52) and T16_8x8_hi (mode 53)
  auto shuffleModeLo = rewriter.create<LLVM::ConstantOp>(
      loc, i32ty, rewriter.getI32IntegerAttr(52));
  auto xa = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
      loc, VectorType::get({16}, i32ty), xlI32, xhI32, shuffleModeLo);

  auto shuffleModeHi = rewriter.create<LLVM::ConstantOp>(
      loc, i32ty, rewriter.getI32IntegerAttr(53));
  auto xb = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
      loc, VectorType::get({16}, i32ty), xlI32, xhI32, shuffleModeHi);

  // Convert back to bfloat16
  auto xaBF16 = forceCastValueToType(rewriter, loc, xa, v32bf16Ty);
  auto xbBF16 = forceCastValueToType(rewriter, loc, xb, v32bf16Ty);

  // Helper to extract and broadcast 8 elements to 32, then shuffle with T16_4x8
  auto extractBroadcastShuffle = [&](Value src, int idx) -> Value {
    SmallVector<int64_t> extractMask;
    int startIdx = idx * 8;
    for (int i = 0; i < 8; ++i)
      extractMask.push_back(startIdx + i);
    // Broadcast by repeating 4 times to get 32 elements
    for (int rep = 0; rep < 3; ++rep) {
      for (int i = 0; i < 8; ++i)
        extractMask.push_back(startIdx + i);
    }
    auto broadcasted =
        rewriter.create<vector::ShuffleOp>(loc, src, src, extractMask);

    // Apply T16_4x8 shuffle pattern (mode 29)
    auto broadI32 = forceCastValueToType(rewriter, loc, broadcasted,
                                         VectorType::get({16}, i32ty));
    auto shuffleMode4x8 = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty, rewriter.getI32IntegerAttr(29));
    auto shuffled = rewriter.create<xllvm::VectorShuffleAIE2pIntrOp>(
        loc, VectorType::get({16}, i32ty), broadI32, broadI32, shuffleMode4x8);

    return forceCastValueToType(rewriter, loc, shuffled, v32bf16Ty);
  };

  // Prepare 8 row vectors from xa and xb
  SmallVector<Value> rowVectors;
  for (int i = 0; i < 4; ++i)
    rowVectors.push_back(extractBroadcastShuffle(xaBF16, i));
  for (int i = 0; i < 4; ++i)
    rowVectors.push_back(extractBroadcastShuffle(xbBF16, i));

  // Helper to extract and broadcast 4 elements to 32 (for RHS columns)
  auto extractBroadcast4 = [&](Value src, int idx) -> Value {
    SmallVector<int64_t> mask;
    int startIdx = idx * 4;
    // Repeat the 4 elements 8 times to get 32 elements
    for (int rep = 0; rep < 8; ++rep) {
      for (int i = 0; i < 4; ++i)
        mask.push_back(startIdx + i);
    }
    return rewriter.create<vector::ShuffleOp>(loc, src, src, mask);
  };

  // Prepare 8 column vectors from RHS
  SmallVector<Value> colVectors;
  for (int i = 0; i < 8; ++i)
    colVectors.push_back(extractBroadcast4(rhs32bf16, i));

  // Perform 8 MAC operations with conf=60 (no zero_acc)
  auto conf60 = rewriter.create<LLVM::ConstantOp>(
      loc, i32ty, rewriter.getI32IntegerAttr(60));

  Value acc = acc32f32;
  for (int i = 0; i < 8; ++i) {
    acc = rewriter.create<xllvm::MacConfBF16I512ACC1024AIE2pIntrOp>(
        loc, v32f32Ty, rowVectors[i], colVectors[i], acc, conf60);
  }

  return acc;
}

class MatMulOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MatMulOp_AIE2P> {
  using ConvertOpToLLVMPattern<aievec::MatMulOp_AIE2P>::ConvertOpToLLVMPattern;
  struct DecodedMatMulOp {
    typedef enum {
      BF16_8x8x8_I1024_ACC2048,
      BF16_4x8x8_I1024_ACC1024,
      BF16_8x1x8_I512_ACC2048,
      BF16_4x8x4_I512_ACC512,
      BF16_8x8x4_I512_ACC1024,
      I8_8x8x8_I512_ACC2048,
      I16_8x2x8_I1024_ACC2048,
      UNSUPPORTED
    } Kind;
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

    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto rhsVecTy = cast<VectorType>(rhs.getType());
    auto accVecTy = cast<VectorType>(acc.getType());

    // Check for AIE2p integer matmul
    if (isa<IntegerType>(lhsVecTy.getElementType()) &&
        isa<IntegerType>(rhsVecTy.getElementType()) &&
        isa<IntegerType>(accVecTy.getElementType())) {

      auto lhsIntTy = cast<IntegerType>(lhsVecTy.getElementType());
      auto rhsIntTy = cast<IntegerType>(rhsVecTy.getElementType());
      auto accIntTy = cast<IntegerType>(accVecTy.getElementType());

      int lhsLanes = getVectorLaneSize(lhsVecTy);
      int rhsLanes = getVectorLaneSize(rhsVecTy);
      int accLanes = getVectorLaneSize(accVecTy);

      // Check for <8x8xi8> x <8x8xi8> + <8x8xi32>
      if (lhsIntTy.getWidth() == 8 && rhsIntTy.getWidth() == 8 &&
          accIntTy.getWidth() == 32 && lhsLanes == 64 && rhsLanes == 64 &&
          accLanes == 64) {
        // Uses I512.I512.ACC2048 (64 lanes of i8 -> 64 lanes of i32)
        return {DecodedMatMulOp::Kind::I8_8x8x8_I512_ACC2048, lhs, rhs, acc, 8};
      }

      // Check for <8x2xi16> x <2x8xi16> + <8x8xi32>
      // Note: Vectors are <8x8xi16> shape, but only lower <8x2xi16> and
      // <2x8xi16> contain data
      if (lhsIntTy.getWidth() == 16 && rhsIntTy.getWidth() == 16 &&
          accIntTy.getWidth() == 32 && lhsLanes == 16 && rhsLanes == 16 &&
          accLanes == 64) {
        // Uses I1024.I1024.ACC2048 (64 lanes of i16 -> 64 lanes of i32)
        return {DecodedMatMulOp::Kind::I16_8x2x8_I1024_ACC2048, lhs, rhs, acc,
                24};
      }
    }

    // Check for AIE2p bf16 matmul
    if (isa<BFloat16Type>(lhsVecTy.getElementType()) &&
        isa<BFloat16Type>(rhsVecTy.getElementType()) &&
        isa<Float32Type>(accVecTy.getElementType())) {

      // Determine input size and accumulator size to select the right variant
      int lhsLanes = getVectorLaneSize(lhsVecTy);
      int rhsLanes = getVectorLaneSize(rhsVecTy);
      int accLanes = getVectorLaneSize(accVecTy);

      // I512 inputs (32 lanes each) with ACC512 (16 lanes)
      if (lhsLanes == 32 && rhsLanes == 32 && accLanes == 16) {
        // Uses I512.I512.ACC512 (16 lanes of f32)
        return {DecodedMatMulOp::Kind::BF16_4x8x4_I512_ACC512, lhs, rhs, acc,
                60};
      }
      // Special case for 8x8x4 matmul: <8x8xbf16> x <8x4xbf16> + <8x4xf32>
      else if (lhsLanes == 64 && rhsLanes == 32 && accLanes == 32) {
        // Uses I512.I512.ACC1024 for each MAC operation
        return {DecodedMatMulOp::Kind::BF16_8x8x4_I512_ACC1024, lhs, rhs, acc,
                60};
      }
      // Special case for 4x8x8 matmul: <4x8xbf16> x <8x8xbf16> + <4x8xf32>
      else if (lhsLanes == 32 && rhsLanes == 64 && accLanes == 32) {
        // Uses BFP16 format via mac_8x8_8x8T_conf
        return {DecodedMatMulOp::Kind::BF16_4x8x8_I1024_ACC1024, lhs, rhs, acc,
                780};
      }
      // Special case for 8x1x8 matmul: <8x1xbf16> x <1x8xbf16> + <8x8xf32>
      else if (lhsLanes == 8 && rhsLanes == 8 && accLanes == 64) {
        // Outer product: transpose+replicate LHS, replicate RHS, use
        // mac_elem_64_conf
        return {DecodedMatMulOp::Kind::BF16_8x1x8_I512_ACC2048, lhs, rhs, acc,
                60};
      }
      // I1024 inputs (64 lanes each)
      else if (lhsLanes == 64 && rhsLanes == 64 && accLanes == 64) {
        // Uses I1024.I1024.ACC2048 (64 lanes of f32)
        return {DecodedMatMulOp::Kind::BF16_8x8x8_I1024_ACC2048, lhs, rhs, acc,
                60};
      }
    }

    return {DecodedMatMulOp::Kind::UNSUPPORTED, lhs, rhs, acc, -1};
  }
  LogicalResult
  matchAndRewrite(aievec::MatMulOp_AIE2P op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto decodedMatMulOp = decodeMatMulOp(adaptor);
    if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::UNSUPPORTED) {
      op.emitWarning() << "aievec.matmul_aie2p conversion is not supported for "
                          "this type combination.\n";
      return failure();
    }
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

    if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::I8_8x8x8_I512_ACC2048) {
      // <8x8xi8> x <8x8xi8> + <8x8xi32>
      // Signature: <32 x i64> @llvm.aie2p.I512.I512.ACC2048.mac.conf(
      //              <16 x i32>, <32 x i16>, <32 x i64>, i32)
      // Bitcast LHS <64 x i8> -> <16 x i32>
      // Bitcast RHS <64 x i8> -> <32 x i16>
      // Bitcast ACC <64 x i32> -> <32 x i64>
      matMulResVal =
          rewriter
              .create<xllvm::MacConfI512ACC2048AIE2pIntrOp>(
                  loc, VectorType::get({32}, rewriter.getI64Type()),
                  forceCastOperandsToSignature(
                      rewriter, loc, operands,
                      {VectorType::get({16}, rewriter.getI32Type()),
                       VectorType::get({32}, rewriter.getI16Type()),
                       VectorType::get({32}, rewriter.getI64Type()), i32ty}))
              .getResult();
    } else if (decodedMatMulOp.kind ==
               DecodedMatMulOp::Kind::I16_8x2x8_I1024_ACC2048) {
      // <8x2xi16> x <2x8xi16> + <8x8xi32>
      // Input vectors are 16 lanes each, need to pad to 64 lanes for intrinsic
      // Signature: <32 x i64> @llvm.aie2p.I1024.I1024.ACC2048.mac.conf(
      //              <32 x i32>, <64 x i16>, <32 x i64>, i32)

      // Pad LHS from <16 x i16> to <64 x i16> using shuffle
      SmallVector<int64_t> lhsPadMask;
      for (int i = 0; i < 16; ++i)
        lhsPadMask.push_back(i);
      for (int i = 16; i < 64; ++i)
        lhsPadMask.push_back(-1); // undef/poison
      auto lhsPadded = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, lhsPadMask);

      // Pad RHS from <16 x i16> to <64 x i16> using shuffle
      SmallVector<int64_t> rhsPadMask;
      for (int i = 0; i < 16; ++i)
        rhsPadMask.push_back(i);
      for (int i = 16; i < 64; ++i)
        rhsPadMask.push_back(-1); // undef/poison
      auto rhsPadded = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, rhsPadMask);

      // Update operands with padded vectors
      SmallVector<Value> paddedOperands(
          {lhsPadded, rhsPadded, decodedMatMulOp.acc, confCst});

      // Bitcast LHS <64 x i16> -> <32 x i32>
      // Keep RHS as <64 x i16>
      // Bitcast ACC <64 x i32> -> <32 x i64>
      matMulResVal =
          rewriter
              .create<xllvm::MacConfI1024ACC2048AIE2pIntrOp>(
                  loc, VectorType::get({32}, rewriter.getI64Type()),
                  forceCastOperandsToSignature(
                      rewriter, loc, paddedOperands,
                      {VectorType::get({32}, rewriter.getI32Type()),
                       VectorType::get({64}, rewriter.getI16Type()),
                       VectorType::get({32}, rewriter.getI64Type()), i32ty}))
              .getResult();
    } else if (decodedMatMulOp.kind ==
               DecodedMatMulOp::Kind::BF16_8x8x8_I1024_ACC2048) {
      // <8x8xbf16> x <8x8xbf16> + <8x8xf32>
      // This implements the 8×8×8 BF16 matmul using BFP16 format
      // Following the aie_api reference implementation that converts to BFP16

      auto v32f32Ty = VectorType::get({32}, rewriter.getF32Type());

      // Step 1: Convert LHS v64bfloat16 to v64accfloat (in two v32 chunks)
      SmallVector<int64_t> firstHalfMask, secondHalfMask;
      for (int i = 0; i < 32; ++i) {
        firstHalfMask.push_back(i);
        secondHalfMask.push_back(32 + i);
      }

      auto lhs32bf16_lo = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, firstHalfMask);
      auto lhs32bf16_hi = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, secondHalfMask);

      auto lhs32f32_lo =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, lhs32bf16_lo);
      auto lhs32f32_hi =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, lhs32bf16_hi);

      // Concat to v64accfloat
      SmallVector<int64_t> concatMask;
      for (int i = 0; i < 64; ++i)
        concatMask.push_back(i);
      auto lhs64f32 = rewriter.create<vector::ShuffleOp>(
          loc, lhs32f32_lo, lhs32f32_hi, concatMask);

      // Step 2: Convert RHS v64bfloat16 to v64accfloat (in two v32 chunks)
      auto rhs32bf16_lo = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, firstHalfMask);
      auto rhs32bf16_hi = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, secondHalfMask);

      auto rhs32f32_lo =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, rhs32bf16_lo);
      auto rhs32f32_hi =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, rhs32bf16_hi);

      auto rhs64f32 = rewriter.create<vector::ShuffleOp>(
          loc, rhs32f32_lo, rhs32f32_hi, concatMask);

      // Step 3: Transpose RHS 8×8 matrix using vshuffle intrinsics
      auto rhsTransposed =
          transpose8x8WithVShuffle(rewriter, loc, i32ty, rhs64f32);

      // Step 4: Use shared BFP16 8×8 matmul helper
      auto conf780 = rewriter.create<LLVM::ConstantOp>(
          loc, i32ty, rewriter.getI32IntegerAttr(780));

      matMulResVal = performBFP16_8x8MatMul(
          rewriter, loc, i32ty, lhs64f32, rhsTransposed,
          forceCastValueToType(rewriter, loc, decodedMatMulOp.acc,
                               VectorType::get({64}, rewriter.getI32Type())),
          conf780);
    } else if (decodedMatMulOp.kind ==
               DecodedMatMulOp::Kind::BF16_4x8x8_I1024_ACC1024) {
      // <4x8xbf16> x <8x8xbf16> + <4x8xf32>
      // LHS: 32 lanes, RHS: 64 lanes, ACC: 32 lanes
      // Similar to 8×8×8 but only use first 32 lanes of LHS

      auto v32f32Ty = VectorType::get({32}, rewriter.getF32Type());

      // Step 1: Convert LHS v32bfloat16 to v32accfloat, then pad to v64accfloat
      auto lhs32f32 =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, decodedMatMulOp.lhs);

      // Pad v32accfloat to v64accfloat using shuffle
      SmallVector<int64_t> lhsPadMask;
      for (int i = 0; i < 32; ++i)
        lhsPadMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        lhsPadMask.push_back(-1); // poison
      auto lhs64f32 = rewriter.create<vector::ShuffleOp>(loc, lhs32f32,
                                                         lhs32f32, lhsPadMask);

      // Step 2: Convert RHS v64bfloat16 to v64accfloat (in two v32 chunks)
      SmallVector<int64_t> firstHalfMask, secondHalfMask;
      for (int i = 0; i < 32; ++i) {
        firstHalfMask.push_back(i);
        secondHalfMask.push_back(32 + i);
      }

      auto rhs32bf16_lo = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, firstHalfMask);
      auto rhs32bf16_hi = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, secondHalfMask);

      auto rhs32f32_lo =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, rhs32bf16_lo);
      auto rhs32f32_hi =
          rewriter.create<xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp>(
              loc, v32f32Ty, rhs32bf16_hi);

      // Concat to v64accfloat
      SmallVector<int64_t> concatMask;
      for (int i = 0; i < 64; ++i)
        concatMask.push_back(i);
      auto rhs64f32 = rewriter.create<vector::ShuffleOp>(
          loc, rhs32f32_lo, rhs32f32_hi, concatMask);

      // Step 3: Transpose RHS 8×8 matrix using vshuffle intrinsics
      auto rhsTransposed =
          transpose8x8WithVShuffle(rewriter, loc, i32ty, rhs64f32);

      // Step 4: Pad ACC from 32 to 64 i32
      SmallVector<int64_t> accPadMask;
      for (int i = 0; i < 32; ++i)
        accPadMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        accPadMask.push_back(-1); // poison
      auto acc64i32 = rewriter.create<vector::ShuffleOp>(
          loc,
          forceCastValueToType(rewriter, loc, decodedMatMulOp.acc,
                               VectorType::get({32}, rewriter.getI32Type())),
          forceCastValueToType(rewriter, loc, decodedMatMulOp.acc,
                               VectorType::get({32}, rewriter.getI32Type())),
          accPadMask);

      // Step 5: Use shared BFP16 8×8 matmul helper
      auto result64i32 = performBFP16_8x8MatMul(
          rewriter, loc, i32ty, lhs64f32, rhsTransposed, acc64i32, confCst);

      // Step 6: Extract first 32 elements
      SmallVector<int64_t> extractMask;
      for (int i = 0; i < 32; ++i)
        extractMask.push_back(i);
      matMulResVal = rewriter.create<vector::ShuffleOp>(
          loc, result64i32, result64i32, extractMask);
    } else if (decodedMatMulOp.kind ==
               DecodedMatMulOp::Kind::BF16_8x1x8_I512_ACC2048) {
      // <8x1xbf16> x <1x8xbf16> + <8x8xf32>
      // Outer product: grow_replicate both to 64, transpose LHS, use
      // mac_elem_64_conf

      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());

      // Step 1: Replicate LHS from 8 to 64 elements (replicate 8 times)
      SmallVector<int64_t> lhsReplicateMask;
      for (int rep = 0; rep < 8; ++rep) {
        for (int i = 0; i < 8; ++i)
          lhsReplicateMask.push_back(i);
      }
      auto lhs64bf16 = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, lhsReplicateMask);

      // Step 2: Transpose LHS as 8×8 matrix
      SmallVector<int64_t> transposeMask;
      for (int c = 0; c < 8; ++c) {
        for (int r = 0; r < 8; ++r) {
          transposeMask.push_back(r * 8 + c);
        }
      }
      auto lhs64bf16Transposed = rewriter.create<vector::ShuffleOp>(
          loc, lhs64bf16, lhs64bf16, transposeMask);

      // Step 3: Replicate RHS from 8 to 64 elements (replicate 8 times)
      SmallVector<int64_t> rhsReplicateMask;
      for (int rep = 0; rep < 8; ++rep) {
        for (int i = 0; i < 8; ++i)
          rhsReplicateMask.push_back(i);
      }
      auto rhs64bf16 = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, rhsReplicateMask);

      // Step 4: Use mac_elem_64_conf (which is
      // MacConfBF16I512ACC2048AIE2pIntrOp)
      matMulResVal = rewriter.create<xllvm::MacConfBF16I512ACC2048AIE2pIntrOp>(
          loc, v64f32Ty, lhs64bf16Transposed, rhs64bf16, decodedMatMulOp.acc,
          confCst);
    } else if (decodedMatMulOp.kind ==
               DecodedMatMulOp::Kind::BF16_4x8x4_I512_ACC512) {
      // 4×8×4 matmul: <4x8xbf16> x <8x4xbf16> + <4x4xf32>
      // Following the reference pattern: a.grow<64>(), b,
      // acc.grow<32>().extract<16>(0) We pad LHS 32→64, pad ACC 16→32, call
      // 8×8×4 impl, then extract 32→16

      // Pad LHS from 32 to 64 bfloat16 using shuffle
      SmallVector<int64_t> lhsPadMask;
      for (int i = 0; i < 32; ++i)
        lhsPadMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        lhsPadMask.push_back(-1); // poison/undef
      auto lhsPadded = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, lhsPadMask);

      // Pad ACC from 16 to 32 float using shuffle
      SmallVector<int64_t> accPadMask;
      for (int i = 0; i < 16; ++i)
        accPadMask.push_back(i);
      for (int i = 16; i < 32; ++i)
        accPadMask.push_back(-1); // poison/undef
      auto accPadded = rewriter.create<vector::ShuffleOp>(
          loc, decodedMatMulOp.acc, decodedMatMulOp.acc, accPadMask);

      // Call the shared 8×8×4 helper with padded inputs
      Value acc32 = perform8x8x4MatMul(rewriter, loc, i32ty, lhsPadded,
                                       decodedMatMulOp.rhs, accPadded);

      // Extract first 16 elements from 32-element result
      SmallVector<int64_t> extractMask;
      for (int i = 0; i < 16; ++i)
        extractMask.push_back(i);
      matMulResVal =
          rewriter.create<vector::ShuffleOp>(loc, acc32, acc32, extractMask);
    } else if (decodedMatMulOp.kind ==
               DecodedMatMulOp::Kind::BF16_8x8x4_I512_ACC1024) {
      // Special 8×8×4 matmul: <8x8xbf16> x <8x4xbf16> + <8x4xf32>
      // Uses shared helper function
      matMulResVal =
          perform8x8x4MatMul(rewriter, loc, i32ty, decodedMatMulOp.lhs,
                             decodedMatMulOp.rhs, decodedMatMulOp.acc);
    }

    // Cast from flattened result back to original accumulator shape
    auto castFromAcc =
        forceCastValueToType(rewriter, loc, matMulResVal, accFlattenedVecTy);
    // Reshape back to original shape
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(),
                                                     castFromAcc);
    return success();
  }
};

// This pattern folds aievec.cast op. For AIE2, the accumulators are in 32/64
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

class ShuffleOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ShuffleOp> {
  using ConvertOpToLLVMPattern<aievec::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = shuffleOp.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto i32ty = rewriter.getI32Type();
    auto v16xi32ty = VectorType::get({16}, i32ty);
    if (!rhs)
      rhs = rewriter.create<xllvm::UndefV16I32IntrOp>(loc, v16xi32ty);

    auto modeAttrVal =
        rewriter
            .create<LLVM::ConstantOp>(loc, i32ty,
                                      static_cast<int32_t>(shuffleOp.getMode()))
            .getResult();
    auto vShuffleVal = rewriter
                           .create<xllvm::VectorShuffleIntrOp>(
                               loc, v16xi32ty,
                               forceCastOperandsToSignature(
                                   rewriter, loc,
                                   /*operands=*/{lhs, rhs, modeAttrVal},
                                   /*signature=*/{v16xi32ty, v16xi32ty, i32ty}))
                           .getResult();

    vShuffleVal = forceCastValueToType(rewriter, loc, vShuffleVal,
                                       shuffleOp.getResult().getType());

    rewriter.replaceOp(shuffleOp, vShuffleVal);

    return success();
  }
};

// Convert aievec.exp to xllvm.exp2 intrinsic for AIE2P
// Uses the identity: exp(x) = exp2(x * log2(e))
class ExpOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ExpOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExpOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = expOp.getLoc();
    auto srcType = cast<VectorType>(adaptor.getSource().getType());
    auto srcElemType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);

    // Only support v16bfloat16 for now
    if (laneSize != 16 || !srcElemType.isBF16())
      return expOp.emitWarning()
             << "aievec.exp conversion only supports v16bfloat16.\n";

    // Step 1: Create bf16 constant for log2(e) ≈ 1.442695
    auto log2eBF16Const = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getBF16Type(),
        rewriter.getFloatAttr(rewriter.getBF16Type(), 1.442695));

    // Broadcast log2(e) to v16bfloat16
    SmallVector<int64_t> broadcastMask;
    for (int i = 0; i < 16; ++i)
      broadcastMask.push_back(0);

    auto v1bf16 = rewriter.create<LLVM::UndefOp>(
        loc, VectorType::get({1}, rewriter.getBF16Type()));
    auto v1bf16Inserted = rewriter.create<LLVM::InsertElementOp>(
        loc, v1bf16, log2eBF16Const,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0));

    auto log2eVec = rewriter.create<vector::ShuffleOp>(
        loc, v1bf16Inserted, v1bf16Inserted, broadcastMask);

    // Step 2: Multiply input by log2(e) in bf16 domain using MulElemOp
    // This will use the I512.I512.ACC512 bf16 mul intrinsic
    auto v16bf16Ty = VectorType::get({16}, rewriter.getBF16Type());
    auto v16f32Ty = VectorType::get({16}, rewriter.getF32Type());

    // Multiply in bf16: x * log2(e)
    auto mulResult = rewriter.create<aievec::MulElemOp>(
        loc, v16f32Ty, adaptor.getSource(), log2eVec);

    // Step 3: Call exp2 intrinsic
    // exp2 takes v16float and returns v16bfloat16
    auto exp2Op =
        rewriter.create<xllvm::Exp2AIE2pIntrOp>(loc, v16bf16Ty, mulResult);

    rewriter.replaceOp(expOp, exp2Op.getResult());

    return success();
  }
};

void populateAIEVecToLLVMCommonConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  // clang-format off
  // Patterns that work for all backends (AIE1, AIE2, AIE2p)
  patterns.add<AddOpConversion,
               SubOpConversion,
               FMAOpConversion,
               MulOpConversion,
               UPDOpConversion,
               ExtOpConversion,
               SelectOpConversion,
               PackOpConversion,
               UnpackOpConversion,
               BroadcastOpConversion,
               BroadcastScalarOpConversion,
               FMAElemOpConversion,
               MatMulOpConversion,
               FoldAIECastOps,
               ShuffleOpConversion>(converter);
  // clang-format on
}

void populateAIEVecToLLVMAIE2ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    Aie2Fp32Emulation aie2Fp32EmulationOption) {
  // Patterns specific to AIE2 backend
  patterns.add<AddElemOpAIE2Conversion, SubElemOpAIE2Conversion>(converter);
  patterns.add<MulElemOpConversion>(converter, aie2Fp32EmulationOption);
  patterns.add<UPSOpAIE2Conversion, SRSOpAIE2Conversion>(converter);
  patterns.add<ShiftOpConversion>(converter);
  patterns.add<MaxOpConversion, MinOpConversion>(converter);
  patterns.add<ExtractElemOpConversion>(converter);
  patterns.add<ConcatOpConversion>(converter);
}

// AIE2p version of ExtractElemOp conversion using LLVM extractelement
class ExtractElemOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ExtElemOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExtElemOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ExtElemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // AIE2p doesn't have dedicated vextract intrinsics, so use LLVM
    // extractelement
    Value extracted = rewriter.create<LLVM::ExtractElementOp>(
        loc, adaptor.getSource(), adaptor.getIndex());

    rewriter.replaceOp(op, extracted);
    return success();
  }
};

// AIE2p version of ConcatOp conversion using vector.shuffle
class ConcatOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ConcatOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ConcatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> sources = adaptor.getSources();

    if (sources.empty()) {
      op.emitWarning() << "aievec.concat with no sources is not supported.\n";
      return failure();
    }

    // AIE2p doesn't have dedicated concat intrinsics, use vector.shuffle
    Value result = sources[0];

    // Build shuffle mask that concatenates all sources
    auto srcType = cast<VectorType>(sources[0].getType());
    int64_t srcLanes = getVectorLaneSize(srcType);

    if (sources.size() == 2) {
      // Concatenate two vectors using shuffle
      SmallVector<int64_t> mask;
      for (int64_t i = 0; i < srcLanes * 2; ++i)
        mask.push_back(i);

      result =
          rewriter.create<vector::ShuffleOp>(loc, sources[0], sources[1], mask);
    } else if (sources.size() == 4) {
      // Concatenate four vectors: first concat pairs, then concat results
      SmallVector<int64_t> pairMask;
      for (int64_t i = 0; i < srcLanes * 2; ++i)
        pairMask.push_back(i);

      auto pair0 = rewriter.create<vector::ShuffleOp>(loc, sources[0],
                                                      sources[1], pairMask);
      auto pair1 = rewriter.create<vector::ShuffleOp>(loc, sources[2],
                                                      sources[3], pairMask);

      SmallVector<int64_t> finalMask;
      for (int64_t i = 0; i < srcLanes * 4; ++i)
        finalMask.push_back(i);

      result = rewriter.create<vector::ShuffleOp>(loc, pair0, pair1, finalMask);
    } else {
      op.emitWarning() << "aievec.concat with " << sources.size()
                       << " operands is not supported for AIE2p.\n";
      return failure();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateAIEVecToLLVMAIE2pConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  // Patterns specific to AIE2p backend
  patterns.add<AddElemOpAIE2pConversion, SubElemOpAIE2pConversion>(converter);
  patterns.add<MulElemOpAIE2pConversion>(converter);
  patterns.add<UPSOpAIE2pConversion, SRSOpAIE2pConversion>(converter);
  patterns.add<MatMulOpAIE2pConversion>(converter);
  patterns.add<ShiftOpAIE2pConversion>(converter);
  patterns.add<MaxOpAIE2pConversion, MinOpAIE2pConversion>(converter);
  patterns.add<ExtractElemOpAIE2pConversion>(converter);
  patterns.add<ConcatOpAIE2pConversion>(converter);
  patterns.add<ExpOpAIE2pConversion>(converter);
  patterns.add<BroadcastScalarOpAIE2pConversion>(converter);
}

void populateAIEVecToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    Aie2Fp32Emulation aie2Fp32EmulationOption, StringRef aieTarget) {
  populateAIEVecToLLVMCommonConversionPatterns(converter, patterns);
  if (aieTarget == "aie2p")
    populateAIEVecToLLVMAIE2pConversionPatterns(converter, patterns);
  else
    populateAIEVecToLLVMAIE2ConversionPatterns(converter, patterns,
                                               aie2Fp32EmulationOption);
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

    populateAIEVecToLLVMConversionPatterns(converter, patterns,
                                           aie2Fp32Emulation, aieTarget);

    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<xilinx::aievec::AIEVecDialect,
                             xilinx::aievec::aie1::AIEVecAIE1Dialect>();
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect,
                           xilinx::xllvm::XLLVMDialect, ub::UBDialect>();
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
