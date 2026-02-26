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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/TypeUtilities.h"
#include <sstream>

using namespace mlir;

namespace xilinx::aievec {

inline static Value bitcastValueToType(OpBuilder &builder, Location loc,
                                       Value val, Type dstTy) {
  return LLVM::BitcastOp::create(builder, loc, dstTy, val).getResult();
}

// This function emits the instructions required to widen a 128b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<4xi32> to respect the intrinsic signature.
inline static Value widen128bVectorValueTo512b(OpBuilder &builder, Location loc,
                                               Value val) {
  return xllvm::VectorSetI512I128IntrOp::create(
             builder, loc, VectorType::get({16}, builder.getI32Type()),
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
      LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), (int32_t)0);
  return xllvm::VectorSetI512I256IntrOp::create(
             builder, loc, VectorType::get({16}, builder.getI32Type()),
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
      val = vector::ShapeCastOp::create(builder, loc, flatSrcVecTy, val);

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
      val = vector::ShapeCastOp::create(builder, loc, dstVecTy, val);

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

// Utility function to get or create a noinline scalar helper function.
// This is used to create optimization barriers that prevent LLVM from
// re-vectorizing unrolled scalar operations.
//
// Parameters:
//   - module: The parent module to insert the function into
//   - rewriter: The pattern rewriter
//   - opName: Base name of the operation (e.g., "fdiv", "addf", "mulf")
//   - device: Target device ("aie2", "aie2p", etc.)
//   - argTypes: Input argument types
//   - resultType: Return type
//   - bodyBuilder: Lambda that builds the function body given (builder, loc,
//   args)
//
// Returns: The helper function (created or existing)
//
// Function naming convention: __<device>_scalar_<opName>
// Example: __aie2p_scalar_fdiv, __aie2_scalar_addf
static LLVM::LLVMFuncOp getOrCreateScalarHelperFunc(
    ModuleOp module, OpBuilder &rewriter, StringRef opName, StringRef device,
    TypeRange argTypes, Type resultType,
    std::function<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {

  // Build function name: __<device>_scalar_<opName>
  std::string funcName = "__" + device.str() + "_scalar_" + opName.str();

  // Check if function already exists
  auto helperFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (helperFunc)
    return helperFunc;

  // Create new function
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  // Convert TypeRange to SmallVector<Type> for LLVM::LLVMFunctionType::get
  SmallVector<Type> argTypesVec(argTypes.begin(), argTypes.end());

  helperFunc = LLVM::LLVMFuncOp::create(
      rewriter, rewriter.getUnknownLoc(), funcName,
      LLVM::LLVMFunctionType::get(resultType, argTypesVec));

  // Mark as noinline to act as optimization barrier
  helperFunc->setAttr("passthrough", rewriter.getArrayAttr(
                                         {rewriter.getStringAttr("noinline")}));

  // Add function body
  auto *entryBlock = helperFunc.addEntryBlock(rewriter);
  OpBuilder::InsertionGuard bodyGuard(rewriter);
  rewriter.setInsertionPointToStart(entryBlock);

  // Collect function arguments
  SmallVector<Value> args;
  for (unsigned i = 0; i < argTypes.size(); ++i)
    args.push_back(entryBlock->getArgument(i));

  // Call the body builder with the function arguments
  bodyBuilder(rewriter, rewriter.getUnknownLoc(), args);

  return helperFunc;
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
      auto confCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(decodedAddElemOp.conf));
      SmallVector<Value> operands(
          {adaptor.getLhs(), adaptor.getRhs(), confCst});

      auto addElemOp = xllvm::AddAccFloatAIE2IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
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
      auto confCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(decodedSubElemOp.conf));
      SmallVector<Value> operands(
          {adaptor.getLhs(), adaptor.getRhs(), confCst});

      auto subElemOp = xllvm::SubAccFloatAIE2IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
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
    enum class Kind {
      FP32_FP32_FP32_16x1x1x1,
      FP32_FP32_FP32_32x1x1x1,
      UNSUPPORTED
    };
    Kind kind;
    int conf;
  };

  static DecodedAddElemOp decodeAddElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();
    int laneSize = getVectorLaneSize(lhsVecTy);

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedAddElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 32) {
        // FP32 add_elem
        if (laneSize == 16) {
          return {DecodedAddElemOp::Kind::FP32_FP32_FP32_16x1x1x1, /*conf*/ 60};
        } else if (laneSize == 32) {
          return {DecodedAddElemOp::Kind::FP32_FP32_FP32_32x1x1x1, /*conf*/ 60};
        }
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

    // Handle the FP32 add_elem for AIE2p (16-lane)
    // We need to expand <16xf32> to <64xf32> for the ACC2048 intrinsic
    if (decodedAddElemOp.kind ==
        DecodedAddElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      // Step 1: Bitcast <16 x float> to <8 x i64>
      auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
      auto lhsI64 =
          LLVM::BitcastOp::create(rewriter, loc, v8i64Ty, adaptor.getLhs());
      auto rhsI64 =
          LLVM::BitcastOp::create(rewriter, loc, v8i64Ty, adaptor.getRhs());

      // Step 2: Shuffle <8 x i64> to <32 x i64> (expand with poison values)
      auto v32i64Ty = VectorType::get({32}, rewriter.getI64Type());
      SmallVector<int64_t> expandMask = {0, 1, 2, 3, 4, 5, 6, 7};
      for (int i = 8; i < 32; ++i)
        expandMask.push_back(-1); // poison values

      auto lhsExpanded =
          vector::ShuffleOp::create(rewriter, loc, lhsI64, lhsI64, expandMask);
      auto rhsExpanded =
          vector::ShuffleOp::create(rewriter, loc, rhsI64, rhsI64, expandMask);

      // Step 3: Bitcast <32 x i64> to <64 x float>
      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());
      auto lhsF32 =
          LLVM::BitcastOp::create(rewriter, loc, v64f32Ty, lhsExpanded);
      auto rhsF32 =
          LLVM::BitcastOp::create(rewriter, loc, v64f32Ty, rhsExpanded);

      // Step 4: Call the ACC2048 intrinsic with conf=60
      auto confCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(60));

      // Create the intrinsic call
      auto addResult = xllvm::AddACC2048AccFloatAIE2pIntrOp::create(
          rewriter, loc, v64f32Ty, lhsF32, rhsF32, confCst);

      // Step 5: Bitcast <64 x float> back to <32 x i64>
      auto resultI64 =
          LLVM::BitcastOp::create(rewriter, loc, v32i64Ty, addResult);

      // Step 6: Shuffle to extract first 8 elements <32 x i64> -> <8 x i64>
      SmallVector<int64_t> extractMask = {0, 1, 2, 3, 4, 5, 6, 7};
      auto resultExtracted = vector::ShuffleOp::create(rewriter, loc, resultI64,
                                                       resultI64, extractMask);

      // Step 7: Bitcast <8 x i64> back to <16 x float>
      auto v16f32Ty = VectorType::get({16}, rewriter.getF32Type());
      auto finalResult =
          LLVM::BitcastOp::create(rewriter, loc, v16f32Ty, resultExtracted);

      rewriter.replaceOp(op, finalResult);
      return success();
    }

    // Handle the FP32 add_elem for AIE2p (32-lane)
    // Use ACC2048 intrinsic by padding to 64 lanes
    if (decodedAddElemOp.kind ==
        DecodedAddElemOp::Kind::FP32_FP32_FP32_32x1x1x1) {
      // Pad from <32 x float> to <64 x float> using shuffle
      SmallVector<int64_t> padMask;
      for (int i = 0; i < 32; ++i)
        padMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        padMask.push_back(-1); // poison/undef

      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());
      auto lhsPadded = vector::ShuffleOp::create(
          rewriter, loc, adaptor.getLhs(), adaptor.getLhs(), padMask);
      auto rhsPadded = vector::ShuffleOp::create(
          rewriter, loc, adaptor.getRhs(), adaptor.getRhs(), padMask);

      // Call ACC2048 intrinsic
      auto confCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(60));
      auto addResult = xllvm::AddACC2048AccFloatAIE2pIntrOp::create(
          rewriter, loc, v64f32Ty, lhsPadded, rhsPadded, confCst);

      // Extract first 32 elements from 64-element result
      SmallVector<int64_t> extractMask;
      for (int i = 0; i < 32; ++i)
        extractMask.push_back(i);
      auto finalResult = vector::ShuffleOp::create(rewriter, loc, addResult,
                                                   addResult, extractMask);

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
    enum class Kind {
      FP32_FP32_FP32_16x1x1x1,
      FP32_FP32_FP32_32x1x1x1,
      UNSUPPORTED
    };
    Kind kind;
    int conf;
  };

  static DecodedSubElemOp decodeSubElemOp(OpAdaptor op) {
    auto lhs = op.getLhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = lhsVecTy.getElementType();
    unsigned lhsBitWidth = lhsScaTy.getIntOrFloatBitWidth();
    int laneSize = getVectorLaneSize(lhsVecTy);

    // Integer types
    if (llvm::isa<IntegerType>(lhsScaTy)) {
      return {DecodedSubElemOp::Kind::UNSUPPORTED, -1};
    } else {
      // Float types
      if (lhsBitWidth == 32) {
        // FP32 sub_elem
        if (laneSize == 16) {
          return {DecodedSubElemOp::Kind::FP32_FP32_FP32_16x1x1x1, /*conf*/ 60};
        } else if (laneSize == 32) {
          return {DecodedSubElemOp::Kind::FP32_FP32_FP32_32x1x1x1, /*conf*/ 60};
        }
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

    // Handle the FP32 sub_elem for AIE2p (16-lane)
    // We need to expand <16xf32> to <64xf32> for the ACC2048 intrinsic
    if (decodedSubElemOp.kind ==
        DecodedSubElemOp::Kind::FP32_FP32_FP32_16x1x1x1) {
      // Step 1: Bitcast <16 x float> to <8 x i64>
      auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
      auto lhsI64 =
          LLVM::BitcastOp::create(rewriter, loc, v8i64Ty, adaptor.getLhs());
      auto rhsI64 =
          LLVM::BitcastOp::create(rewriter, loc, v8i64Ty, adaptor.getRhs());

      // Step 2: Shuffle <8 x i64> to <32 x i64> (expand with poison values)
      auto v32i64Ty = VectorType::get({32}, rewriter.getI64Type());
      SmallVector<int64_t> expandMask = {0, 1, 2, 3, 4, 5, 6, 7};
      for (int i = 8; i < 32; ++i)
        expandMask.push_back(-1); // poison values

      auto lhsExpanded =
          vector::ShuffleOp::create(rewriter, loc, lhsI64, lhsI64, expandMask);
      auto rhsExpanded =
          vector::ShuffleOp::create(rewriter, loc, rhsI64, rhsI64, expandMask);

      // Step 3: Bitcast <32 x i64> to <64 x float>
      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());
      auto lhsF32 =
          LLVM::BitcastOp::create(rewriter, loc, v64f32Ty, lhsExpanded);
      auto rhsF32 =
          LLVM::BitcastOp::create(rewriter, loc, v64f32Ty, rhsExpanded);

      // Step 4: Call the ACC2048 intrinsic with conf=60
      auto confCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(60));

      // Create the intrinsic call
      auto subResult = xllvm::SubACC2048AccFloatAIE2pIntrOp::create(
          rewriter, loc, v64f32Ty, lhsF32, rhsF32, confCst);

      // Step 5: Bitcast <64 x float> back to <32 x i64>
      auto resultI64 =
          LLVM::BitcastOp::create(rewriter, loc, v32i64Ty, subResult);

      // Step 6: Shuffle to extract first 8 elements <32 x i64> -> <8 x i64>
      SmallVector<int64_t> extractMask = {0, 1, 2, 3, 4, 5, 6, 7};
      auto resultExtracted = vector::ShuffleOp::create(rewriter, loc, resultI64,
                                                       resultI64, extractMask);

      // Step 7: Bitcast <8 x i64> back to <16 x float>
      auto v16f32Ty = VectorType::get({16}, rewriter.getF32Type());
      auto finalResult =
          LLVM::BitcastOp::create(rewriter, loc, v16f32Ty, resultExtracted);

      rewriter.replaceOp(op, finalResult);
      return success();
    }

    // Handle the FP32 sub_elem for AIE2p (32-lane)
    // Use ACC2048 intrinsic by padding to 64 lanes
    if (decodedSubElemOp.kind ==
        DecodedSubElemOp::Kind::FP32_FP32_FP32_32x1x1x1) {
      // Pad from <32 x float> to <64 x float> using shuffle
      SmallVector<int64_t> padMask;
      for (int i = 0; i < 32; ++i)
        padMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        padMask.push_back(-1); // poison/undef

      auto v64f32Ty = VectorType::get({64}, rewriter.getF32Type());
      auto lhsPadded = vector::ShuffleOp::create(
          rewriter, loc, adaptor.getLhs(), adaptor.getLhs(), padMask);
      auto rhsPadded = vector::ShuffleOp::create(
          rewriter, loc, adaptor.getRhs(), adaptor.getRhs(), padMask);

      // Call ACC2048 intrinsic
      auto confCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(60));
      auto subResult = xllvm::SubACC2048AccFloatAIE2pIntrOp::create(
          rewriter, loc, v64f32Ty, lhsPadded, rhsPadded, confCst);

      // Extract first 32 elements from 64-element result
      SmallVector<int64_t> extractMask;
      for (int i = 0; i < 32; ++i)
        extractMask.push_back(i);
      auto finalResult = vector::ShuffleOp::create(rewriter, loc, subResult,
                                                   subResult, extractMask);

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
      func = LLVM::LLVMFuncOp::create(
          rewriter, rewriter.getUnknownLoc(), intrinsicName,
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
    auto xstartVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), startType, rewriter.getI32IntegerAttr(x.start));
    auto ystartVal = LLVM::ConstantOp::create(rewriter, op->getLoc(), startType,
                                              rewriter.getI32IntegerAttr(0));
    auto zstartVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), startType, rewriter.getI32IntegerAttr(z.start));
    auto xoffsetsVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)x.offsets, (int32_t)x.offsets_hi}));
    auto zoffsetsVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)z.offsets, (int32_t)z.offsets_hi}));
    auto confVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), confType,
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
      func = LLVM::LLVMFuncOp::create(
          rewriter, rewriter.getUnknownLoc(), intrinsicName,
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
    auto xstartVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), startType, rewriter.getI32IntegerAttr(x.start));
    auto ystartVal = LLVM::ConstantOp::create(rewriter, op->getLoc(), startType,
                                              rewriter.getI32IntegerAttr(0));
    auto zstartVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), startType, rewriter.getI32IntegerAttr(z.start));
    auto xoffsetsVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)x.offsets, (int32_t)x.offsets_hi}));
    auto zoffsetsVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)z.offsets, (int32_t)z.offsets_hi}));
    auto confVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), confType,
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
    auto zeroCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto a0 = adaptor.getLhs();
    auto a1 = xllvm::VectorBroadcast32I512IntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI32Type()), zeroCst);
    auto b0 = adaptor.getRhs();
    auto b1 = xllvm::UndefV16I32IntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI32Type()));

    // 4* Shuffle
    auto a_lo = xllvm::VectorShuffleIntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI32Type()), a0, a1,
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(2)));
    auto a_hi = xllvm::VectorShuffleIntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI32Type()), a0, a1,
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(3)));
    auto b_lo = xllvm::VectorShuffleIntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI32Type()), b0, b1,
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(2)));
    auto b_hi = xllvm::VectorShuffleIntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI32Type()), b0, b1,
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(3)));
    // MUL + 3 * MAC
    auto mulConfCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
            /*sgn_x=*/1, /*sgn_y=*/1, /*amode=*/1, /*bmode=*/3,
            /*variant=*/2, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0, /*sub_mask=*/0)));
    auto mulConfOp = xllvm::MulConfAcc64IntrOp::create(
        rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
        forceCastOperandsToSignature(
            rewriter, loc,
            /*operands=*/{a_hi, b_hi, mulConfCst},
            /*signature=*/
            {VectorType::get({64}, rewriter.getI8Type()),
             VectorType::get({16}, rewriter.getI32Type()),
             rewriter.getI32Type()}));

    auto createMacConfOp = [&](SmallVector<Value> operands,
                               int macConf) -> Value {
      operands.push_back(
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(macConf)));
      return xllvm::MacConfAcc64IntrOp::create(
                 rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
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
    auto zeroCst =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getBF16Type(),
                                 rewriter.getZeroAttr(rewriter.getBF16Type()));
    auto aZeros = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto bZeros = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto cZeros = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto dZeros = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto eZeros = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto fZeros = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), zeroCst);
    auto oneCst =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getBF16Type(),
                                 rewriter.getOneAttr(rewriter.getBF16Type()));
    auto dummy0 = xllvm::VectorBroadcast16BF512IntrOp::create(
        rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), oneCst);
    auto zeroCstI32 = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto mscMacMulConfCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
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
      auto v1ToBF16 = xllvm::Vector16AccFloatToV16BF16AIE2IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
          inputBitCasted);
      auto a = xllvm::UpdBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), aZeros,
          v1ToBF16, zeroCstI32);

      // v16accfloat acc0 = msc_elem_16_2(a, dummy0, (v16accfloat)v1);
      auto acc0 = xllvm::MscConfBF16IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()), a, dummy0,
          inputBitCasted, mscMacMulConfCst);

      // b = insert(b,0,to_v16bfloat16(acc0));
      auto acc0ToBF16 = xllvm::Vector16AccFloatToV16BF16AIE2IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()), acc0);
      auto b = xllvm::UpdBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), bZeros,
          acc0ToBF16, zeroCstI32);

      // c = insert(c,0,to_v16bfloat16(msc_elem_16_2(b, dummy0, acc0)));
      auto acc0Mscb = xllvm::MscConfBF16IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()), b, dummy0,
          acc0, mscMacMulConfCst);
      auto acc0MscbToBF16 = xllvm::Vector16AccFloatToV16BF16AIE2IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
          acc0Mscb);
      auto c = xllvm::UpdBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), cZeros,
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
      return xllvm::MacConfBF16IntrOp::create(
                 rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
                 lhs, rhs, acc, mscMacMulConfCst)
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
      auto afMul = xllvm::MulConfBF16IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()), a, f,
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
      auto bdMul = xllvm::MulConfBF16IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()), b, d,
          mscMacMulConfCst);
      finalMacVal = createMacOps(a, d, createMacOps(a, e, bdMul));
    } else {
      // aie2Fp32EmulationOption == Aie2Fp32Emulation::AccuracySafe
      // Most accurate option since input fp32 number is split in to 3 bfloat16
      // numbers to extract all the bits of the mantissa. float a*b would
      // require 9 mac operations due to 3 bfloat16 splits each.
      auto cfMul = xllvm::MulConfBF16IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()), c, f,
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
    auto confCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(decodedMulElemOp.conf));
    Value mulElemOp = nullptr;
    SmallVector<Value> operands({adaptor.getLhs(), adaptor.getRhs(), confCst});

    // create xllvm intrinsic
    if (decodedMulElemOp.kind == DecodedMulElemOp::Kind::I16_I16_I32_32x1x1x1 ||
        decodedMulElemOp.kind == DecodedMulElemOp::Kind::I8_I8_I32_32x1x2x1) {
      mulElemOp = xllvm::MulConfAcc32IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({64}, rewriter.getI8Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (decodedMulElemOp.kind ==
               DecodedMulElemOp::Kind::BF16_BF16_FP32_16x1x2x1) {
      // Create zero vector using the exact pattern from working reference:
      // vbroadcast16.I512(0) -> bitcast to bf16 -> extract lower 256 bits
      auto zero32 = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      auto zeros_i16 = xllvm::VectorBroadcast16I512IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getI16Type()), zero32);
      auto zeros_bf16 = LLVM::BitcastOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
          zeros_i16);
      auto zeroVec = xllvm::ExtBF256BF512IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
          zeros_bf16, zero32);

      // Use set+upd pattern to match working reference
      auto idx1 = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(1));

      // Set lhs at lower 256 bits, then update upper 256 bits with zeros
      auto lhsSet = xllvm::VectorSetBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
          adaptor.getLhs(), zero32);
      auto lhsConcat = xllvm::UpdBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), lhsSet,
          zeroVec, idx1);

      // Set rhs at lower 256 bits, then update upper 256 bits with zeros
      auto rhsSet = xllvm::VectorSetBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
          adaptor.getRhs(), zero32);
      auto rhsConcat = xllvm::UpdBF512BF256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()), rhsSet,
          zeroVec, idx1);

      // Call bf.mul16.conf with padded vectors
      mulElemOp = xllvm::MulConfBF16IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI64Type()), lhsConcat,
          rhsConcat, confCst);
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
    auto confCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
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

      auto lhsPadded = vector::ShuffleOp::create(
          rewriter, loc, adaptor.getLhs(), adaptor.getLhs(), padMask);
      auto rhsPadded = vector::ShuffleOp::create(
          rewriter, loc, adaptor.getRhs(), adaptor.getRhs(), padMask);

      SmallVector<Value> operands({lhsPadded, rhsPadded, confCst});

      // Call I512.I512.ACC512 intrinsic
      mulElemOp = xllvm::MulConfBF16I512ACC512AIE2pIntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getF32Type()),
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
      mulElemOp = xllvm::MulConfBF16I512ACC1024AIE2pIntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getF32Type()),
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
      mulElemOp = xllvm::MulConfBF16I1024ACC2048AIE2pIntrOp::create(
          rewriter, loc, VectorType::get({64}, rewriter.getF32Type()),
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

// AIE2p version of FMAElemOp conversion. Uses native F32 accumulators
// and AIE2p-specific MAC intrinsics.
class FMAElemOpAIE2pConversion
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
    auto accTy = cast<VectorType>(acc.getType());
    auto flatLhsTy = getFlattenedVectorType(lhsTy);
    auto flatAccTy = getFlattenedVectorType(accTy);

    // Flatten operands, if needed
    if (lhsTy != flatLhsTy)
      lhs = vector::ShapeCastOp::create(rewriter, loc, flatLhsTy, lhs);
    if (cast<VectorType>(rhs.getType()) != flatLhsTy)
      rhs = vector::ShapeCastOp::create(rewriter, loc, flatLhsTy, rhs);
    if (accTy != flatAccTy)
      acc = vector::ShapeCastOp::create(rewriter, loc, flatAccTy, acc);

    if (!flatLhsTy.getElementType().isBF16()) {
      fmaOp.emitWarning()
          << "aievec.mac_elem AIE2p conversion only supports bf16 inputs.\n";
      return failure();
    }

    Type i32ty = rewriter.getI32Type();
    auto confCst = LLVM::ConstantOp::create(
        rewriter, loc, i32ty,
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
            /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
            /*sub_mask=*/0)));

    unsigned lhsLanes = flatLhsTy.getNumElements();
    Value macIntrOp = nullptr;

    if (lhsLanes == 16) {
      // 16-lane bf16: pad to v32bf16, use I512.I512.ACC512 intrinsic
      SmallVector<int64_t> padMask;
      for (int i = 0; i < 16; ++i)
        padMask.push_back(i);
      for (int i = 16; i < 32; ++i)
        padMask.push_back(-1); // poison

      auto lhsPadded =
          vector::ShuffleOp::create(rewriter, loc, lhs, lhs, padMask);
      auto rhsPadded =
          vector::ShuffleOp::create(rewriter, loc, rhs, rhs, padMask);

      auto v32bf16Ty = VectorType::get({32}, rewriter.getBF16Type());
      auto v16f32Ty = VectorType::get({16}, rewriter.getF32Type());
      macIntrOp = xllvm::MacConfBF16I512ACC512AIE2pIntrOp::create(
          rewriter, loc, v16f32Ty,
          forceCastOperandsToSignature(
              rewriter, loc, {lhsPadded, rhsPadded, acc, confCst},
              {v32bf16Ty, v32bf16Ty, v16f32Ty, i32ty}));
    } else if (lhsLanes == 32) {
      // 32-lane bf16: direct, use I512.I512.ACC1024 intrinsic
      auto v32bf16Ty = VectorType::get({32}, rewriter.getBF16Type());
      auto v32f32Ty = VectorType::get({32}, rewriter.getF32Type());
      macIntrOp = xllvm::MacConfBF16I512ACC1024AIE2pIntrOp::create(
          rewriter, loc, v32f32Ty,
          forceCastOperandsToSignature(
              rewriter, loc, {lhs, rhs, acc, confCst},
              {v32bf16Ty, v32bf16Ty, v32f32Ty, i32ty}));
    } else {
      fmaOp.emitWarning()
          << "aievec.mac_elem AIE2p conversion: unsupported lane count "
          << lhsLanes << ".\n";
      return failure();
    }

    // Recast/Reshape result
    auto resVal = forceCastValueToType(rewriter, loc, macIntrOp, flatAccTy);
    if (flatAccTy != accTy)
      resVal = vector::ShapeCastOp::create(rewriter, loc, accTy, resVal);

    rewriter.replaceOp(fmaOp, resVal);
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
      opSrcVal = vector::ShapeCastOp::create(rewriter, op.getLoc(), fltSrcVecTy,
                                             opSrcVal)
                     .getResult();

    // create xllvm intrinsic
    // Integer types
    Value upsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      auto shiftCst =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(op.getShift()));

      SmallVector<Value> operands({opSrcVal, shiftCst, signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 32) {
          // v16int16 -> v16acc32
          upsIntrOp = xllvm::Acc32V16I256UpsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64) {
          // v8int32 -> v8acc64
          upsIntrOp = xllvm::Acc64V8I256UpsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
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
          upsIntrOp = xllvm::Acc32V32I512UpsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 32) {
          // v16int32 -> v16acc64
          upsIntrOp = xllvm::Acc64V16I512UpsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16) {
          // v16int16 -> v16acc64
          upsIntrOp = xllvm::Acc64V16I256UpsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 8) {
          // v32int8 -> v32acc32
          upsIntrOp = xllvm::Acc32V32I256UpsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
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
        upsIntrOp = xllvm::Vector16BF16ToV16AccFloatAIE2IntrOp::create(
            rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
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
        auto indexZeroCst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(0));
        auto indexOneCst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(1));
        auto extractUps = [&](Value source, Value index) -> Value {
          auto extOp = xllvm::ExtI256I512IntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return xllvm::Vector16BF16ToV16AccFloatAIE2IntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({16}, rewriter.getBF16Type())}));
        };
        auto resLo = extractUps(opSrcVal, indexZeroCst);
        auto resHi = extractUps(opSrcVal, indexOneCst);
        // Concat the two 512-bit vector to a 1024-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        upsIntrOp = xllvm::ConcatI1024I512IntrOp::create(
            rewriter, loc, VectorType::get({32}, rewriter.getI32Type()),
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
      upsIntrOp = LLVM::BitcastOp::create(rewriter, loc, flatResTy, upsIntrOp);

    if (flatResTy != resultType)
      upsIntrOp =
          vector::ShapeCastOp::create(rewriter, loc, resultType, upsIntrOp);

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
      opSrcVal = vector::ShapeCastOp::create(rewriter, op.getLoc(), fltSrcVecTy,
                                             opSrcVal)
                     .getResult();

    // create xllvm intrinsic
    // Integer types
    Value upsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      auto shiftCst =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(op.getShift()));

      SmallVector<Value> operands({opSrcVal, shiftCst, signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 32) {
          // v16int16 -> v16acc32
          upsIntrOp = xllvm::Acc32V16I256UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64) {
          // v8int32 -> v8acc64
          upsIntrOp = xllvm::Acc64V8I256UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
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
          upsIntrOp = xllvm::Acc32V32I512UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 32 &&
                   srcVectorSize == 512) {
          // v16int32 -> v16acc64
          upsIntrOp = xllvm::Acc64V16I512UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16 &&
                   srcVectorSize == 256) {
          // v16int16 -> v16acc64
          upsIntrOp = xllvm::Acc64V16I256UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 8 &&
                   srcVectorSize == 256) {
          // v32int8 -> v32acc32
          upsIntrOp = xllvm::Acc32V32I256UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI32Type()),
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
          upsIntrOp = xllvm::Acc32V64I512UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({64}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({64}, rewriter.getI8Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16 &&
                   srcVectorSize == 512) {
          // v32int16 -> v32acc64
          upsIntrOp = xllvm::Acc64V32I512UpsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 16 &&
                   srcVectorSize == 1024) {
          // v64int16 -> v64acc32
          // Extract 2 chunks of v32int16 and convert each to v32acc32
          auto index0Cst =
              LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(0));
          auto index1Cst =
              LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(1));

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

            auto extOp = vector::ShuffleOp::create(rewriter, loc, v32i32Source,
                                                   v32i32Source, shuffleMask);

            return xllvm::Acc32V32I512UpsAIE2pIntrOp::create(
                rewriter, loc, VectorType::get({32}, rewriter.getI32Type()),
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
              vector::ShuffleOp::create(rewriter, loc, res0, res1, concatMask);
        }
      }
    } else {
      // Float types
      // AIE2p uses native F32 types, AIE2 uses packed I64 types
      if (resultVectorSize == 512) {
        // v16bfloat16 -> v16accfloat
        upsIntrOp = xllvm::Vector16BF16ToV16AccFloatAIE2pIntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getF32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {opSrcVal},
                {VectorType::get({16}, rewriter.getBF16Type())}));
      } else if (resultVectorSize == 1024) {
        // v32bfloat16 -> v32accfloat
        upsIntrOp = xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
            rewriter, loc, VectorType::get({32}, rewriter.getF32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {opSrcVal},
                {VectorType::get({32}, rewriter.getBF16Type())}));
      } else if (resultVectorSize == 2048) {
        // v64bfloat16 -> v64accfloat
        // Extract 2 chunks of v32bfloat16 and convert each to v32accfloat
        auto index0Cst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(0));
        auto index1Cst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(1));

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

          auto extOp = vector::ShuffleOp::create(rewriter, loc, v32i32Source,
                                                 v32i32Source, shuffleMask);

          return xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getF32Type()),
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
        upsIntrOp = vector::ShuffleOp::create(rewriter, loc, v32i32Res0,
                                              v32i32Res1, concatMask);
      }
    }

    if (!upsIntrOp) {
      op.emitWarning() << "aievec.ups is not supported.\n";
      return failure();
    }

    // create bitcast for result if needed
    if (flatResTy != upsIntrOp.getType())
      upsIntrOp = LLVM::BitcastOp::create(rewriter, loc, flatResTy, upsIntrOp);

    if (flatResTy != resultType)
      upsIntrOp =
          vector::ShapeCastOp::create(rewriter, loc, resultType, upsIntrOp);

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
      // create constant for sign from the op's sign attribute
      auto signCst =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(op.getSign()));

      // create xllvm intrinsic
      SmallVector<Value> operands(
          {adaptor.getSource(), adaptor.getShift(), signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 16) {
          srsIntrOp = xllvm::I512V32Acc32SrsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32) {
          srsIntrOp = xllvm::I512V16Acc64SrsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
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
          srsIntrOp = xllvm::I256V16Acc32SrsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          srsIntrOp = xllvm::I256V32Acc32SrsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          srsIntrOp = xllvm::I256V16Acc64SrsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          srsIntrOp = xllvm::I256V8Acc64SrsAIE2IntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      }
    } else {
      // Float types
      if (resultVectorSize == 256) {
        srsIntrOp = xllvm::Vector16AccFloatToV16BF16AIE2IntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
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
        auto indexZeroCst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(0));
        auto indexOneCst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(1));
        auto extractSrs = [&](Value source, Value index) -> Value {
          auto extOp = xllvm::ExtI512I1024IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return xllvm::Vector16AccFloatToV16BF16AIE2IntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({8}, rewriter.getI64Type())}));
        };
        auto resLo = extractSrs(adaptor.getSource(), indexZeroCst);
        auto resHi = extractSrs(adaptor.getSource(), indexOneCst);
        // Concat the two 256-bit vector to a 512-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        srsIntrOp = xllvm::ConcatI512I256IntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
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
      // create constant for sign from the op's sign attribute
      auto signCst =
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(op.getSign()));

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
          srsIntrOp = xllvm::I512V32Acc32SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          // v32acc64 -> v32int16
          srsIntrOp = xllvm::I512V32Acc64SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          // v16acc64 -> v16int32
          srsIntrOp = xllvm::I512V16Acc64SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          // v64acc32 -> v64int8
          srsIntrOp = xllvm::I512V64Acc32SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({64}, rewriter.getI8Type()),
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
          srsIntrOp = xllvm::I256V16Acc32SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          // v32acc32 -> v32int8
          srsIntrOp = xllvm::I256V32Acc32SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          // v16acc64 -> v16int16
          srsIntrOp = xllvm::I256V16Acc64SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          // v8acc64 -> v8int32
          srsIntrOp = xllvm::I256V8Acc64SrsAIE2pIntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI32Type()),
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
          auto index0Cst =
              LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(0));
          auto index1Cst =
              LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(1));

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

            auto extOp = vector::ShuffleOp::create(rewriter, loc, v64i32Source,
                                                   v64i32Source, shuffleMask);

            return xllvm::I512V32Acc32SrsAIE2pIntrOp::create(
                rewriter, loc, VectorType::get({32}, rewriter.getI16Type()),
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
          srsIntrOp = vector::ShuffleOp::create(rewriter, loc, v16i32Res0,
                                                v16i32Res1, concatMask);
        }
      }
    } else {
      // Float types
      // AIE2p uses native F32 types, AIE2 uses packed I64 types
      if (resultVectorSize == 256) {
        // v16accfloat -> v16bfloat16
        srsIntrOp = xllvm::Vector16AccFloatToV16BF16AIE2pIntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getSource()},
                {VectorType::get({16}, rewriter.getF32Type())}));
      } else if (resultVectorSize == 512) {
        // v32accfloat -> v32bfloat16
        srsIntrOp = xllvm::Vector32AccFloatToV32BF16AIE2pIntrOp::create(
            rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getSource()},
                {VectorType::get({32}, rewriter.getF32Type())}));
      } else if (resultVectorSize == 1024) {
        // v64accfloat -> v64bfloat16
        // Extract 2 chunks of v32accfloat and convert each to v32bfloat16
        auto index0Cst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(0));
        auto index1Cst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(1));

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

          auto extOp = vector::ShuffleOp::create(rewriter, loc, v64i32Source,
                                                 v64i32Source, shuffleMask);

          return xllvm::Vector32AccFloatToV32BF16AIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
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
        srsIntrOp = vector::ShuffleOp::create(rewriter, loc, v16i32Res0,
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
          LLVM::BitcastOp::create(rewriter, op->getLoc(), vectorPtrType, ptr);
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
          LLVM::BitcastOp::create(rewriter, op->getLoc(), vectorPtrType, ptr);
      auto loadValue =
          LLVM::LoadOp::create(rewriter, op->getLoc(), loadType, castedPtr, 1);

      // Get set up for the intrinsic
      std::string intrinsicName = getIntrinsicName(op, loadSize);

      // If the intrinsic declaration doesn't exist, create it
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
          StringAttr::get(context, intrinsicName));

      if (!func) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        func = LLVM::LLVMFuncOp::create(
            rewriter, rewriter.getUnknownLoc(), intrinsicName,
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
        // LLVM::UndefOp::create(rewriter, op->getLoc(), resultType);

        std::stringstream ss;
        ss << "llvm.aie." << getVectorTypeString(resultType) << ".undef";
        std::string intrinsicName = ss.str();

        auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
            StringAttr::get(rewriter.getContext(), intrinsicName));

        if (!func) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(module.getBody());
          func = LLVM::LLVMFuncOp::create(
              rewriter, rewriter.getUnknownLoc(), intrinsicName,
              LLVM::LLVMFunctionType::get(resultType, {}));
        }
        destValue =
            LLVM::CallOp::create(rewriter, op->getLoc(), func, ValueRange{})
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
      concatOp = xllvm::ConcatI512I256IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, adaptor.getSources(),
              {VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type())}));
    } else if (srcVectorSize == 256 && resultVectorSize == 1024) {
      concatOp = xllvm::ConcatI1024I256IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, adaptor.getSources(),
              {VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type()),
               VectorType::get({8}, rewriter.getI32Type())}));
    } else if (srcVectorSize == 512 && resultVectorSize == 1024) {
      concatOp = xllvm::ConcatI1024I512IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getI32Type()),
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
    auto indexCst =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(op.getIndex()));

    // create xllvm intrinsic
    SmallVector<Value> operands({adaptor.getSource(), indexCst});
    Value extOp = nullptr;
    // Integer types
    if (resultVectorSize == 256 && srcVectorSize == 512) {
      extOp = xllvm::ExtI256I512IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 512 && srcVectorSize == 1024) {
      extOp = xllvm::ExtI512I1024IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 256 && srcVectorSize == 1024) {
      extOp = xllvm::ExtI256I1024IntrOp::create(
          rewriter, loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 128 && srcVectorSize == 512) {
      auto shiftOp = adaptor.getSource();
      if (op.getIndex() > 0) {
        auto undefOp = xllvm::UndefV16I32IntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getI32Type()));
        auto stepCst =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(0));
        auto shiftCst = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(op.getIndex() * 16));
        SmallVector<Value> shiftOperands{adaptor.getSource(), undefOp, stepCst,
                                         shiftCst};
        // Right shift the source vector in index * 16 bytes (i.e. in index *
        // 128 bits). The integer index is expected to be 0 to 3.
        shiftOp = xllvm::VectorShiftI512I512IntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, shiftOperands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
      }
      // The underlying intrinsic takes a source vector and extract the lowest
      // 128-bit. i.e. it always extracts the input vector with index = 0.
      extOp = xllvm::ExtI128I512IntrOp::create(
          rewriter, loc, VectorType::get({4}, rewriter.getI32Type()),
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

// AIE2p version of ExtOp conversion using vector.shuffle
class ExtOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ExtOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value src = adaptor.getSource();
    VectorType srcType = cast<VectorType>(src.getType());
    VectorType resultType = cast<VectorType>(op.getResult().getType());

    int srcLanes = getVectorLaneSize(srcType);
    int resultLanes = getVectorLaneSize(resultType);

    // Verify this is extracting half the vector
    if (srcLanes != 2 * resultLanes) {
      op.emitWarning() << "aievec.ext with non-half extraction is not "
                          "supported for AIE2p.\n";
      return failure();
    }

    // Build shuffle mask based on index
    // index 0: extract lower half [0, 1, ..., resultLanes-1]
    // index 1: extract upper half [resultLanes, ..., srcLanes-1]
    SmallVector<int64_t> shuffleMask;
    int startIdx = op.getIndex() * resultLanes;
    for (int i = 0; i < resultLanes; ++i) {
      shuffleMask.push_back(startIdx + i);
    }

    // Use vector.shuffle to extract the half
    auto extracted =
        vector::ShuffleOp::create(rewriter, loc, src, src, shuffleMask);

    rewriter.replaceOp(op, extracted);
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
      func = LLVM::LLVMFuncOp::create(
          rewriter, rewriter.getUnknownLoc(), intrinsicName,
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
    auto selectVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), selectType, rewriter.getI32IntegerAttr(select));
    auto xstartVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), startType, rewriter.getI32IntegerAttr(x.start));
    auto ystartVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), startType, rewriter.getI32IntegerAttr(y.start));
    auto xoffsetsVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)x.offsets, (int32_t)x.offsets_hi}));
    auto yoffsetsVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), offsetsType,
        rewriter.getI32VectorAttr({(int32_t)y.offsets, (int32_t)y.offsets_hi}));
    auto confVal = LLVM::ConstantOp::create(
        rewriter, op->getLoc(), confType,
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
      func = LLVM::LLVMFuncOp::create(
          rewriter, rewriter.getUnknownLoc(), intrinsicName,
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
      auto cmpCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        maxOp = xllvm::VectorMaxLt8IntrOp::create(
            rewriter, loc,
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
        maxOp = xllvm::VectorMaxLt16IntrOp::create(
            rewriter, loc,
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
        maxOp = xllvm::VectorMaxLt32IntrOp::create(
            rewriter, loc,
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
        maxOp = xllvm::VectorMaxLtBf16IntrOp::create(
            rewriter, loc,
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
      auto cmpCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        minOp = xllvm::VectorMinGe8IntrOp::create(
            rewriter, loc,
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
        minOp = xllvm::VectorMinGe16IntrOp::create(
            rewriter, loc,
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
        minOp = xllvm::VectorMinGe32IntrOp::create(
            rewriter, loc,
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
        minOp = xllvm::VectorMinGeBf16IntrOp::create(
            rewriter, loc,
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
      auto cmpCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        maxOp = xllvm::VectorMaxLt8AIE2pIntrOp::create(
            rewriter, loc,
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
        maxOp = xllvm::VectorMaxLt16AIE2pIntrOp::create(
            rewriter, loc,
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
        maxOp = xllvm::VectorMaxLt32AIE2pIntrOp::create(
            rewriter, loc,
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
        maxOp = xllvm::VectorMaxLtBf16AIE2pIntrOp::create(
            rewriter, loc,
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
      auto cmpCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs(), cmpCst};
      if (resultBitWidth == 8) {
        minOp = xllvm::VectorMinGe8AIE2pIntrOp::create(
            rewriter, loc,
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
        minOp = xllvm::VectorMinGe16AIE2pIntrOp::create(
            rewriter, loc,
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
        minOp = xllvm::VectorMinGe32AIE2pIntrOp::create(
            rewriter, loc,
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
        minOp = xllvm::VectorMinGeBf16AIE2pIntrOp::create(
            rewriter, loc,
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
        src = LLVM::SExtOp::create(rewriter, loc, rewriter.getI32Type(), src);
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
        // Use llvm.aie2.vbroadcast32.I512 with bitcasts (float -> i32 -> float)
        // Following the pattern: %0 = bitcast float %b to i32
        //                        %1 = tail call <16 x i32>
        //                        @llvm.aie2.vbroadcast32.I512(i32 %0) %2 =
        //                        bitcast <16 x i32> %1 to <16 x float>
        auto srcAsI32 = bitcastValueToType(rewriter, loc, adaptor.getSource(),
                                           rewriter.getI32Type());
        auto broadcastI32 = xllvm::VectorBroadcast32I512IntrOp::create(
            rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
            srcAsI32);
        auto resultF32 =
            bitcastValueToType(rewriter, loc, broadcastI32,
                               VectorType::get({16}, rewriter.getF32Type()));
        rewriter.replaceOp(op, resultF32);
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
        src = LLVM::SExtOp::create(rewriter, loc, resultScaTy, src);
      } else if (srcBitWidth > resultBitWidth) {
        src = LLVM::TruncOp::create(rewriter, loc, resultScaTy, src);
      }
    }

    // Create poison vector of the result type
    auto poisonVec = LLVM::PoisonOp::create(rewriter, loc, resultType);

    // Insert scalar at position 0
    auto idx0 = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                         rewriter.getI64IntegerAttr(0));
    auto insertedVec = LLVM::InsertElementOp::create(rewriter, loc, resultType,
                                                     poisonVec, src, idx0);

    // Create shufflevector mask with all zeros (broadcast position 0 to all
    // lanes)
    SmallVector<int64_t> broadcastMask(resultLanes, 0);
    auto broadcastVec = vector::ShuffleOp::create(rewriter, loc, insertedVec,
                                                  insertedVec, broadcastMask);

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
    auto stepCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // create xllvm intrinsic
    Value shiftOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getLhs(), adaptor.getRhs(), stepCst, adaptor.getShift()});
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // Integer types
      shiftOp = xllvm::VectorShiftI512I512IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      // Float types
      shiftOp = xllvm::VectorShiftBF512BF512IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
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
    auto stepCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // create xllvm intrinsic
    Value shiftOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getLhs(), adaptor.getRhs(), stepCst, adaptor.getShift()});
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // Integer types - use AIE2p intrinsic
      shiftOp = xllvm::VectorShiftI512I512AIE2pIntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      // Float types - use AIE2p intrinsic
      shiftOp = xllvm::VectorShiftBF512BF512AIE2pIntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getBF16Type()),
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
    auto signCst = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    // create xllvm intrinsic
    Value extElemOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getSource(), adaptor.getIndex(), signCst});
    if (resultBitWidth == 8) {
      extElemOp = xllvm::VectorExtractElem8I512IntrOp::create(
          rewriter, loc, rewriter.getI32Type(),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({64}, rewriter.getI8Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else if (resultBitWidth == 16) {
      extElemOp = xllvm::VectorExtractElem16I512IntrOp::create(
          rewriter, loc, rewriter.getI32Type(),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI16Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else if (resultBitWidth == 32) {
      extElemOp = xllvm::VectorExtractElem32I512IntrOp::create(
          rewriter, loc, rewriter.getI32Type(),
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
        extElemOp = LLVM::TruncOp::create(rewriter, loc, rewriter.getI16Type(),
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
      lhs = vector::ShapeCastOp::create(rewriter, loc, flatLhsTy, lhs);
    if (rhsTy != flatRhsTy)
      rhs = vector::ShapeCastOp::create(rewriter, loc, flatRhsTy, rhs);
    if (accTy != flatAccTy)
      acc = vector::ShapeCastOp::create(rewriter, loc, flatAccTy, acc);

    // Build vmac configuration constant
    Type i32ty = rewriter.getI32Type();
    auto confCst = LLVM::ConstantOp::create(
        rewriter, loc, i32ty,
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
            /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
            /*sub_mask=*/0)));

    // Pad 16-lane bf16 operands to 32-lane using set+upd intrinsics.
    // forceCastOperandsToSignature only does bitwise reinterpretation, which
    // leaves garbage in the upper lanes. The MAC intrinsic requires properly
    // zero-padded v32bf16 inputs.
    auto v32bf16Ty = VectorType::get({32}, rewriter.getBF16Type());
    if (flatLhsTy.getElementType().isBF16() &&
        flatLhsTy.getNumElements() < 32) {
      auto zero32 = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                             rewriter.getI32IntegerAttr(0));
      auto zeros_i16 = xllvm::VectorBroadcast16I512IntrOp::create(
          rewriter, loc, VectorType::get({32}, rewriter.getI16Type()), zero32);
      auto zeros_bf16 =
          LLVM::BitcastOp::create(rewriter, loc, v32bf16Ty, zeros_i16);
      auto zeroVec = xllvm::ExtBF256BF512IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getBF16Type()),
          zeros_bf16, zero32);

      auto idx1 = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                           rewriter.getI32IntegerAttr(1));

      auto lhsSet = xllvm::VectorSetBF512BF256IntrOp::create(
          rewriter, loc, v32bf16Ty, lhs, zero32);
      lhs = xllvm::UpdBF512BF256IntrOp::create(rewriter, loc, v32bf16Ty, lhsSet,
                                               zeroVec, idx1);

      auto rhsSet = xllvm::VectorSetBF512BF256IntrOp::create(
          rewriter, loc, v32bf16Ty, rhs, zero32);
      rhs = xllvm::UpdBF512BF256IntrOp::create(rewriter, loc, v32bf16Ty, rhsSet,
                                               zeroVec, idx1);
    }

    // Insert vmac intrinsic
    auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
    auto macIntrOp = xllvm::MacConfBF16IntrOp::create(
        rewriter, loc, v8i64Ty,
        forceCastOperandsToSignature(rewriter, loc, {lhs, rhs, acc, confCst},
                                     {v32bf16Ty, v32bf16Ty, v8i64Ty, i32ty}));

    // Recast/Reshape result
    auto resVal =
        forceCastValueToType(rewriter, loc, macIntrOp.getResult(), flatAccTy);
    if (flatAccTy != accTy)
      resVal = vector::ShapeCastOp::create(rewriter, loc, accTy, resVal);

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
    decodedMatMulOp.lhs = vector::ShapeCastOp::create(
        rewriter, loc, lhsFlattenedVecTy, decodedMatMulOp.lhs);
    auto rhsFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.rhs.getType()));
    decodedMatMulOp.rhs = vector::ShapeCastOp::create(
        rewriter, loc, rhsFlattenedVecTy, decodedMatMulOp.rhs);
    auto accFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.acc.getType()));
    decodedMatMulOp.acc = vector::ShapeCastOp::create(
        rewriter, loc, accFlattenedVecTy, decodedMatMulOp.acc);

    Type i32ty = rewriter.getI32Type();
    auto confCst = LLVM::ConstantOp::create(
        rewriter, loc, i32ty, rewriter.getI32IntegerAttr(decodedMatMulOp.conf));
    SmallVector<Value> operands({decodedMatMulOp.lhs, decodedMatMulOp.rhs,
                                 decodedMatMulOp.acc, confCst});
    Value matMulResVal;
    if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::BF16)
      matMulResVal =
          xllvm::MacConfBF16IntrOp::create(
              rewriter, loc, VectorType::get({8}, rewriter.getI64Type()),
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
        matMulResVal = xllvm::MacConfAcc32IntrOp::create(
                           rewriter, loc, v16xi64ty,
                           forceCastOperandsToSignature(rewriter, loc, operands,
                                                        intrFuncSig))
                           .getResult();
      else
        matMulResVal = xllvm::MacConfAcc64IntrOp::create(
                           rewriter, loc, v16xi64ty,
                           forceCastOperandsToSignature(rewriter, loc, operands,
                                                        intrFuncSig))
                           .getResult();
    }

    auto castFromAcc =
        bitcastValueToType(rewriter, loc, matMulResVal, accFlattenedVecTy);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(),
                                                     castFromAcc);

    return success();
  }
};

// Helper function to transpose RHS in bf16 format and convert to accfloat
// Input: vXxbf16 (X must be 64), Output: v64accfloat
static Value transposeAndConvertRHS(OpBuilder &rewriter, Location loc,
                                    Type i32ty, Value rhs64bf16) {
  auto v32f32Ty = VectorType::get({32}, rewriter.getF32Type());

  // Transpose RHS 8x8 matrix in bf16 format (more efficient)
  // Cast v64bf16 to v32i32 for transpose operations
  auto rhs64i32 = forceCastValueToType(
      rewriter, loc, rhs64bf16, VectorType::get({32}, rewriter.getI32Type()));

  // Extract two <16 x i32> chunks
  SmallVector<int64_t> chunk0Mask, chunk1Mask;
  for (int i = 0; i < 16; ++i) {
    chunk0Mask.push_back(i);
    chunk1Mask.push_back(16 + i);
  }
  auto rhs16i32_0 =
      vector::ShuffleOp::create(rewriter, loc, rhs64i32, rhs64i32, chunk0Mask);
  auto rhs16i32_1 =
      vector::ShuffleOp::create(rewriter, loc, rhs64i32, rhs64i32, chunk1Mask);

  // Apply vshuffle with modes 52 and 53
  auto shuffleMode52 = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                                rewriter.getI32IntegerAttr(52));
  auto shuffleMode53 = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                                rewriter.getI32IntegerAttr(53));

  auto shuffled52 = xllvm::VectorShuffleAIE2pIntrOp::create(
      rewriter, loc, VectorType::get({16}, i32ty), rhs16i32_0, rhs16i32_1,
      shuffleMode52);
  auto shuffled53 = xllvm::VectorShuffleAIE2pIntrOp::create(
      rewriter, loc, VectorType::get({16}, i32ty), rhs16i32_0, rhs16i32_1,
      shuffleMode53);

  // Concatenate to get transposed v32i32
  SmallVector<int64_t> transposeConcatMask;
  for (int i = 0; i < 32; ++i)
    transposeConcatMask.push_back(i);
  auto rhsTransposedI32 = vector::ShuffleOp::create(
      rewriter, loc, shuffled52, shuffled53, transposeConcatMask);
  auto rhsTransposedBF16 =
      forceCastValueToType(rewriter, loc, rhsTransposedI32,
                           VectorType::get({64}, rewriter.getBF16Type()));

  // Convert transposed RHS v64bfloat16 to v64accfloat (in two v32 chunks)
  SmallVector<int64_t> firstHalfMask, secondHalfMask;
  for (int i = 0; i < 32; ++i) {
    firstHalfMask.push_back(i);
    secondHalfMask.push_back(32 + i);
  }

  auto rhsT32bf16_lo = vector::ShuffleOp::create(
      rewriter, loc, rhsTransposedBF16, rhsTransposedBF16, firstHalfMask);
  auto rhsT32bf16_hi = vector::ShuffleOp::create(
      rewriter, loc, rhsTransposedBF16, rhsTransposedBF16, secondHalfMask);

  auto rhsT32f32_lo = xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
      rewriter, loc, v32f32Ty, rhsT32bf16_lo);
  auto rhsT32f32_hi = xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
      rewriter, loc, v32f32Ty, rhsT32bf16_hi);

  // Concat to v64accfloat
  SmallVector<int64_t> concatMask;
  for (int i = 0; i < 64; ++i)
    concatMask.push_back(i);
  return vector::ShuffleOp::create(rewriter, loc, rhsT32f32_lo, rhsT32f32_hi,
                                   concatMask);
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

  auto lhsBFP = xllvm::Vector64AccFloatToV64BFP16EBS8AIE2pIntrOp::create(
      rewriter, loc, bfpStructTy, lhs64f32);
  auto rhsBFP = xllvm::Vector64AccFloatToV64BFP16EBS8AIE2pIntrOp::create(
      rewriter, loc, bfpStructTy, rhs64f32Transposed);

  // Extract mantissa and exponent
  auto lhsData = LLVM::ExtractValueOp::create(rewriter, loc, lhsBFP, 0);
  auto lhsExp = LLVM::ExtractValueOp::create(rewriter, loc, lhsBFP, 1);
  auto rhsData = LLVM::ExtractValueOp::create(rewriter, loc, rhsBFP, 0);
  auto rhsExp = LLVM::ExtractValueOp::create(rewriter, loc, rhsBFP, 1);

  // Perform BFP16 matmul
  return xllvm::MacConfBFP576ACC2048AIE2pIntrOp::create(
      rewriter, loc, v64i32Ty, lhsData, lhsExp, rhsData, rhsExp, acc64i32,
      confCst);
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
      vector::ShuffleOp::create(rewriter, loc, lhs64bf16, lhs64bf16, lowerMask);
  auto xh =
      vector::ShuffleOp::create(rewriter, loc, lhs64bf16, lhs64bf16, upperMask);

  // Cast to v16xi32 for shuffle intrinsic
  auto xlI32 =
      forceCastValueToType(rewriter, loc, xl, VectorType::get({16}, i32ty));
  auto xhI32 =
      forceCastValueToType(rewriter, loc, xh, VectorType::get({16}, i32ty));

  // Shuffle with T16_8x8_lo (mode 52) and T16_8x8_hi (mode 53)
  auto shuffleModeLo = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                                rewriter.getI32IntegerAttr(52));
  auto xa = xllvm::VectorShuffleAIE2pIntrOp::create(
      rewriter, loc, VectorType::get({16}, i32ty), xlI32, xhI32, shuffleModeLo);

  auto shuffleModeHi = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                                rewriter.getI32IntegerAttr(53));
  auto xb = xllvm::VectorShuffleAIE2pIntrOp::create(
      rewriter, loc, VectorType::get({16}, i32ty), xlI32, xhI32, shuffleModeHi);

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
        vector::ShuffleOp::create(rewriter, loc, src, src, extractMask);

    // Apply T16_4x8 shuffle pattern (mode 29)
    auto broadI32 = forceCastValueToType(rewriter, loc, broadcasted,
                                         VectorType::get({16}, i32ty));
    auto shuffleMode4x8 = LLVM::ConstantOp::create(
        rewriter, loc, i32ty, rewriter.getI32IntegerAttr(29));
    auto shuffled = xllvm::VectorShuffleAIE2pIntrOp::create(
        rewriter, loc, VectorType::get({16}, i32ty), broadI32, broadI32,
        shuffleMode4x8);

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
    return vector::ShuffleOp::create(rewriter, loc, src, src, mask);
  };

  // Prepare 8 column vectors from RHS
  SmallVector<Value> colVectors;
  for (int i = 0; i < 8; ++i)
    colVectors.push_back(extractBroadcast4(rhs32bf16, i));

  // Perform 8 MAC operations with conf=60 (no zero_acc)
  auto conf60 = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                         rewriter.getI32IntegerAttr(60));

  Value acc = acc32f32;
  for (int i = 0; i < 8; ++i) {
    acc = xllvm::MacConfBF16I512ACC1024AIE2pIntrOp::create(
        rewriter, loc, v32f32Ty, rowVectors[i], colVectors[i], acc, conf60);
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
        return {DecodedMatMulOp::Kind::I8_8x8x8_I512_ACC2048, lhs, rhs, acc,
                776};
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
    decodedMatMulOp.lhs = vector::ShapeCastOp::create(
        rewriter, loc, lhsFlattenedVecTy, decodedMatMulOp.lhs);
    auto rhsFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.rhs.getType()));
    decodedMatMulOp.rhs = vector::ShapeCastOp::create(
        rewriter, loc, rhsFlattenedVecTy, decodedMatMulOp.rhs);
    auto accFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.acc.getType()));
    decodedMatMulOp.acc = vector::ShapeCastOp::create(
        rewriter, loc, accFlattenedVecTy, decodedMatMulOp.acc);
    Type i32ty = rewriter.getI32Type();
    auto confCst = LLVM::ConstantOp::create(
        rewriter, loc, i32ty, rewriter.getI32IntegerAttr(decodedMatMulOp.conf));

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
          xllvm::MacConfI512ACC2048AIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI64Type()),
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
      auto lhsPadded = vector::ShuffleOp::create(
          rewriter, loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, lhsPadMask);

      // Pad RHS from <16 x i16> to <64 x i16> using shuffle
      SmallVector<int64_t> rhsPadMask;
      for (int i = 0; i < 16; ++i)
        rhsPadMask.push_back(i);
      for (int i = 16; i < 64; ++i)
        rhsPadMask.push_back(-1); // undef/poison
      auto rhsPadded = vector::ShuffleOp::create(
          rewriter, loc, decodedMatMulOp.rhs, decodedMatMulOp.rhs, rhsPadMask);

      // Update operands with padded vectors
      SmallVector<Value> paddedOperands(
          {lhsPadded, rhsPadded, decodedMatMulOp.acc, confCst});

      // Bitcast LHS <64 x i16> -> <32 x i32>
      // Keep RHS as <64 x i16>
      // Bitcast ACC <64 x i32> -> <32 x i64>
      matMulResVal =
          xllvm::MacConfI1024ACC2048AIE2pIntrOp::create(
              rewriter, loc, VectorType::get({32}, rewriter.getI64Type()),
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

      auto lhs32bf16_lo =
          vector::ShuffleOp::create(rewriter, loc, decodedMatMulOp.lhs,
                                    decodedMatMulOp.lhs, firstHalfMask);
      auto lhs32bf16_hi =
          vector::ShuffleOp::create(rewriter, loc, decodedMatMulOp.lhs,
                                    decodedMatMulOp.lhs, secondHalfMask);

      auto lhs32f32_lo = xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
          rewriter, loc, v32f32Ty, lhs32bf16_lo);
      auto lhs32f32_hi = xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
          rewriter, loc, v32f32Ty, lhs32bf16_hi);

      // Concat to v64accfloat
      SmallVector<int64_t> concatMask;
      for (int i = 0; i < 64; ++i)
        concatMask.push_back(i);
      auto lhs64f32 = vector::ShuffleOp::create(rewriter, loc, lhs32f32_lo,
                                                lhs32f32_hi, concatMask);

      // Step 2: Transpose RHS and convert to accfloat using shared helper
      auto rhsTransposed =
          transposeAndConvertRHS(rewriter, loc, i32ty, decodedMatMulOp.rhs);

      // Step 4: Use shared BFP16 8×8 matmul helper
      auto conf780 = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                              rewriter.getI32IntegerAttr(780));

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
      auto lhs32f32 = xllvm::Vector32BF16ToV32AccFloatAIE2pIntrOp::create(
          rewriter, loc, v32f32Ty, decodedMatMulOp.lhs);

      // Pad v32accfloat to v64accfloat using shuffle
      SmallVector<int64_t> lhsPadMask;
      for (int i = 0; i < 32; ++i)
        lhsPadMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        lhsPadMask.push_back(-1); // poison
      auto lhs64f32 = vector::ShuffleOp::create(rewriter, loc, lhs32f32,
                                                lhs32f32, lhsPadMask);

      // Step 2: Transpose RHS and convert to accfloat using shared helper
      auto rhsTransposed =
          transposeAndConvertRHS(rewriter, loc, i32ty, decodedMatMulOp.rhs);

      // Step 4: Pad ACC from 32 to 64 i32
      SmallVector<int64_t> accPadMask;
      for (int i = 0; i < 32; ++i)
        accPadMask.push_back(i);
      for (int i = 32; i < 64; ++i)
        accPadMask.push_back(-1); // poison
      auto acc64i32 = vector::ShuffleOp::create(
          rewriter, loc,
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
      matMulResVal = vector::ShuffleOp::create(rewriter, loc, result64i32,
                                               result64i32, extractMask);
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
      auto lhs64bf16 =
          vector::ShuffleOp::create(rewriter, loc, decodedMatMulOp.lhs,
                                    decodedMatMulOp.lhs, lhsReplicateMask);

      // Step 2: Transpose LHS as 8×8 matrix
      SmallVector<int64_t> transposeMask;
      for (int c = 0; c < 8; ++c) {
        for (int r = 0; r < 8; ++r) {
          transposeMask.push_back(r * 8 + c);
        }
      }
      auto lhs64bf16Transposed = vector::ShuffleOp::create(
          rewriter, loc, lhs64bf16, lhs64bf16, transposeMask);

      // Step 3: Replicate RHS from 8 to 64 elements (replicate 8 times)
      SmallVector<int64_t> rhsReplicateMask;
      for (int rep = 0; rep < 8; ++rep) {
        for (int i = 0; i < 8; ++i)
          rhsReplicateMask.push_back(i);
      }
      auto rhs64bf16 =
          vector::ShuffleOp::create(rewriter, loc, decodedMatMulOp.rhs,
                                    decodedMatMulOp.rhs, rhsReplicateMask);

      // Step 4: Use mac_elem_64_conf (which is
      // MacConfBF16I512ACC2048AIE2pIntrOp)
      matMulResVal = xllvm::MacConfBF16I512ACC2048AIE2pIntrOp::create(
          rewriter, loc, v64f32Ty, lhs64bf16Transposed, rhs64bf16,
          decodedMatMulOp.acc, confCst);
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
      auto lhsPadded = vector::ShuffleOp::create(
          rewriter, loc, decodedMatMulOp.lhs, decodedMatMulOp.lhs, lhsPadMask);

      // Pad ACC from 16 to 32 float using shuffle
      SmallVector<int64_t> accPadMask;
      for (int i = 0; i < 16; ++i)
        accPadMask.push_back(i);
      for (int i = 16; i < 32; ++i)
        accPadMask.push_back(-1); // poison/undef
      auto accPadded = vector::ShuffleOp::create(
          rewriter, loc, decodedMatMulOp.acc, decodedMatMulOp.acc, accPadMask);

      // Call the shared 8×8×4 helper with padded inputs
      Value acc32 = perform8x8x4MatMul(rewriter, loc, i32ty, lhsPadded,
                                       decodedMatMulOp.rhs, accPadded);

      // Extract first 16 elements from 32-element result
      SmallVector<int64_t> extractMask;
      for (int i = 0; i < 16; ++i)
        extractMask.push_back(i);
      matMulResVal =
          vector::ShuffleOp::create(rewriter, loc, acc32, acc32, extractMask);
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

  // Helper to check if a value is a constant zero
  static bool isConstantZero(Value val) {
    DenseElementsAttr denseAttr;

    // Check for both arith.constant and llvm.mlir.constant
    if (auto arithConstOp = val.getDefiningOp<arith::ConstantOp>()) {
      denseAttr = dyn_cast<DenseElementsAttr>(arithConstOp.getValue());
    } else if (auto llvmConstOp = val.getDefiningOp<LLVM::ConstantOp>()) {
      denseAttr = dyn_cast<DenseElementsAttr>(llvmConstOp.getValue());
    }

    if (!denseAttr || !denseAttr.isSplat())
      return false;

    auto splatAttr = denseAttr.getSplatValue<Attribute>();
    if (auto floatAttr = dyn_cast<FloatAttr>(splatAttr))
      return floatAttr.getValue().isZero();
    if (auto intAttr = dyn_cast<IntegerAttr>(splatAttr))
      return intAttr.getValue().isZero();

    return false;
  }

  LogicalResult
  matchAndRewrite(aievec::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special handling for isResAcc=true with zero constant source
    // The backend cannot handle zeroinitializer for accumulator types,
    // so we must use the vbroadcast.zero.acc1024 intrinsic instead
    if (!castOp.getIsResAcc() || !isConstantZero(adaptor.getSource())) {
      // Default behavior: fold the cast
      rewriter.replaceOp(castOp, adaptor.getSource());
      return success();
    }

    Location loc = castOp.getLoc();
    auto srcVecType = cast<VectorType>(castOp.getSource().getType());
    Type srcElemType = srcVecType.getElementType();
    int lanes = getVectorLaneSize(srcVecType);

    // For f32 vectors (accfloat), use vbroadcast.zero.acc1024
    if (srcElemType.isF32() && lanes == 16) {
      // Call vbroadcast.zero.acc1024 to get vector<16xi64>
      auto zeroAcc1024 = xllvm::VectorBroadcastZeroAcc1024IntrOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getI64Type()));

      // Extract lower 8 elements to get vector<8xi64> (512-bit accumulator)
      SmallVector<int64_t> extractMask = {0, 1, 2, 3, 4, 5, 6, 7};
      auto zeroAcc512 = vector::ShuffleOp::create(rewriter, loc, zeroAcc1024,
                                                  zeroAcc1024, extractMask);

      // Bitcast back to vector<16xf32> to match the cast result type
      auto result = LLVM::BitcastOp::create(
          rewriter, loc, VectorType::get({16}, rewriter.getF32Type()),
          zeroAcc512);

      rewriter.replaceOp(castOp, result);
      return success();
    }

    // Fallback: fold the cast (should not reach here for supported cases)
    rewriter.replaceOp(castOp, adaptor.getSource());
    return success();
  }
};

// AIE2p version of FoldAIECastOps
class FoldAIECastOpsAIE2p
    : public mlir::ConvertOpToLLVMPattern<aievec::CastOp> {
  using ConvertOpToLLVMPattern<aievec::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Fold the cast.
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
      rhs = xllvm::UndefV16I32IntrOp::create(rewriter, loc, v16xi32ty);

    auto modeAttrVal =
        LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                 static_cast<int32_t>(shuffleOp.getMode()))
            .getResult();
    auto vShuffleVal = xllvm::VectorShuffleIntrOp::create(
                           rewriter, loc, v16xi32ty,
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

// Convert aievec.inv to xllvm.intr.aie2p.inv intrinsic for AIE2P
// Scalar f32: direct conversion to xllvm.intr.aie2p.inv
// Vector f32: unroll into scalar xllvm.intr.aie2p.inv operations
class InvOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::InvOp> {
public:
  using ConvertOpToLLVMPattern<aievec::InvOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::InvOp invOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = invOp.getLoc();
    auto operandType = adaptor.getSource().getType();

    // Handle scalar f32 inverse
    if (operandType.isF32()) {
      auto invResult = xllvm::InvAIE2pIntrOp::create(
          rewriter, loc, rewriter.getF32Type(), adaptor.getSource());
      rewriter.replaceOp(invOp, invResult);
      return success();
    }

    // Handle vector<N x f32> inverse
    auto vecType = dyn_cast<VectorType>(operandType);
    if (!vecType || !vecType.getElementType().isF32())
      return failure();

    // Unroll vector inverse into scalar operations
    int numElements = getVectorLaneSize(vecType);
    Value result = LLVM::PoisonOp::create(rewriter, loc, vecType);

    for (int i = 0; i < numElements; ++i) {
      // Extract element i
      auto indexCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i));
      auto extractedElem = LLVM::ExtractElementOp::create(
          rewriter, loc, adaptor.getSource(), indexCst);

      // Call xllvm.intr.aie2p.inv on the scalar
      auto invResult = xllvm::InvAIE2pIntrOp::create(
          rewriter, loc, rewriter.getF32Type(), extractedElem);

      // Insert result back into vector
      result = LLVM::InsertElementOp::create(rewriter, loc, vecType, result,
                                             invResult, indexCst);
    }

    rewriter.replaceOp(invOp, result);
    return success();
  }
};

// Convert aievec.exp to xllvm.exp2 intrinsic for AIE2P
// Uses the identity: exp(x) = exp2(x * log2(e))
// Supports both lane-16 and lane-32 bf16 vectors
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

    // Support v16bfloat16 and v32bfloat16
    if ((laneSize != 16 && laneSize != 32) || !srcElemType.isBF16())
      return expOp.emitWarning()
             << "aievec.exp conversion only supports v16bfloat16 and "
                "v32bfloat16.\n";

    // Step 1: Create bf16 constant for log2(e) ≈ 1.442695
    auto log2eBF16Const = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getBF16Type(),
        rewriter.getFloatAttr(rewriter.getBF16Type(), 1.442695));

    // Broadcast log2(e) to match input lane size
    SmallVector<int64_t> broadcastMask;
    for (unsigned i = 0; i < laneSize; ++i)
      broadcastMask.push_back(0);

    auto v1bf16 = LLVM::UndefOp::create(
        rewriter, loc, VectorType::get({1}, rewriter.getBF16Type()));
    auto v1bf16Inserted = LLVM::InsertElementOp::create(
        rewriter, loc, v1bf16, log2eBF16Const,
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), 0));

    auto log2eVec = vector::ShuffleOp::create(rewriter, loc, v1bf16Inserted,
                                              v1bf16Inserted, broadcastMask);

    // Step 2: Multiply input by log2(e) in bf16 domain using MulElemOp
    // For lane-16: uses I512.I512.ACC512
    // For lane-32: uses I512.I512.ACC1024
    auto resultF32Ty =
        VectorType::get({(int64_t)laneSize}, rewriter.getF32Type());
    auto mulResult = aievec::MulElemOp::create(rewriter, loc, resultF32Ty,
                                               adaptor.getSource(), log2eVec);

    // Step 3: Call exp2 intrinsic based on lane size
    Value exp2Result;
    auto v16bf16Ty = VectorType::get({16}, rewriter.getBF16Type());

    if (laneSize == 16) {
      // Lane-16: Single exp2 call
      // exp2 takes v16float and returns v16bfloat16
      exp2Result =
          xllvm::Exp2AIE2pIntrOp::create(rewriter, loc, v16bf16Ty, mulResult);
    } else {
      // Lane-32: Split-and-recombine pattern
      // Split v32float into two v16float halves
      SmallVector<int64_t> lowerMask, upperMask;
      for (int i = 0; i < 16; ++i) {
        lowerMask.push_back(i);      // indices 0-15
        upperMask.push_back(16 + i); // indices 16-31
      }

      auto lowerHalf = vector::ShuffleOp::create(rewriter, loc, mulResult,
                                                 mulResult, lowerMask);
      auto upperHalf = vector::ShuffleOp::create(rewriter, loc, mulResult,
                                                 mulResult, upperMask);

      // Call exp2 on each half separately
      auto exp2Lower =
          xllvm::Exp2AIE2pIntrOp::create(rewriter, loc, v16bf16Ty, lowerHalf);
      auto exp2Upper =
          xllvm::Exp2AIE2pIntrOp::create(rewriter, loc, v16bf16Ty, upperHalf);

      // Recombine the two v16bfloat16 results into v32bfloat16
      SmallVector<int64_t> combineMask;
      for (int i = 0; i < 32; ++i)
        combineMask.push_back(i);

      exp2Result = vector::ShuffleOp::create(rewriter, loc, exp2Lower,
                                             exp2Upper, combineMask);
    }

    rewriter.replaceOp(expOp, exp2Result);

    return success();
  }
};

// Convert math.rsqrt (scalar f32 or vector f32) to xllvm.intr.aie2p.invsqrt
// Scalar f32: direct conversion to xllvm.intr.aie2p.invsqrt
// Vector f32: unroll into scalar xllvm.intr.aie2p.invsqrt operations
class RsqrtOpAIE2pConversion
    : public mlir::ConvertOpToLLVMPattern<math::RsqrtOp> {
public:
  using ConvertOpToLLVMPattern<math::RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = rsqrtOp.getLoc();
    auto operandType = adaptor.getOperand().getType();

    // Handle scalar f32 rsqrt
    if (operandType.isF32()) {
      auto rsqrtResult = xllvm::InvsqrtAIE2pIntrOp::create(
          rewriter, loc, rewriter.getF32Type(), adaptor.getOperand());
      rewriter.replaceOp(rsqrtOp, rsqrtResult);
      return success();
    }

    // Handle vector<N x f32> rsqrt
    auto vecType = dyn_cast<VectorType>(operandType);
    if (!vecType || !vecType.getElementType().isF32())
      return failure();

    // Unroll vector rsqrt into scalar operations
    int numElements = getVectorLaneSize(vecType);
    Value result = LLVM::PoisonOp::create(rewriter, loc, vecType);

    for (int i = 0; i < numElements; ++i) {
      // Extract element i
      auto indexCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i));
      auto extractedElem = LLVM::ExtractElementOp::create(
          rewriter, loc, adaptor.getOperand(), indexCst);

      // Call xllvm.intr.aie2p.invsqrt on the scalar
      auto rsqrtResult = xllvm::InvsqrtAIE2pIntrOp::create(
          rewriter, loc, rewriter.getF32Type(), extractedElem);

      // Insert result back into vector
      result = LLVM::InsertElementOp::create(rewriter, loc, vecType, result,
                                             rsqrtResult, indexCst);
    }

    rewriter.replaceOp(rsqrtOp, result);
    return success();
  }
};

// Convert arith.divf for vector<N x f32> to unrolled scalar divisions
// Uses a noinline helper function call as a barrier to prevent LLVM
// re-vectorization. Scalar f32 divisions are handled by downstream passes.
class FdivOpConversion : public mlir::ConvertOpToLLVMPattern<arith::DivFOp> {
public:
  using ConvertOpToLLVMPattern<arith::DivFOp>::ConvertOpToLLVMPattern;

  FdivOpConversion(const LLVMTypeConverter &typeConverter, StringRef device)
      : ConvertOpToLLVMPattern(typeConverter), deviceName(device.str()) {}

  std::string deviceName;

  LogicalResult
  matchAndRewrite(arith::DivFOp divOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = divOp.getLoc();
    auto lhsType = adaptor.getLhs().getType();

    // Only handle vector<N x f32> fdiv
    // Scalar f32 fdiv is handled by downstream passes
    auto vecType = dyn_cast<VectorType>(lhsType);
    if (!vecType || !vecType.getElementType().isF32())
      return failure();

    auto rhsType = adaptor.getRhs().getType();
    auto rhsVecType = dyn_cast<VectorType>(rhsType);
    if (!rhsVecType || rhsVecType != vecType)
      return failure();

    // Get or create the noinline scalar fdiv helper function using utility
    auto module = divOp->getParentOfType<ModuleOp>();
    auto f32Ty = rewriter.getF32Type();

    auto helperFunc = getOrCreateScalarHelperFunc(
        module, rewriter, "fdiv", deviceName,
        /*argTypes=*/{f32Ty, f32Ty},
        /*resultType=*/f32Ty,
        /*bodyBuilder=*/[](OpBuilder &builder, Location loc, ValueRange args) {
          auto divResult =
              arith::DivFOp::create(builder, loc, args[0], args[1]);
          LLVM::ReturnOp::create(builder, loc, ValueRange{divResult});
        });

    // Unroll vector fdiv into scalar helper function calls
    int numElements = getVectorLaneSize(vecType);
    Value result = LLVM::PoisonOp::create(rewriter, loc, vecType);

    for (int i = 0; i < numElements; ++i) {
      // Extract element i from both lhs and rhs
      auto indexCst = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i));
      auto lhsElem = LLVM::ExtractElementOp::create(rewriter, loc,
                                                    adaptor.getLhs(), indexCst);
      auto rhsElem = LLVM::ExtractElementOp::create(rewriter, loc,
                                                    adaptor.getRhs(), indexCst);

      // Call noinline helper function (acts as barrier to prevent
      // re-vectorization)
      auto divResult = LLVM::CallOp::create(rewriter, loc, helperFunc,
                                            ValueRange{lhsElem, rhsElem})
                           ->getResult(0);

      // Insert result back into vector
      result = LLVM::InsertElementOp::create(rewriter, loc, vecType, result,
                                             divResult, indexCst);
    }

    rewriter.replaceOp(divOp, result);
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
               SelectOpConversion,
               PackOpConversion,
               UnpackOpConversion,
               BroadcastOpConversion,
               BroadcastScalarOpConversion,
               MatMulOpConversion,
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
  patterns.add<ExtOpConversion>(converter);
  patterns.add<ExtractElemOpConversion>(converter);
  patterns.add<ConcatOpConversion>(converter);
  patterns.add<FMAElemOpConversion>(converter);
  patterns.add<FoldAIECastOps>(converter);
  patterns.add<FdivOpConversion>(converter, "aie2");
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
    Value extracted = LLVM::ExtractElementOp::create(
        rewriter, loc, adaptor.getSource(), adaptor.getIndex());

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

      result = vector::ShuffleOp::create(rewriter, loc, sources[0], sources[1],
                                         mask);
    } else if (sources.size() == 4) {
      // Concatenate four vectors: first concat pairs, then concat results
      SmallVector<int64_t> pairMask;
      for (int64_t i = 0; i < srcLanes * 2; ++i)
        pairMask.push_back(i);

      auto pair0 = vector::ShuffleOp::create(rewriter, loc, sources[0],
                                             sources[1], pairMask);
      auto pair1 = vector::ShuffleOp::create(rewriter, loc, sources[2],
                                             sources[3], pairMask);

      SmallVector<int64_t> finalMask;
      for (int64_t i = 0; i < srcLanes * 4; ++i)
        finalMask.push_back(i);

      result =
          vector::ShuffleOp::create(rewriter, loc, pair0, pair1, finalMask);
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
  patterns.add<FMAElemOpAIE2pConversion>(converter);
  patterns.add<UPSOpAIE2pConversion, SRSOpAIE2pConversion>(converter);
  patterns.add<MatMulOpAIE2pConversion>(converter);
  patterns.add<ShiftOpAIE2pConversion>(converter);
  patterns.add<MaxOpAIE2pConversion, MinOpAIE2pConversion>(converter);
  patterns.add<ExtOpAIE2pConversion>(converter);
  patterns.add<ExtractElemOpAIE2pConversion>(converter);
  patterns.add<ConcatOpAIE2pConversion>(converter);
  patterns.add<ExpOpAIE2pConversion>(converter);
  patterns.add<InvOpAIE2pConversion>(converter);
  patterns.add<BroadcastScalarOpAIE2pConversion>(converter);
  patterns.add<RsqrtOpAIE2pConversion>(converter);
  patterns.add<FdivOpConversion>(converter, "aie2p");
  patterns.add<FoldAIECastOpsAIE2p>(converter);
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

// Configure legalization rules shared by AIE2 and AIE2p
static void configureAIEVecToLLVMLegalizations(LLVMConversionTarget &target) {
  // Vector f32 divf is illegal (needs unrolling to scalar divf)
  // Scalar f32 divf is legal (handled by downstream passes)
  target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp divOp) {
    auto resultType = divOp.getType();
    if (auto vecType = dyn_cast<VectorType>(resultType)) {
      // Vector f32 divf is illegal and needs conversion
      return !vecType.getElementType().isF32();
    }
    // Scalar divf is legal
    return true;
  });
}

struct ConvertAIEVecToLLVMPass
    : ConvertAIEVecToLLVMBase<ConvertAIEVecToLLVMPass> {
  ConvertAIEVecToLLVMPass() = default;
  ConvertAIEVecToLLVMPass(const xilinx::ConvertAIEVecToLLVMOptions &options) {
    aieTarget = options.aieTarget;
    aie2Fp32Emulation = options.aie2Fp32Emulation;
  }

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

    // Configure legalizations for AIE2/AIE2p
    configureAIEVecToLLVMLegalizations(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass() {
  return std::make_unique<ConvertAIEVecToLLVMPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass(
    const xilinx::ConvertAIEVecToLLVMOptions &options) {
  return std::make_unique<ConvertAIEVecToLLVMPass>(options);
}

} // namespace xilinx::aievec
