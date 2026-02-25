//===-VectorToAIEVecConversions.cpp - Vector to AIEVec convs. ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains conversions from the Vector dialect into the AIEVec
// dialect. Conversions assume that the Vector dialect has been rectricted
// to ops that can be translated to a sequence of valid AIEVec ops.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"

#include "aie/Dialect/AIEVec/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include <bitset>
#include <optional>
#include <tuple>

#define DEBUG_TYPE "lower-vector-to-aievec"

using namespace llvm;
using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static bool isNarrowingOp(Operation *op) {
  if (isa<arith::TruncFOp>(op) || isa<arith::TruncIOp>(op))
    return true;

  if (auto srsOp = dyn_cast<aievec::SRSOp>(op)) {
    auto *srsOpSrcOp = srsOp.getSource().getDefiningOp();
    if (isa<aievec::UPSOp>(srsOpSrcOp) || isa<aievec::CastOp>(srsOpSrcOp))
      return true;
  }
  return false;
}

// Check if a TruncIOp is part of a shrsi+[clamp]+trunc chain that can be
// lowered to a compound SRS pattern. Only returns true for chains that the
// ShiftClampTruncToSRSPattern rewrite will actually match:
// - shrsi → trunci (no clamp)
// - shrsi → maxsi → minsi → trunci (full clamp pair)
// - shrsi → minsi → maxsi → trunci (reversed clamp pair)
// Does NOT return true for a single min or max without a matching pair,
// to avoid marking trunci illegal when no rewrite can handle it.
static bool isSRSCompoundCandidate(arith::TruncIOp trunciOp) {
  Value source = trunciOp.getIn();

  // Case 1: direct shrsi → trunci
  if (source.getDefiningOp<arith::ShRSIOp>())
    return true;

  // Case 2: minsi(maxsi(shrsi(...))) → trunci
  if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
    if (auto maxsiOp = minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>()) {
      if (maxsiOp.getLhs().getDefiningOp<arith::ShRSIOp>())
        return true;
    }
  }

  // Case 3: maxsi(minsi(shrsi(...))) → trunci (reversed order)
  if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
    if (auto minsiOp = maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>()) {
      if (minsiOp.getLhs().getDefiningOp<arith::ShRSIOp>())
        return true;
    }
  }

  return false;
}

// Check if a ShRSIOp's result feeds into a trunci (possibly through
// clamp ops), meaning the compound pattern will consume it.
static bool shrsiUsedByCompoundSRS(arith::ShRSIOp rsOp) {
  for (Operation *user : rsOp->getUsers()) {
    // Direct: shrsi → trunci (validate full chain via isSRSCompoundCandidate)
    if (auto truncOp = dyn_cast<arith::TruncIOp>(user))
      if (isSRSCompoundCandidate(truncOp))
        return true;
    // Through clamp: shrsi → maxsi → minsi → trunci
    //            or: shrsi → minsi → maxsi → trunci
    if (isa<arith::MaxSIOp, arith::MinSIOp>(user)) {
      for (Operation *user2 : user->getUsers()) {
        if (auto truncOp2 = dyn_cast<arith::TruncIOp>(user2))
          if (isSRSCompoundCandidate(truncOp2))
            return true;
        if (isa<arith::MaxSIOp, arith::MinSIOp>(user2)) {
          for (Operation *user3 : user2->getUsers()) {
            if (auto truncOp3 = dyn_cast<arith::TruncIOp>(user3))
              if (isSRSCompoundCandidate(truncOp3))
                return true;
          }
        }
      }
    }
  }
  return false;
}

// Check if a scalar maxsi/minsi is sandwiched in a compound SRS chain
// (between shrsi and trunci). Used to keep the op legal so the scalar
// compound SRS pattern can consume the entire chain.
static bool scalarClampInCompoundSRS(Operation *op) {
  if (!isa<arith::MaxSIOp, arith::MinSIOp>(op))
    return false;
  // Only apply to scalar types
  if (isa<VectorType>(op->getResult(0).getType()))
    return false;
  for (Operation *user : op->getUsers()) {
    if (auto truncOp = dyn_cast<arith::TruncIOp>(user)) {
      if (isSRSCompoundCandidate(truncOp))
        return true;
    }
    if (isa<arith::MaxSIOp, arith::MinSIOp>(user)) {
      for (Operation *user2 : user->getUsers()) {
        if (auto truncOp2 = dyn_cast<arith::TruncIOp>(user2)) {
          if (isSRSCompoundCandidate(truncOp2))
            return true;
        }
      }
    }
  }
  return false;
}

// Given a Value, if it is defined by a widening op (arith:ExtSIOp,
// arith::ExtUIOp, arith::ExtFOp, aievec::UPSOp + aievec::SRSOp,
// aievec::UPSOp + aievec::CastOp), return the source of the widening op.
static std::optional<Value> getSourceOfWideningOp(Value src) {
  if (auto extSIOp = src.getDefiningOp<arith::ExtSIOp>())
    return extSIOp.getIn();
  if (auto extUIOp = src.getDefiningOp<arith::ExtUIOp>())
    return extUIOp.getIn();
  if (auto extFOp = src.getDefiningOp<arith::ExtFOp>())
    return extFOp.getIn();
  if (auto srsOp = src.getDefiningOp<aievec::SRSOp>()) {
    // Conversion through AIE intrinsics takes two steps:
    //     1) Load to accumulator: aievec.ups
    //     2) Move from accumulator: aievec.srs
    auto srsSource = srsOp.getSource();
    if (srsSource)
      if (auto upsOp = srsSource.getDefiningOp<aievec::UPSOp>())
        return upsOp.getSource();
  }
  if (auto castOp = src.getDefiningOp<aievec::CastOp>()) {
    // Conversion through AIE intrinsics can also take the following two steps:
    //     1) Load to accumulator: aievec.ups
    //     2) Move from accumulator: aievec.cast
    auto castSource = castOp.getSource();
    if (castSource)
      if (auto upsOp = castSource.getDefiningOp<aievec::UPSOp>())
        return upsOp.getSource();
  }
  return std::optional<Value>();
}

// Given a Value, if it is defined by a narrowing op (arith::TruncFOp,
// arith::TruncIOp), return the source of the narrowing op.
static std::optional<Value> getSourceOfNarrowingOp(Value src) {
  if (auto truncFOp = src.getDefiningOp<arith::TruncFOp>())
    return truncFOp.getIn();
  if (auto truncIOp = src.getDefiningOp<arith::TruncIOp>())
    return truncIOp.getIn();
  return std::optional<Value>();
}

//===----------------------------------------------------------------------===//
// Type conversion utilities with narrowing/widening optimization awareness
//===----------------------------------------------------------------------===//

// Smart widen a value to target type. If the value was previously narrowed
// from the target type, reuse the original source to avoid truncf->extf chains.
static Value widenValueWithNarrowingCheck(Value val, Type targetType,
                                          Location loc,
                                          ConversionPatternRewriter &rewriter) {
  // Check if this value was narrowed from the target type
  if (auto narrowedSrc = getSourceOfNarrowingOp(val)) {
    if (narrowedSrc->getType() == targetType)
      return *narrowedSrc; // Reuse the original value (skip truncf->extf)
  }

  // Otherwise, create the widening op if needed
  if (val.getType() == targetType)
    return val;

  return arith::ExtFOp::create(rewriter, loc, targetType, val);
}

// Result structure for smart narrowing operation
struct NarrowingResult {
  Value narrowedValue;     // The narrowed value (or original if optimized)
  bool skipNarrowing;      // True if we should skip creating truncf
  Operation *wideningUser; // The widening op to replace (if skipNarrowing)
};

// Smart narrow a value to target type. If the result will be immediately
// widened back, skip both truncf and extf operations.
static NarrowingResult
narrowValueWithWideningCheck(Operation *srcOp, Value val, Type targetType,
                             Location loc,
                             ConversionPatternRewriter &rewriter) {
  NarrowingResult result;
  result.narrowedValue = val;
  result.skipNarrowing = false;
  result.wideningUser = nullptr;

  // Check if srcOp will be immediately widened back
  if (srcOp->hasOneUse()) {
    Operation *user = *srcOp->getUsers().begin();
    if (auto extfOp = dyn_cast<arith::ExtFOp>(user)) {
      // The result will be widened - skip both truncf and extf
      result.skipNarrowing = true;
      result.wideningUser = extfOp;
      return result;
    }
  }

  // Normal case: create the narrowing op
  result.narrowedValue =
      arith::TruncFOp::create(rewriter, loc, targetType, val);
  return result;
}

// High-level helper to perform a binary operation on bf16 values in f32
// precision. This function handles:
// 1. Smart widening of operands (reuses f32 source if narrowed from f32)
// 2. Executing the operation in f32
// 3. Smart narrowing back to bf16 (skips truncf->extf if result is widened)
static void
performBF16BinaryOpInF32(Value lhs, Value rhs, Operation *srcOp, Location loc,
                         ConversionPatternRewriter &rewriter,
                         std::function<Value(Value, Value)> opBuilder) {
  Type f32Type = rewriter.getF32Type();

  // Smart widen both operands (reuse f32 source if narrowed from f32)
  Value lhsF32 = widenValueWithNarrowingCheck(lhs, f32Type, loc, rewriter);
  Value rhsF32 = widenValueWithNarrowingCheck(rhs, f32Type, loc, rewriter);

  // Perform operation in f32
  Value resultF32 = opBuilder(lhsF32, rhsF32);

  // Smart narrow back to bf16 (skip if result will be widened)
  auto narrowResult = narrowValueWithWideningCheck(
      srcOp, resultF32, lhs.getType(), loc, rewriter);

  if (narrowResult.skipNarrowing) {
    // Replace the widening user directly with f32 result
    rewriter.replaceOp(narrowResult.wideningUser, resultF32);
    rewriter.eraseOp(srcOp);
  } else {
    rewriter.replaceOp(srcOp, narrowResult.narrowedValue);
  }
}

// Given the LHS and RHS of an `arith::AddIOp`, if one of them is defined by an
// `arith::MulIOp`, return a tuple with the `lhs`, `rhs`, and `acc` of the MAC
// operation that can replace them.
static std::optional<std::tuple<Value, Value, Value>>
extractMACOperandsFromAddOperands(Value addLhs, Value addRhs) {
  auto *lhsDefOp = addLhs.getDefiningOp();
  auto *rhsDefOp = addRhs.getDefiningOp();
  arith::MulIOp mulOp = nullptr;
  Value acc;
  if (lhsDefOp) {
    mulOp = dyn_cast<arith::MulIOp>(lhsDefOp);
    acc = addRhs;
  }
  if (!mulOp && rhsDefOp) {
    mulOp = dyn_cast<arith::MulIOp>(rhsDefOp);
    acc = addLhs;
  }
  if (mulOp)
    return std::make_tuple(mulOp.getLhs(), mulOp.getRhs(), acc);

  // If the MulIOp has been already translated to aievec::aie1::MulOp:
  auto lhsSrsOp = addLhs.getDefiningOp<aievec::SRSOp>();
  auto rhsSrsOp = addRhs.getDefiningOp<aievec::SRSOp>();
  aievec::aie1::MulOp aieMulOp = nullptr;
  if (lhsSrsOp) {
    aieMulOp = lhsSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>();
    acc = addRhs;
  }
  if (!aieMulOp && rhsSrsOp) {
    aieMulOp = rhsSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>();
    acc = addLhs;
  }
  if (aieMulOp)
    return std::make_tuple(aieMulOp.getLhs(), aieMulOp.getRhs(), acc);
  return {};
}

// Convert a input value to a target vector type. This function can insert
// multiple aievec ops depending on the combination of input and output vector
// types.
static std::optional<Value>
convertValueToTargetTypeAIE2(ConversionPatternRewriter &rewriter, Location loc,
                             Value inputVal, VectorType tgtType) {
  auto srcType = cast<VectorType>(inputVal.getType());
  auto srcElemType = srcType.getElementType();
  unsigned srcBitWidth = srcElemType.getIntOrFloatBitWidth();
  unsigned srcLaneSize = getVectorLaneSize(srcType);

  auto tgtElemType = tgtType.getElementType();
  unsigned tgtBitWidth = tgtElemType.getIntOrFloatBitWidth();
  unsigned tgtLaneSize = getVectorLaneSize(tgtType);

  if (srcType == tgtType)
    return inputVal;

  if ((srcElemType == tgtElemType) && (srcLaneSize != tgtLaneSize)) {
    // TODO: relax the condition below?
    if ((srcLaneSize == 16 && tgtLaneSize == 32 &&
         isa<FloatType>(srcElemType)) ||
        (srcLaneSize == 32 && tgtLaneSize == 64 &&
         isa<IntegerType>(srcElemType))) {
      auto zeroConstOp = arith::ConstantOp::create(
          rewriter, loc, srcType.getElementType(),
          rewriter.getZeroAttr(srcType.getElementType()));
      auto broadcastZeroOp = aievec::BroadcastScalarOp::create(
          rewriter, loc, tgtType, zeroConstOp->getResult(0));
      auto extOp = aievec::ExtOp::create(rewriter, loc, srcType,
                                         broadcastZeroOp->getResult(0), 0);

      SmallVector<Value> inputSources = {inputVal, extOp->getResult(0)};
      auto concatOp =
          aievec::ConcatOp::create(rewriter, loc, tgtType, inputSources);

      return concatOp.getResult();
    }
  } else if ((srcElemType != tgtElemType) && (srcLaneSize == tgtLaneSize) &&
             isa<IntegerType>(srcElemType) && isa<IntegerType>(tgtElemType)) {
    if (srcBitWidth == 16 && tgtBitWidth == 32 && srcLaneSize == 16) {
      // Case 1: vector<16xi16> to vector<16xi32> conversion by aievec.ups +
      // aievec.cast
      auto accType = getVectorOpDestType(srcType, /*AIE2 =*/true);
      auto upsOp = aievec::UPSOp::create(rewriter, loc, accType, inputVal);
      auto castOp = aievec::CastOp::create(
          rewriter, loc, tgtType, upsOp.getResult(), /*isResAcc*/ false);
      return castOp.getResult();
    }

    if (srcBitWidth == 8 && tgtBitWidth == 32 && srcLaneSize == 16) {
      // Case 2: vector<16xi8> to vector<16xi32> conversion by aievec.concat +
      // aievec.ups + aievec.cast + aievec.ext
      auto concatOutType = createVectorType(32, srcElemType);
      auto concatOp =
          aievec::ConcatOp::create(rewriter, loc, concatOutType,
                                   SmallVector<Value>({inputVal, inputVal}));
      auto accType = getVectorOpDestType(concatOutType, /*AIE2 =*/true);
      auto upsOp =
          aievec::UPSOp::create(rewriter, loc, accType, concatOp.getResult());
      auto castType = createVectorType(32, tgtElemType);
      auto castOp = aievec::CastOp::create(
          rewriter, loc, castType, upsOp.getResult(), /*isResAcc*/ false);
      auto extOp =
          aievec::ExtOp::create(rewriter, loc, tgtType, castOp.getResult(), 0);
      return extOp.getResult();
    }

    if (srcBitWidth == 8 && tgtBitWidth == 16 && srcLaneSize == 32) {
      // Case 3: vector<32xi8> to vector<32xi16> conversion by aievec.unpack
      auto unpackOp =
          aievec::UnpackOp::create(rewriter, loc, tgtType, inputVal);
      return unpackOp.getResult();
    }
  }

  return std::nullopt;
}

// Return the list of attributes that configure an `aievec.select` op to
// perform a rotation of the input vector by `rotation` number of elements.
// The attribute values depend on the vector type of the select operation.
static SmallVector<NamedAttribute>
buildAttributeListForRotationSelectOp(PatternRewriter &rewriter, VectorType vTy,
                                      int64_t rotation) {
  unsigned width = 0;
  auto elemTy = vTy.getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    width = intTy.getWidth();
  StringAttr attr0 = rewriter.getStringAttr("0");
  StringAttr attr0x06040200 = rewriter.getStringAttr("0x06040200");
  StringAttr attr0x0e0c0a08 = rewriter.getStringAttr("0x0e0c0a08");
  StringAttr attr0x2103 = rewriter.getStringAttr("0x2103");
  StringAttr attr0x3210 = rewriter.getStringAttr("0x3210");
  StringAttr selectAttrName = rewriter.getStringAttr("select");
  StringAttr xoffsetsAttrName = rewriter.getStringAttr("xoffsets");
  StringAttr xoffsetsHiAttrName = rewriter.getStringAttr("xoffsets_hi");
  StringAttr xsquareAttrName = rewriter.getStringAttr("xsquare");
  StringAttr xstartAttrName = rewriter.getStringAttr("xstart");
  StringAttr yoffsetsAttrName = rewriter.getStringAttr("yoffsets");
  StringAttr yoffsetsHiAttrName = rewriter.getStringAttr("yoffsets_hi");
  StringAttr ysquareAttrName = rewriter.getStringAttr("ysquare");
  StringAttr ystartAttrName = rewriter.getStringAttr("ystart");

  switch (width) {
  case 16: {
    if (rotation % 2) {
      int64_t xstart = rotation + 1;
      int64_t ystart = rotation - 1;
      return SmallVector<NamedAttribute, 9>(
          {{selectAttrName, rewriter.getStringAttr("0x11111111")},
           {xoffsetsAttrName, attr0x06040200},
           {xoffsetsHiAttrName, attr0x0e0c0a08},
           {xsquareAttrName, attr0x2103},
           {xstartAttrName, rewriter.getStringAttr(std::to_string(xstart))},
           {yoffsetsAttrName, rewriter.getStringAttr("0x0503010f")},
           {yoffsetsHiAttrName, rewriter.getStringAttr("0x0d0b0907")},
           {ysquareAttrName, attr0x2103},
           {ystartAttrName, rewriter.getStringAttr(std::to_string(ystart))}});
    }
    return SmallVector<NamedAttribute, 9>(
        {{selectAttrName, attr0},
         {xoffsetsAttrName, attr0x06040200},
         {xoffsetsHiAttrName, attr0x0e0c0a08},
         {xsquareAttrName, attr0x3210},
         {xstartAttrName, rewriter.getStringAttr(std::to_string(rotation))},
         {yoffsetsAttrName, attr0},
         {yoffsetsHiAttrName, attr0},
         {ysquareAttrName, attr0},
         {ystartAttrName, attr0}});
  }
  case 32:
    return SmallVector<NamedAttribute, 7>(
        {{selectAttrName, attr0},
         {xoffsetsAttrName, rewriter.getStringAttr("0x76543210")},
         {xsquareAttrName, attr0x3210},
         {xstartAttrName, rewriter.getStringAttr(std::to_string(rotation))},
         {yoffsetsAttrName, attr0},
         {ysquareAttrName, attr0},
         {ystartAttrName, attr0}});
  default:
    llvm::report_fatal_error("Unexpected width!");
  }

  return {};
}

namespace xilinx::aievec {

SmallVector<NamedAttribute>
buildFMAOpSplatAttrForElemTy(aievec::aie1::FMAOp fmaOp, int64_t bcastPos,
                             int64_t step = 1) {
  unsigned width = 0;
  auto elemTy = fmaOp.getLhs().getType().getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    width = intTy.getWidth();
  auto *ctx = fmaOp.getContext();
  switch (width) {
  case 16:
    // NOTE: The pattern is:
    //       acc[0]  = x[0]  * z[bcastPos] + x[16] * z[bcastPos+step]
    //       acc[1]  = x[1]  * z[bcastPos] + x[17] * z[bcastPos+step]
    //       acc[2]  = x[2]  * z[bcastPos] + x[18] * z[bcastPos+step]
    //       acc[3]  = x[3]  * z[bcastPos] + x[19] * z[bcastPos+step]
    //       acc[4]  = x[4]  * z[bcastPos] + x[20] * z[bcastPos+step]
    //       acc[5]  = x[5]  * z[bcastPos] + x[21] * z[bcastPos+step]
    //       acc[6]  = x[6]  * z[bcastPos] + x[22] * z[bcastPos+step]
    //       acc[7]  = x[7]  * z[bcastPos] + x[23] * z[bcastPos+step]
    //       acc[8]  = x[8]  * z[bcastPos] + x[24] * z[bcastPos+step]
    //       acc[9]  = x[9]  * z[bcastPos] + x[25] * z[bcastPos+step]
    //       acc[10] = x[10] * z[bcastPos] + x[26] * z[bcastPos+step]
    //       acc[11] = x[11] * z[bcastPos] + x[27] * z[bcastPos+step]
    //       acc[12] = x[12] * z[bcastPos] + x[28] * z[bcastPos+step]
    //       acc[13] = x[13] * z[bcastPos] + x[29] * z[bcastPos+step]
    //       acc[14] = x[14] * z[bcastPos] + x[30] * z[bcastPos+step]
    //       acc[15] = x[15] * z[bcastPos] + x[31] * z[bcastPos+step]
    return SmallVector<NamedAttribute, 11>(
        {{fmaOp.getXstartAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getXoffsetsAttrName(), StringAttr::get(ctx, "0x73727170")},
         {fmaOp.getXoffsetsHiAttrName(), StringAttr::get(ctx, "0x77767574")},
         {fmaOp.getXstepAttrName(), fmaOp.getXstepAttr()},
         {fmaOp.getXsquareAttrName(), StringAttr::get(ctx, "0x3120")},
         {fmaOp.getZstartAttrName(),
          StringAttr::get(ctx, std::to_string(bcastPos))},
         {fmaOp.getZoffsetsAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getZoffsetsHiAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getZstepAttrName(), StringAttr::get(ctx, std::to_string(step))},
         {fmaOp.getZsquareAttrName(), fmaOp.getZsquareAttr()},
         {fmaOp.getFmsubAttrName(), fmaOp.getFmsubAttr()}});
  case 32:
    return SmallVector<NamedAttribute, 11>(
        {{fmaOp.getXstartAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getXoffsetsAttrName(), StringAttr::get(ctx, "0x76543210")},
         {fmaOp.getXoffsetsHiAttrName(), fmaOp.getXoffsetsHiAttr()},
         {fmaOp.getXstepAttrName(), fmaOp.getXstepAttr()},
         {fmaOp.getXsquareAttrName(), fmaOp.getXsquareAttr()},
         {fmaOp.getZstartAttrName(),
          StringAttr::get(ctx, std::to_string(bcastPos))},
         {fmaOp.getZoffsetsAttrName(), StringAttr::get(ctx, "0x00000000")},
         {fmaOp.getZoffsetsHiAttrName(), fmaOp.getZoffsetsHiAttr()},
         {fmaOp.getZstepAttrName(), fmaOp.getZstepAttr()},
         {fmaOp.getZsquareAttrName(), fmaOp.getZsquareAttr()},
         {fmaOp.getFmsubAttrName(), fmaOp.getFmsubAttr()}});
  default:
    llvm::report_fatal_error("Unexpected width!");
  }

  return {};
}

} // namespace xilinx::aievec

template <typename SrcOpTy, typename AIEv2ElemOp>
static LogicalResult genAddElemAIE2(ConversionPatternRewriter &rewriter,
                                    Value lval, Value rval, VectorType srcType,
                                    SrcOpTy srcOp) {
  auto lCastOp = aievec::CastOp::create(rewriter, srcOp.getLoc(), srcType, lval,
                                        /*isResAcc*/ true);
  auto rCastOp = aievec::CastOp::create(rewriter, srcOp.getLoc(), srcType, rval,
                                        /*isResAcc*/ true);
  auto elemOp = AIEv2ElemOp::create(
      rewriter, srcOp.getLoc(), lCastOp->getResult(0).getType(),
      lCastOp->getResult(0), rCastOp->getResult(0));
  rewriter.replaceOpWithNewOp<aievec::CastOp>(
      srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
  return success();
}

static arith::CmpIPredicate
convertToIntegerPredicate(arith::CmpFPredicate pred) {
  switch (pred) {
  case CmpFPredicate::UEQ:
  case CmpFPredicate::OEQ:
    return CmpIPredicate::eq;
  case CmpFPredicate::UGT:
    return CmpIPredicate::ugt;
  case CmpFPredicate::OGT:
    return CmpIPredicate::sgt;
  case CmpFPredicate::UGE:
    return CmpIPredicate::uge;
  case CmpFPredicate::OGE:
    return CmpIPredicate::sge;
  case CmpFPredicate::ULT:
    return CmpIPredicate::ult;
  case CmpFPredicate::OLT:
    return CmpIPredicate::slt;
  case CmpFPredicate::ULE:
    return CmpIPredicate::ule;
  case CmpFPredicate::OLE:
    return CmpIPredicate::sle;
  case CmpFPredicate::UNE:
  case CmpFPredicate::ONE:
    return CmpIPredicate::ne;
  default:
    llvm::report_fatal_error("Unexpected predicate!");
  }
}

static arith::CmpIPredicate
convertToIntegerPredicate(arith::CmpIPredicate pred) {
  return pred;
}

static aievec::CmpOp createCmpOpAIE2(ConversionPatternRewriter &rewriter,
                                     CmpIPredicate pred, Location loc,
                                     Type type, Value lhs, Value rhs) {
  switch (pred) {
  case CmpIPredicate::eq:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "eq");
  case CmpIPredicate::ne:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ne");
  case CmpIPredicate::slt:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "slt");
  case CmpIPredicate::ult:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ult");
  case CmpIPredicate::sle:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "sle");
  case CmpIPredicate::ule:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ule");
  case CmpIPredicate::sgt:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "sgt");
  case CmpIPredicate::ugt:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "ugt");
  case CmpIPredicate::sge:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "sge");
  case CmpIPredicate::uge:
    return aievec::CmpOp::create(rewriter, loc, type, lhs, rhs, "uge");
  }
  return nullptr;
}

template <typename DstOpTy>
static aievec::ExtElemOp
generateAIEVecOpsForReductionOp(ConversionPatternRewriter &rewriter,
                                vector::ReductionOp srcOp, int shiftIndex,
                                Value curValue) {
  assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
         "shiftIndex must be power of 2");

  Location loc = srcOp.getLoc();
  auto vType = dyn_cast<VectorType>(curValue.getType());
  Type scalarType = vType.getElementType();
  Type vecType = curValue.getType();
  DstOpTy curOp = nullptr;
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();

  for (int id = shiftIndex; id > 0; id /= 2) {
    auto constOp = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(id * elWidth / 8));

    auto shiftBytesOp = aievec::ShiftOp::create(
        rewriter, loc, vecType, curValue, curValue, constOp.getResult());

    curOp = DstOpTy::create(rewriter, loc, vecType, curValue,
                            shiftBytesOp.getResult());

    curValue = curOp.getResult();
  }

  auto zeroConstOp =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  return aievec::ExtElemOp::create(rewriter, loc, scalarType, curOp,
                                   zeroConstOp.getResult());
}

// Helper to pad a v16bf16 vector to v32bf16 by concatenating with a splat of
// the given infinity value. Used by min/max reduction patterns.
// Returns {paddedVector, newLaneSize}.
static std::pair<Value, unsigned>
padV16ToV32WithInfinity(ConversionPatternRewriter &rewriter, Location loc,
                        Value inputVec, Type scalarType, bool negativeInf) {
  VectorType v32bf16Type = createVectorType(32, scalarType);
  VectorType v16bf16Type = createVectorType(16, scalarType);

  // Create a scalar infinity constant
  auto infAttr = rewriter.getFloatAttr(
      scalarType,
      APFloat::getInf(cast<FloatType>(scalarType).getFloatSemantics(),
                      negativeInf));
  auto splatInf = arith::ConstantOp::create(rewriter, loc, infAttr).getResult();

  // Broadcast to v32bf16, then extract upper half (which is also infinity)
  auto infVec =
      aievec::BroadcastScalarOp::create(rewriter, loc, v32bf16Type, splatInf);
  auto infUpperHalf =
      aievec::ExtOp::create(rewriter, loc, v16bf16Type, infVec, 1);

  // Concatenate input with infinity padding
  Value paddedVec =
      aievec::ConcatOp::create(rewriter, loc, v32bf16Type,
                               ValueRange{inputVec, infUpperHalf.getResult()});

  return {paddedVec, 32};
}

static func::FuncOp getOrInsertFuncDecl(ConversionPatternRewriter &rewriter,
                                        Operation *parentSymbolTableOp,
                                        StringRef funcName, TypeRange inTypes,
                                        TypeRange outTypes) {

  mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(
      &parentSymbolTableOp->getRegions().front().getBlocks().front());
  SymbolTable st = SymbolTable(parentSymbolTableOp);
  func::FuncOp fnOpLookup = st.lookup<func::FuncOp>(funcName);
  func::FuncOp fnOp;
  // if the function is already declared, use the existing function, don't
  // declare multiple times
  if (fnOpLookup != NULL) {
    fnOp = fnOpLookup;
  } else {
    StringAttr t1 = rewriter.getStringAttr("sym_visibility");
    StringAttr t2 = rewriter.getStringAttr("private");
    NamedAttribute funcAccess = NamedAttribute(t1, t2);
    FunctionType fnType =
        mlir::FunctionType::get(rewriter.getContext(), inTypes, outTypes);
    fnOp = func::FuncOp::create(rewriter, parentSymbolTableOp->getLoc(),
                                funcName, fnType, funcAccess);
  }
  return fnOp;
}

//===----------------------------------------------------------------------===//
// Wide vector splitting utility
//===----------------------------------------------------------------------===//

// Utility function to split a wide vector operation (e.g., v32bf16) into two
// half-width operations (e.g., v16bf16) and concatenate the results.
// This pattern is common for AIE2 when hardware only supports 16-lane
// operations but we need to process 32-lane vectors.
//
// Template parameters:
//   SrcOpTy - The source operation type (e.g., arith::MulFOp, math::ExpOp)
//
// Parameters:
//   srcOp - The source operation to replace
//   wideInputs - The wide input values (e.g., lhs and rhs for binary ops)
//   halfType - The half-width vector type (e.g., vector<16xbf16>)
//   wideType - The wide vector type (e.g., vector<32xbf16>)
//   rewriter - The pattern rewriter
//   processHalves - Callback that processes the lower and upper halves
//                   and returns a pair of half-width results to be concatenated
//
// The callback signature is:
//   std::pair<Value, Value>(ArrayRef<std::pair<Value, Value>> halfInputs,
//                           Location loc, ConversionPatternRewriter &rewriter)
// where halfInputs[i] is {lowerHalf, upperHalf} for each wideInput
template <typename SrcOpTy, typename Func>
static void splitWideVectorOp(SrcOpTy srcOp, ArrayRef<Value> wideInputs,
                              VectorType halfType, VectorType wideType,
                              ConversionPatternRewriter &rewriter,
                              Func &&processHalves) {

  Location loc = srcOp.getLoc();

  // Extract lower and upper halves for each wide input
  SmallVector<std::pair<Value, Value>> halfInputs;
  halfInputs.reserve(wideInputs.size());
  for (Value wideInput : wideInputs) {
    auto lowerHalf =
        aievec::ExtOp::create(rewriter, loc, halfType, wideInput, 0);
    auto upperHalf =
        aievec::ExtOp::create(rewriter, loc, halfType, wideInput, 1);
    halfInputs.emplace_back(lowerHalf.getResult(), upperHalf.getResult());
  }

  // Process halves using the callback
  auto [lowResult, highResult] = processHalves(halfInputs, loc, rewriter);

  // Concatenate results
  SmallVector<Value> concatSources = {lowResult, highResult};
  rewriter.replaceOpWithNewOp<aievec::ConcatOp>(srcOp, wideType, concatSources);
}

// Simplified version for unary operations
template <typename SrcOpTy>
static void splitWideUnaryVectorOp(
    SrcOpTy srcOp, Value wideInput, VectorType halfType, VectorType wideType,
    ConversionPatternRewriter &rewriter,
    std::function<Value(Value, Location, ConversionPatternRewriter &)>
        processHalf) {

  splitWideVectorOp<SrcOpTy>(
      srcOp, {wideInput}, halfType, wideType, rewriter,
      [&processHalf](ArrayRef<std::pair<Value, Value>> halfInputs, Location loc,
                     ConversionPatternRewriter &rewriter) {
        auto [lowerHalf, upperHalf] = halfInputs[0];
        Value lowResult = processHalf(lowerHalf, loc, rewriter);
        Value highResult = processHalf(upperHalf, loc, rewriter);
        return std::make_pair(lowResult, highResult);
      });
}

// Simplified version for binary operations
template <typename SrcOpTy>
static void splitWideBinaryVectorOp(
    SrcOpTy srcOp, Value lhs, Value rhs, VectorType halfType,
    VectorType wideType, ConversionPatternRewriter &rewriter,
    std::function<Value(Value, Value, Location, ConversionPatternRewriter &)>
        processHalf) {

  splitWideVectorOp<SrcOpTy>(
      srcOp, {lhs, rhs}, halfType, wideType, rewriter,
      [&processHalf](ArrayRef<std::pair<Value, Value>> halfInputs, Location loc,
                     ConversionPatternRewriter &rewriter) {
        auto [lhsLow, lhsHigh] = halfInputs[0];
        auto [rhsLow, rhsHigh] = halfInputs[1];
        Value lowResult = processHalf(lhsLow, rhsLow, loc, rewriter);
        Value highResult = processHalf(lhsHigh, rhsHigh, loc, rewriter);
        return std::make_pair(lowResult, highResult);
      });
}

//===----------------------------------------------------------------------===//
// Math operation matching utilities
//===----------------------------------------------------------------------===//

// Check if math.exp op matches AIE2 LUT-based exp constraints
static bool matchExpOpForAIE2LUT(math::ExpOp::Adaptor adaptor) {
  auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());

  if (!srcType)
    return false;

  Type scalarType = srcType.getElementType();
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();
  unsigned laneSize = getVectorLaneSize(srcType);
  // AIE2 LUT-based exp: supports v16bf16 and v32bf16
  return isa<FloatType>(scalarType) && (laneSize == 16 || laneSize == 32) &&
         elWidth == 16;
}

// Check if math.exp op matches AIE2P exp constraints
static bool matchExpOpForAIE2P(math::ExpOp::Adaptor adaptor) {
  auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());

  if (!srcType)
    return false;

  Type scalarType = srcType.getElementType();
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();
  unsigned laneSize = getVectorLaneSize(srcType);
  // AIE2P exp: supports v16bf16 and v32bf16
  return scalarType.isBF16() && (laneSize == 16 || laneSize == 32) &&
         elWidth == 16;
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

// This pattern fold `vector.extract` and `vector.broadcast` into
// `aievec.broadcast` for AIE2
struct FoldVectorExtractAndSplatToAIEBroadcast
    : OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto extOp = adaptor.getSource().getDefiningOp<vector::ExtractOp>();

    if (!extOp)
      return failure();

    auto src = extOp.getSource();
    auto pos = extOp.getStaticPosition();
    int64_t posVal = pos[0];
    auto srcVecType = cast<VectorType>(src.getType());
    auto resultType = cast<VectorType>(bcastOp.getResult().getType());
    if (srcVecType != resultType) {
      if (srcVecType.getNumElements() != 2 * resultType.getNumElements())
        return failure();
      auto half = static_cast<int8_t>(posVal / resultType.getNumElements());
      posVal -= half * resultType.getNumElements();
      src = aievec::ExtOp::create(rewriter, extOp.getLoc(), resultType, src,
                                  rewriter.getI8IntegerAttr(half))
                .getResult();
    }

    unsigned elWidth = resultType.getElementType().getIntOrFloatBitWidth();

    if (unsigned laneSize = getVectorLaneSize(resultType);
        laneSize * elWidth == 512) {
      // Common use case for the broadcast_elem intrinsic
      rewriter.replaceOpWithNewOp<aievec::BroadcastOp>(bcastOp, resultType, src,
                                                       posVal);
    } else if (laneSize * elWidth == 256) {
      // e.g. need v16bf16 due to the subsequent v16accfloat operation
      VectorType aievecBcastType =
          createVectorType(512 / elWidth, resultType.getElementType());
      auto concatOp =
          aievec::ConcatOp::create(rewriter, bcastOp.getLoc(), aievecBcastType,
                                   SmallVector<Value>({src, src}));
      auto aieBcastOp = aievec::BroadcastOp::create(
          rewriter, bcastOp.getLoc(), aievecBcastType, concatOp.getResult(),
          posVal);
      rewriter.replaceOpWithNewOp<aievec::ExtOp>(bcastOp, resultType,
                                                 aieBcastOp.getResult(), 0);
    } else if (laneSize * elWidth == 1024) {
      // e.g. need v32int32 due to the subsequent v32acc32 operation
      VectorType aievecBcastType =
          createVectorType(512 / elWidth, resultType.getElementType());
      auto half = static_cast<int8_t>(posVal / resultType.getNumElements());
      posVal -= half * resultType.getNumElements();
      auto extOp =
          aievec::ExtOp::create(rewriter, bcastOp.getLoc(), aievecBcastType,
                                src, rewriter.getI8IntegerAttr(half));
      auto aieBcastOp = aievec::BroadcastOp::create(rewriter, bcastOp.getLoc(),
                                                    aievecBcastType,
                                                    extOp.getResult(), posVal);
      rewriter.replaceOpWithNewOp<aievec::ConcatOp>(
          bcastOp, resultType,
          SmallVector<Value>({aieBcastOp.getResult(), aieBcastOp.getResult()}));
    } else {
      return failure();
    }

    return success();
  }
};

struct ConvertSplatToAIEBroadcast : OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getSource().getDefiningOp<vector::ExtractOp>())
      return failure();

    auto resultType = cast<VectorType>(bcastOp.getResult().getType());
    auto flatResultType = getFlattenedVectorType(resultType);
    Type scalarType = resultType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    auto src = bcastOp.getSource();

    if (laneSize * elWidth == 512) {
      Value newOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), flatResultType, src);
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    if (laneSize * elWidth == 256) {
      VectorType vecType = createVectorType(512 / elWidth, scalarType);
      auto aieBcastOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), vecType, src);
      Value newOp =
          aievec::ExtOp::create(rewriter, bcastOp.getLoc(), flatResultType,
                                aieBcastOp.getResult(), 0);
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    if (laneSize * elWidth == 1024) {
      VectorType vecType = createVectorType(512 / elWidth, scalarType);
      auto aieBcastOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), vecType, src);
      Value newOp = aievec::ConcatOp::create(
          rewriter, bcastOp.getLoc(), flatResultType,
          SmallVector<Value>({aieBcastOp.getResult(), aieBcastOp.getResult()}));
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    return failure();
  }
};

// This pattern replaces `arith.muli`+`arith.addi` on vectors with
// `aievec.mac_elem`. This pattern works for AIE2.
struct ConvertMulAddToAIEVecFMAElemOpPattern
    : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulAddToAIEVecFMAElemOpPattern(MLIRContext *context,
                                        unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(addOp.getType());
    if (!resultType)
      return failure();

    // Verify it can be replaced by a MAC
    auto res =
        extractMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    // Verify the vector type is supported by AIE2
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || resultElWidth != 16) &&
        (laneSize != 16 || resultElWidth != 32))
      return failure();

    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIE2 =*/true);
    auto upsOp = aievec::UPSOp::create(rewriter, addOp.getLoc(), accType, acc,
                                       shiftParam);
    auto fmaElemOp = aievec::FMAElemOp::create(
        rewriter, addOp.getLoc(), accType, lhs, rhs, upsOp.getResult(),
        /*fmsub=*/false);

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, addOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        addOp, resultType, fmaElemOp.getResult(), shiftParamOp.getResult());

    return success();
  }

  unsigned shiftParam;
};

// Convert `vector.fma` to `aievec.mac_elem`. Only `vector<16xf32>` and
// `vector<16xbf16>` operand types are supported. In the case of vectors with
// `f32` elemental type, this pattern will try to match `bf16` to `f32`
// widening ops in the `lhs` and `rhs` operands, or fail otherwise.
// TODO: When sign extensions are not found, a conversion from `f32` to `bf16`
// TODO: can be inserted to emulate `f32` fma with `bf16` logic.
struct ConvertVectorFMAOpToAIEVecFMAElemOpPattern
    : OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertVectorFMAOpToAIEVecFMAElemOpPattern(MLIRContext *context,
                                             unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify the vector type is supported by AIE2
    auto resVecTy = cast<VectorType>(fmaOp.getType());
    auto resElemTy = resVecTy.getElementType();
    unsigned numElems = getVectorLaneSize(resVecTy);

    if (numElems != 16 || (!resElemTy.isF32() && !resElemTy.isBF16()))
      return rewriter.notifyMatchFailure(
          fmaOp, "Unsupported operand types in vector.fma lowering.");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value acc = adaptor.getAcc();
    if (resElemTy.isBF16())
      acc = aievec::UPSOp::create(rewriter, fmaOp.getLoc(),
                                  VectorType::get({16}, rewriter.getF32Type()),
                                  acc, shiftParam);
    else {
      lhs = getSourceOfWideningOp(lhs).value_or(nullptr);
      rhs = getSourceOfWideningOp(rhs).value_or(nullptr);
      if (!lhs || !rhs)
        return rewriter.notifyMatchFailure(
            fmaOp, "vector.fma operands are f32, and they don't come from "
                   "arith.extf on bf16; can't lower to aievec.");
      if (!cast<VectorType>(lhs.getType()).getElementType().isBF16() ||
          !cast<VectorType>(rhs.getType()).getElementType().isBF16())
        return rewriter.notifyMatchFailure(
            fmaOp, "vector.fma operands come from arith.extf, but the source "
                   "of the widening op is not bf16; can't lower to aievec.");
    }
    Value newOp =
        aievec::FMAElemOp::create(rewriter, fmaOp.getLoc(), acc.getType(), lhs,
                                  rhs, acc, /*fmsub=*/false);

    if (resElemTy.isBF16()) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, fmaOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
      newOp = aievec::SRSOp::create(rewriter, fmaOp.getLoc(), resVecTy, newOp,
                                    shiftParamOp);
    }

    rewriter.replaceOp(fmaOp, newOp);

    return success();
  }

  unsigned shiftParam;
};

// This pattern replaces `arith.mulf` on vectors with
// `aievec.mul_elem`. This pattern works for AIE2.
struct ConvertMulFToAIEVecMulElemOpPattern
    : OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulFToAIEVecMulElemOpPattern(MLIRContext *context,
                                      unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::MulFOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(mulOp.getType());
    if (!resultType)
      return failure();

    // FIXME: Verify it is not a part of FMA
    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    if (mulOp->hasOneUse() && llvm::any_of(mulOp->getUsers(), isAddOp))
      return failure();

    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();

    unsigned laneSize = getVectorLaneSize(resultType);

    // Handle vector<32xbf16> by splitting into two vector<16xbf16> operations
    if (laneSize == 32 && resultElWidth == 16) {
      VectorType halfType = createVectorType(16, resultType.getElementType());
      unsigned localShiftParam = shiftParam;

      splitWideBinaryVectorOp<arith::MulFOp>(
          mulOp, adaptor.getLhs(), adaptor.getRhs(), halfType, resultType,
          rewriter,
          [localShiftParam](Value lhsHalf, Value rhsHalf, Location loc,
                            ConversionPatternRewriter &rewriter) -> Value {
            Type accType = getVectorOpDestType(
                cast<VectorType>(lhsHalf.getType()), /*AIE2 =*/true);
            auto mulElemOp = aievec::MulElemOp::create(rewriter, loc, accType,
                                                       lhsHalf, rhsHalf);
            auto shiftParamOp = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(localShiftParam));
            auto srsOp = aievec::SRSOp::create(
                rewriter, loc, cast<VectorType>(lhsHalf.getType()),
                mulElemOp.getResult(), shiftParamOp.getResult());
            return srsOp.getResult();
          });
      return success();
    }

    // bfloat16 and float type (laneSize == 16)
    if (laneSize != 16 || (resultElWidth != 16 && resultElWidth != 32))
      return failure();

    // Decide the accType for aievec.mul_elem based on mulOp's lhs & rhs
    auto lval = adaptor.getLhs();
    auto rval = adaptor.getRhs();
    lval = getSourceOfWideningOp(lval).value_or(lval);
    rval = getSourceOfWideningOp(rval).value_or(rval);
    auto lSrcType = cast<VectorType>(lval.getType());
    auto rSrcType = cast<VectorType>(rval.getType());
    unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
    unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();
    Type accType = getVectorOpDestType(lSrcType, /*AIE2 =*/true);
    if (rBitWidth > lBitWidth) {
      accType = getVectorOpDestType(rSrcType, /*AIE2 =*/true);
    }
    // Only support the same lhs/rhs type at the moment
    if (lSrcType != rSrcType) {
      return failure();
    }

    // Prepare lhr/rhs for the aievec.mul_elem op
    unsigned bitWidth = (rBitWidth > lBitWidth) ? rBitWidth : lBitWidth;
    Type srcElemType = (rBitWidth > lBitWidth) ? rSrcType.getElementType()
                                               : lSrcType.getElementType();
    unsigned numLanes = 0;
    if (isa<FloatType>(srcElemType) && (bitWidth == 16 || bitWidth == 32)) {
      numLanes = 16;
    } else if (isa<IntegerType>(srcElemType) &&
               (bitWidth == 8 || bitWidth == 16)) {
      numLanes = 32;
    } else if (isa<IntegerType>(srcElemType) && (bitWidth == 32)) {
      numLanes = 16;
    } else {
      return failure();
    }
    VectorType targetInputType = createVectorType(numLanes, srcElemType);
    if (targetInputType != lSrcType) {
      lval = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(), lval,
                                          targetInputType)
                 .value();
    }
    if (targetInputType != rSrcType) {
      rval = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(), rval,
                                          targetInputType)
                 .value();
    }
    if (!lval || !rval)
      return failure();

    // Create an aievec.mul_elem op
    auto mulElemOp = aievec::MulElemOp::create(rewriter, mulOp.getLoc(),
                                               accType, lval, rval);

    // Create an aievec.cast or an aievec.srs op
    auto mulElemResultType = mulElemOp.getType();
    auto mulElemResultElWidth =
        mulElemResultType.getElementType().getIntOrFloatBitWidth();

    if (mulElemResultElWidth == resultElWidth) {
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          mulOp, resultType, mulElemOp.getResult(), /*isResAcc*/ false);
    } else if (mulElemResultElWidth > resultElWidth) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, mulOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          mulOp, resultType, mulElemOp.getResult(), shiftParamOp.getResult());
    } else {
      return failure();
    }

    return success();
  }

  unsigned shiftParam;
};

// This pattern replaces `arith.muli` on vectors with
// `aievec.mul_elem`. This pattern works for AIE2.
struct ConvertMulIToAIEVecMulElemOpPattern
    : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertMulIToAIEVecMulElemOpPattern(MLIRContext *context,
                                      unsigned shiftParam = 0)
      : OpConversionPattern(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::MulIOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    auto resultType = dyn_cast<VectorType>(mulOp.getType());
    if (!resultType)
      return failure();

    // FIXME: Verify it is not a part of MAC
    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    if (mulOp->hasOneUse() && llvm::any_of(mulOp->getUsers(), isAddOp))
      return failure();

    // Verify the vector type is supported by AIE2
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
        ((laneSize != 16 && laneSize != 32) || resultElWidth != 32))
      return failure();

    // Decide the accType for aievec.mul_elem based on mulOp's lhs & rhs
    auto lval = adaptor.getLhs();
    auto rval = adaptor.getRhs();

    lval = getSourceOfWideningOp(lval).value_or(lval);
    rval = getSourceOfWideningOp(rval).value_or(rval);

    auto lSrcType = cast<VectorType>(lval.getType());
    auto rSrcType = cast<VectorType>(rval.getType());
    unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
    unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();
    Type accType = getVectorOpDestType(lSrcType, /*AIE2 =*/true);
    if (rBitWidth > lBitWidth) {
      accType = getVectorOpDestType(rSrcType, /*AIE2 =*/true);
    }

    // Prepare lhr/rhs for the aievec.mul_elem op
    unsigned bitWidth = (rBitWidth > lBitWidth) ? rBitWidth : lBitWidth;
    Type srcElemType = (rBitWidth > lBitWidth) ? rSrcType.getElementType()
                                               : lSrcType.getElementType();
    unsigned numLanes = 0;
    if (isa<FloatType>(srcElemType) && (bitWidth == 16 || bitWidth == 32)) {
      numLanes = 16;
    } else if (isa<IntegerType>(srcElemType) &&
               (bitWidth == 8 || bitWidth == 16)) {
      numLanes = 32;
    } else if (isa<IntegerType>(srcElemType) && (bitWidth == 32)) {
      numLanes = 16;
    } else {
      return failure();
    }
    VectorType targetInputType = createVectorType(numLanes, srcElemType);
    if (targetInputType != lSrcType) {
      lval = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(), lval,
                                          targetInputType)
                 .value();
    }
    if (targetInputType != rSrcType) {
      rval = convertValueToTargetTypeAIE2(rewriter, mulOp.getLoc(), rval,
                                          targetInputType)
                 .value();
    }
    if (!lval || !rval)
      return failure();

    // Create an aievec.mul_elem op
    auto mulElemOp = aievec::MulElemOp::create(rewriter, mulOp.getLoc(),
                                               accType, lval, rval);

    // Create an aievec.cast or an aievec.srs op
    auto mulElemResultType = mulElemOp.getType();
    auto mulElemResultElWidth =
        mulElemResultType.getElementType().getIntOrFloatBitWidth();

    if (mulElemResultElWidth == resultElWidth) {
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          mulOp, resultType, mulElemOp.getResult(), /*isResAcc*/ false);
    } else if (mulElemResultElWidth > resultElWidth) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, mulOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          mulOp, resultType, mulElemOp.getResult(), shiftParamOp.getResult());
    } else {
      return failure();
    }

    return success();
  }

  unsigned shiftParam;
};

// This pattern folds an extract + broadcast feeding into an
// `aievec::aie1::FMAOp` into the op, using the shuffle attributes.
struct FoldSplatToFMAOp : OpConversionPattern<aievec::aie1::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatOp =
        dyn_cast<aievec::ConcatOp>(adaptor.getLhs().getDefiningOp());
    if (!concatOp)
      return failure();
    vector::BroadcastOp bcastOp = nullptr;
    auto *concatDefOp = concatOp.getSources()[0].getDefiningOp();
    if (concatDefOp)
      bcastOp = dyn_cast<vector::BroadcastOp>(concatDefOp);
    Value lhs = adaptor.getRhs();
    if (!bcastOp) {
      bcastOp = dyn_cast<vector::BroadcastOp>(adaptor.getRhs().getDefiningOp());
      if (!bcastOp)
        return failure();
      lhs = concatOp.getSources()[0];
    }
    auto extOp =
        dyn_cast<vector::ExtractOp>(bcastOp.getSource().getDefiningOp());
    if (!extOp)
      return failure();

    auto rhs = extOp.getSource();
    auto concatVecType = cast<VectorType>(concatOp.getResult().getType());
    auto zvec =
        arith::ConstantOp::create(rewriter, concatOp.getLoc(), lhs.getType(),
                                  rewriter.getZeroAttr(lhs.getType()));
    auto lhsX2 =
        aievec::ConcatOp::create(rewriter, concatOp.getLoc(), concatVecType,
                                 SmallVector<Value, 2>({lhs, zvec}))
            .getResult();
    // XXX: We assume a 1D vector
    auto pos = extOp.getStaticPosition();
    int64_t zstart = pos[0];
    auto fmaOpAttr = buildFMAOpSplatAttrForElemTy(fmaOp, zstart);
    rewriter.replaceOpWithNewOp<aievec::aie1::FMAOp>(
        fmaOp, TypeRange({fmaOp.getResult().getType()}),
        ValueRange({lhsX2, rhs, adaptor.getAcc()}), fmaOpAttr);

    return success();
  }
};

struct ConvertMulAddToAIEVecFMAOpPattern
    : OpConversionPattern<aievec::aie1::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::aie1::AddOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecType = cast<VectorType>(addOp.getType());

    auto res =
        extractMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    SmallVector<int64_t, 4> concatVecShape(vecType.getShape().begin(),
                                           vecType.getShape().end());
    concatVecShape[vecType.getRank() - 1] *= 2;
    auto concatVecType =
        VectorType::get(concatVecShape, vecType.getElementType());
    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIE2 =*/false);
    auto lhsX2 =
        aievec::ConcatOp::create(rewriter, addOp.getLoc(), concatVecType,
                                 SmallVector<Value, 2>(2, lhs))
            .getResult();
    auto upsOp = aievec::UPSOp::create(rewriter, addOp.getLoc(), accType, acc);
    auto fmaOp = aievec::aie1::FMAOp::create(
        rewriter, addOp.getLoc(), accType, lhsX2, rhs, upsOp.getResult(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xstep=*/"",
        /*xsquare=*/"", /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"",
        /*zstep=*/"", /*zsquare=*/"", /*fmsub=*/false);
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, addOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        addOp, vecType, fmaOp.getResult(), shiftParamOp.getResult());
    return success();
  }
};

// This pattern replaces `vector.transfer_read` with `aievec.upd`. Right now,
// it performs a naïve direct translation. This needs to be expanded to
// support more complex scenarios.
struct LowerVectorTransferReadToAIEUPD
    : OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LowerVectorTransferReadToAIEUPD(MLIRContext *context, int64_t minVectorSize,
                                  int64_t maxVectorSize, int64_t alignment,
                                  int64_t maxLoadSize)
      : OpConversionPattern(context), minVectorSize(minVectorSize),
        maxVectorSize(maxVectorSize), vectorAlignment(alignment),
        maxLoadSize(maxLoadSize) {}

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Masked loads
    if (readOp.getMask())
      return readOp.emitError() << "AIE doesn't support masked loads.";

    // Non-contiguous loads
    AffineMap map = readOp.getPermutationMap();
    if (!map.isMinorIdentity())
      return failure();

    // Splats
    if (map.isConstant())
      return failure();

    // Misaligned accesses
    auto vType = readOp.getVectorType();
    if (getTransferReadAlignmentOffset(adaptor, vType, vectorAlignment)
            .value_or(0) != 0)
      return failure();

    // Invalid vector size.
    // We can handle cases where the vector size is:
    //   1) the minimum vector size
    //   2) a square multiple of the alignment size and up to the maximum
    //      vector size.
    int64_t vSize = vType.getNumElements() * vType.getElementTypeBitWidth();
    if (vSize > maxVectorSize ||
        (vSize % vectorAlignment && vSize != minVectorSize))
      return failure();
    // We can deal with linked update instructions when the vector size is
    // exactly twice the load size. This could change in future architectures
    if (vSize > maxLoadSize && vSize != maxLoadSize * 2)
      return failure();
    int64_t multiplicity = vSize / vectorAlignment;
    if ((vSize > minVectorSize) && std::bitset<8>(multiplicity).count() != 1)
      return failure();

    auto updOp = xilinx::aievec::UPDOp::create(
        rewriter, readOp.getLoc(), vType, adaptor.getBase(),
        adaptor.getIndices(), 0, 0, TypedValue<VectorType>(nullptr));
    if (vSize > maxLoadSize) {
      updOp = xilinx::aievec::UPDOp::create(
          rewriter, readOp.getLoc(), vType, adaptor.getBase(),
          adaptor.getIndices(), maxLoadSize, 1, updOp.getResult());
    }
    rewriter.replaceOp(readOp, updOp.getResult());

    return success();
  }

  int64_t minVectorSize, maxVectorSize, vectorAlignment, maxLoadSize;
};

// XXX: Notice that this template doesn't verify that the vector element type
// XXX: is supported by the target architecture.
template <typename SrcOpTy, typename DstOpTy>
struct OneToOneVectorOpToAIEVecOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOpTy>(
        srcOp, srcOp.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xsquare=*/"",
        /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"", /*zsquare=*/"");
    return success();
  }
};

struct LowerVectorAddIOpToAIEVecAddOp : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = addOp.getType();
    if (!isa<VectorType>(resType))
      return failure();

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto *lhsDefOp = lhs.getDefiningOp();
    auto *rhsDefOp = rhs.getDefiningOp();
    if ((isa_and_nonnull<arith::MulIOp>(lhsDefOp)) ||
        (isa_and_nonnull<arith::MulIOp>(rhsDefOp)))
      return failure();

    rewriter.replaceOpWithNewOp<aievec::aie1::AddOp>(
        addOp, resType, lhs, rhs,
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xsquare=*/"",
        /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"", /*zsquare=*/"");
    return success();
  }
};

using LowerVectorAddFOpToAIEVecAddOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::AddFOp, aievec::aie1::AddOp>;
using LowerVectorMulFOpToAIEVecMulOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::MulFOp, aievec::aie1::MulOp>;
using LowerVectorSubIOpToAIEVecSubOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::SubIOp, aievec::aie1::SubOp>;
using LowerVectorSubFOpToAIEVecSubOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::SubFOp, aievec::aie1::SubOp>;

struct LowerVectorMulIOpToAIEVecMulOp : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::MulIOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = dyn_cast<VectorType>(mulOp.getType());
    if (!resTy)
      return failure();
    auto accTy = getVectorOpDestType(resTy, /*AIE2 =*/false);
    auto newMulOp = aievec::aie1::MulOp::create(
        rewriter, mulOp.getLoc(), accTy, adaptor.getLhs(), adaptor.getRhs());
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, mulOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        mulOp, resTy, newMulOp.getResult(), shiftParamOp.getResult());
    return success();
  }
};

template <typename SrcOpTy, typename DstOpTy>
struct LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp
    : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    // A set recording the vector lane size and element width we are supporting
    // for AIE2.
    llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
    laneSizeElWidthPairSet.insert({64, 8});
    laneSizeElWidthPairSet.insert({32, 16});
    laneSizeElWidthPairSet.insert({16, 32});
    laneSizeElWidthPairSet.insert({32, 32});

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsDefOp = lhs.getDefiningOp();
    auto rhsDefOp = rhs.getDefiningOp();
    // Check if this is part of a MAC/FMA pattern (mul + add).
    // We only skip conversion if BOTH operands could potentially be part of an
    // FMA pattern (i.e., neither is a constant). Constants can never be the
    // multiply result in an FMA, so we should allow conversion in those cases.
    bool lhsIsMul = lhsDefOp && (isa<arith::MulIOp>(lhsDefOp) ||
                                 isa<arith::MulFOp>(lhsDefOp));
    bool rhsIsMul = rhsDefOp && (isa<arith::MulIOp>(rhsDefOp) ||
                                 isa<arith::MulFOp>(rhsDefOp));
    bool lhsIsConst = lhsDefOp && isa<arith::ConstantOp>(lhsDefOp);
    bool rhsIsConst = rhsDefOp && isa<arith::ConstantOp>(rhsDefOp);

    // Only fail if we have a multiply that could be part of FMA, and the other
    // operand is NOT a constant
    if ((lhsIsMul && !rhsIsConst) || (rhsIsMul && !lhsIsConst))
      return failure();

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    // Integer cases
    if (isa<IntegerType>(scalarType)) {
      if (!laneSizeElWidthPairSet.count(
              std::make_pair(laneSize, resultElWidth)))
        return failure();

      // If the ops are defined without extension ops and with supported data
      // type, the arith::AddI or arith::SubI can be directly replaced with
      // aievec::AddElem or aievec::SubElem.
      if (!lhsDefOp && !rhsDefOp) {
        if (laneSize * resultElWidth == 512) {
          rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs,
                                               rhs);
          return success();
        }
        return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs, resultType,
                                                srcOp);
      }

      // If element width is 32, we need to consider sign extension cases
      if (resultElWidth == 32) {
        auto lhsExt = getSourceOfWideningOp(lhs).value_or(nullptr);
        auto rhsExt = getSourceOfWideningOp(rhs).value_or(nullptr);

        if (!lhsExt && !rhsExt) {
          if (laneSize * resultElWidth == 512) {
            rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs,
                                                 rhs);
            return success();
          }
          return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                  resultType, srcOp);
        }

        if (lhsExt && rhsExt) {
          auto lval = lhsExt;
          auto rval = rhsExt;
          VectorType lSrcType = cast<VectorType>(lval.getType());

          Type accType = getVectorOpDestType(lSrcType, /*AIE2 =*/true);
          auto lUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
          auto rUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);
          auto elemOp = DstOpTy::create(
              rewriter, srcOp.getLoc(), lUpsOp->getResult(0).getType(),
              lUpsOp->getResult(0), rUpsOp->getResult(0));
          rewriter.replaceOpWithNewOp<aievec::CastOp>(
              srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
          return success();
        }

        if (!lhsExt || !rhsExt) {
          auto lval = lhsExt ? lhsExt : lhs;
          auto rval = rhsExt ? rhsExt : rhs;
          auto extVal = lhsExt ? lval : rval;
          VectorType vType = cast<VectorType>(extVal.getType());
          unsigned bitWidth = vType.getElementType().getIntOrFloatBitWidth();

          if (bitWidth != 8 && bitWidth != 16) {
            return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                    resultType, srcOp);
          }

          if (bitWidth * laneSize != 256) {
            return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                    resultType, srcOp);
          }

          Type accType = nullptr;

          if (bitWidth == 8) {
            accType = getVectorOpDestType(vType, /*AIE2 =*/true);
            Value valToUps = lhsExt ? lval : rval;
            Value valToCast = lhsExt ? rval : lval;
            auto upsOp = aievec::UPSOp::create(rewriter, srcOp.getLoc(),
                                               accType, valToUps);
            auto castOp =
                aievec::CastOp::create(rewriter, srcOp.getLoc(), resultType,
                                       valToCast, /*isResAcc*/ true);
            Value lhsToElemOp =
                lhsExt ? upsOp->getResult(0) : castOp->getResult(0);
            Value rhsToElemOp =
                lhsExt ? castOp->getResult(0) : upsOp->getResult(0);
            auto elemOp = DstOpTy::create(rewriter, srcOp.getLoc(),
                                          upsOp->getResult(0).getType(),
                                          lhsToElemOp, rhsToElemOp);
            rewriter.replaceOpWithNewOp<aievec::CastOp>(
                srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
            return success();
          }

          if (bitWidth == 16) {
            accType = getVectorOpDestType(resultType, /*AIE2 =*/true);
            auto lUpsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
            auto rUpsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);

            auto elemOp = DstOpTy::create(
                rewriter, srcOp.getLoc(), lUpsOp->getResult(0).getType(),
                lUpsOp->getResult(0), rUpsOp->getResult(0));

            auto shiftParamOp = arith::ConstantOp::create(
                rewriter, srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
            rewriter.replaceOpWithNewOp<aievec::SRSOp>(
                srcOp, srcOp.getType(), elemOp.getResult(),
                shiftParamOp.getResult());
            return success();
          }
        }
      } else {
        rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs, rhs);
        return success();
      }
    }
    // Float types
    else {
      if (laneSize != 16 && laneSize != 32)
        return failure();

      // v32f32: split into two v16f32 ops
      if (laneSize == 32 && resultElWidth == 32) {
        VectorType halfType = createVectorType(16, scalarType);

        splitWideBinaryVectorOp<SrcOpTy>(
            srcOp, lhs, rhs, halfType, resultType, rewriter,
            [](Value lhsHalf, Value rhsHalf, Location loc,
               ConversionPatternRewriter &rewriter) -> Value {
              VectorType halfVecType = cast<VectorType>(lhsHalf.getType());
              // For f32, use cast to acc, add_elem/sub_elem, cast back
              auto lCastOp = aievec::CastOp::create(rewriter, loc, halfVecType,
                                                    lhsHalf, /*isResAcc*/ true);
              auto rCastOp = aievec::CastOp::create(rewriter, loc, halfVecType,
                                                    rhsHalf, /*isResAcc*/ true);
              auto elemOp = DstOpTy::create(
                  rewriter, loc, lCastOp->getResult(0).getType(),
                  lCastOp->getResult(0), rCastOp->getResult(0));
              auto resCastOp = aievec::CastOp::create(
                  rewriter, loc, halfVecType, elemOp.getResult(),
                  /*isResAcc*/ false);
              return resCastOp.getResult();
            });
        return success();
      }

      // v32bf16: split into two v16bf16 ops
      if (laneSize == 32 && resultElWidth == 16) {
        VectorType halfType = createVectorType(16, scalarType);

        splitWideBinaryVectorOp<SrcOpTy>(
            srcOp, lhs, rhs, halfType, resultType, rewriter,
            [](Value lhsHalf, Value rhsHalf, Location loc,
               ConversionPatternRewriter &rewriter) -> Value {
              VectorType halfVecType = cast<VectorType>(lhsHalf.getType());
              Type accType = getVectorOpDestType(halfVecType, /*AIE2 =*/true);
              auto lUpsOp =
                  aievec::UPSOp::create(rewriter, loc, accType, lhsHalf);
              auto rUpsOp =
                  aievec::UPSOp::create(rewriter, loc, accType, rhsHalf);
              auto elemOp =
                  DstOpTy::create(rewriter, loc, lUpsOp->getResult(0).getType(),
                                  lUpsOp->getResult(0), rUpsOp->getResult(0));
              auto shiftParamOp = arith::ConstantOp::create(
                  rewriter, loc, rewriter.getI32IntegerAttr(0));
              auto srsOp = aievec::SRSOp::create(rewriter, loc, halfVecType,
                                                 elemOp.getResult(),
                                                 shiftParamOp.getResult());
              return srsOp.getResult();
            });
        return success();
      }

      // Now we know laneSize == 16 for remaining float cases
      // v16float or v16bf16 with extension op case
      if (resultElWidth == 32) {
        if (!lhsDefOp && !rhsDefOp) {
          return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                  resultType, srcOp);
        }

        auto lhsExt = getSourceOfWideningOp(lhs).value_or(nullptr);
        auto rhsExt = getSourceOfWideningOp(rhs).value_or(nullptr);
        // v16float
        if (!lhsExt && !rhsExt) {
          return genAddElemAIE2<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                  resultType, srcOp);
        }

        // v16bf16 with two extension ops
        if (lhsExt && rhsExt) {
          auto lval = lhsExt;
          auto rval = rhsExt;
          VectorType vType = cast<VectorType>(lval.getType());

          Type accType = getVectorOpDestType(vType, /*AIE2 =*/true);
          auto lUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
          auto rUpsOp =
              aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);
          auto elemOp = DstOpTy::create(
              rewriter, srcOp.getLoc(), lUpsOp->getResult(0).getType(),
              lUpsOp->getResult(0), rUpsOp->getResult(0));
          rewriter.replaceOpWithNewOp<aievec::CastOp>(srcOp, srcOp.getType(),
                                                      elemOp.getResult());
          return success();
        }

        // v16bf16 with one extension op
        if (!lhsExt || !rhsExt) {
          auto lval = lhsExt ? lhsExt : lhs;
          auto rval = rhsExt ? rhsExt : rhs;
          auto extVal = lhsExt ? lval : rval;
          VectorType vType = cast<VectorType>(extVal.getType());
          Type accType = getVectorOpDestType(vType, /*AIE2 =*/true);

          aievec::UPSOp upsOp;
          aievec::CastOp castOp;
          if (lhsExt) {
            upsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lval);
            castOp = aievec::CastOp::create(rewriter, srcOp.getLoc(),
                                            resultType, rval,
                                            /*isResAcc*/ true);
          } else {
            upsOp =
                aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rval);
            castOp = aievec::CastOp::create(rewriter, srcOp.getLoc(),
                                            resultType, lval,
                                            /*isResAcc*/ true);
          }

          auto elemOp = DstOpTy::create(
              rewriter, srcOp.getLoc(), upsOp->getResult(0).getType(),
              upsOp->getResult(0), castOp->getResult(0));

          rewriter.replaceOpWithNewOp<aievec::CastOp>(
              srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);

          return success();
        }
      }

      // v16bfloat16
      Type accType = getVectorOpDestType(resultType, /*AIE2 =*/true);
      auto lUpsOp =
          aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, lhs);
      auto rUpsOp =
          aievec::UPSOp::create(rewriter, srcOp.getLoc(), accType, rhs);
      auto elemOp = DstOpTy::create(rewriter, srcOp.getLoc(),
                                    lUpsOp->getResult(0).getType(),
                                    lUpsOp->getResult(0), rUpsOp->getResult(0));
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          srcOp, srcOp.getType(), elemOp.getResult(), shiftParamOp.getResult());

      return success();
    }

    return failure();
  }
};

using LowerVectorAddIOpToAIEVecAddElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::AddIOp,
                                                    aievec::AddElemOp>;
using LowerVectorSubIOpToAIEVecSubElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::SubIOp,
                                                    aievec::SubElemOp>;
using LowerVectorAddFOpToAIEVecAddElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::AddFOp,
                                                    aievec::AddElemOp>;
using LowerVectorSubFOpToAIEVecSubElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::SubFOp,
                                                    aievec::SubElemOp>;

template <typename SrcOpTy, typename DstOpTy>
struct LowerVectorMinMaxOpToAIEVecMinMaxOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    // A set recording the element width we are supporting for AIE2.
    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (!elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512)
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(),
                                         adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

using LowerVectorMinSIOpToAIEVecMinOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MinSIOp, aievec::MinOp>;
using LowerVectorMaxSIOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaxSIOp, aievec::MaxOp>;
// Promote scalar arith.maxsi/arith.minsi to vector aievec.max/aievec.min
// to avoid the AIE2 G_SELECT legalizer crash on scalar i32 select.
template <typename SrcOpTy, typename DstOpTy>
struct LowerScalarMinMaxToAIEVecMinMaxOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match scalar integer types (reject vectors)
    Type resultType = srcOp.getType();
    if (isa<VectorType>(resultType))
      return failure();

    auto intType = dyn_cast<IntegerType>(resultType);
    if (!intType)
      return failure();

    unsigned elWidth = intType.getWidth();
    if (elWidth != 8 && elWidth != 16 && elWidth != 32)
      return failure();

    unsigned numLanes = 512 / elWidth;
    VectorType vecType = createVectorType(numLanes, intType);
    Location loc = srcOp.getLoc();

    // Broadcast both scalars to 512-bit vectors
    auto lhsBcast = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                      adaptor.getLhs());
    auto rhsBcast = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                      adaptor.getRhs());

    // Apply vector min/max
    auto vecOp = DstOpTy::create(rewriter, loc, vecType, lhsBcast.getResult(),
                                 rhsBcast.getResult());

    // Extract element 0 back to scalar
    auto zeroIdx =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(
        srcOp, intType, vecOp.getResult(), zeroIdx.getResult());
    return success();
  }
};

using LowerScalarMinSIOpToAIEVecMinOp =
    LowerScalarMinMaxToAIEVecMinMaxOp<arith::MinSIOp, aievec::MinOp>;
using LowerScalarMaxSIOpToAIEVecMaxOp =
    LowerScalarMinMaxToAIEVecMinMaxOp<arith::MaxSIOp, aievec::MaxOp>;

using LowerVectorMinimumFOpToAIEVecMinOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MinimumFOp, aievec::MinOp>;
using LowerVectorMaximumFOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaximumFOp, aievec::MaxOp>;
using LowerVectorMaxNumFFOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaxNumFOp, aievec::MaxOp>;

template <typename SrcOpTy, typename CmpTy>
struct LowerVectorCmpOpToAIEVecCmpOp : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType lhsType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!lhsType)
      return failure();

    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = lhsType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    if (!elWidthSet.count(elWidth) || laneSize * elWidth != 512)
      return failure();

    // Unsigned int and unsigned long long are acceptable type.
    Type type =
        mlir::IntegerType::get(srcOp.getContext(), laneSize <= 32 ? 32 : 64,
                               mlir::IntegerType::Unsigned);

    Location loc = srcOp.getLoc();
    Value lhs = srcOp.getLhs();
    Value rhs = srcOp.getRhs();
    CmpTy pred = srcOp.getPredicate();

    arith::CmpIPredicate ipred = convertToIntegerPredicate(pred);

    aievec::CmpOp aieCmpOp =
        createCmpOpAIE2(rewriter, ipred, loc, type, lhs, rhs);

    if (!aieCmpOp)
      return failure();

    VectorType resultType = dyn_cast<VectorType>(srcOp.getResult().getType());
    // Convert vector i1 type to unsigned interger type by built-in unrealized
    // conversion cast op.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        srcOp, resultType, aieCmpOp.getResult());

    return success();
  }
};

using LowerVectorCmpIOpToAIEVecCmpOp =
    LowerVectorCmpOpToAIEVecCmpOp<arith::CmpIOp, CmpIPredicate>;
using LowerVectorCmpFOpToAIEVecCmpOp =
    LowerVectorCmpOpToAIEVecCmpOp<arith::CmpFOp, CmpFPredicate>;

struct LowerVectorSelectOpToAIEVecSelOp : OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (!elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512)
      return failure();

    Type type =
        mlir::IntegerType::get(srcOp.getContext(), laneSize <= 32 ? 32 : 64,
                               mlir::IntegerType::Unsigned);

    auto convertOp = UnrealizedConversionCastOp::create(
        rewriter, srcOp.getLoc(), type, adaptor.getCondition());

    rewriter.replaceOpWithNewOp<aievec::SelOp>(
        srcOp, srcOp.getResult().getType(), srcOp.getTrueValue(),
        srcOp.getFalseValue(), convertOp.getResult(0));

    return success();
  }
};

struct LowerVectorReductionMinOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::MINSI &&
                                     kind != vector::CombiningKind::MINUI &&
                                     kind != vector::CombiningKind::MINIMUMF &&
                                     kind != vector::CombiningKind::MINNUMF)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);
    unsigned vectorSize = laneSize * elWidth;

    // Support 512-bit vectors directly, and 256-bit bf16 vectors by padding
    if (vectorSize != 512 && !(vectorSize == 256 && scalarType.isBF16()))
      return failure();

    Location loc = srcOp.getLoc();
    Value inputVec = srcOp.getVector();

    // For 256-bit bf16 (v16bf16), pad to 512-bit (v32bf16) with +inf
    if (vectorSize == 256) {
      std::tie(inputVec, laneSize) = padV16ToV32WithInfinity(
          rewriter, loc, srcOp.getVector(), scalarType, /*negativeInf=*/false);
    }

    int shiftIndex = laneSize / 2;
    auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::MinOp>(
        rewriter, srcOp, shiftIndex, inputVec);

    if (srcOp.getAcc()) {
      Value reduceResult = reduceResultOp.getResult();
      Value acc = srcOp.getAcc();

      // If accumulator is bf16, use the high-level helper for bf16->f32->bf16
      if (acc.getType().isBF16()) {
        // Define the min operation to be performed in f32
        auto minOpBuilder = [&](Value lhs, Value rhs) -> Value {
          auto cmpOp = arith::CmpFOp::create(
              rewriter, srcOp.getLoc(), arith::CmpFPredicate::OLT, lhs, rhs);
          return arith::SelectOp::create(rewriter, srcOp.getLoc(), cmpOp, lhs,
                                         rhs);
        };

        // Use helper to handle bf16->f32 conversion, perform min, and convert
        // back
        performBF16BinaryOpInF32(reduceResult, acc, srcOp, srcOp.getLoc(),
                                 rewriter, minOpBuilder);
      } else {
        // Non-bf16 path: perform min using cmpf and select
        auto cmpOp =
            arith::CmpFOp::create(rewriter, srcOp.getLoc(),
                                  arith::CmpFPredicate::OLT, reduceResult, acc);
        rewriter.replaceOpWithNewOp<arith::SelectOp>(srcOp, cmpOp, reduceResult,
                                                     acc);
      }
    } else {
      rewriter.replaceOp(srcOp, reduceResultOp);
    }
    return success();
  }
};

struct LowerVectorReductionMaxOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::MAXSI &&
                                     kind != vector::CombiningKind::MAXUI &&
                                     kind != vector::CombiningKind::MAXIMUMF &&
                                     kind != vector::CombiningKind::MAXNUMF)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);
    unsigned vectorSize = laneSize * elWidth;

    // Support 512-bit vectors directly, and 256-bit bf16 vectors by padding
    // Only bf16 is supported for the 256-bit padding path (not f16)
    if (vectorSize != 512 && !(vectorSize == 256 && scalarType.isBF16()))
      return failure();

    Location loc = srcOp.getLoc();
    Value inputVec = srcOp.getVector();

    // For 256-bit bf16 (v16bf16), pad to 512-bit (v32bf16) with -inf
    if (vectorSize == 256) {
      std::tie(inputVec, laneSize) = padV16ToV32WithInfinity(
          rewriter, loc, srcOp.getVector(), scalarType, /*negativeInf=*/true);
    }

    int shiftIndex = laneSize / 2;
    auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::MaxOp>(
        rewriter, srcOp, shiftIndex, inputVec);

    if (srcOp.getAcc()) {
      Value reduceResult = reduceResultOp.getResult();
      Value acc = srcOp.getAcc();

      // If accumulator is bf16, use the high-level helper for bf16->f32->bf16
      if (acc.getType().isBF16()) {
        // Define the max operation to be performed in f32
        auto maxOpBuilder = [&](Value lhs, Value rhs) -> Value {
          auto cmpOp = arith::CmpFOp::create(
              rewriter, srcOp.getLoc(), arith::CmpFPredicate::OGT, lhs, rhs);
          return arith::SelectOp::create(rewriter, srcOp.getLoc(), cmpOp, lhs,
                                         rhs);
        };

        // Use helper to handle bf16->f32 conversion, perform max, and convert
        // back
        performBF16BinaryOpInF32(reduceResult, acc, srcOp, srcOp.getLoc(),
                                 rewriter, maxOpBuilder);
      } else {
        // Non-bf16 path: perform max directly
        auto cmpOp =
            arith::CmpFOp::create(rewriter, srcOp.getLoc(),
                                  arith::CmpFPredicate::OGT, reduceResult, acc);
        rewriter.replaceOpWithNewOp<arith::SelectOp>(srcOp, cmpOp, reduceResult,
                                                     acc);
      }
    } else {
      rewriter.replaceOp(srcOp, reduceResultOp);
    }
    return success();
  }
};

struct LowerVectorReductionAddIntOp : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);
    llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
    laneSizeElWidthPairSet.insert({64, 8});
    laneSizeElWidthPairSet.insert({32, 16});
    laneSizeElWidthPairSet.insert({32, 32});
    laneSizeElWidthPairSet.insert({16, 32});

    if (!isa<IntegerType>(scalarType) ||
        !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
      return failure();

    int shiftIndex = laneSize / 2;
    if (laneSize == 32 && elWidth == 32) {
      Location loc = srcOp.getLoc();
      VectorType vecType = createVectorType(laneSize / 2, scalarType);

      auto lExtOp =
          aievec::ExtOp::create(rewriter, loc, vecType, srcOp.getVector(), 0);
      auto rExtOp =
          aievec::ExtOp::create(rewriter, loc, vecType, srcOp.getVector(), 1);
      auto addElemOp =
          aievec::AddElemOp::create(rewriter, loc, lExtOp.getResult().getType(),
                                    lExtOp.getResult(), rExtOp.getResult());
      shiftIndex /= 2;
      auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::AddElemOp>(
          rewriter, srcOp, shiftIndex, addElemOp.getResult());
      if (srcOp.getAcc())
        rewriter.replaceOpWithNewOp<arith::AddIOp>(
            srcOp, reduceResultOp.getResult(), srcOp.getAcc());
      else
        rewriter.replaceOp(srcOp, reduceResultOp);
    } else {
      auto reduceResultOp = generateAIEVecOpsForReductionOp<aievec::AddElemOp>(
          rewriter, srcOp, shiftIndex, srcOp.getVector());
      if (srcOp.getAcc())
        rewriter.replaceOpWithNewOp<arith::AddIOp>(
            srcOp, reduceResultOp.getResult(), srcOp.getAcc());
      else
        rewriter.replaceOp(srcOp, reduceResultOp);
    }

    return success();
  }
};

struct LowerVectorReductionAddFloatOp
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    if (!isa<FloatType>(scalarType) || laneSize != 16 || elWidth != 32)
      return failure();

    int shiftIndex = laneSize / 2;
    assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
           "shiftIndex must be power of 2");

    Location loc = srcOp.getLoc();
    Value curValue = srcOp.getVector();
    aievec::CastOp curOp = nullptr;

    for (int id = shiftIndex; id > 0; id /= 2) {
      auto constOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(id * elWidth / 8));

      auto shiftBytesOp = aievec::ShiftOp::create(
          rewriter, loc, vType, curValue, curValue, constOp.getResult());

      auto lCastOp = aievec::CastOp::create(rewriter, loc, vType, curValue,
                                            /*isResAcc*/ true);
      auto rCastOp =
          aievec::CastOp::create(rewriter, loc, vType, shiftBytesOp.getResult(),
                                 /*isResAcc*/ true);
      auto elemOp = aievec::AddElemOp::create(
          rewriter, loc, lCastOp.getResult().getType(), lCastOp.getResult(),
          rCastOp.getResult());
      curOp = aievec::CastOp::create(rewriter, loc, vType, elemOp.getResult(),
                                     /*isResAcc*/ false);
      curValue = curOp.getResult();
    }

    auto zeroConstOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto reduceResultOp = aievec::ExtElemOp::create(
        rewriter, srcOp.getLoc(), scalarType, curOp, zeroConstOp.getResult());

    if (srcOp.getAcc())
      rewriter.replaceOpWithNewOp<arith::AddFOp>(
          srcOp, reduceResultOp.getResult(), srcOp.getAcc());
    else
      rewriter.replaceOp(srcOp, reduceResultOp);
    return success();
  }
};

// AIE2-specific bf16 ADD reduction - requires concat to v32bf16 before ext_elem
// due to aie2 ext_elem limitation
struct LowerVectorReductionAddBfloat16OpAIE2
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD) {
      return failure();
    }

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    // Support both lane=16 and lane=32 for bf16
    if (!isa<FloatType>(scalarType) || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16) {
      return failure();
    }

    Location loc = srcOp.getLoc();
    Value curValue = srcOp.getVector();
    VectorType currentVType = vType; // Track current working vector type

    // For lane=32, split into two v16bf16 halves and add them
    if (laneSize == 32) {
      VectorType halfType = createVectorType(laneSize / 2, scalarType);
      auto lowerHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 0);
      auto upperHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 1);

      Type accType = getVectorOpDestType(halfType, /*AIE2 =*/true);
      auto lUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, lowerHalf.getResult());
      auto rUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, upperHalf.getResult());
      auto addElemOp = aievec::AddElemOp::create(
          rewriter, loc, accType, lUpsOp.getResult(), rUpsOp.getResult());
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));
      auto srsOp =
          aievec::SRSOp::create(rewriter, loc, halfType, addElemOp.getResult(),
                                shiftParamOp.getResult());
      curValue = srsOp.getResult();
      currentVType = halfType; // Update to v16bf16 after split
    }

    int shiftIndex = 8; // Always 8 since we work with v16bf16
    Type accType = getVectorOpDestType(cast<VectorType>(curValue.getType()),
                                       /*AIE2 =*/true);
    unsigned accWidth =
        dyn_cast<VectorType>(accType).getElementType().getIntOrFloatBitWidth();

    auto upsOp = aievec::UPSOp::create(rewriter, loc, accType, curValue);
    curValue = upsOp.getResult();

    aievec::AddElemOp curOp = nullptr;

    for (int id = shiftIndex; id > 0; id /= 2) {
      auto constOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(id * accWidth / 8));
      auto shiftBytesOp = aievec::ShiftOp::create(
          rewriter, loc, accType, curValue, curValue, constOp, true);
      curOp = aievec::AddElemOp::create(rewriter, loc, accType, curValue,
                                        shiftBytesOp.getResult());
      curValue = curOp.getResult();
    }

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
    // Use currentVType instead of vType to ensure lane count matches
    auto srsOp =
        aievec::SRSOp::create(rewriter, loc, currentVType, curOp.getResult(),
                              shiftParamOp.getResult());

    // AIE2 ext_elem requires v32bf16, so concat v16bf16 to v32bf16
    VectorType vecType = createVectorType(32, scalarType);
    SmallVector<Value> concatSources = {srsOp.getResult(), srsOp.getResult()};
    auto concatOp =
        aievec::ConcatOp::create(rewriter, loc, vecType, concatSources);

    auto zeroConstOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto reduceResultOp =
        aievec::ExtElemOp::create(rewriter, srcOp.getLoc(), scalarType,
                                  concatOp, zeroConstOp.getResult());

    if (srcOp.getAcc())
      rewriter.replaceOpWithNewOp<arith::AddFOp>(
          srcOp, reduceResultOp.getResult(), srcOp.getAcc());
    else
      rewriter.replaceOp(srcOp, reduceResultOp);
    return success();
  }
};

// AIE2P-specific bf16 ADD reduction - can extract directly from v16bf16
struct LowerVectorReductionAddBfloat16OpAIE2P
    : OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto kind = srcOp.getKind(); kind != vector::CombiningKind::ADD)
      return failure();

    auto vType = cast<VectorType>(srcOp.getVector().getType());
    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    // Support both lane=16 and lane=32 for bf16
    if (!isa<FloatType>(scalarType) || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16)
      return failure();

    Location loc = srcOp.getLoc();
    int shiftIndex = laneSize / 2;
    Value inputToReduce = srcOp.getVector();

    // For lane=32, split into two v16bf16 halves, add them, then reduce
    if (laneSize == 32) {
      VectorType halfType = createVectorType(laneSize / 2, scalarType);

      // Extract lower and upper halves
      auto lowerHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 0);
      auto upperHalf =
          aievec::ExtOp::create(rewriter, loc, halfType, srcOp.getVector(), 1);

      // Add the two halves together
      Type accType = getVectorOpDestType(halfType, /*AIE2 =*/true);
      auto lUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, lowerHalf.getResult());
      auto rUpsOp =
          aievec::UPSOp::create(rewriter, loc, accType, upperHalf.getResult());
      auto addElemOp = aievec::AddElemOp::create(
          rewriter, loc, accType, lUpsOp.getResult(), rUpsOp.getResult());
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));
      auto srsOp =
          aievec::SRSOp::create(rewriter, loc, halfType, addElemOp.getResult(),
                                shiftParamOp.getResult());

      inputToReduce = srsOp.getResult();
      shiftIndex = 8;
    }

    // Perform reduction using utility
    Type accType = getVectorOpDestType(
        cast<VectorType>(inputToReduce.getType()), /*AIE2 =*/true);
    unsigned accWidth =
        dyn_cast<VectorType>(accType).getElementType().getIntOrFloatBitWidth();

    auto upsOp = aievec::UPSOp::create(rewriter, loc, accType, inputToReduce);
    Value curValue = upsOp.getResult();

    aievec::AddElemOp curOp = nullptr;
    for (int id = shiftIndex; id > 0; id /= 2) {
      auto constOp = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(id * accWidth / 8));
      auto shiftBytesOp = aievec::ShiftOp::create(
          rewriter, loc, accType, curValue, curValue, constOp, true);
      curOp = aievec::AddElemOp::create(rewriter, loc, accType, curValue,
                                        shiftBytesOp.getResult());
      curValue = curOp.getResult();
    }

    // Extract element 0 from the f32 accumulator
    // The loop has already fully reduced the vector to a single value in
    // element 0
    auto zeroConstOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto extractedF32 = aievec::ExtElemOp::create(
        rewriter, srcOp.getLoc(), rewriter.getF32Type(), curOp.getResult(),
        zeroConstOp.getResult());

    // Convert extracted f32 to bf16
    auto reduceResultBF16 = arith::TruncFOp::create(
        rewriter, srcOp.getLoc(), scalarType, extractedF32.getResult());

    if (srcOp.getAcc())
      rewriter.replaceOpWithNewOp<arith::AddFOp>(srcOp, reduceResultBF16,
                                                 srcOp.getAcc());
    else
      rewriter.replaceOp(srcOp, reduceResultBF16);
    return success();
  }
};

// Convert a `vector.extract_strided_slice` op on 1D vectors into an
// `aievec.select` + `aievec.ext` op.
struct LowerVectorExtractStridedSliceOpAIEv1Pattern
    : OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vType = extractOp.getSourceVectorType();
    if (vType.getRank() != 1)
      return failure();

    int64_t stride = cast<IntegerAttr>(adaptor.getStrides()[0]).getInt();
    if (stride != 1)
      return failure();

    // AIE doesn't support select operations on i8
    if (getElementSizeInBits(vType) == 8)
      return extractOp.emitError()
             << "AIEv1 doesn't support select ops on int8 types";

    // We only accept the case where we are extracting a slice half the size of
    // the input vector.
    int64_t size = cast<IntegerAttr>(adaptor.getSizes()[0]).getInt();
    if (vType.getNumElements() != 2 * size)
      return failure();

    int64_t offset = cast<IntegerAttr>(adaptor.getOffsets()[0]).getInt();
    auto selectOp = aievec::aie1::SelectOp::create(
        rewriter, extractOp.getLoc(), vType, adaptor.getSource(),
        buildAttributeListForRotationSelectOp(rewriter, vType, offset));
    rewriter.replaceOpWithNewOp<aievec::aie1::ExtOp>(
        extractOp, extractOp.getType(), selectOp.getResult(),
        rewriter.getI8IntegerAttr(0));
    return success();
  }
};

// Convert a `vector.extract_strided_slice` op on 1D vectors into an
// `aievec.shift` op.
struct LowerVectorExtractStridedSliceOpAIE2Pattern
    : OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vType = cast<VectorType>(adaptor.getSource().getType());
    if (vType.getRank() != 1)
      return failure();

    int64_t stride = cast<IntegerAttr>(adaptor.getStrides()[0]).getInt();
    if (stride != 1)
      return failure();

    // We only accept the case where we are extracting a slice half the size of
    // the input vector.
    int64_t size = cast<IntegerAttr>(adaptor.getSizes()[0]).getInt();
    if (vType.getNumElements() != 2 * size)
      return failure();

    auto shortVecType = cast<VectorType>(extractOp.getResult().getType());
    auto bottomHalf =
        aievec::ExtOp::create(rewriter, extractOp.getLoc(), shortVecType,
                              adaptor.getSource(), rewriter.getI8IntegerAttr(0))
            .getResult();
    auto topHalf =
        aievec::ExtOp::create(rewriter, extractOp.getLoc(), shortVecType,
                              adaptor.getSource(), rewriter.getI8IntegerAttr(1))
            .getResult();
    int64_t offset = cast<IntegerAttr>(adaptor.getOffsets()[0]).getInt();
    int32_t shiftBytes = offset * getElementSizeInBits(vType) / 8;
    auto shiftBytesConstOp = arith::ConstantOp::create(
        rewriter, extractOp.getLoc(), rewriter.getIntegerType(32),
        rewriter.getI32IntegerAttr(shiftBytes));
    rewriter.replaceOpWithNewOp<aievec::ShiftOp>(
        extractOp, shortVecType, bottomHalf, topHalf, shiftBytesConstOp);

    return success();
  }
};

// Replaces a short UPD op with a wide one followed by an ext op of the bottom
// half.
struct ExpandUPDToUPDAndExtPattern : OpConversionPattern<aievec::UPDOp> {
  using OpConversionPattern::OpConversionPattern;

  ExpandUPDToUPDAndExtPattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(aievec::UPDOp updOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify that we haven't already expanded this one
    if (updOp->hasOneUse() && isa<aievec::ExtOp>(*updOp->getUsers().begin()))
      return failure();

    auto vecType = cast<VectorType>(updOp.getType());
    SmallVector<int64_t, 4> vecShape(vecType.getShape().begin(),
                                     vecType.getShape().end());
    vecShape[vecType.getRank() - 1] *= 2;
    auto longVecType = VectorType::get(vecShape, vecType.getElementType());
    auto newUpdOp = aievec::UPDOp::create(
        rewriter, updOp.getLoc(), longVecType, adaptor.getSource(),
        adaptor.getIndices(), adaptor.getOffset(), adaptor.getIndex(),
        adaptor.getVector());
    rewriter.replaceOpWithNewOp<aievec::ExtOp>(
        updOp, vecType, newUpdOp.getResult(), rewriter.getI8IntegerAttr(0));

    return success();
  }
};

// Replaces a wide UPD op followed by an ext op of the bottom half with a short
// UPD op.
struct FuseExtIntoUPDPattern : OpConversionPattern<aievec::ExtOp> {
  using OpConversionPattern::OpConversionPattern;

  FuseExtIntoUPDPattern(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(aievec::ExtOp extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify we are extracting the lower half...
    if (extOp.getIndex() != 0)
      return failure();
    // ...of a UPDOp
    auto updOp = dyn_cast<aievec::UPDOp>(extOp.getSource().getDefiningOp());
    if (!updOp)
      return failure();

    // Verify that this is a direct upd -> ext pattern
    if (!updOp->hasOneUse())
      return failure();

    rewriter.replaceOpWithNewOp<aievec::UPDOp>(
        extOp, extOp.getType(), updOp.getSource(), updOp.getIndices(),
        updOp.getOffset(), updOp.getIndex(), updOp.getVector());

    return success();
  }
};

// Convert math.exp to aievec.exp for AIE2P (will be further lowered to exp2
// intrinsic)
struct ConvertMathExpToAIEVecExpOpPattern : OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchExpOpForAIE2P(adaptor))
      return failure();

    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    rewriter.replaceOpWithNewOp<aievec::ExpOp>(expOp, srcType,
                                               adaptor.getOperand());
    return success();
  }
};

struct ComputeExpOpByLUTLLVMPattern : OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!matchExpOpForAIE2LUT(adaptor))
      return failure();

    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    unsigned laneSize = getVectorLaneSize(srcType);
    Location loc = expOp.getLoc();
    StringRef funcName = "getExpBf16";

    VectorType v16bf16Ty = mlir::VectorType::get({16}, rewriter.getBF16Type());
    VectorType v8i64Ty = mlir::VectorType::get({8}, rewriter.getI64Type());
    func::FuncOp fnOp = getOrInsertFuncDecl(
        rewriter, expOp->getParentWithTrait<OpTrait::SymbolTable>(), funcName,
        TypeRange{v16bf16Ty}, TypeRange{v8i64Ty});

    // Handle v32bf16 by splitting into two v16bf16 operations
    if (laneSize == 32) {
      splitWideUnaryVectorOp<math::ExpOp>(
          expOp, adaptor.getOperand(), v16bf16Ty, srcType, rewriter,
          [&fnOp](Value halfInput, Location loc,
                  ConversionPatternRewriter &rewriter) -> Value {
            VectorType v16bf16Ty =
                mlir::VectorType::get({16}, rewriter.getBF16Type());
            auto callOp = func::CallOp::create(rewriter, loc, fnOp,
                                               SmallVector<Value>{halfInput});
            Type accType = getVectorOpDestType(v16bf16Ty, /*AIE2 =*/true);
            auto resCastOp = vector::BitCastOp::create(rewriter, loc, accType,
                                                       callOp.getResults());
            auto shiftParamOp = arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(0));
            auto srsOp = aievec::SRSOp::create(rewriter, loc, v16bf16Ty,
                                               resCastOp.getResult(),
                                               shiftParamOp.getResult());
            return srsOp.getResult();
          });
      return success();
    }

    // Handle v16bf16 directly
    SmallVector<Value> expOperands = {adaptor.getOperand()};

    Type accTypeNative = getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto callOp = func::CallOp::create(rewriter, loc, fnOp, expOperands);
    auto resCastOp = vector::BitCastOp::create(rewriter, loc, accTypeNative,
                                               callOp.getResults());
    auto shiftParamOp =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        expOp, srcType, resCastOp.getResult(), shiftParamOp.getResult());

    return success();
  }
};
// Lower ExpOp to function call
struct ComputeExpOpByLUTPattern : OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp expOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchExpOpForAIE2LUT(adaptor))
      return failure();
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    StringRef includeName = "lut_based_ops.h";
    auto moduleOp = expOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(expOp);

    auto v16bf16OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, expOp.getLoc(), v16bf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> expOperands = {opaquedOperand};

    Type accTypeNative = getVectorOpDestType(srcType, /*AIE2 =*/true);
    Type v16accf32OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "v16accfloat");
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, expOp.getLoc(), TypeRange{v16accf32OpaqueTy}, "getExpBf16",
        nullptr, nullptr, expOperands);
    auto resCastOp = UnrealizedConversionCastOp::create(
        rewriter, expOp.getLoc(), accTypeNative, callOp.getResults());
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, expOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        expOp, srcType, resCastOp.getResult(0), shiftParamOp.getResult());

    return success();
  }
};

// Lower the inverse of a float to a function call (CPP backend)
// Convert the pattern-
//  %cst = arith.constant 1.000000e+00 : f32
//  %0 = arith.divf %cst, %arg1 : f32
//  %1 = arith.truncf %0 : f32 to bf16
// to -
//  %0 = emitc.call "getInvBf16"(%0) : f32 -> bf16;
struct ComputeInvOpByLUTPattern : OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp divOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getLhs().getType();
    if (!divOp->hasOneUse() || isa<VectorType>(srcType) ||
        !isa<FloatType>(srcType))
      return failure();

    if (!isNarrowingOp(*divOp->getUsers().begin()))
      return failure();

    auto fType = cast<FloatType>(srcType);
    if (fType.getWidth() != 32)
      return failure();

    auto constOp = dyn_cast<arith::ConstantOp>(divOp.getLhs().getDefiningOp());
    if (!constOp ||
        cast<FloatAttr>(constOp.getValue()).getValue().convertToDouble() !=
            1.0f)
      return failure();

    StringRef includeName = "lut_based_ops.h";
    auto moduleOp = divOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    auto truncOp = cast<arith::TruncFOp>(*divOp->getUsers().begin());

    rewriter.setInsertionPoint(truncOp);
    Type bf16OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "bfloat16");
    SmallVector<Value> invOperands = {adaptor.getRhs()};
    auto callOp = emitc::CallOpaqueOp::create(rewriter, truncOp.getLoc(),
                                              bf16OpaqueTy, "getInvBf16",
                                              nullptr, nullptr, invOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        truncOp, TypeRange{truncOp.getResult().getType()}, callOp.getResults());
    rewriter.eraseOp(divOp);

    return success();
  }
};

// Lower the inverse of a float to aievec.inv (LLVMIR backend for AIE2P)
// Supports both scalar f32 and vector<Nxf32> types.
// Convert the pattern-
//  %cst = arith.constant 1.000000e+00 : f32
//  %0 = arith.divf %cst, %arg1 : f32
// to -
//  %0 = aievec.inv %arg1 : f32
// Also supports:
//  %cst = arith.constant dense<1.0> : vector<16xf32>
//  %0 = arith.divf %cst, %arg1 : vector<16xf32>
// to -
//  %0 = aievec.inv %arg1 : vector<16xf32>
struct ConvertDivFToAIEVecInvOpPattern : OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp divOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = adaptor.getLhs().getType();

    // Check if LHS is defined by an operation
    auto *defOp = divOp.getLhs().getDefiningOp();
    if (!defOp)
      return failure();

    auto constOp = dyn_cast<arith::ConstantOp>(defOp);
    if (!constOp)
      return failure();

    // Handle scalar f32 case
    if (auto fType = dyn_cast<FloatType>(srcType)) {
      if (fType.getWidth() != 32)
        return failure();

      auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
      if (!floatAttr || !floatAttr.getValue().isExactlyValue(1.0))
        return failure();

      rewriter.replaceOpWithNewOp<aievec::InvOp>(divOp, srcType,
                                                 adaptor.getRhs());
      return success();
    }

    // Handle vector f32 case
    if (auto vecType = dyn_cast<VectorType>(srcType)) {
      auto elemType = vecType.getElementType();
      if (!elemType.isF32())
        return failure();

      // Check for supported vector sizes (16 or 32 lanes)
      unsigned laneSize = getVectorLaneSize(vecType);
      if (laneSize != 16 && laneSize != 32)
        return failure();

      // Check if it's a splat of 1.0
      auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
      if (!denseAttr || !denseAttr.isSplat())
        return failure();

      if (!denseAttr.getSplatValue<APFloat>().isExactlyValue(1.0))
        return failure();

      rewriter.replaceOpWithNewOp<aievec::InvOp>(divOp, vecType,
                                                 adaptor.getRhs());
      return success();
    }

    return failure();
  }
};

// Convert math.tanh to a function call to compute tanh(x) by look up tables
struct ComputeTanhOpByLUTPattern : OpConversionPattern<math::TanhOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::TanhOp tanhOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(tanhOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || laneSize != 16)
      return failure();

    StringRef includeName = "lut_based_ops.h";
    auto moduleOp = tanhOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(tanhOp);
    Type v16bf16OpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, tanhOp.getLoc(), v16bf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> tanhOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(rewriter, tanhOp.getLoc(),
                                              v16bf16OpaqueTy, "getTanhBf16",
                                              nullptr, nullptr, tanhOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        tanhOp, TypeRange{tanhOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.sqrt to a function call to compute sqrt(x) for v16bfloat16 and
// v32bfloat16 types
struct ComputeSqrtOpAIE2Pattern : OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SqrtOp sqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(sqrtOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = sqrtOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(sqrtOp);
    Type vLNbf16OpaqueTy;
    if (laneSize == 16)
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, sqrtOp.getLoc(), vLNbf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> sqrtOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, sqrtOp.getLoc(), TypeRange{vLNbf16OpaqueTy}, "getSqrtBf16",
        nullptr, nullptr, sqrtOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        sqrtOp, TypeRange{sqrtOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

struct ComputeRsqrtOpLLVMAIE2Pattern : OpConversionPattern<math::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(srcType);

    // Only support v16bf16 for LLVM backend
    if (!isa<FloatType>(scalarType) || laneSize != 16 || elWidth != 16)
      return failure();

    StringRef funcName = "getRsqrtBf16";

    VectorType v16bf16Ty = mlir::VectorType::get({16}, rewriter.getBF16Type());
    VectorType v8i64Ty = mlir::VectorType::get({8}, rewriter.getI64Type());
    func::FuncOp fnOp = getOrInsertFuncDecl(
        rewriter, rsqrtOp->getParentWithTrait<OpTrait::SymbolTable>(), funcName,
        TypeRange{v16bf16Ty}, TypeRange{v8i64Ty});

    SmallVector<Value> rsqrtOperands = {adaptor.getOperand()};

    Type accTypeNative = getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto callOp =
        func::CallOp::create(rewriter, rsqrtOp.getLoc(), fnOp, rsqrtOperands);
    auto resCastOp = vector::BitCastOp::create(
        rewriter, rsqrtOp.getLoc(), accTypeNative, callOp.getResults());
    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, rsqrtOp.getLoc(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        rsqrtOp, srcType, resCastOp.getResult(), shiftParamOp.getResult());

    return success();
  }
};

// Convert math.rsqrt to a function call to compute 1.0f / sqrt(x) for
// v16bfloat16 and v32bfloat16 types
struct ComputeRsqrtOpPattern : OpConversionPattern<math::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = rsqrtOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(rsqrtOp);
    Type vLNbf16OpaqueTy;
    if (laneSize == 16)
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, rsqrtOp.getLoc(), vLNbf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> rsqrtOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, rsqrtOp.getLoc(), TypeRange{vLNbf16OpaqueTy}, "getRsqrtBf16",
        nullptr, nullptr, rsqrtOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        rsqrtOp, TypeRange{rsqrtOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.erf to a function call to compute erf(x) for v16bfloat16 and
// v32bfloat16 types
struct ComputeErfOpPattern : OpConversionPattern<math::ErfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ErfOp erfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(erfOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = erfOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(erfOp);
    Type vLNbf16OpaqueTy;
    if (laneSize == 16)
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vLNbf16OpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(
            rewriter, erfOp.getLoc(), vLNbf16OpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> erfOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, erfOp.getLoc(), TypeRange{vLNbf16OpaqueTy}, "getErfBf16",
        nullptr, nullptr, erfOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        erfOp, TypeRange{erfOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.absf and math.absi to a function call to compute abs(x) for
// v16bfloat16, v32bfloat16, v16float, v16int32, v32int16 and v64int8 types
template <typename SrcOpTy>
struct ComputeAbsOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy absOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(absOp.getOperand().getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();

    unsigned laneSize = getVectorLaneSize(vecTy);
    unsigned elWidth = elemTy.getIntOrFloatBitWidth();

    StringRef includeName = "vec_math.h";
    auto moduleOp = absOp->template getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(absOp);
    std::ostringstream typeName;
    typeName << "v" << laneSize;
    if (isa<FloatType>(elemTy)) {
      if (elWidth == 16)
        typeName << "bfloat16";
      else
        typeName << "float";
    } else
      typeName << "int" << elWidth;
    Type vecOpaqueTy =
        emitc::OpaqueType::get(rewriter.getContext(), typeName.str());
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, absOp.getLoc(),
                                           vecOpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> absOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(rewriter, absOp.getLoc(),
                                              TypeRange{vecOpaqueTy}, "getAbs",
                                              nullptr, nullptr, absOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        absOp, TypeRange{absOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

using ComputeAbsFOpPattern = ComputeAbsOpPattern<math::AbsFOp>;
using ComputeAbsIOpPattern = ComputeAbsOpPattern<math::AbsIOp>;

template <typename SrcOpTy>
struct LowerExtOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy extOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = dyn_cast<VectorType>(extOp.getIn().getType());
    VectorType dstType = dyn_cast<VectorType>(extOp.getOut().getType());

    Type scalarType = dstType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    auto accType =
        isa<IntegerType>(scalarType) && (elWidth == 32 || elWidth == 64)
            ? dstType
            : getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto upsOp =
        aievec::UPSOp::create(rewriter, extOp.getLoc(), accType, extOp.getIn());

    if (dstType.getElementType().getIntOrFloatBitWidth() == 16) {
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, extOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          extOp, dstType, upsOp.getResult(), shiftParamOp.getResult());
    } else
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          extOp, dstType, upsOp.getResult(), /*isResAcc*/ false);

    return success();
  }
};

using LowerExtFOpPattern = LowerExtOpPattern<arith::ExtFOp>;
using LowerExtSIOpPattern = LowerExtOpPattern<arith::ExtSIOp>;

template <typename SrcOpTy>
struct LowerTruncOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy truncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = dyn_cast<VectorType>(truncOp.getIn().getType());
    VectorType dstType = dyn_cast<VectorType>(truncOp.getOut().getType());
    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    auto accType =
        isa<IntegerType>(scalarType) && (elWidth == 32 || elWidth == 64)
            ? srcType
            : getVectorOpDestType(srcType, /*AIE2 =*/true);

    auto shiftParamOp = arith::ConstantOp::create(
        rewriter, truncOp.getLoc(), rewriter.getI32IntegerAttr(0));
    if (elWidth == 16) {
      auto upsOp = aievec::UPSOp::create(rewriter, truncOp.getLoc(), accType,
                                         truncOp.getIn());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          truncOp, dstType, upsOp.getResult(), shiftParamOp.getResult());
    } else {
      auto castOp = aievec::CastOp::create(rewriter, truncOp.getLoc(), accType,
                                           truncOp.getIn(), true);
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          truncOp, dstType, castOp.getResult(), shiftParamOp.getResult());
    }

    return success();
  }
};

using LowerTruncFOpPattern = LowerTruncOpPattern<arith::TruncFOp>;
using LowerTruncIOpPattern = LowerTruncOpPattern<arith::TruncIOp>;

// If `op` is the last operation in the sequence:
//     %0 = unrealized_conversion_cast <%IN> : <native type>, !emitc.opaque_type
//     %1 = emitc.call_opaque <funcName>, %0...
//     %2 = unrealized_conversion_cast %1 : !emitc.opaque_type, <native type>
// return the value <%IN>.
static std::optional<Value>
getUnOpaquedOperandOfEmitCOpaqueCallOp(Operation *op, StringRef funcName) {
  auto uccOp = dyn_cast<UnrealizedConversionCastOp>(op);
  if (!uccOp)
    return {};

  auto inVal = uccOp.getInputs()[0];
  if (!isa<emitc::OpaqueType>(inVal.getType()))
    return {};

  auto callOp = inVal.getDefiningOp<emitc::CallOpaqueOp>();
  if (callOp.getCallee() != funcName)
    return {};

  auto callOperandsUccOp =
      callOp.getOperands()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!callOperandsUccOp)
    return {};

  return callOperandsUccOp.getInputs()[0];
}

// Check there is an operation chain like-
//
//      %cst_0 = arith.constant dense<1.000000e+00> : vector<16xbf16>
//      %cst_1 = arith.constant 0.000000e+00 : bf16
//      %0 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>,
//      vector<16xbf16>
//      %1 = arith.negf %0 : vector<16xbf16>
//      %2 = math.exp %1 : vector<16xbf16>
//      %3 = arith.addf %2, %cst_0 : vector<16xbf16>
//      %4 = arith.divf %cst_0, %3 : vector<16xbf16>
//
// so that this operation chain can be converted to a function call to compute
// sigmoid value for v16bfloat16 and v32bfloat16 types
template <typename DivFOpTy>
static bool hasSigmoidComputationChain(DivFOpTy divfOp, arith::NegFOp &negOp) {
  auto constOp = dyn_cast<arith::ConstantOp>(divfOp.getLhs().getDefiningOp());
  if (!constOp)
    return false;

  auto cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;

  if (cstDense.template getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  Operation *addLvalOp;
  Operation *addRvalOp;
  // divfOp's rval could be an arith::AddFOp or the pattern like-
  // %1 = aievec.ups %a
  // %2 = aievec.ups %b;
  // %3 = aievec.add_elem %1, %2
  // %4 = aievec.srs %3;
  auto addOp = dyn_cast<arith::AddFOp>(divfOp.getRhs().getDefiningOp());
  if (!addOp) {
    auto srsOp = dyn_cast<aievec::SRSOp>(divfOp.getRhs().getDefiningOp());
    if (!srsOp)
      return false;

    auto addElemOp =
        dyn_cast<aievec::AddElemOp>(srsOp.getSource().getDefiningOp());
    if (!addElemOp)
      return false;

    auto lUpsOp = dyn_cast<aievec::UPSOp>(addElemOp.getLhs().getDefiningOp());
    auto rUpsOp = dyn_cast<aievec::UPSOp>(addElemOp.getRhs().getDefiningOp());
    if (!lUpsOp || !rUpsOp)
      return false;

    addLvalOp = lUpsOp.getSource().getDefiningOp();
    addRvalOp = rUpsOp.getSource().getDefiningOp();
    // One of add operation's operand is a constant op and another operand could
    // be arith::ExpOp or the combination of emitc.call and aievec.srs
    auto addDefOp = isa<arith::ConstantOp>(addLvalOp)
                        ? dyn_cast<aievec::SRSOp>(addRvalOp)
                        : dyn_cast<aievec::SRSOp>(addLvalOp);
    if (!addDefOp)
      addLvalOp = isa<arith::ConstantOp>(addLvalOp)
                      ? dyn_cast<math::ExpOp>(addRvalOp)
                      : dyn_cast<math::ExpOp>(addLvalOp);
    else
      addLvalOp = addDefOp.getSource().getDefiningOp();

    addRvalOp = isa<arith::ConstantOp>(addLvalOp)
                    ? lUpsOp.getSource().getDefiningOp()
                    : rUpsOp.getSource().getDefiningOp();
  } else {
    addLvalOp = addOp.getLhs().getDefiningOp();
    addRvalOp = addOp.getRhs().getDefiningOp();
  }

  if (!addLvalOp || !addRvalOp)
    return false;

  auto addLvalExpOp = dyn_cast<math::ExpOp>(addLvalOp);
  auto addRvalExpOp = dyn_cast<math::ExpOp>(addRvalOp);
  auto addLvalExpOpIn =
      getUnOpaquedOperandOfEmitCOpaqueCallOp(addLvalOp, "getExpBf16")
          .value_or(nullptr);
  auto addRvalExpOpIn =
      getUnOpaquedOperandOfEmitCOpaqueCallOp(addRvalOp, "getExpBf16")
          .value_or(nullptr);
  if (!addLvalExpOpIn && addLvalExpOp)
    addLvalExpOpIn = addLvalExpOp.getOperand();
  if (!addRvalExpOpIn && addRvalExpOp)
    addRvalExpOpIn = addRvalExpOp.getOperand();

  if (!((addLvalExpOpIn && isa<arith::ConstantOp>(addRvalOp)) ||
        (addRvalExpOpIn && isa<arith::ConstantOp>(addLvalOp))))
    return false;

  constOp = isa<arith::ConstantOp>(addLvalOp)
                ? cast<arith::ConstantOp>(addLvalOp)
                : cast<arith::ConstantOp>(addRvalOp);

  cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;
  if (cstDense.template getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  auto expOperand = addLvalExpOpIn ? addLvalExpOpIn : addRvalExpOpIn;

  negOp = expOperand.getDefiningOp<arith::NegFOp>();

  return negOp != nullptr;
}

// Convert the operation chain like-
//
//      %cst_0 = arith.constant dense<1.000000e+00> : vector<16xbf16>
//      %cst_1 = arith.constant 0.000000e+00 : bf16
//      %0 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>,
//      vector<16xbf16>
//      %1 = arith.negf %0 : vector<16xbf16>
//      %2 = math.exp %1 :vector<16xbf16>
//      %3 = arith.addf %2, %cst_0 : vector<16xbf16>
//      %4 = arith.divf %cst_0, %3 : vector<16xbf16>
//
// to a function call to compute sigmoid value for v16bfloat16 and
// v32bfloat16 types
struct ComputeSigmoidOpPattern : OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp divfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    arith::NegFOp negOp = nullptr;
    if (!hasSigmoidComputationChain(adaptor, negOp))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = divfOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(divfOp);
    Type vecOpaqueTy;
    if (laneSize == 16)
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, divfOp.getLoc(),
                                           vecOpaqueTy, negOp.getOperand())
            .getResult(0);
    SmallVector<Value> sigmoidOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, divfOp.getLoc(), TypeRange{vecOpaqueTy}, "getSigmoidBf16",
        nullptr, nullptr, sigmoidOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        divfOp, TypeRange{adaptor.getLhs().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.ceil to a function call to compute ceil(x) for v16bfloat16
struct ComputeCeilOpPattern : OpConversionPattern<math::CeilOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::CeilOp ceilOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(ceilOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = ceilOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(ceilOp);
    Type vecOpaqueTy;
    if (laneSize == 16)
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, ceilOp.getLoc(),
                                           vecOpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> ceilOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, ceilOp.getLoc(), TypeRange{vecOpaqueTy}, "getCeilBf16",
        nullptr, nullptr, ceilOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        ceilOp, TypeRange{ceilOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert math.floor to a function call to compute floor(x) for v16bfloat16
struct ComputeFloorOpPattern : OpConversionPattern<math::FloorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::FloorOp floorOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(floorOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
      return failure();

    StringRef includeName = "vec_math.h";
    auto moduleOp = floorOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToStart(
        &moduleOp.getRegion().getBlocks().front());
    emitc::IncludeOp::create(rewriter, moduleOp.getLoc(), includeName, false);

    rewriter.setInsertionPoint(floorOp);
    Type vecOpaqueTy;
    if (laneSize == 16)
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v16bfloat16");
    else
      vecOpaqueTy =
          emitc::OpaqueType::get(rewriter.getContext(), "v32bfloat16");
    auto opaquedOperand =
        UnrealizedConversionCastOp::create(rewriter, floorOp.getLoc(),
                                           vecOpaqueTy, adaptor.getOperand())
            .getResult(0);
    SmallVector<Value> floorOperands = {opaquedOperand};
    auto callOp = emitc::CallOpaqueOp::create(
        rewriter, floorOp.getLoc(), TypeRange{vecOpaqueTy}, "getFloorBf16",
        nullptr, nullptr, floorOperands);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        floorOp, TypeRange{floorOp.getResult().getType()}, callOp.getResults());

    return success();
  }
};

// Convert arith.negf to aievec.neg to negate the vector for v16bfloat16 and
// v16float types.
struct ComputeNegOpPattern : OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::NegFOp negOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(negOp.getOperand().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return failure();

    if (unsigned laneSize = getVectorLaneSize(srcType); laneSize != 16)
      return failure();

    Location loc = negOp.getLoc();
    auto accType = getVectorOpDestType(srcType, /*AIE2 =*/true);

    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth == 16) {
      auto upsOp =
          aievec::UPSOp::create(rewriter, loc, accType, adaptor.getOperand());
      auto aieNegOp =
          aievec::NegOp::create(rewriter, loc, accType, upsOp.getResult());
      auto shiftParamOp = arith::ConstantOp::create(
          rewriter, negOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          negOp, srcType, aieNegOp.getResult(), shiftParamOp.getResult());
    } else {
      auto castOp = aievec::CastOp::create(
          rewriter, loc, accType, adaptor.getOperand(), /*isResAcc*/ true);
      auto aieNegOp =
          aievec::NegOp::create(rewriter, loc, accType, castOp.getResult());
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          negOp, srcType, aieNegOp.getResult(), /*isResAcc*/ false);
    }

    return success();
  }
};

// Check whether the value of constant operation is int type and the dense value
// is -1.
static bool hasConstNegOneValue(arith::ConstantOp constOp, unsigned elWidth) {
  if (!constOp)
    return false;

  auto cstDense = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;

  if (elWidth == 32)
    return cstDense.getSplatValue<int32_t>() == -1;
  if (elWidth == 16)
    return cstDense.getSplatValue<int16_t>() == -1;
  if (elWidth == 8)
    return cstDense.getSplatValue<int8_t>() == -1;
  return false;
}

// Convert arith.xori to aievec.bxor to compute bitwise xor of two vectors for
// integer types
struct ComputeBxorAndBnegOpPattern : OpConversionPattern<arith::XOrIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp xorOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(xorOp.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (laneSize * elWidth != 512)
      return failure();

    auto lhsConstOp =
        dyn_cast<arith::ConstantOp>(xorOp.getLhs().getDefiningOp());
    auto rhsConstOp =
        dyn_cast<arith::ConstantOp>(xorOp.getRhs().getDefiningOp());

    // If one of operands in xorOp is a constant -1, xorOp will be replaced with
    // aievec::BnegOp.
    if ((lhsConstOp && hasConstNegOneValue(lhsConstOp, elWidth)) ||
        (rhsConstOp && hasConstNegOneValue(rhsConstOp, elWidth))) {
      Value val = hasConstNegOneValue(lhsConstOp, elWidth) ? adaptor.getRhs()
                                                           : adaptor.getLhs();
      rewriter.replaceOpWithNewOp<aievec::BnegOp>(xorOp, srcType, val);
    } else
      rewriter.replaceOpWithNewOp<aievec::BxorOp>(
          xorOp, srcType, adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

template <typename SrcOpTy, typename DstOpTy>
struct ComputeBandAndBorOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return failure();

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (laneSize * elWidth != 512)
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getResult().getType(),
                                         adaptor.getLhs(), adaptor.getRhs());

    return success();
  }
};

using ComputeBorOpPattern =
    ComputeBandAndBorOpPattern<arith::OrIOp, aievec::BorOp>;
using ComputeBandOpPattern =
    ComputeBandAndBorOpPattern<arith::AndIOp, aievec::BandOp>;

// Convert arith.shrsi to a combination of aievec.ups and aievec.srs to compute
// arithmetic right shift for integer types. Currently, only support the shift
// value with a broadcast vector.
struct ComputeSignedIntRightShiftOpPattern
    : OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp rsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(adaptor.getLhs().getType());
    if (!srcType)
      return failure();

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (laneSize * elWidth != 512)
      return failure();

    auto bcastOp =
        dyn_cast<aievec::BroadcastOp>(adaptor.getRhs().getDefiningOp());
    if (!bcastOp)
      return failure();

    auto constOp =
        arith::ConstantOp::create(rewriter, bcastOp.getLoc(),
                                  rewriter.getI32IntegerAttr(bcastOp.getIdx()));
    auto extElemOp = aievec::ExtElemOp::create(
        rewriter, bcastOp.getLoc(), scalarType, bcastOp, constOp.getResult());
    Location loc = rsOp.getLoc();

    // The vector with v64int8 type can be divided into two v32int8 vectors and
    // be processed individually and be concatenated at the end.
    if (elWidth == 8) {
      VectorType halfSrcType = createVectorType(laneSize / 2, scalarType);
      auto rsOpLow = aievec::ExtOp::create(rewriter, loc, halfSrcType,
                                           adaptor.getLhs(), 0);
      auto rsOpHigh = aievec::ExtOp::create(rewriter, loc, halfSrcType,
                                            adaptor.getLhs(), 1);
      Type accType = getVectorOpDestType(halfSrcType, /*AIE2 =*/true);
      auto upsOpLow =
          aievec::UPSOp::create(rewriter, loc, accType, rsOpLow.getResult());
      auto srsOpLow =
          aievec::SRSOp::create(rewriter, loc, halfSrcType,
                                upsOpLow.getResult(), extElemOp.getResult());
      auto upsOpHigh =
          aievec::UPSOp::create(rewriter, loc, accType, rsOpHigh.getResult());
      auto srsOpHigh =
          aievec::SRSOp::create(rewriter, loc, halfSrcType,
                                upsOpHigh.getResult(), extElemOp.getResult());
      SmallVector<Value> inputSources = {srsOpLow.getResult(),
                                         srsOpHigh.getResult()};
      rewriter.replaceOpWithNewOp<aievec::ConcatOp>(rsOp, srcType,
                                                    inputSources);
    } else {
      Type accType = getVectorOpDestType(srcType, /*AIE2 =*/true);
      auto upsOp =
          aievec::UPSOp::create(rewriter, loc, accType, adaptor.getLhs());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          rsOp, srcType, upsOp.getResult(), extElemOp.getResult());
    }

    return success();
  }
};

// Recognize the compound shift+clamp+truncate pattern and lower it to
// aievec.ups + aievec.srs. This maps the standard quantized neural network
// "shift-round-saturate + clamp" idiom to the AIE2 SRS hardware unit.
//
// Pattern A (with clamp):
//   %shifted  = arith.shrsi %wide, %shift_splat
//   %clamped0 = arith.maxsi %shifted, %c_lo
//   %clamped  = arith.minsi %clamped0, %c_hi
//   %result   = arith.trunci %clamped
//
// Pattern B (without clamp):
//   %shifted  = arith.shrsi %wide, %shift_splat
//   %result   = arith.trunci %shifted
//
// Both lower to: aievec.ups + aievec.srs
struct ShiftClampTruncToSRSPattern : OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  ShiftClampTruncToSRSPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpConversionPattern(context, benefit) {}

  // Try to extract a scalar integer splat value from a Value.
  // Returns std::nullopt if the value is not a constant splat.
  static std::optional<int64_t> getConstantSplatValue(Value val) {
    auto defOp = val.getDefiningOp<arith::ConstantOp>();
    if (!defOp)
      return std::nullopt;
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(defOp.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return std::nullopt;
    return denseAttr.getSplatValue<APInt>().getSExtValue();
  }

  // Try to extract the shift amount from the right operand of shrsi.
  // Accepts either a constant splat vector or an aievec.broadcast of a
  // constant.
  static std::optional<Value>
  getShiftValue(Value rhs, ConversionPatternRewriter &rewriter, Location loc) {
    // Case 1: constant splat vector
    if (auto constOp = rhs.getDefiningOp<arith::ConstantOp>()) {
      auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
      if (denseAttr && denseAttr.isSplat()) {
        int64_t shiftVal = denseAttr.getSplatValue<APInt>().getSExtValue();
        return arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getI32IntegerAttr(shiftVal))
            .getResult();
      }
    }
    // Case 2: aievec.broadcast
    if (auto bcastOp = dyn_cast<aievec::BroadcastOp>(rhs.getDefiningOp())) {
      auto constOp = arith::ConstantOp::create(
          rewriter, bcastOp.getLoc(),
          rewriter.getI32IntegerAttr(bcastOp.getIdx()));
      return aievec::ExtElemOp::create(rewriter, bcastOp.getLoc(),
                                       rewriter.getI32Type(), bcastOp,
                                       constOp.getResult())
          .getResult();
    }
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(arith::TruncIOp truncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = dyn_cast<VectorType>(truncOp.getOut().getType());
    if (!dstType)
      return failure();

    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(dstScalarType))
      return failure();

    // Walk backward through optional clamp chain
    Value source = adaptor.getIn();
    int32_t sign = 1; // default: signed

    // Check for minsi(maxsi(...), hi) or maxsi(minsi(...), lo) clamp pattern
    arith::MinSIOp minOp = nullptr;
    arith::MaxSIOp maxOp = nullptr;

    if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
      if (auto maxsiOp = minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>()) {
        minOp = minsiOp;
        maxOp = maxsiOp;
        source = maxOp.getLhs();
      }
    } else if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
      if (auto minsiOp = maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>()) {
        maxOp = maxsiOp;
        minOp = minsiOp;
        source = minOp.getLhs();
      }
    }

    // If we found a clamp, verify it's a valid saturation range
    if (minOp && maxOp) {
      auto loVal = getConstantSplatValue(maxOp.getRhs());
      auto hiVal = getConstantSplatValue(minOp.getRhs());
      if (!loVal || !hiVal)
        return failure();

      unsigned dstBits = dstScalarType.getIntOrFloatBitWidth();
      // Guard against UB from shifting into or past the sign bit.
      if (dstBits == 0 || dstBits > 63)
        return failure();
      uint64_t one = 1ULL;
      int64_t unsignedLo = 0;
      int64_t unsignedHi = static_cast<int64_t>((one << dstBits) - 1);
      int64_t signedLo = -static_cast<int64_t>(one << (dstBits - 1));
      int64_t signedHi = static_cast<int64_t>((one << (dstBits - 1)) - 1);

      if (*loVal == unsignedLo && *hiVal == unsignedHi) {
        sign = 0; // unsigned saturation
      } else if (*loVal == signedLo && *hiVal == signedHi) {
        sign = 1; // signed saturation
      } else {
        // Clamp range doesn't match standard saturation — don't match
        return failure();
      }
    }

    // Now source should be the shrsi result
    auto shrsiOp = source.getDefiningOp<arith::ShRSIOp>();
    if (!shrsiOp)
      return failure();

    auto srcType = dyn_cast<VectorType>(shrsiOp.getLhs().getType());
    if (!srcType)
      return failure();

    Type srcScalarType = srcType.getElementType();
    if (!isa<IntegerType>(srcScalarType))
      return failure();

    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    if (dstElWidth >= srcElWidth)
      return failure();

    Location loc = truncOp.getLoc();

    // Extract the shift amount
    auto shiftVal = getShiftValue(shrsiOp.getRhs(), rewriter, loc);
    if (!shiftVal)
      return failure();

    // Get the wide input (pre-shift)
    Value wideInput = shrsiOp.getLhs();

    unsigned laneSize = getVectorLaneSize(srcType);
    bool needsPadding = (laneSize % 16 != 0);

    VectorType paddedSrcType = srcType;
    VectorType paddedDstType = dstType;
    unsigned paddedLanes = laneSize;

    if (needsPadding) {
      // Round up to nearest multiple of 16
      paddedLanes = ((laneSize + 15) / 16) * 16;
      paddedSrcType = createVectorType(paddedLanes, srcScalarType);
      paddedDstType = createVectorType(paddedLanes, dstScalarType);

      // Zero-pad the input using insert_strided_slice
      auto zeroAttr = rewriter.getZeroAttr(paddedSrcType);
      auto zeroPad =
          arith::ConstantOp::create(rewriter, loc, zeroAttr).getResult();
      SmallVector<int64_t> offsets(1, 0);
      SmallVector<int64_t> strides(1, 1);
      wideInput = vector::InsertStridedSliceOp::create(
                      rewriter, loc, wideInput, zeroPad, offsets, strides)
                      .getResult();
    }

    // Determine accumulator type and create the accumulator value.
    // For i16 source: use UPS to widen to accumulator type.
    // For i32/i64 source: use CastOp (marks as accumulator without widening),
    // matching the approach in LowerTruncOpPattern.
    Type accScalarType = paddedSrcType.getElementType();
    unsigned accElWidth = accScalarType.getIntOrFloatBitWidth();
    Value accValue;
    if (accElWidth == 16) {
      Type accType = getVectorOpDestType(paddedSrcType, /*AIE2=*/true);
      accValue =
          aievec::UPSOp::create(rewriter, loc, accType, wideInput).getResult();
    } else {
      // For i32/i64: CastOp with isResAcc=true marks as accumulator
      accValue = aievec::CastOp::create(rewriter, loc, paddedSrcType, wideInput,
                                        /*isResAcc=*/true)
                     .getResult();
    }
    auto srsOp = aievec::SRSOp::create(rewriter, loc, paddedDstType, accValue,
                                       *shiftVal, sign);

    Value result = srsOp.getResult();

    if (needsPadding) {
      // Extract original lanes from the padded result
      SmallVector<int64_t> offsets(1, 0);
      SmallVector<int64_t> sizes = {static_cast<int64_t>(laneSize)};
      SmallVector<int64_t> strides(1, 1);
      result = vector::ExtractStridedSliceOp::create(rewriter, loc, result,
                                                     offsets, sizes, strides)
                   .getResult();
    }

    rewriter.replaceOp(truncOp, result);

    // Erase the intermediate clamp/shift ops if they have no other uses.
    // These must be cleaned up because shrsi on 512-bit vectors is marked
    // illegal and would cause conversion failure if left as dead ops.
    SmallVector<Operation *, 3> opsToErase;
    if (minOp && minOp->use_empty())
      opsToErase.push_back(minOp);
    if (maxOp && maxOp->use_empty())
      opsToErase.push_back(maxOp);
    if (shrsiOp->use_empty())
      opsToErase.push_back(shrsiOp);
    for (Operation *op : opsToErase)
      rewriter.eraseOp(op);

    return success();
  }
};

// Promote scalar shrsi + [clamp] + trunci chain to a vectorized SRS sequence.
// Anchored on scalar arith::TruncIOp. Fuses the entire chain into:
//   broadcast_scalar -> cast(isResAcc) -> srs(narrowed, shift, sign) ->
//   ext_elem
// This prevents scalar arith.trunci i32->i8 from reaching the AIE2 backend
// where it crashes (the backend only supports 32->16/20/32 truncations).
struct LowerScalarShiftClampTruncToSRS : OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  // Try to extract a scalar integer constant value from a Value.
  static std::optional<int64_t> getScalarConstantValue(Value val) {
    auto defOp = val.getDefiningOp<arith::ConstantOp>();
    if (!defOp)
      return std::nullopt;
    auto intAttr = dyn_cast<IntegerAttr>(defOp.getValue());
    if (!intAttr)
      return std::nullopt;
    return intAttr.getInt();
  }

  LogicalResult
  matchAndRewrite(arith::TruncIOp truncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match scalar types (vector compound pattern handles vectors)
    Type dstType = truncOp.getOut().getType();
    if (isa<VectorType>(dstType))
      return failure();

    auto dstIntType = dyn_cast<IntegerType>(dstType);
    if (!dstIntType)
      return failure();

    unsigned dstBits = dstIntType.getWidth();
    if (dstBits != 8 && dstBits != 16)
      return failure();

    auto srcIntType = dyn_cast<IntegerType>(truncOp.getIn().getType());
    if (!srcIntType || srcIntType.getWidth() != 32)
      return failure();

    // Walk backward through optional clamp chain
    Value source = truncOp.getIn();
    int32_t sign = 1; // default: signed

    arith::MinSIOp minOp = nullptr;
    arith::MaxSIOp maxOp = nullptr;

    if (auto minsiOp = source.getDefiningOp<arith::MinSIOp>()) {
      if (auto maxsiOp = minsiOp.getLhs().getDefiningOp<arith::MaxSIOp>()) {
        minOp = minsiOp;
        maxOp = maxsiOp;
        source = maxOp.getLhs();
      }
    } else if (auto maxsiOp = source.getDefiningOp<arith::MaxSIOp>()) {
      if (auto minsiOp = maxsiOp.getLhs().getDefiningOp<arith::MinSIOp>()) {
        maxOp = maxsiOp;
        minOp = minsiOp;
        source = minOp.getLhs();
      }
    }

    // If we found a clamp, verify it's a valid saturation range
    if (minOp && maxOp) {
      auto loVal = getScalarConstantValue(maxOp.getRhs());
      auto hiVal = getScalarConstantValue(minOp.getRhs());
      if (!loVal || !hiVal)
        return failure();

      if (dstBits == 0 || dstBits > 63)
        return failure();
      uint64_t one = 1ULL;
      int64_t unsignedLo = 0;
      int64_t unsignedHi = static_cast<int64_t>((one << dstBits) - 1);
      int64_t signedLo = -static_cast<int64_t>(one << (dstBits - 1));
      int64_t signedHi = static_cast<int64_t>((one << (dstBits - 1)) - 1);

      if (*loVal == unsignedLo && *hiVal == unsignedHi) {
        sign = 0; // unsigned saturation
      } else if (*loVal == signedLo && *hiVal == signedHi) {
        sign = 1; // signed saturation
      } else {
        return failure();
      }
    }

    // Now source should be the shrsi result
    auto shrsiOp = source.getDefiningOp<arith::ShRSIOp>();
    if (!shrsiOp)
      return failure();

    // Verify shrsi operand types
    if (!isa<IntegerType>(shrsiOp.getLhs().getType()))
      return failure();

    Location loc = truncOp.getLoc();

    // Get the pre-shift value and shift amount
    Value preShiftVal = shrsiOp.getLhs();
    Value shiftVal = shrsiOp.getRhs();

    // Create 512-bit vector type: vector<16xi32>
    unsigned srcLanes = 512 / srcIntType.getWidth(); // 16 for i32
    VectorType bcastVecType = createVectorType(srcLanes, srcIntType);

    // Broadcast pre-shift value to 512-bit vector
    auto bcast = aievec::BroadcastScalarOp::create(rewriter, loc, bcastVecType,
                                                   preShiftVal);

    Value accValue;
    VectorType srsOutType;

    if (dstBits == 8) {
      // i32→i8: The SRS intrinsic I256V32Acc32Srs needs 1024-bit source
      // (vector<32xi32>, cast to vector<16xi64> internally) and produces
      // vector<32xi8> (256-bit). Concat two broadcast copies to get 1024 bits.
      unsigned accLanes = srcLanes * 2; // 32
      VectorType accVecType =
          createVectorType(accLanes, srcIntType); // vector<32xi32>
      auto concatSrc = aievec::ConcatOp::create(
          rewriter, loc, accVecType,
          SmallVector<Value>({bcast.getResult(), bcast.getResult()}));
      accValue =
          aievec::CastOp::create(rewriter, loc, accVecType,
                                 concatSrc.getResult(), /*isResAcc=*/true)
              .getResult();
      srsOutType = createVectorType(accLanes, dstIntType); // vector<32xi8>
    } else {
      // i32→i16: The SRS intrinsic I256V16Acc32Srs needs 512-bit source
      // (vector<16xi32>, cast to vector<8xi64> internally) and produces
      // vector<16xi16> (256-bit). 512-bit broadcast works directly.
      accValue = aievec::CastOp::create(rewriter, loc, bcastVecType,
                                        bcast.getResult(), /*isResAcc=*/true)
                     .getResult();
      srsOutType = createVectorType(srcLanes, dstIntType); // vector<16xi16>
    }

    // SRS: accumulator → narrowed output with shift and sign
    auto srsOp = aievec::SRSOp::create(rewriter, loc, srsOutType, accValue,
                                       shiftVal, sign);

    // ExtElem needs 512-bit source. SRS output is 256-bit, so concat to 512.
    unsigned extLanes = 512 / dstBits; // 64 for i8, 32 for i16
    VectorType extVecType = createVectorType(extLanes, dstIntType);
    auto concatForExt = aievec::ConcatOp::create(
        rewriter, loc, extVecType,
        SmallVector<Value>({srsOp.getResult(), srsOp.getResult()}));

    // Extract element 0 back to scalar
    auto zeroIdx =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(
        truncOp, dstIntType, concatForExt.getResult(), zeroIdx.getResult());

    // Erase dead intermediate ops
    SmallVector<Operation *, 3> opsToErase;
    if (minOp && minOp->use_empty())
      opsToErase.push_back(minOp);
    if (maxOp && maxOp->use_empty())
      opsToErase.push_back(maxOp);
    if (shrsiOp->use_empty())
      opsToErase.push_back(shrsiOp);
    for (Operation *op : opsToErase)
      rewriter.eraseOp(op);

    return success();
  }
};

// Promote scalar arith.shrsi to vector aievec.ups + aievec.srs to prevent
// LLVM's SLP vectorizer from creating sub-512-bit vector shifts that the
// AIE2 backend cannot legalize (G_LSHR on <4 x s32>).
struct LowerScalarShRSIToAIEVecUPSSRS : OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp rsOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match scalar i32
    Type resultType = rsOp.getType();
    if (isa<VectorType>(resultType))
      return failure();

    auto intType = dyn_cast<IntegerType>(resultType);
    if (!intType || intType.getWidth() != 32)
      return failure();

    Location loc = rsOp.getLoc();
    VectorType vecType = createVectorType(16, intType); // vector<16xi32>

    // Broadcast scalar value to 512-bit vector
    auto lhsBcast = aievec::BroadcastScalarOp::create(rewriter, loc, vecType,
                                                      adaptor.getLhs());

    // UPS: vector<16xi32> -> accumulator type (vector<16xi64>)
    Type accType = getVectorOpDestType(vecType, /*AIE2=*/true);
    auto upsOp =
        aievec::UPSOp::create(rewriter, loc, accType, lhsBcast.getResult());

    // SRS: accumulator + i32 shift -> vector<16xi32>
    auto srsOp = aievec::SRSOp::create(rewriter, loc, vecType,
                                       upsOp.getResult(), adaptor.getRhs());

    // Extract element 0 back to scalar
    auto zeroIdx =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(
        rsOp, intType, srsOp.getResult(), zeroIdx.getResult());
    return success();
  }
};

// Convert a `vector.contract` op to an `aievec.matmul` op for AIE2 or
// `aievec.matmul_aie2p` for AIE2P
template <typename MatMulOpTy>
struct LowerVectorContractionOpToAIEVecMatMulPattern
    : OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern::OpConversionPattern;

  LowerVectorContractionOpToAIEVecMatMulPattern(MLIRContext *context,
                                                bool matMoveToAcc = true)
      : OpConversionPattern(context), matMoveToAcc(matMoveToAcc) {}

  Value reshapeLeadingUnitDims(OpBuilder &b, Value v) const {
    auto vecTy = dyn_cast<VectorType>(v.getType());
    if (!vecTy)
      return v;
    auto vecShape = vecTy.getShape();

    size_t numLeadUnitDims = 0;
    while (numLeadUnitDims < vecShape.size() && vecShape[numLeadUnitDims] == 1)
      numLeadUnitDims++;

    if (!numLeadUnitDims)
      return v;

    SmallVector<int64_t> newShape(vecShape.begin() + numLeadUnitDims,
                                  vecShape.end());
    auto newVecTy = VectorType::get(newShape, vecTy.getElementType());
    return vector::ShapeCastOp::create(b, v.getLoc(), newVecTy, v).getResult();
  }

  LogicalResult
  matchAndRewrite(vector::ContractionOp contractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = reshapeLeadingUnitDims(rewriter, adaptor.getLhs());
    auto rhs = reshapeLeadingUnitDims(rewriter, adaptor.getRhs());
    auto acc = reshapeLeadingUnitDims(rewriter, adaptor.getAcc());
    bool bReshapedAcc = (acc != adaptor.getAcc());

    if (matMoveToAcc)
      acc = aievec::CastOp::create(rewriter, contractOp.getLoc(), acc.getType(),
                                   acc, true);

    auto matmulOp = MatMulOpTy::create(rewriter, contractOp.getLoc(),
                                       acc.getType(), lhs, rhs, acc);
    Value result;
    {
      // Replace diagnostics handler to silence errors when verifying the
      // validity of the matmul ops being generated.
      ScopedDiagnosticHandler diagHandler(
          contractOp.getContext(), [](Diagnostic &) { return success(); });
      if (failed(matmulOp.verifyInvariants())) {
        rewriter.eraseOp(matmulOp);
        // There is a possibility that, when the linalg op is converted to
        // contractions, lower precisions operands are cast to the target
        // precision outside the contraction. For those cases, we check.
        lhs = adaptor.getLhs();
        auto wideLhsValue = getSourceOfWideningOp(lhs).value_or(nullptr);
        if (wideLhsValue)
          lhs = reshapeLeadingUnitDims(rewriter, wideLhsValue);

        rhs = adaptor.getRhs();
        auto wideRhsValue = getSourceOfWideningOp(rhs).value_or(nullptr);
        if (wideRhsValue)
          rhs = reshapeLeadingUnitDims(rewriter, wideRhsValue);

        matmulOp = MatMulOpTy::create(rewriter, contractOp.getLoc(),
                                      acc.getType(), lhs, rhs, acc);
        if (failed(matmulOp.verifyInvariants()))
          return failure();
      }
    }
    result = matmulOp.getResult();

    if (matMoveToAcc)
      result = aievec::CastOp::create(rewriter, contractOp.getLoc(),
                                      acc.getType(), result, false);
    if (bReshapedAcc)
      result = vector::ShapeCastOp::create(rewriter, contractOp.getLoc(),
                                           adaptor.getAcc().getType(), result);
    rewriter.replaceOp(contractOp, result);

    return success();
  }

  bool matMoveToAcc;
};

using LowerVectorContractionOpToAIEVecMatMulOpAIE2 =
    LowerVectorContractionOpToAIEVecMatMulPattern<aievec::MatMulOp>;
using LowerVectorContractionOpToAIEVecMatMulOpAIE2P =
    LowerVectorContractionOpToAIEVecMatMulPattern<aievec::MatMulOp_AIE2P>;

// Convert a `vector.transpose` op to an `aievec.shuffle` op for AIE2.
struct LowerVectorTransposeOpToAIEVecShuffleOpPattern
    : OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::TransposeOp transpOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = transpOp.getResultVectorType();
    auto resShape = resTy.getShape();
    auto elemTyBitWidth = resTy.getElementTypeBitWidth();
    auto vBitWidth = std::accumulate(resShape.begin(), resShape.end(),
                                     elemTyBitWidth, std::multiplies<>());
    if (vBitWidth != 512)
      return failure();

    if (elemTyBitWidth != 8 && elemTyBitWidth != 16 && elemTyBitWidth != 32)
      return failure();

    // Verify leading dimensions are all 1.
    for (int64_t i = 0; i < static_cast<int64_t>(resShape.size() - 2); ++i)
      if (resShape[i] != 1)
        return failure();

    // Only permutation of the 2 innermost dimensions are supported.
    ArrayRef<int64_t> perm = transpOp.getPermutation();
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size() - 2); ++i)
      if (perm[i] != i)
        return failure();
    if (perm.back() != static_cast<int64_t>(perm.size() - 2))
      return failure();

    auto shuffleMode = aievec::ShuffleMode::T32_4X4;
    if (elemTyBitWidth == 8) {
      switch (resShape.back()) {
      case 4:
        shuffleMode = aievec::ShuffleMode::T8_4X16;
        break;
      case 8:
        shuffleMode = aievec::ShuffleMode::T8_8X8;
        break;
      case 16:
        shuffleMode = aievec::ShuffleMode::T8_16X4;
        break;
      default:
        return failure();
      }
    } else if (elemTyBitWidth == 16) {
      switch (resShape.back()) {
      case 2:
        shuffleMode = aievec::ShuffleMode::T16_2X16;
        break;
      case 4:
        shuffleMode = aievec::ShuffleMode::T16_4X8;
        break;
      case 8:
        shuffleMode = aievec::ShuffleMode::T16_8X4;
        break;
      case 16:
        shuffleMode = aievec::ShuffleMode::T16_16X2;
        break;
      default:
        return failure();
      }
    } else if (resShape.back() != 4)
      return failure();

    auto flatVecTy =
        VectorType::get({512 / elemTyBitWidth}, resTy.getElementType());
    auto loc = transpOp.getLoc();
    auto flatInput = vector::ShapeCastOp::create(rewriter, loc, flatVecTy,
                                                 adaptor.getVector());
    auto shuffOp = aievec::ShuffleOp::create(rewriter, loc, flatVecTy,
                                             flatInput, nullptr, shuffleMode);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(transpOp, resTy, shuffOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern collection
//===----------------------------------------------------------------------===//

static void populateAIEVecCommonConversionPatterns(RewritePatternSet &patterns,
                                                   TargetBackend backend) {
  // clang-format off
  patterns.add<LowerExtFOpPattern,
               LowerExtSIOpPattern,
               LowerTruncFOpPattern,
               LowerTruncIOpPattern>(patterns.getContext());
  // clang-format on
}

static void populateAIEVecV1ConversionPatterns(RewritePatternSet &patterns,
                                               TargetBackend backend) {
  patterns.add<LowerVectorTransferReadToAIEUPD>(patterns.getContext(), 128, 512,
                                                128, 256);
  // clang-format off
  patterns.add<LowerVectorAddIOpToAIEVecAddOp,
               LowerVectorSubIOpToAIEVecSubOp,
               LowerVectorMulIOpToAIEVecMulOp,
               LowerVectorAddFOpToAIEVecAddOp,
               LowerVectorSubFOpToAIEVecSubOp,
               LowerVectorMulFOpToAIEVecMulOp,
               ConvertMulAddToAIEVecFMAOpPattern,
               FoldSplatToFMAOp,
               LowerVectorExtractStridedSliceOpAIEv1Pattern>(patterns.getContext());
  // clang-format on
}

// Populate common conversion patterns for AIE2 and AIE2P
static void
populateAIEVecV2CommonConversionPatterns(RewritePatternSet &patterns,
                                         TargetBackend backend) {
  // clang-format off
  // TODO: Reorder these alphabetically
  if (backend == TargetBackend::CPP) {
    patterns.add<
        LowerVectorTransferReadToAIEUPD
      >(patterns.getContext(), 128, 1024, 256, 1024);
    patterns.add<
        ComputeExpOpByLUTPattern,
        ComputeInvOpByLUTPattern,
        ComputeRsqrtOpPattern,
        LowerVectorAddFOpToAIEVecAddElemOp,
        LowerVectorSubFOpToAIEVecSubElemOp,
        LowerVectorAddIOpToAIEVecAddElemOp,
        LowerVectorSubIOpToAIEVecSubElemOp
      >(patterns.getContext());
  } else if (backend == TargetBackend::LLVMIR){
      patterns.add<
      LowerVectorAddFOpToAIEVecAddElemOp,
      LowerVectorSubFOpToAIEVecSubElemOp
      >(patterns.getContext());
  }
  // Add the compound shift+clamp+trunc→SRS pattern with higher benefit
  // so it takes priority over the individual shrsi and trunci patterns.
  patterns.add<ShiftClampTruncToSRSPattern>(patterns.getContext(),
                                            /*benefit=*/2);
  // Scalar version of compound SRS with even higher benefit.
  patterns.add<LowerScalarShiftClampTruncToSRS>(patterns.getContext(),
                                                /*benefit=*/3);
  patterns.add<
      ComputeTanhOpByLUTPattern,
      ComputeSqrtOpAIE2Pattern,
      ComputeErfOpPattern,
      ComputeAbsFOpPattern,
      ComputeAbsIOpPattern,
      ComputeSigmoidOpPattern,
      ComputeCeilOpPattern,
      ComputeFloorOpPattern,
      ComputeNegOpPattern,
      ComputeBxorAndBnegOpPattern,
      ComputeBorOpPattern,
      ComputeBandOpPattern,
      ComputeSignedIntRightShiftOpPattern,
      LowerScalarShRSIToAIEVecUPSSRS,
      ConvertMulIToAIEVecMulElemOpPattern,
      ConvertMulFToAIEVecMulElemOpPattern,
      LowerVectorMinSIOpToAIEVecMinOp,
      LowerScalarMinSIOpToAIEVecMinOp,
      LowerVectorMinimumFOpToAIEVecMinOp,
      LowerVectorMaxSIOpToAIEVecMaxOp,
      LowerScalarMaxSIOpToAIEVecMaxOp,
      LowerVectorMaximumFOpToAIEVecMaxOp,
      LowerVectorMaxNumFFOpToAIEVecMaxOp,
      LowerVectorCmpIOpToAIEVecCmpOp,
      LowerVectorCmpFOpToAIEVecCmpOp,
      LowerVectorSelectOpToAIEVecSelOp,
      LowerVectorReductionMinOp,
      LowerVectorReductionMaxOp,
      LowerVectorReductionAddIntOp,
      LowerVectorReductionAddFloatOp,
      FoldVectorExtractAndSplatToAIEBroadcast,
      ConvertSplatToAIEBroadcast,
      ConvertMulAddToAIEVecFMAElemOpPattern,
      ConvertVectorFMAOpToAIEVecFMAElemOpPattern,
      LowerVectorExtractStridedSliceOpAIE2Pattern,
      LowerVectorTransposeOpToAIEVecShuffleOpPattern
      >(patterns.getContext());
  // clang-format on
}

static void populateAIEVecV2ConversionPatterns(RewritePatternSet &patterns,
                                               TargetBackend backend) {
  populateAIEVecV2CommonConversionPatterns(patterns, backend);
  patterns.add<LowerVectorContractionOpToAIEVecMatMulOpAIE2>(
      patterns.getContext(), backend == TargetBackend::CPP);
  patterns.add<LowerVectorReductionAddBfloat16OpAIE2>(patterns.getContext());
  // For AIE2 with LLVMIR backend, use LUT-based exp and rsqrt
  if (backend == TargetBackend::LLVMIR) {
    patterns.add<ComputeExpOpByLUTLLVMPattern, ComputeRsqrtOpLLVMAIE2Pattern>(
        patterns.getContext());
  }
}

// AIE2p-specific version of ConvertSplatToAIEBroadcast that supports direct
// 256-bit broadcasts without extract
struct ConvertSplatToAIEBroadcastAIE2p
    : OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getSource().getDefiningOp<vector::ExtractOp>())
      return failure();

    auto resultType = cast<VectorType>(bcastOp.getResult().getType());
    auto flatResultType = getFlattenedVectorType(resultType);
    Type scalarType = resultType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    auto src = bcastOp.getSource();

    // AIE2p supports both 256-bit and 512-bit broadcast directly
    if (laneSize * elWidth == 512 || laneSize * elWidth == 256) {
      Value newOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), flatResultType, src);
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    if (laneSize * elWidth == 1024) {
      VectorType vecType = createVectorType(512 / elWidth, scalarType);
      auto aieBcastOp = aievec::BroadcastScalarOp::create(
          rewriter, bcastOp.getLoc(), vecType, src);
      Value newOp = aievec::ConcatOp::create(
          rewriter, bcastOp.getLoc(), flatResultType,
          SmallVector<Value>({aieBcastOp.getResult(), aieBcastOp.getResult()}));
      if (resultType != flatResultType)
        newOp = vector::ShapeCastOp::create(rewriter, bcastOp.getLoc(),
                                            resultType, newOp);
      rewriter.replaceOp(bcastOp, newOp);
      return success();
    }

    return failure();
  }
};

static void populateAIEVecV2PConversionPatterns(RewritePatternSet &patterns,
                                                TargetBackend backend) {
  populateAIEVecV2CommonConversionPatterns(patterns, backend);
  patterns.add<LowerVectorContractionOpToAIEVecMatMulOpAIE2P>(
      patterns.getContext(), backend == TargetBackend::CPP);
  // AIE2p-specific broadcast pattern that handles 256-bit directly
  patterns.add<ConvertSplatToAIEBroadcastAIE2p>(patterns.getContext());
  patterns.add<LowerVectorReductionAddBfloat16OpAIE2P>(patterns.getContext());
  // For AIE2P with LLVMIR backend, use aievec.exp and aievec.inv
  // math.rsqrt is kept legal and will be lowered in AIEVecToLLVM pass
  if (backend == TargetBackend::LLVMIR) {
    patterns.add<ConvertMathExpToAIEVecExpOpPattern,
                 ConvertDivFToAIEVecInvOpPattern>(patterns.getContext());
  }
}

//===----------------------------------------------------------------------===//
// Legalizations
//===----------------------------------------------------------------------===//

// TODO: Review the validity of these legalizations beyond basic cases.

static bool isInSigmoidOperationChain(math::ExpOp expOp) {
  if (!expOp.getOperand().getDefiningOp<arith::NegFOp>())
    return false;

  arith::AddFOp addOp = nullptr;
  for (Operation *user : expOp->getUsers()) {
    addOp = dyn_cast<arith::AddFOp>(user);
    if (addOp)
      break;
  }

  if (!addOp)
    return false;

  auto *addLvalOp = addOp.getLhs().getDefiningOp();
  auto *addRvalOp = addOp.getRhs().getDefiningOp();
  if (!((isa<math::ExpOp>(addLvalOp) && isa<arith::ConstantOp>(addRvalOp)) ||
        (isa<math::ExpOp>(addRvalOp) && isa<arith::ConstantOp>(addLvalOp))))
    return false;

  auto constOp = isa<arith::ConstantOp>(addLvalOp)
                     ? cast<arith::ConstantOp>(addLvalOp)
                     : cast<arith::ConstantOp>(addRvalOp);

  auto cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;

  if (cstDense.getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  arith::DivFOp divOp = nullptr;
  for (Operation *user : addOp->getUsers()) {
    divOp = dyn_cast<arith::DivFOp>(user);
    if (divOp)
      break;
  }

  if (!divOp)
    return false;

  constOp = dyn_cast<arith::ConstantOp>(divOp.getLhs().getDefiningOp());
  if (!constOp)
    return false;
  cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;
  if (cstDense.getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  return true;
}

static void configureAIEVecCommonLegalizations(ConversionTarget &target,
                                               TargetBackend backend) {
  target
      .addLegalDialect<xilinx::aievec::aie1::AIEVecAIE1Dialect,
                       xilinx::aievec::AIEVecDialect, arith::ArithDialect,
                       ub::UBDialect, emitc::EmitCDialect, func::FuncDialect>();
  if (backend == TargetBackend::CPP) {
    target.addIllegalOp<vector::TransferReadOp>();
  }
  target.addIllegalOp<vector::ExtractStridedSliceOp>();
  target.addLegalOp<vector::BitCastOp>();

  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    return srcElWidth != 16 || srcLaneSize != 16 || dstElWidth != 32 ||
           dstLaneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    return srcLaneSize != 32 || (dstElWidth <= srcElWidth) ||
           (dstLaneSize != srcLaneSize);
  });

  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    return srcElWidth != 32 || srcLaneSize != 16 || dstElWidth != 16 ||
           dstLaneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();

    return srcLaneSize != 32 || (dstElWidth >= srcElWidth) ||
           (dstLaneSize != srcLaneSize);
  });

  target.addDynamicallyLegalOp<math::TanhOp>([](math::TanhOp tanhOp) {
    auto srcType = dyn_cast<VectorType>(tanhOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || laneSize != 16;
  });

  target.addDynamicallyLegalOp<math::SqrtOp>([](math::SqrtOp sqrtOp) {
    auto srcType = dyn_cast<VectorType>(sqrtOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<math::ErfOp>([](math::ErfOp erfOp) {
    auto srcType = dyn_cast<VectorType>(erfOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<math::AbsFOp>([](math::AbsFOp absfOp) {
    auto srcType = dyn_cast<VectorType>(absfOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth * laneSize != 512 && elWidth * laneSize != 256;
  });

  target.addDynamicallyLegalOp<math::AbsIOp>([](math::AbsIOp absiOp) {
    auto srcType = dyn_cast<VectorType>(absiOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth * laneSize != 512 && elWidth * laneSize != 256;
  });

  // CPP backend: Mark 1/x pattern as illegal for conversion to inv() via LUT
  // LLVMIR backend: Keep scalar divf legal (handled by downstream passes)
  if (backend == TargetBackend::CPP) {
    target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp divfOp) {
      if (auto srcType = dyn_cast<VectorType>(divfOp.getLhs().getType());
          !srcType) {
        Type scalarType = divfOp.getLhs().getType();
        if (!divfOp->hasOneUse() || !isa<FloatType>(scalarType))
          return true;
        if (!isNarrowingOp(*divfOp->getUsers().begin()))
          return true;

        auto fType = cast<FloatType>(scalarType);
        if (fType.getWidth() != 32)
          return true;

        auto constOp =
            dyn_cast<arith::ConstantOp>(divfOp.getLhs().getDefiningOp());
        if (!constOp ||
            cast<FloatAttr>(constOp.getValue()).getValue().convertToDouble() !=
                1.0f)
          return true;
      } else {
        Type scalarType = srcType.getElementType();
        if (!isa<FloatType>(scalarType))
          return true;

        unsigned laneSize = getVectorLaneSize(srcType);
        unsigned elWidth = scalarType.getIntOrFloatBitWidth();

        if (elWidth != 16 || (laneSize != 16 && laneSize != 32))
          return true;

        arith::NegFOp negOp = nullptr;
        if (!hasSigmoidComputationChain(divfOp, negOp))
          return true;
      }

      return false;
    });
  }

  target.addDynamicallyLegalOp<math::CeilOp>([](math::CeilOp ceilOp) {
    auto srcType = dyn_cast<VectorType>(ceilOp.getOperand().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<math::FloorOp>([](math::FloorOp floorOp) {
    auto srcType = dyn_cast<VectorType>(floorOp.getOperand().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });

  target.addDynamicallyLegalOp<arith::NegFOp>([](arith::NegFOp negOp) {
    auto srcType = dyn_cast<VectorType>(negOp.getOperand().getType());
    if (!srcType)
      return true;
    if (Type scalarType = srcType.getElementType(); !isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::XOrIOp>([](arith::XOrIOp xorOp) {
    auto srcType = dyn_cast<VectorType>(xorOp.getLhs().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::OrIOp>([](arith::OrIOp orOp) {
    auto srcType = dyn_cast<VectorType>(orOp.getLhs().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::ShRSIOp>([](arith::ShRSIOp rsOp) {
    auto srcType = dyn_cast<VectorType>(rsOp.getLhs().getType());
    if (!srcType) {
      // Scalar i32 shrsi: illegal unless it feeds into a compound SRS chain
      // (the compound pattern consumes it via the trunci anchor)
      if (auto intType = dyn_cast<IntegerType>(rsOp.getLhs().getType()))
        if (intType.getWidth() == 32) {
          if (shrsiUsedByCompoundSRS(rsOp))
            return true; // legal — compound pattern will handle
          return false;  // illegal — individual pattern promotes
        }
      return true;
    }

    // If the shrsi feeds into a compound SRS pattern (shrsi+clamp+trunc),
    // keep it legal — the compound pattern will consume it via the trunci.
    if (shrsiUsedByCompoundSRS(rsOp))
      return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::AndIOp>([](arith::AndIOp andOp) {
    auto srcType = dyn_cast<VectorType>(andOp.getLhs().getType());
    if (!srcType)
      return true;
    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  if (backend == TargetBackend::CPP) {
    target.addDynamicallyLegalOp<arith::AddIOp>(
        [](arith::AddIOp op) { return !isa<VectorType>(op.getType()); });
  }
  target.addDynamicallyLegalOp<arith::AddFOp>(
      [](arith::AddFOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubIOp>(
      [](arith::SubIOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubFOp>(
      [](arith::SubFOp op) { return !isa<VectorType>(op.getType()); });
}

static void configureAIEVecV1Legalizations(ConversionTarget &target,
                                           TargetBackend backend) {
  target.addDynamicallyLegalOp<arith::MulIOp>(
      [](arith::MulIOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::MulFOp>(
      [](arith::MulFOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<aievec::aie1::FMAOp>(
      [](xilinx::aievec::aie1::FMAOp op) {
        auto *lhsDefOp = op.getLhs().getDefiningOp();
        aievec::ConcatOp concatOp = nullptr;
        if (lhsDefOp)
          concatOp = dyn_cast<aievec::ConcatOp>(op.getLhs().getDefiningOp());
        if (!concatOp)
          return true;

        vector::BroadcastOp srcBcast = nullptr;
        if (auto *lhsOp = concatOp.getSources()[0].getDefiningOp())
          srcBcast = dyn_cast<vector::BroadcastOp>(lhsOp);
        if (!srcBcast) {
          auto *rhsOp = op.getRhs().getDefiningOp();
          if (!rhsOp)
            return true;
          srcBcast = dyn_cast<vector::BroadcastOp>(rhsOp);
        }

        if (srcBcast)
          if (auto *srcOp = srcBcast.getSource().getDefiningOp())
            return !isa<vector::ExtractOp>(srcOp);

        return true;
      });

  target.addDynamicallyLegalOp<aievec::aie1::AddOp>([](aievec::aie1::AddOp op) {
    auto lSrsOp = op.getLhs().getDefiningOp<aievec::SRSOp>();
    auto rSrsOp = op.getRhs().getDefiningOp<aievec::SRSOp>();
    return (!lSrsOp ||
            !lSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>()) &&
           (!rSrsOp ||
            !rSrsOp.getSource().getDefiningOp<aievec::aie1::MulOp>());
  });
  target.addLegalDialect<memref::MemRefDialect>();
}

static void configureAIEVecV2PLegalizations(ConversionTarget &target,
                                            TargetBackend backend) {
  // AIE2P-specific legalization for rsqrt with LLVMIR backend
  // Vector bf16 rsqrt is illegal (no hardware support)
  // Scalar f32 and vector f32 rsqrt are legal (lowered in AIEVecToLLVM pass)
  if (backend == TargetBackend::LLVMIR) {
    target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp rsqrtOp) {
      auto vecType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
      // Vector bf16 rsqrt is illegal
      if (vecType && vecType.getElementType().isBF16())
        return false;
      // Everything else is legal (scalar f32, vector f32)
      return true;
    });

    // AIE2P-specific legalization for exp with LLVMIR backend
    // v16bf16 and v32bf16 exp are illegal (uses hardware intrinsic)
    target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp expOp) {
      auto srcType = dyn_cast<VectorType>(expOp.getOperand().getType());
      if (!srcType)
        return true;

      Type scalarType = srcType.getElementType();
      unsigned elWidth = scalarType.getIntOrFloatBitWidth();
      unsigned laneSize = getVectorLaneSize(srcType);
      // AIE2P LLVMIR: v16bf16 and v32bf16 are illegal (uses aievec.exp)
      if (!scalarType.isBF16() || (laneSize != 16 && laneSize != 32) ||
          elWidth != 16)
        return true;
      if (expOp->hasOneUse() && isInSigmoidOperationChain(expOp))
        return true;

      return false;
    });

    // AIE2P-specific legalization for divf 1.0/x pattern with LLVMIR backend
    // Scalar f32 or vector<Nxf32> divf with constant 1.0 LHS is illegal
    target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp divfOp) {
      Type srcType = divfOp.getLhs().getType();

      // Check if LHS is defined by a constant operation
      auto constOp =
          dyn_cast_or_null<arith::ConstantOp>(divfOp.getLhs().getDefiningOp());
      if (!constOp)
        return true;

      // Scalar f32 case - check for exactly 1.0
      if (srcType.isF32()) {
        auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
        if (floatAttr && floatAttr.getValue().isExactlyValue(1.0))
          return false; // illegal - will be converted to aievec.inv
        return true;
      }

      // Vector f32 case - check for splat of exactly 1.0
      if (auto vecType = dyn_cast<VectorType>(srcType)) {
        if (vecType.getElementType().isF32()) {
          unsigned laneSize = getVectorLaneSize(vecType);
          if (laneSize == 16 || laneSize == 32) {
            auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
            if (denseAttr && denseAttr.isSplat() &&
                denseAttr.getSplatValue<APFloat>().isExactlyValue(1.0))
              return false; // illegal - will be converted to aievec.inv
          }
        }
      }

      return true;
    });
  }
  // For CPP backend, exp remains legal (uses LUT pattern from common patterns)

  // AIE2P-specific legalization: ExtFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2P-specific legalization: TruncFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2P-specific legalization: ExtSIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2P-specific legalization: TruncIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType) {
      // Scalar trunci: mark illegal if part of compound SRS chain
      if (!srcType && !dstType && isSRSCompoundCandidate(trunciOp))
        return false;
      return true;
    }
    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    // Also mark as illegal if it's part of a shrsi+clamp+trunc SRS pattern,
    // even for sub-AIE-width vectors
    if (isSRSCompoundCandidate(trunciOp))
      return false;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2P-specific legalization: Override AddFOp to support laneSize==32 for
  // float types
  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);

    // For float types, support both laneSize==16 and laneSize==32
    if (isa<FloatType>(scalarType))
      return laneSize != 16 && laneSize != 32;

    // For other types, only laneSize==16 (same as AIE2)
    return laneSize != 16;
  });

  // AIE2P-specific legalization: Override SubFOp to support laneSize==32 for
  // float types
  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);

    // For float types, support both laneSize==16 and laneSize==32
    if (isa<FloatType>(scalarType))
      return laneSize != 16 && laneSize != 32;

    // For other types, only laneSize==16 (same as AIE2)
    return laneSize != 16;
  });
}

static void configureAIEVecV2Legalizations(ConversionTarget &target,
                                           TargetBackend backend) {
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<vector::ShapeCastOp>();

  // A set recording the vector lane size and element width supported
  llvm::SmallSet<std::pair<unsigned, unsigned>, 16> laneSizeElWidthPairSet;
  laneSizeElWidthPairSet.insert({64, 8});
  laneSizeElWidthPairSet.insert({32, 16});
  laneSizeElWidthPairSet.insert({16, 32});
  laneSizeElWidthPairSet.insert({32, 32});

  // A set recording the element width supported
  llvm::SmallSet<unsigned, 16> elWidthSet;
  elWidthSet.insert(8);
  elWidthSet.insert(16);
  elWidthSet.insert(32);

  if (backend == TargetBackend::CPP) {
    target.addDynamicallyLegalOp<arith::AddIOp>([=](arith::AddIOp op) {
      auto resultType = dyn_cast<VectorType>(op.getType());
      if (!resultType)
        return true;

      auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
      unsigned laneSize = getVectorLaneSize(resultType);

      return !laneSizeElWidthPairSet.count(
          std::make_pair(laneSize, resultElWidth));
    });
  }

  target.addDynamicallyLegalOp<arith::SubIOp>([=](arith::SubIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !laneSizeElWidthPairSet.count(
        std::make_pair(laneSize, resultElWidth));
  });

  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();

    // Support laneSize == 16 for f32/bf16
    if (laneSize == 16)
      return false; // illegal - will be converted
    // Support laneSize == 32 for bf16 (split into two v16bf16 ops)
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split
    // Support laneSize == 32 for f32 (split into two v16f32 ops)
    if (laneSize == 32 && resultElWidth == 32)
      return false; // illegal - will be split into two v16f32 ops

    return true; // legal - not supported
  });

  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    Type scalarType = resultType.getElementType();
    unsigned laneSize = getVectorLaneSize(resultType);
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();

    // Support laneSize == 16 for f32/bf16
    if (laneSize == 16)
      return false; // illegal - will be converted
    // Support laneSize == 32 for bf16 (split into two v16bf16 ops)
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split
    // Support laneSize == 32 for f32 (split into two v16f32 ops)
    if (laneSize == 32 && resultElWidth == 32)
      return false; // illegal - will be split into two v16f32 ops

    return true; // legal - not supported
  });

  target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;
    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    // Verify it is not a part of MAC
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return (laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
           ((laneSize != 16 && laneSize != 32) || resultElWidth != 32);
  });

  target.addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    // Verify it is not a part of FMA
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    // Support laneSize == 16 for bf16/f32, and laneSize == 32 for bf16 (split)
    if (laneSize == 16 && (resultElWidth == 16 || resultElWidth == 32))
      return false; // illegal - will be converted
    if (laneSize == 32 && resultElWidth == 16)
      return false; // illegal - will be split into two v16bf16 ops

    return true; // legal - not supported
  });

  target.addDynamicallyLegalOp<arith::MinSIOp>([=](arith::MinSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      // Scalar i8/i16/i32 minsi: illegal unless in compound SRS chain
      if (auto intType = dyn_cast<IntegerType>(op.getType())) {
        unsigned w = intType.getWidth();
        if (w == 8 || w == 16 || w == 32) {
          if (scalarClampInCompoundSRS(op))
            return true; // legal — compound pattern consumes
          return false;  // illegal — individual pattern promotes
        }
      }
      return true;
    }

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MaxSIOp>([=](arith::MaxSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      // Scalar i8/i16/i32 maxsi: illegal unless in compound SRS chain
      if (auto intType = dyn_cast<IntegerType>(op.getType())) {
        unsigned w = intType.getWidth();
        if (w == 8 || w == 16 || w == 32) {
          if (scalarClampInCompoundSRS(op))
            return true; // legal — compound pattern consumes
          return false;  // illegal — individual pattern promotes
        }
      }
      return true;
    }

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MinimumFOp>([=](arith::MinimumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MaximumFOp>([=](arith::MaximumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MaxNumFOp>([=](arith::MaxNumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::MinNumFOp>([=](arith::MinNumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::CmpIOp>([=](arith::CmpIOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType)
      return true;

    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    return !elWidthSet.count(lhsElWidth) || laneSize * lhsElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::CmpFOp>([=](arith::CmpFOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType)
      return true;

    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    return !elWidthSet.count(lhsElWidth) || laneSize * lhsElWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::SelectOp>([=](arith::SelectOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !elWidthSet.count(resultElWidth) || laneSize * resultElWidth != 512;
  });

  target.addDynamicallyLegalOp<vector::ReductionOp>(
      [=](vector::ReductionOp op) {
        if (auto kind = op.getKind(); kind != vector::CombiningKind::ADD &&
                                      kind != vector::CombiningKind::MINSI &&
                                      kind != vector::CombiningKind::MINUI &&
                                      kind != vector::CombiningKind::MINIMUMF &&
                                      kind != vector::CombiningKind::MINNUMF &&
                                      kind != vector::CombiningKind::MAXSI &&
                                      kind != vector::CombiningKind::MAXUI &&
                                      kind != vector::CombiningKind::MAXIMUMF &&
                                      kind != vector::CombiningKind::MAXNUMF)
          return true;

        auto vType = dyn_cast<VectorType>(op.getVector().getType());
        if (!vType)
          return true;

        llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
        laneSizeElWidthPairSet.insert({64, 8});
        laneSizeElWidthPairSet.insert({32, 16});
        laneSizeElWidthPairSet.insert({32, 32});
        laneSizeElWidthPairSet.insert({16, 32});

        Type scalarType = vType.getElementType();
        unsigned elWidth = scalarType.getIntOrFloatBitWidth();
        unsigned laneSize = getVectorLaneSize(vType);

        if (isa<IntegerType>(scalarType) &&
            !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
          return true;

        if (isa<FloatType>(scalarType) && laneSize != 16 && laneSize != 32)
          return true;

        return false;
      });

  // AIE2-specific legalization: ExtFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2-specific legalization: TruncFOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2-specific legalization: ExtSIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType)
      return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  // AIE2-specific legalization: TruncIOp on vector is always illegal
  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType) {
      // Scalar trunci: mark illegal if part of compound SRS chain
      if (!srcType && !dstType && isSRSCompoundCandidate(trunciOp))
        return false;
      return true;
    }
    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    // Also mark as illegal if it's part of a shrsi+clamp+trunc SRS pattern,
    // even for sub-AIE-width vectors
    if (isSRSCompoundCandidate(trunciOp))
      return false;

    unsigned srcLaneSize = getVectorLaneSize(srcType);
    unsigned dstLaneSize = getVectorLaneSize(dstType);
    if ((srcLaneSize % 16 == 0) && (dstLaneSize % 16 == 0))
      return false;

    return true;
  });

  target.addIllegalOp<vector::ContractionOp, vector::TransposeOp,
                      vector::FMAOp>();

  // AIE2-specific legalization: math.exp for v16bf16 and v32bf16 is illegal
  // (uses LUT)
  target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp expOp) {
    auto srcType = dyn_cast<VectorType>(expOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(srcType);
    // AIE2: v16bf16 and v32bf16 are illegal (uses LUT-based lowering)
    if (!isa<FloatType>(scalarType) || (laneSize != 16 && laneSize != 32) ||
        elWidth != 16)
      return true;
    if (expOp->hasOneUse() && isInSigmoidOperationChain(expOp))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp rsqrtOp) {
    auto srcType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
    if (!srcType)
      return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    return elWidth != 16 || (laneSize != 16 && laneSize != 32);
  });
}

//===----------------------------------------------------------------------===//
// Lowering passes
//===----------------------------------------------------------------------===//

/// Lower incoming vector operations into their corresponding AIE vector
/// intrinsics.
struct LowerVectorToAIEVec : PassWrapper<LowerVectorToAIEVec, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorToAIEVec)

  LowerVectorToAIEVec() = default;
  LowerVectorToAIEVec(const LowerVectorToAIEVec &pass) : PassWrapper(pass) {}

  LowerVectorToAIEVec(const LowerVectorToAIEVecOptions &options)
      : LowerVectorToAIEVec() {
    aieTarget = options.aieTarget;
    targetBackend = options.targetBackend;
  }

  // In case we want to register this pass as a standalone pass for test
  // purposes.
  StringRef getArgument() const final { return "test-lower-vector-to-aievec"; }
  StringRef getDescription() const final {
    return "Lower vector operations to AIE vector intrinsics";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, xilinx::aievec::aie1::AIEVecAIE1Dialect,
                xilinx::aievec::AIEVecDialect, arith::ArithDialect,
                memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect,
                emitc::EmitCDialect>();
  }

  Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc(
          "Select AIE version: \"aie\", \"aie2\", or \"aie2p\". This will "
          "determine the vector size and available operations."),
      llvm::cl::init("aie")};

  Option<std::string> targetBackend{
      *this, "target-backend",
      llvm::cl::desc("Select translation backend: \"cpp\" or \"llvmir\". This "
                     "will determine the aievec operations used to convert "
                     "from vector dialect."),
      llvm::cl::init("cpp")};

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    auto aieVersion = AIEArch::AIE;
    if (!aieTarget.empty()) {
      std::string targetStr = aieTarget;
      if (targetStr == "aieml" || targetStr == "aie2")
        aieVersion = AIEArch::AIE2;
      else if (targetStr == "aie2p")
        aieVersion = AIEArch::AIE2P;
      else if (targetStr != "aie") {
        op->emitError() << "unknown AIE target '" << aieTarget << "'";
        return signalPassFailure();
      }
    }

    TargetBackend backend = TargetBackend::CPP;
    if (!targetBackend.empty()) {
      std::string backendStr = targetBackend;
      if (backendStr == "llvmir") {
        backend = TargetBackend::LLVMIR;
        if (aieVersion == AIEArch::AIE) {
          op->emitError() << "targetting LLVM IR is not supported for AIEv1";
          signalPassFailure();
          return;
        }
      } else if (backendStr != "cpp") {
        op->emitError() << "unknown target backend '" << targetBackend << "'";
        signalPassFailure();
        return;
      }
    }

    populateAIEVecCommonConversionPatterns(patterns, backend);
    configureAIEVecCommonLegalizations(target, backend);
    if (aieVersion == AIEArch::AIE) {
      populateAIEVecV1ConversionPatterns(patterns, backend);
      configureAIEVecV1Legalizations(target, backend);
    } else if (aieVersion == AIEArch::AIE2) {
      populateAIEVecV2ConversionPatterns(patterns, backend);
      configureAIEVecV2Legalizations(target, backend);
    } else if (aieVersion == AIEArch::AIE2P) {
      populateAIEVecV2PConversionPatterns(patterns, backend);
      configureAIEVecV2Legalizations(target, backend);
      configureAIEVecV2PLegalizations(target, backend);
    } else {
      llvm_unreachable("AIE version is misconfigured");
    }

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
};

static std::unique_ptr<Pass>
createLowerVectorToAIEVec(const LowerVectorToAIEVecOptions &options) {
  return std::make_unique<LowerVectorToAIEVec>(options);
}

//===---------------------------------------------------------------------------
// Custom canonicalization passes
//===---------------------------------------------------------------------------

// This pass widens UPD ops to twice the width followed by an ext op of the
// bottom half. This can be used together with SimplifyUPDOpsPass to find
// additional common subexpressions with UPDs generated from unaligned
// `transfer_read` ops.
struct ExtendUPDOpsPass : PassWrapper<ExtendUPDOpsPass, OperationPass<>> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.add<ExpandUPDToUPDAndExtPattern>(patterns.getContext());
    target.addLegalDialect<aievec::AIEVecDialect>();
    target.addDynamicallyLegalOp<aievec::UPDOp>([](aievec::UPDOp op) {
      return op.getVector() ||
             (op->hasOneUse() && isa<aievec::UPDOp>(*op->getUsers().begin())) ||
             llvm::all_of(op->getUsers(),
                          [](Operation *op) { return isa<aievec::ExtOp>(op); });
    });

    if (auto *op = getOperation();
        failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

// This pass replaces wide UPD ops that are only used by a single ext op of the
// bottom half. This pass undos the work of ExtendUPDOpsPass.
// TODO: This pass can be extended to work with wide UPD ops that are used by
// TODO: a single ext op of the top half, which might be a good opportunity to
// TODO: further optimize wide UPDs.
struct SimplifyUPDOpsPass : PassWrapper<SimplifyUPDOpsPass, OperationPass<>> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.add<FuseExtIntoUPDPattern>(patterns.getContext());
    target.addLegalDialect<aievec::AIEVecDialect>();
    target.addDynamicallyLegalOp<aievec::ExtOp>([](aievec::ExtOp op) {
      auto *defOp = op.getSource().getDefiningOp();
      return !defOp || !isa<aievec::UPDOp>(defOp) || !defOp->hasOneUse() ||
             op.getIndex() != 0;
    });

    if (auto *op = getOperation();
        failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

//============================================================================//
//=============== Main Vector2AIEVec Pipeline Configuration ==================//
//============================================================================//

void xilinx::aievec::buildLowerVectorToAIEVec(
    OpPassManager &pm, const LowerVectorToAIEVecOptions &options) {
  // Add lowering from `Vector` to `AIEVec`
  pm.addPass(createLowerVectorToAIEVec(options));
  pm.addPass(createCanonicalizerPass());

  // Simplify UPD ops
  pm.addPass(std::make_unique<ExtendUPDOpsPass>());
  pm.addPass(createCSEPass());
  pm.addPass(std::make_unique<SimplifyUPDOpsPass>());
  pm.addPass(createCanonicalizerPass());
}
