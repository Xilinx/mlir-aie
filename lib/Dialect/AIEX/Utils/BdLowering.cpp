//===- BdLowering.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;

namespace xilinx::AIEX {

Value getAsValue(OpBuilder &builder, Location loc, OpFoldResult ofr,
                 Type intType) {
  if (auto constVal = getConstantIntValue(ofr))
    return arith::ConstantOp::create(builder, loc,
                                     IntegerAttr::get(intType, *constVal));
  Value val = cast<Value>(ofr);
  if (val.getType() != intType) {
    unsigned valBits = val.getType().getIntOrFloatBitWidth();
    unsigned tgtBits = intType.getIntOrFloatBitWidth();
    if (valBits > tgtBits)
      val = arith::TruncIOp::create(builder, loc, intType, val);
    else
      val = arith::ExtUIOp::create(builder, loc, intType, val);
  }
  return val;
}

Value buildBdWord(OpBuilder &builder, Location loc,
                  ArrayRef<std::tuple<Value, uint32_t, uint32_t>> fields) {
  auto i32ty = IntegerType::get(builder.getContext(), 32);
  Value result =
      arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, 0));
  for (auto &[val, mask, shift] : fields) {
    Value masked = val;
    if (mask != 0xFFFFFFFF) {
      auto maskConst = arith::ConstantOp::create(builder, loc,
                                                 IntegerAttr::get(i32ty, mask));
      masked = arith::AndIOp::create(builder, loc, val, maskConst);
    }
    if (shift > 0) {
      auto shiftConst = arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(i32ty, shift));
      masked = arith::ShLIOp::create(builder, loc, masked, shiftConst);
    }
    result = arith::OrIOp::create(builder, loc, result, masked);
  }
  return result;
}

HwBdEncoding emitDynamicHwBdEncoding(OpBuilder &builder, Location loc,
                                     const AIE::AIETargetModel &targetModel,
                                     BaseMemRefType bufType,
                                     ArrayRef<OpFoldResult> mixedSizesRev,
                                     ArrayRef<OpFoldResult> mixedStridesRev) {
  auto *ctx = builder.getContext();
  auto i32ty = IntegerType::get(ctx, 32);

  uint64_t elemWidth = bufType.getElementType().getIntOrFloatBitWidth();
  uint32_t addrGran = targetModel.getAddressGenGranularity();

  auto cst = [&](int64_t val) -> Value {
    return arith::ConstantOp::create(builder, loc,
                                     IntegerAttr::get(i32ty, val));
  };

  // Get each input size/stride as an SSA Value (i32)
  Value inSize0 = getAsValue(builder, loc, mixedSizesRev[0], i32ty);
  Value inSize1 = getAsValue(builder, loc, mixedSizesRev[1], i32ty);
  Value inSize2 = getAsValue(builder, loc, mixedSizesRev[2], i32ty);
  Value inSize3 = getAsValue(builder, loc, mixedSizesRev[3], i32ty);
  Value inStride0 = getAsValue(builder, loc, mixedStridesRev[0], i32ty);
  Value inStride1 = getAsValue(builder, loc, mixedStridesRev[1], i32ty);
  Value inStride2 = getAsValue(builder, loc, mixedStridesRev[2], i32ty);
  Value inStride3 = getAsValue(builder, loc, mixedStridesRev[3], i32ty);

  // Hardware d0_size = inputSizes[0] * elemWidth / addrGran
  // NOTE: Must multiply first, then divide to avoid integer truncation.
  // The hardware d0_size field is 10 bits wide (max 1023). The static
  // lowering path applies a linear-mode optimization for contiguous transfers
  // that avoids this limit, but the dynamic path does not.
  Value hwD0Size;
  if (elemWidth == addrGran) {
    hwD0Size = inSize0;
  } else {
    Value scaled = arith::MulIOp::create(builder, loc, inSize0, cst(elemWidth));
    hwD0Size = arith::DivUIOp::create(builder, loc, scaled, cst(addrGran));
  }

  // Warn at compile time if a statically-known d0_size exceeds 10 bits.
  // For a dynamic (SSA) inner size the limit cannot be enforced here: the
  // value is masked into a 10-bit BD field at runtime, so an out-of-range
  // size truncates silently. Warn so the caller knows the bound is now their
  // responsibility (the static path's linear-mode escape hatch does not apply
  // to dynamic transfers).
  if (auto d0Const = getConstantIntValue(mixedSizesRev[0])) {
    uint64_t hwD0 = (*d0Const * elemWidth) / addrGran;
    if (hwD0 > 1023)
      mlir::emitWarning(loc)
          << "hardware d0_size (" << hwD0
          << ") exceeds 10-bit limit (1023); transfer will be truncated. "
             "Consider restructuring to use smaller inner dimensions.";
  } else {
    mlir::emitWarning(loc)
        << "dynamic inner size (d0) cannot be checked against the 10-bit "
           "hardware limit (max 1023 address-gen units); a runtime value that "
           "exceeds it will be silently truncated. Ensure the inner size stays "
           "within range.";
  }

  // Hardware d0_stride: if elemWidth != addrGran, stride = 0; otherwise
  // stride = max(inStride0 - 1, 0) to avoid underflow when stride0 == 0.
  Value hwD0Stride;
  if (elemWidth != addrGran) {
    hwD0Stride = cst(0);
  } else {
    Value strideMinusOne =
        arith::SubIOp::create(builder, loc, inStride0, cst(1));
    Value strideGtZero = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, inStride0, cst(0));
    hwD0Stride = arith::SelectOp::create(builder, loc, strideGtZero,
                                         strideMinusOne, cst(0));
  }

  Value zeroVal = cst(0);
  Value oneVal = cst(1);

  // Scale a stride from element units to address-gen units (no-op when the
  // element width already matches the granularity).
  auto scaleStride = [&](Value inStride) -> Value {
    if (elemWidth == addrGran)
      return inStride;
    Value s = arith::MulIOp::create(builder, loc, inStride, cst(elemWidth));
    return arith::DivUIOp::create(builder, loc, s, cst(addrGran));
  };
  // hw stride = max(scale(inStride) - 1, 0), but only when inSize > 1 (a stride
  // is never applied across a size-1 dimension, so it is forced to 0/"1").
  auto sizeGatedStride = [&](Value inStride, Value inSize) -> Value {
    Value strideMinusOne =
        arith::SubIOp::create(builder, loc, scaleStride(inStride), oneVal);
    Value sizeGt1 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, inSize, oneVal);
    return arith::SelectOp::create(builder, loc, sizeGt1, strideMinusOne,
                                   zeroVal);
  };

  Value hwD1Size = inSize1;
  Value hwD1Stride = sizeGatedStride(inStride1, inSize1);

  Value hwD2Size = inSize2;
  Value hwD2Stride = sizeGatedStride(inStride2, inSize2);

  // iteration_size = inputSizes[3] - 1 when > 1 else 0
  Value hwIterSize;
  {
    Value sizeMinusOne = arith::SubIOp::create(builder, loc, inSize3, oneVal);
    Value sizeGt1 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, inSize3, oneVal);
    hwIterSize =
        arith::SelectOp::create(builder, loc, sizeGt1, sizeMinusOne, zeroVal);
  }

  // iteration_stride = scale(inputStrides[3]) - 1, active only when the
  // iteration dimension both repeats (size > 1) and has a positive stride; a
  // zero stride means "repeat via push queue", which also zeroes iterSize.
  Value hwIterStride;
  {
    Value strideMinusOne =
        arith::SubIOp::create(builder, loc, scaleStride(inStride3), oneVal);
    Value sizeGt1 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, inSize3, oneVal);
    Value strideGt0 = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, inStride3, zeroVal);
    Value active = arith::AndIOp::create(builder, loc, sizeGt1, strideGt0);
    hwIterStride =
        arith::SelectOp::create(builder, loc, active, strideMinusOne, zeroVal);
    hwIterSize =
        arith::SelectOp::create(builder, loc, active, hwIterSize, zeroVal);
  }

  // repeat_count for queue push = max(inputSizes[3] - 1, 0)
  Value rcSub = arith::SubIOp::create(builder, loc, inSize3, cst(1));
  Value rcGtZero = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, inSize3, cst(0));
  Value repeatCount =
      arith::SelectOp::create(builder, loc, rcGtZero, rcSub, cst(0));

  // buffer_length = hwD0Size * d1_size * d2_size
  Value bufLen = arith::MulIOp::create(builder, loc, hwD0Size, hwD1Size);
  bufLen = arith::MulIOp::create(builder, loc, bufLen, hwD2Size);

  return HwBdEncoding{hwD0Size, hwD0Stride, hwD1Size,   hwD1Stride,
                      hwD2Size, hwD2Stride, hwIterSize, hwIterStride,
                      bufLen,   repeatCount};
}

StaticBdPlaceholders
computeStaticBdPlaceholders(ArrayRef<OpFoldResult> mixedSizesRev,
                            ArrayRef<OpFoldResult> mixedStridesRev,
                            uint64_t elemWidth, uint32_t addrGran) {
  auto isConst = [](OpFoldResult ofr) {
    return getConstantIntValue(ofr).has_value();
  };
  auto constOr0 = [](OpFoldResult ofr) -> int64_t {
    return getConstantIntValue(ofr).value_or(0);
  };
  // Scale a value from element units to address-gen units.
  auto scale = [&](int64_t v) -> int64_t {
    return v * static_cast<int64_t>(elemWidth) / static_cast<int64_t>(addrGran);
  };

  bool d0SizeDyn = !isConst(mixedSizesRev[0]);
  bool d1SizeDyn = !isConst(mixedSizesRev[1]);
  bool d2SizeDyn = !isConst(mixedSizesRev[2]);
  bool d3SizeDyn = !isConst(mixedSizesRev[3]);
  bool d0StrideDyn = !isConst(mixedStridesRev[0]);
  bool d1StrideDyn = !isConst(mixedStridesRev[1]);
  bool d2StrideDyn = !isConst(mixedStridesRev[2]);
  bool d3StrideDyn = !isConst(mixedStridesRev[3]);

  StaticBdPlaceholders p;

  if (!d0SizeDyn)
    p.d0Size = scale(constOr0(mixedSizesRev[0]));
  // d0_stride is forced to 0 (hw "1") whenever element width differs from the
  // address granularity; otherwise it is the user stride minus one.
  if (!d0StrideDyn && elemWidth == addrGran)
    p.d0Stride = constOr0(mixedStridesRev[0]) - 1;

  if (!d1SizeDyn)
    p.d1Size = constOr0(mixedSizesRev[1]);
  if (!d1StrideDyn && !d1SizeDyn && constOr0(mixedSizesRev[1]) > 1)
    p.d1Stride = scale(constOr0(mixedStridesRev[1])) - 1;

  if (!d2StrideDyn && !d2SizeDyn && constOr0(mixedSizesRev[2]) > 1)
    p.d2Stride = scale(constOr0(mixedStridesRev[2])) - 1;

  if (!d3SizeDyn) {
    int64_t s3 = constOr0(mixedSizesRev[3]);
    // iteration_size is suppressed when the iteration stride is a known
    // non-positive (repeat-via-queue-push) value.
    bool strideRepeats = !d3StrideDyn && constOr0(mixedStridesRev[3]) <= 0;
    if (s3 > 1 && !strideRepeats)
      p.iterSize = s3 - 1;
  }
  if (!d3StrideDyn && !d3SizeDyn) {
    int64_t s3 = constOr0(mixedSizesRev[3]);
    int64_t st3 = constOr0(mixedStridesRev[3]);
    if (s3 > 1 && st3 > 0)
      p.iterStride = scale(st3) - 1;
  }

  if (!d0SizeDyn && !d1SizeDyn && !d2SizeDyn)
    p.bufLen =
        p.d0Size * constOr0(mixedSizesRev[1]) * constOr0(mixedSizesRev[2]);

  return p;
}

void emitDynamicBdWordOverrides(
    OpBuilder &builder, Location loc, const HwBdEncoding &hw, Value word0BufLen,
    ArrayRef<OpFoldResult> mixedSizesRev,
    ArrayRef<OpFoldResult> mixedStridesRev, uint32_t burstEnc,
    llvm::function_ref<void(uint32_t, Value)> writeWord) {
  auto i32ty = IntegerType::get(builder.getContext(), 32);
  auto cst = [&](int64_t v) {
    return arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, v))
        .getResult();
  };
  auto isDyn = [](OpFoldResult ofr) { return !getConstantIntValue(ofr); };

  bool d0SizeDyn = isDyn(mixedSizesRev[0]);
  bool d1SizeDyn = isDyn(mixedSizesRev[1]);
  bool d2SizeDyn = isDyn(mixedSizesRev[2]);
  bool d3SizeDyn = isDyn(mixedSizesRev[3]);
  bool d0StrideDyn = isDyn(mixedStridesRev[0]);
  bool d1StrideDyn = isDyn(mixedStridesRev[1]);
  bool d2StrideDyn = isDyn(mixedStridesRev[2]);
  bool d3StrideDyn = isDyn(mixedStridesRev[3]);

  // word[0]: buffer_length — dynamic if any of d0/d1/d2 sizes are dynamic.
  if (d0SizeDyn || d1SizeDyn || d2SizeDyn)
    writeWord(0, word0BufLen);

  // word[3]: d0_size [29:20], d0_stride [19:0].
  if (d0SizeDyn || d0StrideDyn)
    writeWord(3,
              buildBdWord(builder, loc,
                          {{hw.d0Size, 0x3FF, 20}, {hw.d0Stride, 0xFFFFF, 0}}));

  // word[4]: burst_length [31:30] (static), d1_size [29:20], d1_stride [19:0].
  if (d1SizeDyn || d1StrideDyn) {
    Value burstVal = cst(static_cast<int64_t>((burstEnc & 0x3u) << 30));
    Value sizeStride = buildBdWord(
        builder, loc, {{hw.d1Size, 0x3FF, 20}, {hw.d1Stride, 0xFFFFF, 0}});
    writeWord(4, arith::OrIOp::create(builder, loc, burstVal, sizeStride));
  }

  // word[5]: AXCache [27:24] (static), d2_stride [19:0].
  if (d2StrideDyn || d2SizeDyn) {
    Value axcache = cst(static_cast<int64_t>((2u & 0xfu) << 24));
    Value strMasked = buildBdWord(builder, loc, {{hw.d2Stride, 0xFFFFF, 0}});
    writeWord(5, arith::OrIOp::create(builder, loc, axcache, strMasked));
  }

  // word[6]: iteration_size [25:20], iteration_stride [19:0].
  if (d3SizeDyn || d3StrideDyn)
    writeWord(
        6, buildBdWord(builder, loc,
                       {{hw.iterSize, 0x3F, 20}, {hw.iterStride, 0xFFFFF, 0}}));
}

} // namespace xilinx::AIEX
