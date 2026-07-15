//===- BdLowering.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;

namespace xilinx::AIEX {

//===----------------------------------------------------------------------===//
// SsaStridePolicy: arith-emitting mirror of ConstStridePolicy.
//===----------------------------------------------------------------------===//

Value SsaStridePolicy::cst(int64_t c) const {
  auto i32ty = builder.getIntegerType(32);
  return arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, c));
}

Value SsaStridePolicy::mul(Value v, int64_t c) const {
  return arith::MulIOp::create(builder, loc, v, cst(c));
}

Value SsaStridePolicy::div(Value v, int64_t c) const {
  // Hardware granularity scaling is exact (verifier enforces divisibility);
  // unsigned division matches the constant policy on the non-negative values
  // that reach here.
  return arith::DivUIOp::create(builder, loc, v, cst(c));
}

Value SsaStridePolicy::sub(Value v, int64_t c) const {
  return arith::SubIOp::create(builder, loc, v, cst(c));
}

Value SsaStridePolicy::selectGT1(Value cond, Value t, Value e) const {
  Value one = cst(1);
  Value gt =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, cond, one);
  return arith::SelectOp::create(builder, loc, gt, t, e);
}

Value SsaStridePolicy::selectGT0(Value cond, Value t, Value e) const {
  Value zero = cst(0);
  Value gt = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                                   cond, zero);
  return arith::SelectOp::create(builder, loc, gt, t, e);
}

//===----------------------------------------------------------------------===//
// Layout helpers.
//===----------------------------------------------------------------------===//

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
  auto i32ty = builder.getIntegerType(32);
  Value result =
      arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, 0));
  for (auto &[val, mask, shift] : fields) {
    Value masked = val;
    if (mask != 0xFFFFFFFF) {
      auto maskConst = arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(i32ty, (int64_t)(int32_t)mask));
      masked = arith::AndIOp::create(builder, loc, masked, maskConst);
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

LogicalResult emitDynamicShimBdWordOverrides(
    OpBuilder &builder, Location loc, const AIE::AIETargetModel &targetModel,
    int tileCol, int tileRow, uint32_t bdId, ArrayRef<OpFoldResult> mixedSizes,
    ArrayRef<OpFoldResult> mixedStrides, uint64_t elemWidth,
    uint32_t burstLength, Value bufLenOverride, Value &repeatCountOut) {
  auto i32ty = builder.getIntegerType(32);

  // Compute the hardware sizes/strides as SSA values via the shared encoder.
  uint32_t gran = targetModel.getAddressGenGranularity();
  SmallVector<OpFoldResult, 4> sizesRev(llvm::reverse(mixedSizes));
  SmallVector<OpFoldResult, 4> stridesRev(llvm::reverse(mixedStrides));
  Value inS[4], inT[4], hwS[4], hwT[4];
  for (int i = 0; i < 4; i++) {
    inS[i] = getAsValue(builder, loc, sizesRev[i], i32ty);
    inT[i] = getAsValue(builder, loc, stridesRev[i], i32ty);
  }
  int64_t innerStride0 = getConstantIntValue(stridesRev[0]).value();
  SsaStridePolicy policy(builder, loc, innerStride0);
  encodeHardwareStridesWraps(policy, elemWidth, gran, inS, inT, hwS, hwT);

  // buffer_length (word[0]): the caller's runtime len if supplied (dma_task),
  // else the d0*d1*d2 hardware-unit size-product (dma_memcpy_nd). hwS[0]
  // already carries the elemWidth/gran scaling; d1/d2 are element counts.
  Value bufLen = bufLenOverride;
  if (!bufLen)
    bufLen = arith::MulIOp::create(
        builder, loc, arith::MulIOp::create(builder, loc, hwS[0], inS[1]),
        inS[2]);

  // Linear-mode decision. This is a RUNTIME-AWARE extension of the static
  // isContiguousTransfer, not a reimplementation: that helper reads a
  // dimension's own size (`sizes[i] > 1`), so it cannot classify a transfer
  // whose d1/d2 SIZE is runtime. Here a runtime outer size is still provably
  // contiguous when its stride equals the product of the (constant) inner sizes
  // -- contiguity never needs a dimension's own size, only the compared stride
  // and the strictly-inner sizes (the common "runtime block count, fixed block
  // shape" case). When a NEEDED operand is runtime, contiguity is undecidable
  // and we fall back to ND mode (safe: a runtime d0/d1 size that could overflow
  // the 10-bit field is guarded below). A contiguous transfer folds d0/d1 into
  // buffer_length, dodging the 10-bit d0_size limit.
  auto cst = [&](OpFoldResult v) { return getConstantIntValue(v); };
  auto knownContiguous = [&]() -> bool {
    auto s0 = cst(stridesRev[0]);
    if (!s0 || *s0 != 1)
      return false;
    auto sz0 = cst(sizesRev[0]);
    auto d1sz = cst(sizesRev[1]);
    auto d1st = cst(stridesRev[1]);
    if (!(d1sz && *d1sz == 1))
      if (!sz0 || !d1st || *d1st != *sz0)
        return false;
    auto d2sz = cst(sizesRev[2]);
    auto d2st = cst(stridesRev[2]);
    if (!(d2sz && *d2sz == 1)) {
      auto prod01 =
          (sz0 && d1sz) ? std::optional<int64_t>(*sz0 * *d1sz) : std::nullopt;
      if (!prod01 || !d2st || *d2st != *prod01)
        return false;
    }
    return true;
  };
  bool isLinear = knownContiguous();

  // Host-side bounds guard for a RUNTIME size landing in a narrow BD field
  // (masking would silently truncate an out-of-range value). Constant sizes are
  // already range-checked by the verifier, so guard only runtime ones, and only
  // for fields actually used: d0/d1 wrap (10-bit) exist in ND mode; the
  // iteration wrap (6-bit) always. The guard is on the hardware value.
  auto guardField = [&](OpFoldResult inSize, Value hwVal, int64_t fieldMax) {
    if (getConstantIntValue(inSize))
      return; // constant: verifier already enforced the bound.
    NpuAssertBdFieldOp::create(builder, loc, hwVal,
                               builder.getI32IntegerAttr(fieldMax));
  };
  if (!isLinear) {
    guardField(sizesRev[0], hwS[0], (1 << 10) - 1);
    guardField(sizesRev[1], hwS[1], (1 << 10) - 1);
  }
  guardField(sizesRev[3], hwS[3], (1 << 6) - 1);

  uint64_t bdAddr = targetModel.getDmaBdAddress(tileCol, tileRow, bdId);
  auto writeWord = [&](uint32_t wordIdx, Value val) {
    NpuWrite32Op::create(
        builder, loc,
        createConstantI32(builder, loc,
                          static_cast<uint32_t>(bdAddr + wordIdx * 4)),
        val, nullptr, nullptr, nullptr);
  };

  // word[0] buffer_length is always overridden (it carries the runtime element
  // count in both linear and ND modes).
  writeWord(0, bufLen);

  // In linear mode the d0/d1/d2 size/stride words stay at their zero template
  // values (matching the static path); only buffer_length + iteration are
  // meaningful. In ND mode, override the size/stride words.
  if (!isLinear) {
    // word[3]: d0_size [29:20], d0_stride [19:0].
    writeWord(3, buildBdWord(builder, loc,
                             {{hwS[0], 0x3FF, 20}, {hwT[0], 0xFFFFF, 0}}));
    // word[4]: burst_length [31:30] (static), d1_size [29:20], d1_stride
    // [19:0].
    Value burst = createConstantI32(
        builder, loc,
        (AIE::getShimBurstLengthEncoding(targetModel, burstLength) & 0x3)
            << 30);
    writeWord(4, arith::OrIOp::create(
                     builder, loc, burst,
                     buildBdWord(builder, loc,
                                 {{hwS[1], 0x3FF, 20}, {hwT[1], 0xFFFFF, 0}})));
    // word[5]: AXCache [27:24] (static), d2_stride [19:0]. Shim d2_size is
    //          always 0 (the template already has it), carried by bufLen.
    Value axcache = createConstantI32(builder, loc, (2u & 0xf) << 24);
    writeWord(5, arith::OrIOp::create(
                     builder, loc, axcache,
                     buildBdWord(builder, loc, {{hwT[2], 0xFFFFF, 0}})));
  }
  // word[6]: iteration_size [25:20], iteration_stride [19:0]. Meaningful in
  // both modes (the outer repeat dimension is independent of linearization).
  writeWord(
      6, buildBdWord(builder, loc, {{hwS[3], 0x3F, 20}, {hwT[3], 0xFFFFF, 0}}));

  // repeat_count for the queue push is the biased hw iteration value (matching
  // the static path's `repeat_count = sizes[3]`), NOT the raw outer size: an
  // outer size of N encodes to (N > 1 ? N - 1 : 0). When the outer size is a
  // compile-time constant, emit a foldable arith.constant so the static
  // push_queue lowering can consume it; only a genuinely runtime outer size
  // yields the SSA-computed hwS[3].
  if (auto outerConst = getConstantIntValue(sizesRev[3])) {
    int64_t r = *outerConst > 1 ? *outerConst - 1 : 0;
    repeatCountOut =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, r));
  } else {
    repeatCountOut = hwS[3];
  }
  return success();
}

} // namespace xilinx::AIEX
