//===- BdLowering.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <limits>

using namespace mlir;

namespace xilinx::AIEX {

//===----------------------------------------------------------------------===//
// SsaStridePolicy: arith-emitting mirror of ConstStridePolicy.
//
// Arithmetic is i32, matching the BD word fields. This agrees with
// ConstStridePolicy's int64 because every value is bounded by a BD field or the
// 32-bit buffer_length, so no intermediate product overflows (a stride big
// enough to overflow exceeds a single shim BD's extent).
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

Value SsaStridePolicy::selectLt(Value a, Value b, Value t, Value e) const {
  Value lt =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, a, b);
  return arith::SelectOp::create(builder, loc, lt, t, e);
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

Value buildArgPlusValue(OpBuilder &builder, Location loc,
                        ArrayRef<OpFoldResult> elementOffsets,
                        ArrayRef<OpFoldResult> strides, int64_t elemWidthBytes,
                        int64_t baseByteOffset) {
  auto i32ty = builder.getIntegerType(32);

  // Fast path: everything constant -> fold to one arith.constant, matching the
  // static lowering byte-for-byte.
  bool allConst = llvm::all_of(elementOffsets,
                               [](OpFoldResult o) {
                                 return getConstantIntValue(o).has_value();
                               }) &&
                  llvm::all_of(strides, [](OpFoldResult s) {
                    return getConstantIntValue(s).has_value();
                  });
  if (allConst) {
    int64_t bytes = baseByteOffset;
    for (auto [o, s] : llvm::zip(elementOffsets, strides)) {
      auto oc = getConstantIntValue(o);
      auto sc = getConstantIntValue(s);
      assert(oc && sc && "allConst already verified these are constant");
      bytes += (*oc) * (*sc) * elemWidthBytes;
    }
    // Reachable only for a constant offset; the runtime path below stays i32.
    if (bytes > std::numeric_limits<uint32_t>::max() || bytes < 0)
      return arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(builder.getIntegerType(64), bytes));
    return arith::ConstantOp::create(builder, loc,
                                     IntegerAttr::get(i32ty, bytes));
  }

  // Runtime path: sum(offset[i] * stride[i]) * elemWidthBytes + base, as arith.
  Value acc =
      arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, 0));
  for (auto [o, s] : llvm::zip(elementOffsets, strides)) {
    Value ov = getAsValue(builder, loc, o, i32ty);
    Value sv = getAsValue(builder, loc, s, i32ty);
    Value prod = arith::MulIOp::create(builder, loc, ov, sv);
    acc = arith::AddIOp::create(builder, loc, acc, prod);
  }
  Value width = arith::ConstantOp::create(
      builder, loc, IntegerAttr::get(i32ty, elemWidthBytes));
  acc = arith::MulIOp::create(builder, loc, acc, width);
  if (baseByteOffset != 0) {
    Value base = arith::ConstantOp::create(
        builder, loc, IntegerAttr::get(i32ty, baseByteOffset));
    acc = arith::AddIOp::create(builder, loc, acc, base);
  }
  return acc;
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

Value getBdRegisterBase(OpBuilder &builder, Location loc,
                        const AIE::AIETargetModel &targetModel, int tileCol,
                        int tileRow, OpFoldResult bdId) {
  auto i32ty = builder.getIntegerType(32);
  if (auto c = getConstantIntValue(bdId))
    return createConstantI32(builder, loc,
                             static_cast<uint32_t>(targetModel.getDmaBdAddress(
                                 tileCol, tileRow, *c)));
  // Runtime bd_id: base + bd_id * bdStride, with base/stride from the (linear)
  // target-model address function.
  uint64_t addrForId0 = targetModel.getDmaBdAddress(tileCol, tileRow, 0);
  uint64_t bdStride =
      targetModel.getDmaBdAddress(tileCol, tileRow, 1) - addrForId0;
  Value bdIdVal = getAsValue(builder, loc, bdId, i32ty);
  return arith::AddIOp::create(
      builder, loc, createConstantI32(builder, loc, addrForId0),
      arith::MulIOp::create(builder, loc, bdIdVal,
                            createConstantI32(builder, loc, bdStride)));
}

namespace {

// What the per-tile packers share, so that only the bit layout is written per
// tile and never the arithmetic. Arrays are innermost-first [d0, d1, d2, iter].
struct EncodedBd {
  Value inS[4], inT[4];  // element counts, as given
  Value hwS[4], hwT[4];  // granule-scaled wraps and -1-biased steps
  Value bufLen;          // buffer_length, in address-gen granules
  Value iterSizeField;   // iteration_size, zeroed for a pure repeat
  Value repeatCount;     // outer-dim repeat for the caller's queue push
  bool isLinear = false; // d0/d1/d2 folded into buffer_length
};

LogicalResult encodeBdCommon(OpBuilder &builder, Location loc,
                             const AIE::AIETargetModel &tm, int tileCol,
                             int tileRow, ArrayRef<OpFoldResult> mixedSizes,
                             ArrayRef<OpFoldResult> mixedStrides,
                             uint64_t elemWidth, Value bufLenOverride,
                             EncodedBd &out) {
  auto i32ty = builder.getIntegerType(32);
  uint32_t gran = tm.getAddressGenGranularity();
  SmallVector<OpFoldResult, 4> sizesRev(llvm::reverse(mixedSizes));
  SmallVector<OpFoldResult, 4> stridesRev(llvm::reverse(mixedStrides));
  for (int i = 0; i < 4; i++) {
    out.inS[i] = getAsValue(builder, loc, sizesRev[i], i32ty);
    out.inT[i] = getAsValue(builder, loc, stridesRev[i], i32ty);
  }
  SsaStridePolicy policy(builder, loc);
  encodeHardwareStridesWraps(policy, elemWidth, gran, out.inS, out.inT, out.hwS,
                             out.hwT);

  // buffer_length: the caller's runtime len if supplied (dma_task), else the
  // d0*d1*d2 hardware-unit size-product (dma_memcpy_nd). hwS[0] already carries
  // the elemWidth/gran scaling; d1/d2 are element counts.
  out.bufLen = bufLenOverride;
  if (!out.bufLen)
    out.bufLen = arith::MulIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, out.hwS[0], out.inS[1]),
        out.inS[2]);

  auto cst = [&](OpFoldResult v) { return getConstantIntValue(v); };
  auto constEq = [&](OpFoldResult v, int64_t c) {
    auto k = cst(v);
    return k && *k == c;
  };
  // A transfer with no data layout transformation at all (mirrors
  // AIEX::isLinearTransfer). Canonicalization zeroes size-1 strides before this
  // runs, hence the stride == 0 tests.
  auto knownLinear = [&]() {
    return constEq(sizesRev[1], 1) && constEq(sizesRev[2], 1) &&
           constEq(stridesRev[0], 1) && constEq(stridesRev[1], 0) &&
           constEq(stridesRev[2], 0);
  };
  // A contiguous row-major scan: each outer stride is the product of the inner
  // sizes (mirrors AIEX::isContiguousTransfer). A runtime operand needed for
  // the test falls back to ND mode.
  auto knownContiguous = [&]() -> bool {
    auto s0 = cst(stridesRev[0]);
    if (!s0 || *s0 != 1)
      return false;
    auto sz0 = cst(sizesRev[0]);
    auto d1sz = cst(sizesRev[1]);
    auto d1st = cst(stridesRev[1]);
    if (!d1sz || *d1sz != 1)
      if (!sz0 || !d1st || *d1st != *sz0)
        return false;
    auto d2sz = cst(sizesRev[2]);
    auto d2st = cst(stridesRev[2]);
    if (!d2sz || *d2sz != 1) {
      auto prod01 =
          (sz0 && d1sz) ? std::optional<int64_t>(*sz0 * *d1sz) : std::nullopt;
      if (!prod01 || !d2st || *d2st != *prod01)
        return false;
    }
    return true;
  };
  // Folding d0/d1/d2 into buffer_length dodges the 10-bit d0 wrap, but only a
  // shim NOC tile's 32-bit length field can absorb a merely-contiguous scan
  // (17 bits on a mem tile, 14 on a core tile). The static path draws the line
  // in the same place, in AIEDMATasksToNPU.cpp's treatAsLinear.
  out.isLinear = knownLinear() ||
                 (tm.isShimNOCTile(tileCol, tileRow) && knownContiguous());

  // Guard a RUNTIME value against a BD field too narrow to hold it; the
  // packer's mask would otherwise truncate it silently on hardware. Constants
  // are verifier-checked instead.
  auto guardField = [&](OpFoldResult in, Value hwVal, int64_t fieldMax) {
    if (getConstantIntValue(in))
      return;
    NpuAssertBdFieldOp::create(builder, loc, hwVal,
                               builder.getI32IntegerAttr(fieldMax));
  };
  int64_t wrapMax = (1ll << tm.getDmaBdWrapBits(tileCol, tileRow)) - 1;
  int64_t iterMax = (1ll << tm.getDmaBdIterBits(tileCol, tileRow)) - 1;
  int64_t stepMax = (1ll << tm.getDmaBdStepBits(tileCol, tileRow)) - 1;
  if (!out.isLinear) {
    guardField(sizesRev[0], out.hwS[0], wrapMax);
    guardField(sizesRev[1], out.hwS[1], wrapMax);
    // The step fields are narrower than the values that reach them (17 bits on
    // a mem tile, 13 on a core tile), so a runtime stride needs the same guard
    // a constant gets from verifyStridesWraps.
    guardField(stridesRev[0], out.hwT[0], stepMax);
    guardField(stridesRev[1], out.hwT[1], stepMax);
    guardField(stridesRev[2], out.hwT[2], stepMax);
  }
  guardField(sizesRev[3], out.hwS[3], iterMax);
  // hwT[3] collapses to 0 unless the iteration wrap is > 1, which is the same
  // condition under which verifyStridesWraps exempts it -- so this guard is
  // inert in the pure-repeat case rather than rejecting it.
  guardField(stridesRev[3], out.hwT[3], stepMax);

  // Neither check fires on a shim NOC tile, whose buffer_length owns the whole
  // 32-bit word -- leaving that stream untouched.
  uint64_t lenMax = tm.getDmaBdMaxLen(tileCol, tileRow);
  if (lenMax < std::numeric_limits<uint32_t>::max()) {
    if (auto constLen = getConstantIntValue(out.bufLen)) {
      if (*constLen < 0 || (uint64_t)*constLen > lenMax)
        return emitError(loc)
               << "buffer length of " << *constLen
               << " address-generation granules exceeds the " << lenMax
               << " this tile's BD buffer_length field can hold.";
    } else {
      NpuAssertBdFieldOp::create(builder, loc, out.bufLen,
                                 builder.getI32IntegerAttr((int64_t)lenMax));
    }
  }

  // Guard a RUNTIME size/stride whose byte extent must be a whole number of
  // granules (mirrors verifyStridesWraps). Guard is on the input element count;
  // constants are verifier-checked.
  int64_t divisor = bdGranuleDivisor(elemWidth, gran);
  auto guardDivisible = [&](OpFoldResult in, Value inVal, bool allowUnit) {
    if (divisor <= 1 || getConstantIntValue(in))
      return;
    NpuAssertBdDivisibleOp::create(builder, loc, inVal, (uint32_t)divisor,
                                   /*allow_unit=*/allowUnit);
  };
  guardDivisible(sizesRev[0], out.inS[0], /*allowUnit=*/false);
  // The innermost stride collapses to hardware 0 when contiguous or
  // granule-aligned; guard with the unit-stride exemption. Outer strides don't
  // collapse.
  guardDivisible(stridesRev[0], out.inT[0], /*allowUnit=*/true);
  for (int i = 1; i < 4; i++)
    guardDivisible(stridesRev[i], out.inT[i], /*allowUnit=*/false);

  // iteration_size. A zero outer stride is a pure repeat (carried by
  // repeat_count), so the field must be 0 like AIEDmaToNpu; gate it on the
  // stride while leaving hwS[3] for repeatCount. hwT[3] already collapses to 0.
  Value zeroI32 = createConstantI32(builder, loc, 0);
  Value iterStridePos = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, out.inT[3], zeroI32);
  out.iterSizeField =
      arith::SelectOp::create(builder, loc, iterStridePos, out.hwS[3], zeroI32);

  // repeat_count for the queue push is the biased outer size (N > 1 ? N - 1 :
  // 0), matching the static path. A constant folds so the static push_queue
  // lowering can consume it; a runtime size yields the SSA hwS[3].
  if (auto outerConst = getConstantIntValue(sizesRev[3])) {
    int64_t r = *outerConst > 1 ? *outerConst - 1 : 0;
    out.repeatCount =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, r));
  } else {
    out.repeatCount = out.hwS[3];
  }
  return success();
}

// The three packers below are the dynamic mirror of the per-tile packing in
// AIEDmaToNpu.cpp's WriteBdToBlockWritePattern, laid out to be read side by
// side against it. dynamic-matches-static-words.mlir enforces that they agree.

// Shim NOC BD: 8 registers.
void packShimBdWords(OpBuilder &builder, Location loc,
                     const AIE::AIETargetModel &tm, const BdTemplateFields &f,
                     const EncodedBd &e, uint32_t burstLength, uint32_t axcache,
                     SmallVectorImpl<Value> &wordsOut) {
  wordsOut.assign(8, createConstantI32(builder, loc, 0));
  // word[0] buffer_length [31:0].
  wordsOut[0] = e.bufLen;
  // word[1] buffer_offset stays 0 (the address patch supplies the pointer).
  // word[2] enable_packet [30], out_of_order_id [29:24], packet_id [23:19],
  // packet_type [18:16].
  wordsOut[2] = createConstantI32(
      builder, loc,
      ((f.enable_packet & 0x1) << 30) | ((f.out_of_order_id & 0x3f) << 24) |
          ((f.packet_id & 0x1f) << 19) | ((f.packet_type & 0x7) << 16));
  // word[4] burst_length [31:30]; d1 fields overlaid below in ND mode.
  wordsOut[4] = createConstantI32(
      builder, loc,
      (AIE::getShimBurstLengthEncoding(tm, burstLength) & 0x3) << 30);
  // word[5] AXCache [27:24]; d2_stride overlaid below in ND mode.
  wordsOut[5] = createConstantI32(builder, loc, (axcache & 0xf) << 24);
  // word[7] next_bd [30:27], use_next_bd [26], valid_bd [25], lock fields.
  wordsOut[7] = createConstantI32(
      builder, loc,
      ((f.next_bd_id & 0xf) << 27) | ((f.use_next_bd & 0x1) << 26) |
          (1u << 25) | ((f.lock_rel_val & 0x7f) << 18) |
          ((f.lock_rel_id & 0xf) << 13) | ((f.lock_acq_enable & 0x1) << 12) |
          ((f.lock_acq_val & 0x7f) << 5) | (f.lock_acq_id & 0xf));

  // Linear mode needs only buffer_length + iteration, so the d0/d1/d2
  // size/stride fields stay zero; words 4/5 OR onto the burst_length / AXCache
  // bits set above.
  if (!e.isLinear) {
    // word[3]: d0_size [29:20], d0_stride [19:0].
    wordsOut[3] = buildBdWord(builder, loc,
                              {{e.hwS[0], 0x3FF, 20}, {e.hwT[0], 0xFFFFF, 0}});
    // word[4]: d1_size [29:20], d1_stride [19:0].
    wordsOut[4] = arith::OrIOp::create(
        builder, loc, wordsOut[4],
        buildBdWord(builder, loc,
                    {{e.hwS[1], 0x3FF, 20}, {e.hwT[1], 0xFFFFF, 0}}));
    // word[5]: d2_stride [19:0]. Shim d2_size is always 0, carried by bufLen.
    wordsOut[5] = arith::OrIOp::create(
        builder, loc, wordsOut[5],
        buildBdWord(builder, loc, {{e.hwT[2], 0xFFFFF, 0}}));
  }
  // word[6]: iteration_size [25:20], iteration_stride [19:0].
  wordsOut[6] = buildBdWord(
      builder, loc, {{e.iterSizeField, 0x3F, 20}, {e.hwT[3], 0xFFFFF, 0}});
}

// Mem tile BD: 8 registers, a different layout from shim throughout.
void packMemTileBdWords(OpBuilder &builder, Location loc,
                        const BdTemplateFields &f, const EncodedBd &e,
                        SmallVectorImpl<Value> &wordsOut) {
  wordsOut.assign(8, createConstantI32(builder, loc, 0));
  // word[0]: enable_packet [31], packet_type [30:28], packet_id [27:23],
  // out_of_order_id [22:17], buffer_length [16:0]. Unlike shim, where
  // buffer_length owns the whole word, here it shares word 0 with the packet
  // header -- so the (possibly runtime) length is OR'd into a constant
  // template. The 17-bit width is guarded in encodeBdCommon.
  wordsOut[0] = arith::OrIOp::create(
      builder, loc,
      createConstantI32(builder, loc,
                        ((f.enable_packet & 0x1) << 31) |
                            ((f.packet_type & 0x7) << 28) |
                            ((f.packet_id & 0x1f) << 23) |
                            ((f.out_of_order_id & 0x3f) << 17)),
      buildBdWord(builder, loc, {{e.bufLen, 0x1FFFF, 0}}));
  // word[1]: d0_zero_before [31:26], next_bd [25:20], use_next_bd [19],
  // buffer_offset [18:0]. buffer_offset stays 0 -- the buffer pointer is
  // written separately by setAddressForSingleBD's masked write, exactly as on
  // the static path. Padding is rejected on this path by the caller.
  wordsOut[1] = createConstantI32(builder, loc,
                                  ((f.next_bd_id & 0x3f) << 20) |
                                      ((f.use_next_bd & 0x1) << 19));
  if (!e.isLinear) {
    // word[2]: d0_size [26:17], d0_stride [16:0].
    wordsOut[2] = buildBdWord(builder, loc,
                              {{e.hwS[0], 0x3FF, 17}, {e.hwT[0], 0x1FFFF, 0}});
    // word[3]: d1_zero_before [31:27], d1_size [26:17], d1_stride [16:0].
    wordsOut[3] = buildBdWord(builder, loc,
                              {{e.hwS[1], 0x3FF, 17}, {e.hwT[1], 0x1FFFF, 0}});
    // word[4]: d2_zero_before [30:27], d2_stride [16:0]. D2_Size is a dead
    // field here: the static packing never writes it either (its `// TODO:
    // D2Size`), the d2 repeat being carried entirely by buffer_length.
    wordsOut[4] = buildBdWord(builder, loc, {{e.hwT[2], 0x1FFFF, 0}});
  }
  // word[5] holds only the zero-after pad fields, all zero here.
  // word[6]: iteration_current [28:23], iteration_size [22:17],
  // iteration_stride [16:0].
  wordsOut[6] = buildBdWord(
      builder, loc, {{e.iterSizeField, 0x3F, 17}, {e.hwT[3], 0x1FFFF, 0}});
  // word[7]: valid_bd [31], lock_rel_val [30:24], lock_rel_id [23:16],
  // lock_acq_enable [15], lock_acq_val [14:8], lock_acq_id [7:0]. Note the
  // 8-bit lock ids against shim's 4: gatherBdTemplateFields adds the mem tile's
  // getLockLocalBaseIndex offset, which does not fit in 4 bits.
  wordsOut[7] = createConstantI32(
      builder, loc,
      (1u << 31) | ((f.lock_rel_val & 0x7f) << 24) |
          ((f.lock_rel_id & 0xff) << 16) | ((f.lock_acq_enable & 0x1) << 15) |
          ((f.lock_acq_val & 0x7f) << 8) | (f.lock_acq_id & 0xff));
}

// Core tile BD: 6 registers.
void packCoreTileBdWords(OpBuilder &builder, Location loc,
                         const BdTemplateFields &f, const EncodedBd &e,
                         SmallVectorImpl<Value> &wordsOut) {
  wordsOut.assign(6, createConstantI32(builder, loc, 0));
  // word[0]: base_address [27:14], buffer_length [13:0]. The address bits stay
  // zero here and are filled in by setAddressForSingleBD's masked write, which
  // preserves the length bits. The 14-bit length is guarded in encodeBdCommon.
  wordsOut[0] = buildBdWord(builder, loc, {{e.bufLen, 0x3FFF, 0}});
  // word[1]: enable_compression [31], enable_packet [30],
  // out_of_order_id [29:24], packet_id [23:19], packet_type [18:16].
  wordsOut[1] = createConstantI32(
      builder, loc,
      ((f.enable_packet & 0x1) << 30) | ((f.out_of_order_id & 0x3f) << 24) |
          ((f.packet_id & 0x1f) << 19) | ((f.packet_type & 0x7) << 16));
  if (!e.isLinear) {
    // word[2]: d1_stride [25:13], d0_stride [12:0].
    wordsOut[2] = buildBdWord(builder, loc,
                              {{e.hwT[1], 0x1FFF, 13}, {e.hwT[0], 0x1FFF, 0}});
    // word[3]: d1_size [28:21], d0_size [20:13], d2_stride [12:0]. Note the
    // 8-bit wraps, against 10 on shim and mem tiles.
    wordsOut[3] = buildBdWord(
        builder, loc,
        {{e.hwS[1], 0xFF, 21}, {e.hwS[0], 0xFF, 13}, {e.hwT[2], 0x1FFF, 0}});
  }
  // word[4]: iteration_current [24:19], iteration_size [18:13],
  // iteration_stride [12:0].
  wordsOut[4] = buildBdWord(
      builder, loc, {{e.iterSizeField, 0x3F, 13}, {e.hwT[3], 0x1FFF, 0}});
  // word[5]: tlast_suppress [31], next_bd [30:27], use_next_bd [26],
  // valid_bd [25], lock_rel_val [24:18], lock_rel_id [16:13],
  // lock_acq_enable [12], lock_acq_val [11:5], lock_acq_id [3:0].
  wordsOut[5] = createConstantI32(
      builder, loc,
      ((f.next_bd_id & 0xf) << 27) | ((f.use_next_bd & 0x1) << 26) |
          (1u << 25) | ((f.lock_rel_val & 0x7f) << 18) |
          ((f.lock_rel_id & 0xf) << 13) | ((f.lock_acq_enable & 0x1) << 12) |
          ((f.lock_acq_val & 0x7f) << 5) | (f.lock_acq_id & 0xf));
}

} // namespace

LogicalResult
buildBdWords(OpBuilder &builder, Location loc,
             const AIE::AIETargetModel &targetModel, int tileCol, int tileRow,
             const BdTemplateFields &f, ArrayRef<OpFoldResult> mixedSizes,
             ArrayRef<OpFoldResult> mixedStrides, uint64_t elemWidth,
             uint32_t burstLength, uint32_t axcache, Value bufLenOverride,
             Value &repeatCountOut, SmallVectorImpl<Value> &wordsOut) {
  EncodedBd e;
  if (failed(encodeBdCommon(builder, loc, targetModel, tileCol, tileRow,
                            mixedSizes, mixedStrides, elemWidth, bufLenOverride,
                            e)))
    return failure();
  repeatCountOut = e.repeatCount;

  if (targetModel.isShimNOCTile(tileCol, tileRow))
    packShimBdWords(builder, loc, targetModel, f, e, burstLength, axcache,
                    wordsOut);
  else if (targetModel.isMemTile(tileCol, tileRow))
    packMemTileBdWords(builder, loc, f, e, wordsOut);
  else if (targetModel.isCoreTile(tileCol, tileRow))
    packCoreTileBdWords(builder, loc, f, e, wordsOut);
  else
    llvm_unreachable("buildBdWords called for a tile type with no DMA BD "
                     "layout (rejected by the caller)");
  return success();
}

} // namespace xilinx::AIEX
