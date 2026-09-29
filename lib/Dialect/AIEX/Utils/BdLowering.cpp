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
// ConstStridePolicy's int64 because encodeBdCommon bounds every runtime operand
// before it reaches the policy, so no intermediate product overflows.
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

// Arrays are innermost-first: [d0, d1, d2, iter].
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
  // The encoding below is i32 arithmetic, and a RUNTIME operand that does not
  // fit it wraps: an i64 is truncated (2^32 + 1 becomes 1), and d0's size and
  // every stride are multiplied by elemWidth before they are scaled down to
  // granules. A wrapped value can then pass the field guards further down, so
  // bound each operand at its original width first. Only the d1..d3 sizes are
  // used unscaled, and an i32 one needs no bound. Every BD field is far
  // narrower than these bounds, so no valid value is rejected here.
  int64_t scaledMax =
      std::numeric_limits<int32_t>::max() / std::max<uint64_t>(elemWidth, 1);
  auto guardOperand = [&](OpFoldResult in, bool scaled) {
    if (getConstantIntValue(in))
      return;
    Value v = cast<Value>(in);
    if (!scaled && v.getType().getIntOrFloatBitWidth() <= 32)
      return;
    NpuAssertBdFieldOp::create(
        builder, loc, v,
        builder.getI32IntegerAttr(
            scaled ? scaledMax : std::numeric_limits<int32_t>::max()));
  };
  for (int i = 0; i < 4; i++) {
    guardOperand(sizesRev[i], /*scaled=*/i == 0);
    guardOperand(stridesRev[i], /*scaled=*/true);
  }
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
  uint64_t lenMax = tm.getDmaBdMaxLen(tileCol, tileRow);
  bool lenGuarded = false;
  out.bufLen = bufLenOverride;
  if (!out.bufLen) {
    auto cst0 = [&](int i) { return getConstantIntValue(sizesRev[i]); };
    bool unitOuter = [&] {
      auto s1 = cst0(1), s2 = cst0(2);
      return s1 && *s1 == 1 && s2 && *s2 == 1;
    }();
    if ((cst0(0) && cst0(1) && cst0(2)) || unitOuter) {
      out.bufLen = arith::MulIOp::create(
          builder, loc,
          arith::MulIOp::create(builder, loc, out.hwS[0], out.inS[1]),
          out.inS[2]);
    } else {
      // A runtime factor can make the i32 product wrap into a length that
      // passes its guard, so multiply in i64 and bound each partial product.
      // hwS[0] is below 2^31 once its operand guard holds, each d1/d2 size
      // below 2^32 unsigned, and a partial product that passed its guard below
      // 2^31, so no product wraps in i64 before the guard that rejects it.
      auto i64ty = builder.getI64Type();
      auto wide = [&](Value v) -> Value {
        return arith::ExtUIOp::create(builder, loc, i64ty, v);
      };
      auto cap = (int64_t)std::min<uint64_t>(
          lenMax, std::numeric_limits<int32_t>::max());
      Value len = wide(out.hwS[0]);
      for (int i = 1; i < 3; i++) {
        if (auto c = cst0(i); c && *c == 1)
          continue;
        len = arith::MulIOp::create(builder, loc, len, wide(out.inS[i]));
        NpuAssertBdFieldOp::create(builder, loc, len,
                                   builder.getI32IntegerAttr(cap));
      }
      out.bufLen = arith::TruncIOp::create(builder, loc, i32ty, len);
      lenGuarded = true;
    }
  }

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
  // 32-bit word, nor on a size-product the i64 path above already bounded.
  if (!lenGuarded && lenMax < std::numeric_limits<uint32_t>::max()) {
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

// dynamic-matches-static-words.mlir holds this to WriteBdToBlockWritePattern.
void packBdWords(OpBuilder &builder, Location loc,
                 const AIE::AIETargetModel &tm, const AIE::DmaBdLayout &layout,
                 const BdTemplateFields &f, const EncodedBd &e,
                 uint32_t burstLength, uint32_t axcache,
                 SmallVectorImpl<Value> &wordsOut) {
  using Field = std::tuple<Value, uint32_t, uint32_t>;
  SmallVector<uint32_t, 8> constWords(layout.numWords, 0);
  SmallVector<SmallVector<Field, 3>, 8> runtimeWords(layout.numWords);
  auto set = [&](const AIE::DmaBdField &field, uint64_t value) {
    if (field.exists())
      constWords[field.word] |= field.place(value);
  };
  auto add = [&](const AIE::DmaBdField &field, Value value) {
    if (field.exists())
      runtimeWords[field.word].push_back({value, field.mask(), field.shift});
  };

  set(layout.enablePacket, f.enable_packet);
  set(layout.packetType, f.packet_type);
  set(layout.packetId, f.packet_id);
  set(layout.outOfOrderId, f.out_of_order_id);
  if (layout.burstLength.exists())
    set(layout.burstLength, AIE::getShimBurstLengthEncoding(tm, burstLength));
  set(layout.axcache, axcache);
  set(layout.nextBd, f.next_bd_id);
  set(layout.useNextBd, f.use_next_bd);
  set(layout.validBd, 1);
  set(layout.lockRelValue, f.lock_rel_val);
  set(layout.lockRelId, f.lock_rel_id);
  set(layout.lockAcqEnable, f.lock_acq_enable);
  set(layout.lockAcqValue, f.lock_acq_val);
  set(layout.lockAcqId, f.lock_acq_id);

  add(layout.bufferLength, e.bufLen);
  if (!e.isLinear) {
    add(layout.d0Size, e.hwS[0]);
    add(layout.d0Stride, e.hwT[0]);
    add(layout.d1Size, e.hwS[1]);
    add(layout.d1Stride, e.hwT[1]);
    add(layout.d2Stride, e.hwT[2]);
  }
  add(layout.iterationSize, e.iterSizeField);
  add(layout.iterationStride, e.hwT[3]);

  wordsOut.clear();
  for (auto [constWord, fields] : llvm::zip(constWords, runtimeWords)) {
    if (fields.empty()) {
      wordsOut.push_back(createConstantI32(builder, loc, constWord));
      continue;
    }
    auto [value, mask, shift] = fields.front();
    if (constWord == 0 && fields.size() == 1 && mask == 0xFFFFFFFF &&
        shift == 0) {
      wordsOut.push_back(value);
      continue;
    }
    Value word = buildBdWord(builder, loc, fields);
    if (constWord != 0)
      word = arith::OrIOp::create(
          builder, loc, createConstantI32(builder, loc, constWord), word);
    wordsOut.push_back(word);
  }
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

  const AIE::DmaBdLayout *layout = targetModel.getDmaBdLayout(tileCol, tileRow);
  assert(layout && "buildBdWords called for a tile with no DMA BD layout "
                   "(rejected by the caller)");
  packBdWords(builder, loc, targetModel, *layout, f, e, burstLength, axcache,
              wordsOut);
  return success();
}

} // namespace xilinx::AIEX
