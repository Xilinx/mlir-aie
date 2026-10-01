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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Iterators.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <numeric>

using namespace mlir;

namespace xilinx::AIEX {

//===----------------------------------------------------------------------===//
// SsaStridePolicy: arith-emitting mirror of ConstStridePolicy.
//
// Arithmetic is i64, like ConstStridePolicy. Operands arrive zero-extended, so
// the comparisons are unsigned; a negative runtime value reads as a huge one,
// which the host-side guards in buildShimBdWords reject before the encoded
// fields are used.
//===----------------------------------------------------------------------===//

Value SsaStridePolicy::cst(int64_t c) const {
  return arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(c));
}

Value SsaStridePolicy::mul(Value v, int64_t c) const {
  return builder.createOrFold<arith::MulIOp>(loc, v, cst(c));
}

Value SsaStridePolicy::div(Value v, int64_t c) const {
  // Hardware granularity scaling is exact (verifier/guards enforce
  // divisibility).
  return builder.createOrFold<arith::DivUIOp>(loc, v, cst(c));
}

Value SsaStridePolicy::sub(Value v, int64_t c) const {
  return builder.createOrFold<arith::SubIOp>(loc, v, cst(c));
}

Value SsaStridePolicy::selectGT1(Value cond, Value t, Value e) const {
  Value gt = builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                 cond, cst(1));
  return builder.createOrFold<arith::SelectOp>(loc, gt, t, e);
}

Value SsaStridePolicy::selectGT0(Value cond, Value t, Value e) const {
  Value gt = builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                 cond, cst(0));
  return builder.createOrFold<arith::SelectOp>(loc, gt, t, e);
}

Value SsaStridePolicy::selectLt(Value a, Value b, Value t, Value e) const {
  Value lt =
      builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, a, b);
  return builder.createOrFold<arith::SelectOp>(loc, lt, t, e);
}

//===----------------------------------------------------------------------===//
// Host-side guards.
//===----------------------------------------------------------------------===//

LogicalResult emitRuntimeCheck(OpBuilder &builder, Location loc, Value cond,
                               const Twine &message) {
  if (auto c = getConstantIntValue(cond)) {
    if (*c)
      return success();
    return emitError(loc) << message;
  }
  cf::AssertOp::create(builder, loc, cond, builder.getStringAttr(message));
  return success();
}

void eraseDeadArith(Operation *root) {
  root->walk<WalkOrder::PostOrder, ReverseIterator>([](Operation *op) {
    if (isa<arith::ArithDialect>(op->getDialect()) && isOpTriviallyDead(op))
      op->erase();
  });
}

namespace {

// Folding unsigned i64 arithmetic for building guard conditions: over
// constant operands a condition folds to a constant, which emitRuntimeCheck
// drops (true) or reports at compile time (false).
struct GuardBuilder {
  OpBuilder &b;
  Location loc;

  Value cst(uint64_t c) {
    return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr((int64_t)c));
  }
  Value add(Value x, Value y) {
    return b.createOrFold<arith::AddIOp>(loc, x, y);
  }
  Value sub(Value x, uint64_t c) {
    return b.createOrFold<arith::SubIOp>(loc, x, cst(c));
  }
  Value mul(Value x, Value y) {
    return b.createOrFold<arith::MulIOp>(loc, x, y);
  }
  Value both(Value x, Value y) {
    return b.createOrFold<arith::AndIOp>(loc, x, y);
  }
  Value either(Value x, Value y) {
    return b.createOrFold<arith::OrIOp>(loc, x, y);
  }
  Value cmp(arith::CmpIPredicate pred, Value x, Value y) {
    return b.createOrFold<arith::CmpIOp>(loc, pred, x, y);
  }
  Value eq(Value x, Value y) { return cmp(arith::CmpIPredicate::eq, x, y); }
  Value eq(Value x, uint64_t c) { return eq(x, cst(c)); }
  Value ule(Value x, uint64_t c) {
    return cmp(arith::CmpIPredicate::ule, x, cst(c));
  }
  // 1 <= x <= hi as one unsigned compare: x - 1 wraps 0 to UINT64_MAX.
  Value inRange1(Value x, uint64_t hi) { return ule(sub(x, 1), hi - 1); }
  Value multipleOf(Value x, uint64_t d) {
    return eq(b.createOrFold<arith::RemUIOp>(loc, x, cst(d)), 0);
  }
  LogicalResult check(Value cond, const Twine &message) {
    return emitRuntimeCheck(b, loc, cond, message);
  }
};

} // namespace

Value getAsI64(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  Type i64 = builder.getI64Type();
  if (auto c = getConstantIntValue(ofr))
    return Value(
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i64, *c)));
  Value v = cast<Value>(ofr);
  if (v.getType().isIndex())
    return Value(arith::IndexCastUIOp::create(builder, loc, i64, v));
  unsigned bits = v.getType().getIntOrFloatBitWidth();
  if (bits < 64)
    return Value(arith::ExtUIOp::create(builder, loc, i64, v));
  if (bits > 64) {
    GuardBuilder g{builder, loc};
    Value max = arith::ConstantOp::create(
        builder, loc,
        IntegerAttr::get(v.getType(), APInt::getMaxValue(64).zext(bits)));
    if (failed(g.check(g.cmp(arith::CmpIPredicate::ule, v, max),
                       "a runtime DMA operand does not fit in 64 bits")))
      return {};
    return Value(arith::TruncIOp::create(builder, loc, i64, v));
  }
  return v;
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
    if (val.getType().isIndex())
      return arith::IndexCastUIOp::create(builder, loc, intType, val);
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
                        int64_t baseByteOffset, uint32_t granuleBytes) {
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
    if (bytes > std::numeric_limits<uint32_t>::max() || bytes < 0)
      return Value(arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(builder.getIntegerType(64), bytes)));
    return Value(arith::ConstantOp::create(builder, loc,
                                           IntegerAttr::get(i32ty, bytes)));
  }

  // Runtime path, in i64: sum(offset[i] * stride[i]) * elemWidthBytes + base.
  // The range is the bounds guard's job (guardWithinHostBuffer); the patch
  // itself only needs the granule alignment the static verifier checks.
  GuardBuilder g{builder, loc};
  Value acc = g.cst(0);
  // A byte offset every term is a multiple of; when it is a whole granule the
  // alignment guard is statically true.
  uint64_t knownMultiple = (uint64_t)baseByteOffset;
  for (auto [o, s] : llvm::zip(elementOffsets, strides)) {
    Value ov = getAsI64(builder, loc, o);
    Value sv = getAsI64(builder, loc, s);
    if (!ov || !sv)
      return {};
    acc = g.add(acc, g.mul(ov, sv));
    uint64_t factor = (uint64_t)elemWidthBytes;
    for (OpFoldResult f : {o, s})
      if (auto c = getConstantIntValue(f))
        factor *= (uint64_t)*c;
    knownMultiple = std::gcd(knownMultiple, factor);
  }
  acc = g.mul(acc, g.cst(elemWidthBytes));
  if (baseByteOffset != 0)
    acc = g.add(acc, g.cst(baseByteOffset));
  if (granuleBytes > 1 && knownMultiple % granuleBytes != 0 &&
      failed(g.check(g.eq(builder.createOrFold<arith::AndIOp>(
                              loc, acc, g.cst(granuleBytes - 1)),
                          0),
                     "a runtime DMA offset is not " + Twine(granuleBytes) +
                         "-byte aligned")))
    return {};
  return acc;
}

LogicalResult guardWithinHostBuffer(
    OpBuilder &builder, Location loc, BaseMemRefType hostBufferType,
    int64_t baseByteOffset, int64_t elemWidthBytes,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> offsetStrides,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  // Elements addressable past the base offset. Capped at 2^32 so every term
  // below is a product of two values under 2^32 and no sum can wrap 64 bits.
  constexpr uint64_t kMaxElems = 1ULL << 32;
  uint64_t cap = kMaxElems;
  if (hostBufferType.hasStaticShape()) {
    int64_t bytes = hostBufferType.getNumElements() *
                    (int64_t)hostBufferType.getElementTypeBitWidth() / 8;
    cap = bytes > baseByteOffset
              ? std::min<uint64_t>((bytes - baseByteOffset) / elemWidthBytes,
                                   kMaxElems)
              : 0;
  }
  if (cap == 0)
    return emitError(loc) << "DMA transfer starts past the end of its host "
                             "buffer";

  // The furthest element touched is sum(offset_k * offsetStride_k) +
  // sum((size_i - 1) * stride_i); it must be below `cap`. A term is in range
  // only if both factors are (a zero factor makes the other irrelevant), which
  // also keeps the product from wrapping.
  GuardBuilder g{builder, loc};
  Value ok, sum = g.cst(0);
  auto addTerm = [&](Value c, Value t) {
    Value inRange = g.both(g.either(g.eq(c, 0), g.ule(t, cap - 1)),
                           g.either(g.eq(t, 0), g.ule(c, cap - 1)));
    ok = ok ? g.both(ok, inRange) : inRange;
    sum = g.add(sum, g.mul(c, t));
  };
  for (auto [o, s] : llvm::zip(offsets, offsetStrides)) {
    Value ov = getAsI64(builder, loc, o);
    Value sv = getAsI64(builder, loc, s);
    if (!ov || !sv)
      return failure();
    addTerm(ov, sv);
  }
  for (auto [sz, st] : llvm::zip(sizes, strides)) {
    Value szv = getAsI64(builder, loc, sz);
    Value stv = getAsI64(builder, loc, st);
    if (!szv || !stv)
      return failure();
    addTerm(g.sub(szv, 1), stv);
  }
  Value cond = ok ? g.both(ok, g.ule(sum, cap - 1)) : g.ule(sum, cap - 1);
  return g.check(cond, "a runtime DMA access runs past the end of its " +
                           Twine(cap) + "-element host buffer");
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
      masked = builder.createOrFold<arith::AndIOp>(loc, masked, maskConst);
    }
    if (shift > 0) {
      auto shiftConst = arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(i32ty, shift));
      masked = builder.createOrFold<arith::ShLIOp>(loc, masked, shiftConst);
    }
    result = builder.createOrFold<arith::OrIOp>(loc, result, masked);
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

LogicalResult
buildShimBdWords(OpBuilder &builder, Location loc,
                 const AIE::AIETargetModel &targetModel,
                 const BdTemplateFields &f, ArrayRef<OpFoldResult> mixedSizes,
                 ArrayRef<OpFoldResult> mixedStrides, uint64_t elemWidth,
                 uint32_t burstLength, uint32_t axcache, OpFoldResult lenElems,
                 Value &repeatCountOut, SmallVectorImpl<Value> &wordsOut) {
  auto i32ty = builder.getIntegerType(32);

  // Shim-NOC BDs are 8 registers wide; slots not set below stay zero.
  wordsOut.assign(8, createConstantI32(builder, loc, 0));
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
      (AIE::getShimBurstLengthEncoding(targetModel, burstLength) & 0x3) << 30);
  // word[5] AXCache [27:24]; d2_stride overlaid below in ND mode.
  wordsOut[5] = createConstantI32(builder, loc, (axcache & 0xf) << 24);
  // word[7] next_bd [30:27], use_next_bd [26], valid_bd [25], lock fields.
  wordsOut[7] = createConstantI32(
      builder, loc,
      ((f.next_bd_id & 0xf) << 27) | ((f.use_next_bd & 0x1) << 26) |
          (1u << 25) | ((f.lock_rel_val & 0x7f) << 18) |
          ((f.lock_rel_id & 0xf) << 13) | ((f.lock_acq_enable & 0x1) << 12) |
          ((f.lock_acq_val & 0x7f) << 5) | (f.lock_acq_id & 0xf));

  uint32_t gran = targetModel.getAddressGenGranularity();
  SmallVector<OpFoldResult, 4> sizesRev(llvm::reverse(mixedSizes));
  SmallVector<OpFoldResult, 4> stridesRev(llvm::reverse(mixedStrides));

  // Linear mode: contiguous transfer (each outer stride == product of inner
  // sizes) folds d0/d1 into buffer_length, dodging the 10-bit d0_size limit. A
  // runtime operand needed for the test falls back to ND mode.
  auto cst = [&](OpFoldResult v) { return getConstantIntValue(v); };
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
  bool isLinear = knownContiguous();

  Value inS[4], inT[4];
  for (int i = 0; i < 4; i++) {
    inS[i] = getAsI64(builder, loc, sizesRev[i]);
    inT[i] = getAsI64(builder, loc, stridesRev[i]);
    if (!inS[i] || !inT[i])
      return failure();
  }

  // Host-side guards: every size and stride must land in its BD field, checked
  // in the element domain before any scaling can wrap. Over constant operands
  // they fold away (the verifiers already checked those) or fail here.
  constexpr auto shim = AIE::AIETileType::ShimNOCTile;
  uint64_t ew = elemWidth;
  uint64_t wrapMax = (1ULL << targetModel.getDmaBdWrapBits(shim)) - 1;
  uint64_t maxLen = targetModel.getDmaBdMaxLen(shim);
  uint64_t maxStride = (1ULL << targetModel.getDmaBdStepBits(shim)) * gran / ew;
  uint64_t maxIterations = targetModel.getMaxBdIterationCount(shim);
  uint64_t maxRepeats = targetModel.getMaxRepeatCount() + 1;
  GuardBuilder g{builder, loc};
  uint64_t sizeMax[4];
  auto checkSize = [&](int i, uint64_t hi) {
    sizeMax[i] = getConstantIntValue(sizesRev[i]).value_or(hi);
    return g.check(g.inRange1(inS[i], hi),
                   (i == 3 ? Twine("a runtime DMA repeat count")
                           : "a runtime DMA d" + Twine(i) + " size") +
                       " must be in [1:" + Twine(hi) + "]");
  };
  auto checkStride = [&](int i, Value inRange, uint64_t hi) {
    return g.check(g.either(g.ule(inS[i], 1), inRange),
                   (i == 3 ? Twine("a runtime DMA iteration stride")
                           : "a runtime DMA d" + Twine(i) + " stride") +
                       " must be in [" + Twine(i == 3 ? 0 : 1) + ":" +
                       Twine(hi) + "] when its size > 1");
  };
  if (failed(checkSize(0, (isLinear ? maxLen : wrapMax) * gran / ew)) ||
      failed(checkSize(1, isLinear ? maxLen : wrapMax)) ||
      failed(checkSize(2, maxLen)) || failed(checkSize(3, maxRepeats)) ||
      failed(g.check(
          g.either(g.eq(inT[3], 0), g.ule(g.sub(inS[3], 1), maxIterations - 1)),
          "a runtime DMA iteration count must be in [1:" +
              Twine(maxIterations) + "]")))
    return failure();
  if (!isLinear)
    for (int i = 0; i < 3; i++)
      if (i > 0 || ew <= gran)
        if (failed(checkStride(i, g.inRange1(inT[i], maxStride), maxStride)))
          return failure();
  if (failed(checkStride(3, g.ule(inT[3], maxStride), maxStride)))
    return failure();

  // A size or stride whose byte extent is not a whole number of granules is
  // unrealizable (mirrors verifyStridesWraps). The innermost stride collapses
  // to hardware 0 when it is the unit (contiguous) stride.
  int64_t divisor = bdGranuleDivisor(elemWidth, gran);
  if (divisor > 1) {
    auto checkMultiple = [&](Value v, Value exempt, const Twine &what) {
      Value cond = g.multipleOf(v, divisor);
      if (exempt)
        cond = g.either(exempt, cond);
      return g.check(cond, "a runtime DMA " + what + " must be a multiple of " +
                               Twine(divisor) + " elements (whole " +
                               Twine(gran / 8) + "-byte granules)");
    };
    if (failed(checkMultiple(inS[0], Value(), "d0 size")) ||
        failed(checkMultiple(inT[0], g.eq(inT[0], 1), "d0 stride")))
      return failure();
    for (int i = 1; i < 4; i++)
      if (failed(checkMultiple(inT[i], Value(),
                               i == 3 ? Twine("iteration stride")
                                      : "d" + Twine(i) + " stride")))
        return failure();
  }

  // Compute the hardware sizes/strides via the shared encoder.
  Value hwS[4], hwT[4];
  SsaStridePolicy policy(builder, loc);
  encodeHardwareStridesWraps(policy, elemWidth, gran, inS, inT, hwS, hwT);

  // buffer_length (word[0]) is the d0*d1*d2 extent in granules. Every size is
  // already bounded by 2^32, so the d0*d1 product cannot wrap, and the final
  // product only counts once d0*d1 is in range. The guard is skipped when the
  // size bounds alone keep it in range. A dma_task's explicit `len` must agree
  // with it.
  Value d0d1 = g.mul(hwS[0], inS[1]);
  Value bufLen = g.mul(d0d1, inS[2]);
  uint64_t lenMax = llvm::SaturatingMultiply(
      llvm::SaturatingMultiply(sizeMax[0] * ew / gran, sizeMax[1]), sizeMax[2]);
  if (lenMax > maxLen) {
    Value lenFits = g.ule(bufLen, maxLen);
    if (bufLen != d0d1)
      lenFits = g.both(g.ule(d0d1, maxLen), lenFits);
    if (failed(g.check(lenFits, "a runtime DMA transfer exceeds the " +
                                    Twine(maxLen) +
                                    "-granule BD buffer_length")))
      return failure();
  }
  if (lenElems) {
    Value lenV = getAsI64(builder, loc, lenElems);
    if (!lenV)
      return failure();
    if (failed(g.check(g.inRange1(lenV, maxLen * gran / ew),
                       "a runtime DMA length must be in [1:" +
                           Twine(maxLen * gran / ew) + "] elements")) ||
        (divisor > 1 &&
         failed(g.check(g.multipleOf(lenV, divisor),
                        "a runtime DMA length must be a multiple of " +
                            Twine(divisor) + " elements (whole " +
                            Twine(gran / 8) + "-byte granules)"))))
      return failure();
    Value lenGranules =
        ew >= gran
            ? g.mul(lenV, g.cst(ew / gran))
            : builder.createOrFold<arith::DivUIOp>(loc, lenV, g.cst(gran / ew));
    if (failed(g.check(g.eq(lenGranules, bufLen),
                       "a runtime DMA length must equal the d0*d1*d2 extent "
                       "of its dimensions")))
      return failure();
  }

  auto asI32 = [&](Value v) {
    return builder.createOrFold<arith::TruncIOp>(loc, i32ty, v);
  };

  // word[6] iteration_size: a zero outer stride is a pure repeat (carried by
  // repeat_count), so both iteration fields must be 0 like AIEDmaToNpu; gate
  // iteration_size on the stride while leaving hwS[3] for repeatCountOut.
  // hwT[3] already collapses to 0.
  Value iterSizeField = policy.selectGT0(inT[3], hwS[3], policy.cst(0));

  // word[0] buffer_length
  wordsOut[0] = asI32(bufLen);

  // Linear mode needs only buffer_length + iteration, so the d0/d1/d2
  // size/stride fields stay zero; words 4/5 OR onto the burst_length / AXCache
  // bits set above.
  if (!isLinear) {
    // word[3]: d0_size [29:20], d0_stride [19:0].
    wordsOut[3] =
        buildBdWord(builder, loc,
                    {{asI32(hwS[0]), 0x3FF, 20}, {asI32(hwT[0]), 0xFFFFF, 0}});
    // word[4]: d1_size [29:20], d1_stride [19:0].
    wordsOut[4] = builder.createOrFold<arith::OrIOp>(
        loc, wordsOut[4],
        buildBdWord(builder, loc,
                    {{asI32(hwS[1]), 0x3FF, 20}, {asI32(hwT[1]), 0xFFFFF, 0}}));
    // word[5]: d2_stride [19:0]. Shim d2_size is always 0, carried by bufLen.
    wordsOut[5] = builder.createOrFold<arith::OrIOp>(
        loc, wordsOut[5],
        buildBdWord(builder, loc, {{asI32(hwT[2]), 0xFFFFF, 0}}));
  }
  // word[6]: iteration_size [25:20], iteration_stride [19:0].
  wordsOut[6] = buildBdWord(
      builder, loc,
      {{asI32(iterSizeField), 0x3F, 20}, {asI32(hwT[3]), 0xFFFFF, 0}});

  // repeat_count for the queue push is the biased outer size (N > 1 ? N - 1 :
  // 0), matching the static path. A constant folds so the static push_queue
  // lowering can consume it; a runtime size yields the SSA hwS[3].
  if (auto outerConst = getConstantIntValue(sizesRev[3])) {
    int64_t r = *outerConst > 1 ? *outerConst - 1 : 0;
    repeatCountOut =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(i32ty, r));
  } else {
    repeatCountOut = asI32(hwS[3]);
  }
  return success();
}

} // namespace xilinx::AIEX
