//===- RegisterField.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/RegisterField.h"

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

FailureOr<uint32_t> AIEX::encodeRegisterField(const RegField &field,
                                              uint32_t value) {
  if (field.mask == 0)
    return failure(); // unknown/unsupported field

  // Every hardware bitfield aie-rt describes is a single contiguous run of
  // set bits, so the field's width is simply the popcount of its mask, and
  // its LSB is the mask's trailing-zero count. Derive both from `mask`
  // unconditionally -- RegField has no separate lsb member for this reason:
  // a caller-supplied lsb could disagree with the mask (e.g. a copy-pasted
  // RegField constant with the mask updated but not the lsb), and that
  // mismatch would silently write the wrong bit position with no
  // diagnostic in a build where assertions are compiled out. Deriving lsb
  // from mask makes that class of bug unrepresentable instead of merely
  // asserted against.
  //
  // The other half of a malformed constant is a non-contiguous mask (e.g.
  // 0x5), whose popcount would describe a width the field does not have.
  // Reject it rather than assert, for the same reason the lsb is derived
  // above: an assert both aborts instead of reaching createMaskWriteField's
  // diagnostic path, and disappears entirely in an assertions-off build --
  // buildAndTestMulti builds that arm of its matrix -- which would leave a bad
  // constant silently encoding against the wrong width.
  if (!llvm::isShiftedMask_32(field.mask))
    return failure(); // non-contiguous mask

  unsigned lsb = static_cast<unsigned>(llvm::countr_zero(field.mask));
  unsigned width = llvm::popcount(field.mask);
  uint64_t maxValue = (uint64_t{1} << width) - 1;
  if (value > maxValue)
    return failure(); // value does not fit in the field's bit width

  return static_cast<uint32_t>(value << lsb) & field.mask;
}

FailureOr<AIEX::NpuMaskWrite32Op>
AIEX::createMaskWriteField(RewriterBase &rewriter, Location loc,
                           Operation *diagOp, const RegField &field,
                           uint32_t value, IntegerAttr colAttr,
                           IntegerAttr rowAttr) {
  FailureOr<uint32_t> data = encodeRegisterField(field, value);
  if (failed(data)) {
    if (field.mask == 0) {
      diagOp->emitOpError() << "unknown register field '" << field.name << "'";
    } else if (!llvm::isShiftedMask_32(field.mask)) {
      diagOp->emitOpError()
          << "register field '" << field.name << "' has a non-contiguous mask";
    } else {
      unsigned width = llvm::popcount(field.mask);
      diagOp->emitOpError() << "value " << value << " does not fit in the "
                            << width << "-bit field '" << field.name << "'";
    }
    return failure();
  }

  // Constants are materialized in named locals, mirroring the reset-pulse
  // lowerings this helper folds together: the emitted IR order must not
  // depend on unspecified C++ argument-evaluation order.
  Value addr = createConstantI32(rewriter, loc, field.regOff);
  Value dataVal = createConstantI32(rewriter, loc, *data);
  Value maskVal = createConstantI32(rewriter, loc, field.mask);
  return rewriter.create<NpuMaskWrite32Op>(loc, addr, dataVal, maskVal, nullptr,
                                           colAttr, rowAttr);
}
