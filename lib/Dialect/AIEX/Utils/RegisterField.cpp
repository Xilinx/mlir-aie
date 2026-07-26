//===- RegisterField.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/RegisterField.h"

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/bit.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

FailureOr<uint32_t> AIEX::encodeRegisterField(const RegField &field,
                                              uint32_t value) {
  if (field.mask == 0)
    return failure(); // unknown/unsupported field

  // Every hardware bitfield aie-rt describes is a single contiguous run of
  // set bits, so the field's width is simply the popcount of its mask, and
  // its LSB is the mask's trailing-zero count. Assert the two are consistent
  // with the RegField the caller built (a mismatch is a bug in the RegField
  // constant, not in an input value) rather than silently mis-deriving the
  // width from an lsb that disagrees with the mask.
  assert(
      llvm::countr_zero(field.mask) == static_cast<int>(field.lsb) &&
      "RegField.lsb does not match the trailing-zero count of RegField.mask");
  unsigned width = llvm::popcount(field.mask);
  uint64_t maxValue = (uint64_t{1} << width) - 1;
  if (value > maxValue)
    return failure(); // value does not fit in the field's bit width

  return static_cast<uint32_t>(value << field.lsb) & field.mask;
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
