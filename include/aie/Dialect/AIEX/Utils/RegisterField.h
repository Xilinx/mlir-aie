//===- RegisterField.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Field-level set-register-field helper over npu.maskwrite32.
//
// Every lowering that pokes a single bitfield of a shared control register
// (CORE_CONTROL, a DMA channel CTRL word, ...) has to know that register's
// field layout: the byte offset, the bit position, and the bit width. Today
// each lowering hand-codes that layout as raw shift/mask constants. That is
// exactly how a write32-that-clobbers-a-sibling-field bug gets in: nothing
// stops a lowering from mistyping a shift or reusing the wrong generation's
// mask, and a plain npu.write32 (instead of npu.maskwrite32) silently
// clobbers every other field packed into the same word.
//
// This header is the single place a (register, field, value) triple turns
// into an npu.maskwrite32. `RegField` describes one bitfield exactly the way
// aie-rt's `XAie_RegFldAttr` does (Lsb, Mask) plus the register's own byte
// offset (aie-rt's `RegOff`); every RegField a caller builds should cite the
// aie-rt reginit table it was read from. `encodeRegisterField` derives the
// shift and validates the value fits the field's width instead of a caller
// hand-deriving either. `createMaskWriteField` is the IR-emitting wrapper
// lowering passes call: it always emits `npu.maskwrite32`, never
// `npu.write32`, so sibling fields packed into the same register word are
// preserved by construction.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_REGISTERFIELD_H
#define AIE_DIALECT_AIEX_UTILS_REGISTERFIELD_H

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace xilinx::AIEX {

// Describes one bitfield of a tile-local control register. Mirrors aie-rt's
// `XAie_RegFldAttr` (Lsb, Mask) plus the register's own `RegOff`, so a
// RegField constant can be read directly off an aie-rt reginit table (e.g.
// `Aie2PCoreCtrlReg` in xaie2pgbl_reginit.c) without any additional
// arithmetic. `mask` is the field's mask already positioned at `lsb` within
// the 32-bit register word (as aie-rt's Mask field always is), and is
// expected to be a single contiguous run of set bits -- every hardware
// bitfield is. A default-constructed RegField (mask == 0) represents an
// unknown/unsupported field: `encodeRegisterField` and `createMaskWriteField`
// both reject it rather than silently writing bit 0.
struct RegField {
  llvm::StringRef name; // for diagnostics, e.g. "CORE_CONTROL.RESET"
  uint32_t regOff = 0;  // tile-local register byte offset (aie-rt RegOff)
  uint32_t lsb = 0;     // field LSB position (aie-rt RegFldAttr::Lsb)
  uint32_t mask = 0;    // field mask, lsb-positioned (aie-rt RegFldAttr::Mask)
};

// Shifts `value` into `field` and validates that it fits: `value` must not
// set any bit outside the field's width, and `field` must be a known field
// (mask != 0). On success, returns the lsb-aligned, mask-confined 32-bit data
// word ready to pair with `field.mask` in an npu.maskwrite32. On failure,
// returns failure() with no diagnostic -- callers with an Operation* to
// anchor a diagnostic on should go through createMaskWriteField instead; this
// overload is also the entry point plain C++ unit tests use, with no MLIR
// context required.
mlir::FailureOr<uint32_t> encodeRegisterField(const RegField &field,
                                              uint32_t value);

// Emits the npu.maskwrite32 that sets `field` on tile (col, row) to `value`.
// Derives the shift and mask from `field` via encodeRegisterField instead of
// a caller hand-deriving either, and always emits npu.maskwrite32 (never
// npu.write32), so a sibling field packed into the same register word is
// preserved by construction.
//
// On success, creates the op with `rewriter` at its current insertion point
// and returns it; the caller decides whether to keep it as-is (rewriter.create
// semantics: this call already performed the create) or fold it into a
// replacement via rewriter.replaceOp(op, result). On failure -- `value` does
// not fit in the field's bit width, or `field` is unknown (mask == 0) --
// emits a diagnostic on `diagOp` and returns failure without creating any IR.
mlir::FailureOr<NpuMaskWrite32Op>
createMaskWriteField(mlir::RewriterBase &rewriter, mlir::Location loc,
                     mlir::Operation *diagOp, const RegField &field,
                     uint32_t value, mlir::IntegerAttr colAttr,
                     mlir::IntegerAttr rowAttr);

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_REGISTERFIELD_H
