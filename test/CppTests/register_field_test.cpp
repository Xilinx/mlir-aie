//===- register_field_test.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exercises AIEX::encodeRegisterField -- the pure mask/shift derivation half
// of the field-level set-register-field helper -- directly, with no
// MLIRContext or IR construction required. The other half,
// AIEX::createMaskWriteField (the npu.maskwrite32-emitting wrapper), is
// exercised at the IR level by test/Passes/lower-core-reset and
// test/Passes/lower-dma-channel-reset, which FileCheck the exact
// address/value/mask constants it produces when AIELowerCoreReset and
// AIELowerDmaChannelReset are refactored onto it.
//
// The RegField constants below are the same ones grounded in the aie-rt
// aie2p reginit tables that AIELowerCoreReset.cpp uses (CORE_CONTROL.RESET,
// CORE_CONTROL.ENABLE) plus one wider field (DMA CTRL.CONTROLLER_ID, an
// 8-bit field at XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID) to
// cover a field wider than 1 bit.

#include "aie/Dialect/AIEX/Utils/RegisterField.h"

#include <cstdio>

using namespace xilinx::AIEX;

namespace {

// CORE_CONTROL.RESET: XAIE2PGBL_CORE_MODULE_CORE_CONTROL_RESET_{LSB,MASK},
// lsb 1, mask 0x2 (aie2p reginit: Aie2PCoreCtrlReg.CtrlRst).
constexpr RegField kCoreCtrlResetField = {"CORE_CONTROL.RESET", 0x32000, 1,
                                          0x2};

// CORE_CONTROL.ENABLE: XAIE2PGBL_CORE_MODULE_CORE_CONTROL_ENABLE_{LSB,MASK},
// lsb 0, mask 0x1 (aie2p reginit: Aie2PCoreCtrlReg.CtrlEn). Shares the same
// register as kCoreCtrlResetField -- exactly the sibling field a write32 (as
// opposed to a maskwrite32) would clobber.
constexpr RegField kCoreCtrlEnableField = {"CORE_CONTROL.ENABLE", 0x32000, 0,
                                           0x1};

// DMA_S2MM_0.CTRL.CONTROLLER_ID:
// XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_{LSB,MASK}, an 8-bit
// field (lsb 8, mask 0xFF00). Wider than the 1-bit reset/enable fields above,
// so it exercises the general width check instead of only the 1-bit case.
constexpr RegField kDmaControllerIdField = {"DMA_CTRL.CONTROLLER_ID", 0x1DE00,
                                            8, 0xFF00};

// A field with mask == 0: the default-constructed / not-yet-populated
// RegField. Every real aie-rt bitfield has a nonzero mask, so this stands in
// for a field that was never looked up -- the "unknown field" negative case.
constexpr RegField kUnknownField = {};

int expectOk(const RegField &field, uint32_t value, uint32_t expectedData,
             const char *label) {
  mlir::FailureOr<uint32_t> data = encodeRegisterField(field, value);
  if (failed(data)) {
    std::printf("FAIL %s: encodeRegisterField(value=%u) unexpectedly failed\n",
                label, value);
    return 1;
  }
  if (*data != expectedData) {
    std::printf(
        "FAIL %s: encodeRegisterField(value=%u) = 0x%x, expected 0x%x\n", label,
        value, *data, expectedData);
    return 1;
  }
  return 0;
}

int expectFail(const RegField &field, uint32_t value, const char *label) {
  mlir::FailureOr<uint32_t> data = encodeRegisterField(field, value);
  if (succeeded(data)) {
    std::printf("FAIL %s: encodeRegisterField(value=%u) unexpectedly succeeded "
                "(data=0x%x)\n",
                label, value, *data);
    return 1;
  }
  return 0;
}

} // namespace

int main() {
  int failures = 0;

  // Positive cases: the 1-bit RESET field, asserted then cleared. These are
  // the exact (field, value) pairs AIELowerCoreReset.cpp feeds through
  // createMaskWriteField for the reset pulse.
  failures += expectOk(kCoreCtrlResetField, /*value=*/1, /*expectedData=*/0x2,
                       "CORE_CONTROL.RESET assert");
  failures += expectOk(kCoreCtrlResetField, /*value=*/0, /*expectedData=*/0x0,
                       "CORE_CONTROL.RESET clear");

  // A different 1-bit field in the same register, lsb 0: confirms the shift
  // is actually applied (not just the mask) -- ENABLE=1 must not collide with
  // RESET's data pattern.
  failures += expectOk(kCoreCtrlEnableField, /*value=*/1, /*expectedData=*/0x1,
                       "CORE_CONTROL.ENABLE set");

  // A wider field: mid-range and max-in-range values.
  failures += expectOk(kDmaControllerIdField, /*value=*/0xAB,
                       /*expectedData=*/0xAB00, "DMA_CTRL.CONTROLLER_ID mid");
  failures += expectOk(kDmaControllerIdField, /*value=*/0xFF,
                       /*expectedData=*/0xFF00, "DMA_CTRL.CONTROLLER_ID max");

  // Negative case: a value that does not fit the field width. 0x100 needs 9
  // bits; CONTROLLER_ID is 8 bits wide (mask 0xFF00), so this must be
  // rejected rather than silently truncated or bleeding into FOT_MODE (the
  // next field up).
  failures += expectFail(kDmaControllerIdField, /*value=*/0x100,
                         "DMA_CTRL.CONTROLLER_ID overflow");
  // Same failure mode on a 1-bit field: value=2 does not fit RESET's 1 bit.
  failures += expectFail(kCoreCtrlResetField, /*value=*/2,
                         "CORE_CONTROL.RESET overflow");

  // Negative case: an unknown field (zero mask) is always rejected,
  // regardless of value -- including value 0, which a shift-only
  // implementation could wrongly accept.
  failures += expectFail(kUnknownField, /*value=*/0, "unknown field, value=0");
  failures += expectFail(kUnknownField, /*value=*/1, "unknown field, value=1");

  if (failures > 0) {
    std::printf("register field encoding: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("register field encoding: all checks passed\n");
  return 0;
}
