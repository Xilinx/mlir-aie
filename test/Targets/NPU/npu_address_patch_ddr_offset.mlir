//===- npu_address_patch_ddr_offset.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiex.npu.address_patch emits a DDR patch (TXN_OPC_DDR_PATCH) whose arg_plus
// word is runtime-dependent for host arguments beyond the first 5:
//
//   - xclbin + instruction-buffer runtime (default, fold=true): the NPU
//     firmware only address-translates the first 5 host buffers, so the
//     0x80000000 AIE-aperture offset is folded into arg_plus for arg_idx >= 5.
//     arg_plus = 0x100 -> 0x80000100 for arg 5; unchanged for args 0..4.
//
//   - full-ELF runtime (--aie-npu-fold-ddr-addr-offset=false): the driver
//     assigns NPU-space device addresses to all host buffers, so no offset is
//     folded. arg_plus stays 0x100 for every arg_idx.

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s --check-prefix=FOLD
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false --aie-npu-fold-ddr-addr-offset=false %s | FileCheck %s --check-prefix=NOFOLD

module {
  aie.device(npu2) {
    aie.runtime_sequence(%a0: memref<8xi32>, %a1: memref<8xi32>, %a2: memref<8xi32>, %a3: memref<8xi32>, %a4: memref<8xi32>, %a5: memref<8xi32>) {
      %off = arith.constant 256 : i32

      // arg_idx 4 (within the firmware-translated set): arg_plus is 0x100 in
      // BOTH modes.
      // FOLD:        00000081
      // FOLD:        00000004
      // FOLD-NEXT:   00000000
      // FOLD-NEXT:   00000100
      // NOFOLD:      00000081
      // NOFOLD:      00000004
      // NOFOLD-NEXT: 00000000
      // NOFOLD-NEXT: 00000100
      aiex.npu.address_patch(%off : i32) {addr = 74560 : ui32, arg_idx = 4 : i32}

      // arg_idx 5 (beyond the firmware-translated set): arg_plus gets the
      // 0x80000000 offset in fold mode (xclbin) and stays 0x100 without it
      // (full-ELF).
      // FOLD:        00000081
      // FOLD:        00000005
      // FOLD-NEXT:   00000000
      // FOLD-NEXT:   80000100
      // NOFOLD:      00000081
      // NOFOLD:      00000005
      // NOFOLD-NEXT: 00000000
      // NOFOLD-NEXT: 00000100
      aiex.npu.address_patch(%off : i32) {addr = 74560 : ui32, arg_idx = 5 : i32}
    }
  }
}
