//===- resident_buffer_reload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A load_pdi reaching the static TXN binary path directly (no
// aie-expand-load-pdi) always resets the array, so two devices in one
// runtime sequence holding a buffer resident on the same global is an error
// here regardless of --reconfig-teardown.

// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false %s 2>&1 | FileCheck %s

// CHECK: 'aiex.npu.load_pdi' op resets the array between @dev1 and @dev2, which both hold a buffer resident on @w; needs --reconfig-teardown
module {
  memref.global "private" @w : memref<16xi32>

  aie.device(npu2) @dev1 {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "buf1", resident = @w} : memref<16xi32>
  }

  aie.device(npu2) @dev2 {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "buf2", resident = @w} : memref<16xi32>
  }

  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @dev1 }
      aiex.npu.load_pdi { device_ref = @dev2 }
    }
  }
}
