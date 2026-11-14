//===- resident_buffer_reload_teardown.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two devices hold a buffer resident on the same global, but each
// configuration is torn down before the next, so the reload does not clear
// the array underneath them.

// RUN: aie-opt --aie-expand-load-pdi="teardown=true" %s | FileCheck %s

// CHECK: aiex.npu.write_teardown @dev1
// CHECK: aiex.npu.write_teardown @dev2
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
