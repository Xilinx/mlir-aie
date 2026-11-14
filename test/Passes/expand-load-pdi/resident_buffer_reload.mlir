//===- resident_buffer_reload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two devices hold a buffer resident on the same global. Reloading from one
// to the other without a teardown clears the array underneath it.

// RUN: aie-opt --aie-expand-load-pdi --verify-diagnostics %s

module {
  memref.global "private" @w : memref<16xi32>

  aie.device(npu2) @dev1 {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "buf1", resident = @w} : memref<16xi32>
  }

  aie.device(npu2) @dev2 {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "buf2", resident = @w} : memref<16xi32>
  }

  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @dev1 }
      // expected-error@+1 {{'aiex.npu.load_pdi' op resets the array between @dev1 and @dev2, which both hold a buffer resident on @w; needs --reconfig-teardown}}
      aiex.npu.load_pdi { device_ref = @dev2 }
    }
  }
}
