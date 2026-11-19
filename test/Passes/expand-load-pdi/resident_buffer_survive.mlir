//===- resident_buffer_survive.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A buffer resident on a global needs the data the last configuration holding
// that global left, so no reset may come between them, whatever holds the
// array in between.

// RUN: aie-opt --aie-expand-load-pdi --verify-diagnostics --split-input-file %s

// A configuration holding nothing resident still resets the array.
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu2) @a {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufa", resident = @w} : memref<16xi32>
  }
  aie.device(npu2) @b {
    %tile = aie.tile(0, 2)
  }
  aie.device(npu2) @c {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufc", resident = @w} : memref<16xi32>
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @a }
      // expected-error@+1 {{resets the array between @a and @c, which both hold a buffer resident on @w; needs --reconfig-teardown}}
      aiex.npu.load_pdi { device_ref = @b }
      aiex.npu.load_pdi { device_ref = @c }
    }
  }
}

// -----

// A load this pass leaves alone resets the array too.
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu2) @a {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufa", resident = @w} : memref<16xi32>
  }
  aie.device(npu2) @b {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufb", resident = @w} : memref<16xi32>
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @a }
      // expected-error@+1 {{resets the array between @a and @b, which both hold a buffer resident on @w; needs --reconfig-teardown}}
      aiex.npu.load_pdi { device_ref = @b, expand_mode = 0 : i32 }
    }
  }
}

// -----

// The array starts configured as the sequence's own device.
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu2) @own {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufown", resident = @w} : memref<16xi32>
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      // expected-error@+1 {{resets the array between @own and @b, which both hold a buffer resident on @w; needs --reconfig-teardown}}
      aiex.npu.load_pdi { device_ref = @b }
    }
  }
  aie.device(npu2) @b {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufb", resident = @w} : memref<16xi32>
  }
}

// -----

// The firmware skips a load of the PDI it loaded last, also across dispatches,
// so the array keeps the resident data.
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu2) @a {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "bufa", resident = @w} : memref<16xi32>
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @a, expand_mode = 0 : i32 }
      aiex.npu.load_pdi { device_ref = @a, expand_mode = 0 : i32 }
    }
  }
}

// -----

// The host runs the sequence again, so its last configuration precedes its
// first.
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu2) @c {
    %tile = aie.tile(0, 2)
    // expected-note@+1 {{resident here}}
    %buf = aie.buffer(%tile) {sym_name = "bufc", resident = @w} : memref<16xi32>
  }
  aie.device(npu2) @b {
    %tile = aie.tile(0, 2)
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @c }
      // expected-error@+1 {{resets the array between @c and @c, which both hold a buffer resident on @w when the sequence runs again; needs --reconfig-teardown}}
      aiex.npu.load_pdi { device_ref = @b }
    }
  }
}
