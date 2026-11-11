//===- aie2ps_core_lowering.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE2PS has no core intrinsic set. A core that needs none lowers; an op that
// needs one is diagnosed.

// RUN: aie-opt --aie-standard-lowering --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK: module @no_intrinsics attributes {llvm.target_triple = "aie2ps"}
// CHECK: memref.global "public" @buf : memref<16xi32>
// CHECK-NOT: func.func private
// CHECK-LABEL: func.func @core_2_3()
// CHECK: memref.get_global @buf
// CHECK: memref.store
// CHECK: return
// CHECK-NOT: aie.
// CHECK-NOT: func.func private
module @no_intrinsics {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %l = aie.lock(%t, 0) {init = 1 : i32}
    %buf = aie.buffer(%t) {sym_name = "buf"} : memref<16xi32>
    // Locks in DMA regions need no intrinsic.
    %m = aie.mem(%t) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %c = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      memref.store %v, %buf[%c0] : memref<16xi32>
      aie.end
    }
  }
}

// -----

module @use_lock {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %l = aie.lock(%t, 0)
    %c = aie.core(%t) {
      %c1 = arith.constant 1 : i32
      // expected-error@+2 {{'aie.use_lock' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.use_lock' that was explicitly marked illegal}}
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      aie.end
    }
  }
}

// -----

// A lock in a function that a core calls needs an intrinsic too.
module @use_lock_in_func {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %l = aie.lock(%t, 0)
    func.func private @kernel(%lock : index) {
      %c1 = arith.constant 1 : i32
      // expected-error@+2 {{'aie.use_lock' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.use_lock' that was explicitly marked illegal}}
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      return
    }
    %c = aie.core(%t) {
      func.call @kernel(%l) : (index) -> ()
      aie.end
    }
  }
}

// -----

module @put_stream {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %c = aie.core(%t) {
      %ch = arith.constant 0 : i32
      // expected-error@+2 {{'aie.put_stream' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.put_stream' that was explicitly marked illegal}}
      aie.put_stream(%ch : i32, %ch : i32)
      aie.end
    }
  }
}

// -----

module @get_stream {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %c = aie.core(%t) {
      %ch = arith.constant 0 : i32
      // expected-error@+2 {{'aie.get_stream' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.get_stream' that was explicitly marked illegal}}
      %v = aie.get_stream(%ch : i32) : i32
      aie.end
    }
  }
}

// -----

module @put_cascade {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %c = aie.core(%t) {
      %v = arith.constant dense<0> : vector<16xi32>
      // expected-error@+2 {{'aie.put_cascade' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.put_cascade' that was explicitly marked illegal}}
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }
  }
}

// -----

module @get_cascade {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %c = aie.core(%t) {
      // expected-error@+2 {{'aie.get_cascade' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.get_cascade' that was explicitly marked illegal}}
      %v = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}

// -----

module @debug {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %c = aie.core(%t) {
      %v = arith.constant 0 : i32
      // expected-error@+2 {{'aie.debug' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.debug' that was explicitly marked illegal}}
      aie.debug(%v : i32)
      aie.end
    }
  }
}

// -----

module @event {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %c = aie.core(%t) {
      // expected-error@+2 {{'aie.event' op has no core lowering for AIE2ps}}
      // expected-error@+1 {{failed to legalize operation 'aie.event' that was explicitly marked illegal}}
      aie.event(0)
      aie.end
    }
  }
}

// -----

// AIE2 and AIE2P set the rounding and saturation modes for SRS at core entry.
module @srs {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    // expected-error@+2 {{'aie.core' op needs the rounding and saturation modes set for aievec.srs or bf16 matmul, which has no lowering for AIE2ps}}
    // expected-error@+1 {{failed to legalize operation 'aie.core' that was explicitly marked illegal}}
    %c = aie.core(%t) {
      %c0 = arith.constant 0 : i32
      %v = arith.constant dense<1.0> : vector<16xf32>
      %srs = aievec.srs %v, %c0 : vector<16xf32>, i32, vector<16xbf16>
      aie.end
    }
  }
}

// -----

// The same holds for SRS in a function the core reaches through calls,
// including through recursion.
module @srs_in_callee {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    func.func private @inner(%n : i32) {
      %c0 = arith.constant 0 : i32
      %v = arith.constant dense<1.0> : vector<16xf32>
      %srs = aievec.srs %v, %c0 : vector<16xf32>, i32, vector<16xbf16>
      func.call @outer(%n) : (i32) -> ()
      return
    }
    func.func private @outer(%n : i32) {
      func.call @inner(%n) : (i32) -> ()
      return
    }
    // expected-error@+2 {{'aie.core' op needs the rounding and saturation modes set for aievec.srs or bf16 matmul, which has no lowering for AIE2ps}}
    // expected-error@+1 {{failed to legalize operation 'aie.core' that was explicitly marked illegal}}
    %c = aie.core(%t) {
      %n = arith.constant 1 : i32
      func.call @outer(%n) : (i32) -> ()
      aie.end
    }
  }
}

// -----

// Locks in a runtime sequence's DMA tasks are erased with the sequence.
// CHECK-LABEL: module @runtime_sequence_lock
// CHECK: func.func @core_2_3()
// CHECK-NOT: aie.use_lock
module @runtime_sequence_lock {
  aie.device(xcve3858) {
    %t = aie.tile(2, 3)
    %buf = aie.buffer(%t) {address = 4096 : i32, sym_name = "seq_buf"} : memref<16xi32>
    %l = aie.lock(%t, 0) {init = 1 : i32}
    %c = aie.core(%t) {
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %task = aiex.dma_configure_task(%t, S2MM, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%l, Acquire, %c1)
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.use_lock(%l, Release, %c1)
        aie.end
      }
    }
  }
}
