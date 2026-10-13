//===- aie-core-int-range-narrowing.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-core-int-range-narrowing %s | FileCheck %s

// Loops in cores and device functions whose bounds fit in i32 get i32
// induction variables, and index arithmetic that fits in i32 is narrowed.
// Loops and arithmetic that might not fit keep `index`, and runtime sequences
// are not touched.

// CHECK-LABEL: aie.device(npu2)
aie.device(npu2) {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  // CHECK: func.func @helper(%[[M:.*]]: memref<1024xi32>)
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
  // CHECK:   %[[IDX:.*]] = arith.index_castui %[[I]] : i32 to index
  // CHECK:   memref.store %{{.*}}, %[[M]][%[[IDX]]]
  func.func @helper(%m: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 7 : i32
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %v, %m[%i] : memref<1024xi32>
    }
    return
  }

  // CHECK: aie.core
  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cmax = arith.constant 9223372036854775807 : index
    %cu32 = arith.constant 4294967295 : index
    %v = arith.constant 7 : i32
    // The INT64_MAX loop that `range_(sys.maxsize)` emits keeps `index`, but
    // the loop nested in it is narrowed.
    // CHECK: scf.for %{{.*}} = %{{.*}} to %c9223372036854775807 step %{{.*}} {
    // CHECK:   scf.for %[[J:.*]] = %c0_i32 to %c32_i32 step %{{.*}} : i32 {
    // CHECK:     %[[JIDX:.*]] = arith.index_castui %[[J]] : i32 to index
    // CHECK:     memref.load %{{.*}}[%[[JIDX]]]
    scf.for %iter = %c0 to %cmax step %c1 {
      scf.for %i = %c0 to %c32 step %c1 {
        %x = memref.load %buf[%i] : memref<1024xi32>
        %y = arith.addi %x, %v : i32
        memref.store %y, %buf[%i] : memref<1024xi32>
      }
      func.call @helper(%buf) : (memref<1024xi32>) -> ()
    }
    // 0xFFFFFFFF does not fit in a signed i32.
    // CHECK: scf.for %{{.*}} = %{{.*}} to %c4294967295 step %{{.*}} {
    scf.for %i = %c0 to %cu32 step %c1 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
    }
    // The bounds and step fit in i32, but the final increment does not
    // (2147483646 + 2), so an i32 counter would wrap. The loop keeps `index`.
    // CHECK: scf.for %{{.*}} = %c2147483646 to %c2147483647 step %c2 {
    %clo = arith.constant 2147483646 : index
    %chi = arith.constant 2147483647 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %clo to %chi step %c2 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
    }
    // The final increment reaches 2147483647 exactly, so this one is narrowed.
    // CHECK: scf.for %{{.*}} = %c2147483645_i32 to %c2147483647_i32 step %{{.*}} : i32 {
    %clo1 = arith.constant 2147483645 : index
    scf.for %i = %clo1 to %chi step %c1 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
    }
    aie.end
  }

  // Index arithmetic and comparisons on the induction variable are narrowed
  // when their results fit in i32, and stay `index` when they might not.
  %tile2 = aie.tile(0, 3)
  %idx = aie.buffer(%tile2) : memref<32xindex>
  // CHECK: aie.core
  // CHECK:   %[[CBIG:.*]] = arith.constant 1099511627776 : index
  // CHECK:   scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
  // CHECK:     %[[KIDX:.*]] = arith.index_castui %[[K]] : i32 to index
  // CHECK:     %[[MUL:.*]] = arith.muli %[[K]], %{{.*}} : i32
  // CHECK:     %[[ADD:.*]] = arith.addi %[[MUL]], %{{.*}} : i32
  // CHECK:     %[[ADDIDX:.*]] = arith.index_castui %[[ADD]] : i32 to index
  // CHECK:     %[[LT:.*]] = arith.cmpi ult, %[[ADD]], %{{.*}} : i32
  // CHECK:     %[[SEL:.*]] = arith.select %[[LT]], %[[ADDIDX]], %{{.*}} : index
  // CHECK:     memref.store %[[SEL]], %{{.*}}[%[[KIDX]]]
  // CHECK:     %[[WMUL:.*]] = arith.muli %[[KIDX]], %[[CBIG]] : index
  // CHECK:     %[[WADD:.*]] = arith.addi %[[WMUL]], %{{.*}} : index
  // CHECK:     %[[WLT:.*]] = arith.cmpi ult, %[[WADD]], %[[CBIG]] : index
  // CHECK:     %[[WSEL:.*]] = arith.select %[[WLT]], %[[WADD]], %{{.*}} : index
  // CHECK:     memref.store %[[WSEL]], %{{.*}}[%[[KIDX]]]
  %core2 = aie.core(%tile2) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cbig = arith.constant 1099511627776 : index
    scf.for %k = %c0 to %c32 step %c1 {
      // At most 31 * 3 + 16, so this fits in i32.
      %m = arith.muli %k, %c3 : index
      %a = arith.addi %m, %c16 : index
      %lt = arith.cmpi ult, %a, %c32 : index
      %s = arith.select %lt, %a, %c0 : index
      memref.store %s, %idx[%k] : memref<32xindex>
      // Up to 31 * 2^40 + 16, which does not.
      %w = arith.muli %k, %cbig : index
      %wa = arith.addi %w, %c16 : index
      %wlt = arith.cmpi ult, %wa, %cbig : index
      %ws = arith.select %wlt, %wa, %c0 : index
      memref.store %ws, %idx[%k] : memref<32xindex>
    }
    aie.end
  }

  // CHECK: aie.runtime_sequence
  // CHECK:   scf.for %[[R:.*]] = %c0 to %c4 step %c1 {
  // CHECK:     arith.addi %[[R]], %c1 : index
  aie.runtime_sequence(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %j = arith.addi %i, %c1 : index
    }
  }
}
