//===- core_outside_constant.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that uses a constant defined in the device gets its own copy of it, so
// the outlined core function does not use the erased device's value. Any other
// value defined outside the core is diagnosed.

// RUN: aie-opt --aie-localize-locks --aie-normalize-address-spaces --aie-standard-lowering --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: aie-opt --aie-localize-locks --aie-normalize-address-spaces --aie-standard-lowering="tilecol=0 tilerow=3" --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=SEL

// CHECK-LABEL: module @use_lock_npu2
// CHECK:       func.func @core_0_2() {
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[LOCK:.*]] = arith.constant 48 : index
// CHECK-NEXT:    %[[ID:.*]] = arith.index_cast %[[LOCK]] : index to i32
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[NEG:.*]] = arith.subi %[[ZERO]], %[[ONE]] : i32
// CHECK-NEXT:    call @llvm.aie2p.acquire(%[[ID]], %[[NEG]])
// CHECK-NEXT:    %[[ID2:.*]] = arith.index_cast %[[LOCK]] : index to i32
// CHECK-NEXT:    call @llvm.aie2p.release(%[[ID2]], %[[ONE]])
// CHECK-NEXT:    return
// SEL-LABEL:   module @use_lock_npu2
// SEL-NOT:     func.func @core_
module @use_lock_npu2 {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %c1 = arith.constant 1 : i32
    %core = aie.core(%tile_0_2) {
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.use_lock(%lock, Release, %c1)
      aie.end
    }
  }
}

// -----

// One constant used by two cores, in a later block, and by a DMA block. Each core
// gets one copy of each at its entry, in walk order; the unselected
// core is erased untouched.

// CHECK-LABEL: module @shared_npu2
// CHECK:       func.func @core_0_3() {
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[TWO:.*]] = arith.constant 2 : i32
// CHECK:         call @llvm.aie2p.release(%{{.*}}, %[[ONE]])
// CHECK:       ^bb1:
// CHECK:         memref.store %[[TWO]]
// CHECK:       func.func @core_0_2() {
// CHECK-NEXT:    %[[ONE2:.*]] = arith.constant 1 : i32
// CHECK:         call @llvm.aie2p.release(%{{.*}}, %[[ONE2]])
// CHECK-NOT:   aie.
// SEL-LABEL:   module @shared_npu2
// SEL-NOT:     func.func @core_0_2
// SEL:         func.func @core_0_3() {
// SEL-NEXT:      arith.constant 1 : i32
// SEL-NEXT:      arith.constant 2 : i32
// SEL-NOT:     func.func @core_0_2
module @shared_npu2 {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %lock = aie.lock(%tile_0_3, 0) {init = 1 : i32}
    %buf = aie.buffer(%tile_0_3) {sym_name = "buf"} : memref<8xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %core_0_2 = aie.core(%tile_0_2) {
      aie.use_lock(%lock, Release, %c1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      aie.use_lock(%lock, Release, %c1)
      cf.br ^bb1
    ^bb1:
      memref.store %c2, %buf[%c0] : memref<8xi32>
      aie.end
    }
    %mem = aie.mem(%tile_0_3) {
      %s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%lock, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @use_lock_aie1
// CHECK:       func.func @core_1_3() {
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[LOCK:.*]] = arith.constant 48 : index
// CHECK-NEXT:    %[[ID:.*]] = arith.index_cast %[[LOCK]] : index to i32
// CHECK-NEXT:    call @llvm.aie.lock.acquire.reg(%[[ID]], %[[ONE]])
// SEL-LABEL:   module @use_lock_aie1
// SEL-NOT:     func.func @core_
module @use_lock_aie1 {
  aie.device(xcvc1902) {
    %tile_1_3 = aie.tile(1, 3)
    %lock = aie.lock(%tile_1_3, 0)
    %c1 = arith.constant 1 : i32
    %core = aie.core(%tile_1_3) {
      aie.use_lock(%lock, Acquire, %c1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @store_aie2ps
// CHECK:       func.func @core_2_3() {
// CHECK-NEXT:    %[[SEVEN:.*]] = arith.constant 7 : i32
// CHECK:         memref.store %[[SEVEN]]
// SEL-LABEL:   module @store_aie2ps
// SEL-NOT:     func.func @core_
module @store_aie2ps {
  aie.device(xcve3858) {
    %tile_2_3 = aie.tile(2, 3)
    %buf = aie.buffer(%tile_2_3) {sym_name = "buf"} : memref<8xi32>
    %c7 = arith.constant 7 : i32
    %core = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      memref.store %c7, %buf[%c0] : memref<8xi32>
      aie.end
    }
  }
}

// -----

module @not_constant {
  aie.device(npu2) {
    %tile_0_3 = aie.tile(0, 3)
    %buf = aie.buffer(%tile_0_3) {sym_name = "buf"} : memref<8xi32>
    %c1 = arith.constant 1 : i32
    // expected-note @below {{value defined here}}
    %c2 = arith.addi %c1, %c1 : i32
    %core = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      // expected-error @below {{'memref.store' op uses a value defined outside its aie.core, which cannot be outlined}}
      memref.store %c2, %buf[%c0] : memref<8xi32>
      aie.end
    }
  }
}
