//===- init_values_shared_mem_array_depth_test.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// Ends that share memory address one pool, so it holds as many objects as the
// deeper end states, even when init_values fill only the producer's two. The
// consumer can then acquire all three it declares, and the producer first
// writes the object the initial contents leave empty.

// CHECK-LABEL: module @init_shared_array_depth_aie2
// CHECK-DAG:   %[[B0:.*]] = aie.buffer(%{{.*}}) {sym_name = "of_buff_0"} : memref<16xi32> = dense<0>
// CHECK-DAG:   %[[B1:.*]] = aie.buffer(%{{.*}}) {sym_name = "of_buff_1"} : memref<16xi32> = dense<1>
// CHECK-DAG:   %[[B2:.*]] = aie.buffer(%{{.*}}) {sym_name = "of_buff_2"} : memref<16xi32>{{ *$}}
// CHECK-DAG:   aie.lock(%{{.*}}) {init = 1 : i32, sym_name = "of_prod_lock_0"}
// CHECK-DAG:   aie.lock(%{{.*}}) {init = 2 : i32, sym_name = "of_cons_lock_0"}
// CHECK:       aie.core
// CHECK:         func.call @fill(%[[B2]])
// CHECK:         func.call @fill(%[[B0]])
// CHECK:       aie.core
// CHECK:         func.call @drain(%[[B0]], %[[B1]], %[[B2]])

module @init_shared_array_depth_aie2 {
  aie.device(npu2) {
    func.func private @fill(memref<16xi32>)
    func.func private @drain(memref<16xi32>, memref<16xi32>, memref<16xi32>)
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    aie.objectfifo @of (%tile02, {%tile03}, [2 : i32, 3 : i32]) : !aie.objectfifo<memref<16xi32>> = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
    %core02 = aie.core(%tile02) {
      %e0 = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      func.call @fill(%e0) : (memref<16xi32>) -> ()
      aie.objectfifo.release @of (Produce, 1)
      %e1 = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      func.call @fill(%e1) : (memref<16xi32>) -> ()
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }
    %core03 = aie.core(%tile03) {
      %a, %b, %c = aie.objectfifo.acquire @of (Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      func.call @drain(%a, %b, %c) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.objectfifo.release @of (Consume, 3)
      aie.end
    }
  }
}

// -----

// With binary locks, only the objects holding initial contents start full.

// CHECK-LABEL: module @init_shared_array_depth_aie1
// CHECK-DAG:   %[[B0:.*]] = aie.buffer(%{{.*}}) {sym_name = "of_buff_0"} : memref<16xi32> = dense<0>
// CHECK-DAG:   %[[B1:.*]] = aie.buffer(%{{.*}}) {sym_name = "of_buff_1"} : memref<16xi32> = dense<1>
// CHECK-DAG:   %[[B2:.*]] = aie.buffer(%{{.*}}) {sym_name = "of_buff_2"} : memref<16xi32>{{ *$}}
// CHECK-DAG:   aie.lock(%{{.*}}) {init = 1 : i32, sym_name = "of_lock_0"}
// CHECK-DAG:   aie.lock(%{{.*}}) {init = 1 : i32, sym_name = "of_lock_1"}
// CHECK-DAG:   %[[L2:.*]] = aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "of_lock_2"}
// CHECK:       aie.core
// CHECK:         aie.use_lock(%[[L2]], Acquire, %{{.*}})
// CHECK:         func.call @fill(%[[B2]])

module @init_shared_array_depth_aie1 {
  aie.device(xcvc1902) {
    func.func private @fill(memref<16xi32>)
    func.func private @drain(memref<16xi32>, memref<16xi32>, memref<16xi32>)
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of (%tile12, {%tile13}, [2 : i32, 3 : i32]) : !aie.objectfifo<memref<16xi32>> = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
    %core12 = aie.core(%tile12) {
      %e0 = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      func.call @fill(%e0) : (memref<16xi32>) -> ()
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }
    %core13 = aie.core(%tile13) {
      %a, %b, %c = aie.objectfifo.acquire @of (Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      func.call @drain(%a, %b, %c) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.objectfifo.release @of (Consume, 3)
      aie.end
    }
  }
}
