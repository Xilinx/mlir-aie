//===- lock_analysis_unassigned_test.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A lock without an ID is a legal state: aie.lock's lockID is an OptionalAttr and
// AIEAssignLockIDs exists to fill it in later. LockAnalysis must therefore tolerate one,
// and must not treat it as reserving an ID. lock_analysis_test.mlir covers the other half
// (pre-existing locks that DO have IDs are respected).

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll --aie-assign-lock-ids %s | FileCheck --check-prefix=ASSIGNED %s

// The unassigned locks reserve nothing, so the objectFifo takes the lowest free IDs on tile(1, 2).
// CHECK: aie.lock @of1_prod_lock_0(%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK: aie.lock @of1_cons_lock_0(%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK: aie.lock @test_prod_lock(%{{.*}}tile_1_2) {init = 1 : i32}
// CHECK: aie.lock @test_cons_lock(%{{.*}}tile_1_2) {init = 0 : i32}

// ...and once AIEAssignLockIDs runs, they get IDs that do not collide with the objectFifo's.
// ASSIGNED-DAG: aie.lock @of1_prod_lock_0(%{{.*}}tile_1_2, 0) {init = 2 : i32}
// ASSIGNED-DAG: aie.lock @of1_cons_lock_0(%{{.*}}tile_1_2, 1) {init = 0 : i32}
// ASSIGNED-DAG: aie.lock @test_prod_lock(%{{.*}}tile_1_2, 2) {init = 1 : i32}
// ASSIGNED-DAG: aie.lock @test_cons_lock(%{{.*}}tile_1_2, 3) {init = 0 : i32}

module @lockAnalysisUnassigned {
   aie.device(xcve2302) {
      %tile12 = aie.tile(1, 2)
      %tile33 = aie.tile(3, 3)

      aie.buffer @test_buff(%tile12) : memref<16xi32>

      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      aie.lock @test_prod_lock(%tile12) {init = 1 : i32}
      aie.lock @test_cons_lock(%tile12) {init = 0 : i32}
      %mem_1_2 = aie.mem(%tile12) {
         %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
         ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(@test_prod_lock, AcquireGreaterEqual, %c1_ul1)
            %test_buff = aie.get_buffer @test_buff : memref<16xi32>
            aie.dma_bd(%test_buff : memref<16xi32> offset = 0 len = 16)
            %c1_ul2 = arith.constant 1 : i32
            aie.use_lock(@test_cons_lock, Release, %c1_ul2)
            aie.next_bd ^bb1
         ^bb2:
            aie.end
      }
   }
}
