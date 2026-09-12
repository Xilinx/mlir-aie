//===- repeat_count_to_shim_test.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// `repeat_count` on an output fifo, where the consumer is a shim tile rather
// than another compute or MemTile. The producer core drains into DDR, so the
// repeat lands on the compute tile's MM2S chain and the shim side stays a
// plain S2MM.

// CHECK-LABEL: @repeatToShim
// CHECK-DAG:     %[[PROD:.*]] = aie.lock(%{{.*}}tile_0_2) {init = 2 : i32, sym_name = "of_out_prod_lock_0"}
// CHECK-DAG:     %[[CONS:.*]] = aie.lock(%{{.*}}tile_0_2) {init = 0 : i32, sym_name = "of_out_cons_lock_0"}
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}shim_noc_tile_0_0, DMA : 0)

// The core holds repeat_count objects at a time, matching the lock's init.
// CHECK:         aie.core(%{{.*}}tile_0_2)
// CHECK:           %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           scf.for
// CHECK:             aie.use_lock(%[[PROD]], AcquireGreaterEqual, %[[C2]])
// CHECK:             aie.use_lock(%[[CONS]], Release, %[[C2]])

// The shim end keeps a plain S2MM record; the repeat stays on the compute tile.
// CHECK:         aie.shim_dma_allocation @of_out_shim_alloc(%{{.*}}shim_noc_tile_0_0, S2MM, 0)
// CHECK:         aie.mem(%{{.*}}tile_0_2)
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 1)

module @repeatToShim {
 aie.device(npu1_1col) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of_out (%tile02, {%tile00}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %v = arith.constant 42 : i32
      scf.for %i = %c0 to %c4 step %c1 {
        %e = aie.objectfifo.acquire @of_out (Produce, 1) : memref<16xi32>
        memref.store %v, %e[%c0] : memref<16xi32>
        aie.objectfifo.release @of_out (Produce, 1)
      }
      aie.end
    }
 }
}
