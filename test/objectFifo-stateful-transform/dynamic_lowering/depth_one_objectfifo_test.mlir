//===- depth_one_objectfifo_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK:  module {
// CHECK:    aie.device(npu1_1col) {
// CHECK:      func.func @passthrough_10_i32(%arg0: memref<10xi32>) {
// CHECK:        return
// CHECK:      }
// CHECK:      %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:      %{{.*}}tile_0_2 = aie.tile(0, 2)
// CHECK:      %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:      %[[VAL_1:.*]] = aie.lock(%{{.*}}tile_0_2) {init = 1 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:      %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_0_2) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:      aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK:      %core_0_2 = aie.core(%{{.*}}tile_0_2) {
// CHECK:        %{{.*}} = arith.constant 10 : index
// CHECK:        %{{.*}} = arith.constant 1 : index
// CHECK:        %{{.*}} = arith.constant 0 : index
// CHECK:        %[[C1:.*]] = arith.constant 1 : i32
// CHECK:        scf.for %arg0 = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %[[C1]])
// CHECK:          func.call @passthrough_10_i32(%[[VAL_0]]) : (memref<10xi32>) -> ()
// CHECK:          aie.use_lock(%[[VAL_1]], Release, %[[C1]])
// CHECK:        }
// CHECK:        aie.end
// CHECK:      } {dynamic_objfifo_lowering = true}
// CHECK:      aie.shim_dma_allocation @input_fifo_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
// CHECK:      %mem_0_2 = aie.mem(%{{.*}}tile_0_2) {
// CHECK:        %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:      ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:        aie.use_lock(%[[VAL_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:        aie.dma_bd(%[[VAL_0]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:        aie.use_lock(%[[VAL_2]], Release, %{{.*}})
// CHECK:        aie.next_bd ^bb1
// CHECK:      ^bb2:  // pred: ^bb0
// CHECK:        aie.end
// CHECK:      }
// CHECK:    }
// CHECK:  }

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @input_fifo(Consume, 1) : memref<10xi32>
        func.call @passthrough_10_i32(%1) : (memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
