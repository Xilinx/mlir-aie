//===- pass_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// `default-dynamic=true` sets the lowering for cores that do not carry an
// explicit `dynamic_objfifo_lowering` attribute. core_0_2 sets that attribute to
// false and so is statically unrolled (loop step 2, buffers bound directly),
// while core_0_4 follows the dynamic default and keeps a rolled loop carrying
// the output/input buffer indices as iter_args.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @passthrough_10_i32(%{{.*}}: memref<10xi32>, %{{.*}}: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK-DAG:           %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           %[[T4:.*]] = aie.tile(0, 4)
// CHECK-DAG:           aie.buffer @[[OF2_B0:output_fifo2_buff_0]](%[[T4]]) : memref<10xi32>
// CHECK-DAG:           aie.buffer @[[OF2_B1:output_fifo2_buff_1]](%[[T4]]) : memref<10xi32>
// CHECK-DAG:           aie.lock @[[OF2_PROD:output_fifo2_prod_lock_0]](%[[T4]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[OF2_CONS:output_fifo2_cons_lock_0]](%[[T4]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[IF2_B0:input_fifo2_cons_buff_0]](%[[T4]]) : memref<10xi32>
// CHECK-DAG:           aie.buffer @[[IF2_B1:input_fifo2_cons_buff_1]](%[[T4]]) : memref<10xi32>
// CHECK-DAG:           aie.lock @[[IF2_PROD:input_fifo2_cons_prod_lock_0]](%[[T4]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[IF2_CONS:input_fifo2_cons_cons_lock_0]](%[[T4]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[OF_B0:output_fifo_buff_0]](%[[T2]]) : memref<10xi32>
// CHECK-DAG:           aie.buffer @[[OF_B1:output_fifo_buff_1]](%[[T2]]) : memref<10xi32>
// CHECK-DAG:           aie.lock @[[OF_PROD:output_fifo_prod_lock_0]](%[[T2]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[OF_CONS:output_fifo_cons_lock_0]](%[[T2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[IF_B0:input_fifo_cons_buff_0]](%[[T2]]) : memref<10xi32>
// CHECK-DAG:           aie.buffer @[[IF_B1:input_fifo_cons_buff_1]](%[[T2]]) : memref<10xi32>
// CHECK-DAG:           aie.lock @[[IF_PROD:input_fifo_cons_prod_lock_0]](%[[T2]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[IF_CONS:input_fifo_cons_cons_lock_0]](%[[T2]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[SHIM]], DMA : 0, %[[T2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[T2]], DMA : 0, %[[SHIM]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[SHIM]], DMA : 1, %[[T4]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[T4]], DMA : 0, %[[SHIM]], DMA : 1)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK-DAG:             %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %{{.*}} = %[[C0]] to %[[C10]] step %[[C2]] {
// CHECK:               aie.use_lock(@[[OF_PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               aie.use_lock(@[[IF_CONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               func.call @passthrough_10_i32(%{{[^ ,)]+}}, %{{[^ ,)]+}}) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(@[[IF_PROD]], Release, %[[C1I]])
// CHECK:               aie.use_lock(@[[OF_CONS]], Release, %[[C1I]])
// CHECK:               aie.use_lock(@[[OF_PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               aie.use_lock(@[[IF_CONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               func.call @passthrough_10_i32(%{{[^ ,)]+}}, %{{[^ ,)]+}}) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(@[[IF_PROD]], Release, %[[C1I]])
// CHECK:               aie.use_lock(@[[OF_CONS]], Release, %[[C1I]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = false}
// CHECK:           %{{.*}} = aie.core(%[[T4]]) {
// CHECK-DAG:             %[[E_C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[E_C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[E_C10:.*]] = arith.constant 10 : index
// CHECK-DAG:             %[[E_C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[O0:.*]] = memref.alloca() : memref<memref<10xi32>>
// CHECK:             memref.store %{{[^ ,)]+}}, %[[O0]][] : memref<memref<10xi32>>
// CHECK:             %[[O1:.*]] = memref.alloca() : memref<memref<10xi32>>
// CHECK:             memref.store %{{[^ ,)]+}}, %[[O1]][] : memref<memref<10xi32>>
// CHECK:             %[[I0:.*]] = memref.alloca() : memref<memref<10xi32>>
// CHECK:             memref.store %{{[^ ,)]+}}, %[[I0]][] : memref<memref<10xi32>>
// CHECK:             %[[I1:.*]] = memref.alloca() : memref<memref<10xi32>>
// CHECK:             memref.store %{{[^ ,)]+}}, %[[I1]][] : memref<memref<10xi32>>
// CHECK:             scf.for %{{.*}} = %[[E_C0]] to %[[E_C10]] step %[[E_C1]] {
// CHECK:               aie.use_lock(@[[OF2_PROD]], AcquireGreaterEqual, %[[E_C1I]])
// CHECK:               %[[OB:.*]] = memref.load %[[O0]][] : memref<memref<10xi32>>
// CHECK:               aie.use_lock(@[[IF2_CONS]], AcquireGreaterEqual, %[[E_C1I]])
// CHECK:               %[[IB:.*]] = memref.load %[[I0]][] : memref<memref<10xi32>>
// CHECK:               func.call @passthrough_10_i32(%[[IB]], %[[OB]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(@[[IF2_PROD]], Release, %[[E_C1I]])
// CHECK:               %[[IX:.*]] = memref.load %[[I0]][] : memref<memref<10xi32>>
// CHECK:               %[[IY:.*]] = memref.load %[[I1]][] : memref<memref<10xi32>>
// CHECK:               memref.store %[[IY]], %[[I0]][] : memref<memref<10xi32>>
// CHECK:               memref.store %[[IX]], %[[I1]][] : memref<memref<10xi32>>
// CHECK:               aie.use_lock(@[[OF2_CONS]], Release, %[[E_C1I]])
// CHECK:               %[[OX:.*]] = memref.load %[[O0]][] : memref<memref<10xi32>>
// CHECK:               %[[OY:.*]] = memref.load %[[O1]][] : memref<memref<10xi32>>
// CHECK:               memref.store %[[OY]], %[[O0]][] : memref<memref<10xi32>>
// CHECK:               memref.store %[[OX]], %[[O1]][] : memref<memref<10xi32>>
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK-DAG:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[SHIM]], MM2S, 0)
// CHECK-DAG:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[SHIM]], S2MM, 0)
// CHECK-DAG:           aie.shim_dma_allocation @input_fifo2_shim_alloc(%[[SHIM]], MM2S, 1)
// CHECK-DAG:           aie.shim_dma_allocation @output_fifo2_shim_alloc(%[[SHIM]], S2MM, 1)
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[IF_B0_ACCESS_0:.*]] = aie.get_buffer @[[IF_B0]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[IF_B0_ACCESS_0]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[IF_B1_ACCESS_1:.*]] = aie.get_buffer @[[IF_B1]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[IF_B1_ACCESS_1]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(@[[OF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[OF_B0_ACCESS_2:.*]] = aie.get_buffer @[[OF_B0]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[OF_B0_ACCESS_2]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[OF_PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(@[[OF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[OF_B1_ACCESS_3:.*]] = aie.get_buffer @[[OF_B1]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[OF_B1_ACCESS_3]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[OF_PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T4]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[IF2_PROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[IF2_B0_ACCESS_4:.*]] = aie.get_buffer @[[IF2_B0]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[IF2_B0_ACCESS_4]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[IF2_CONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[IF2_PROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[IF2_B1_ACCESS_5:.*]] = aie.get_buffer @[[IF2_B1]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[IF2_B1_ACCESS_5]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[IF2_CONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(@[[OF2_CONS]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[OF2_B0_ACCESS_6:.*]] = aie.get_buffer @[[OF2_B0]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[OF2_B0_ACCESS_6]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[OF2_PROD]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(@[[OF2_CONS]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[OF2_B1_ACCESS_7:.*]] = aie.get_buffer @[[OF2_B1]] : memref<10xi32>
// CHECK:             aie.dma_bd(%[[OF2_B1_ACCESS_7]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[OF2_PROD]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    aie.objectfifo @input_fifo2(%tile_0_0, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo2(%tile_0_4, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @output_fifo(Produce, 1) : memref<10xi32>
        %3 = aie.objectfifo.acquire @input_fifo(Consume, 1) : memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = false}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @output_fifo2(Produce, 1) : memref<10xi32>
        %3 = aie.objectfifo.acquire @input_fifo2(Consume, 1) : memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo2(Consume, 1)
        aie.objectfifo.release @output_fifo2(Produce, 1)
      }

      aie.end
    }
  }
}
