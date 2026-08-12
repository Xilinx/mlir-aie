//===- pass_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="dynamic-objFifos=true" %s | FileCheck %s

// The pass-level `dynamic-objFifos=true` flag applies dynamic lowering to every
// core; both cores keep a rolled loop with 4 iter_args (output/input buffer
// indices and held counts). core_0_2's per-core `dynamic_objfifo_lowering =
// false` attribute is preserved on the op but does not override the pass flag.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @passthrough_10_i32(%{{.*}}: memref<10xi32>, %{{.*}}: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[T0:.*]] = aie.tile(0, 0)
// CHECK:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %[[T4:.*]] = aie.tile(0, 4)
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 6) {init = 0 : i32, sym_name = "output_fifo2_cons_prod_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 7) {init = 0 : i32, sym_name = "output_fifo2_cons_cons_lock_0"}
// CHECK:           %[[OF2_B0:.*]] = aie.buffer(%[[T4]]) {sym_name = "output_fifo2_buff_0"} : memref<10xi32>
// CHECK:           %[[OF2_B1:.*]] = aie.buffer(%[[T4]]) {sym_name = "output_fifo2_buff_1"} : memref<10xi32>
// CHECK:           %[[OF2_PROD:.*]] = aie.lock(%[[T4]], 2) {init = 2 : i32, sym_name = "output_fifo2_prod_lock_0"}
// CHECK:           %[[OF2_CONS:.*]] = aie.lock(%[[T4]], 3) {init = 0 : i32, sym_name = "output_fifo2_cons_lock_0"}
// CHECK:           %[[IF2_B0:.*]] = aie.buffer(%[[T4]]) {sym_name = "input_fifo2_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[IF2_B1:.*]] = aie.buffer(%[[T4]]) {sym_name = "input_fifo2_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[IF2_PROD:.*]] = aie.lock(%[[T4]], 0) {init = 2 : i32, sym_name = "input_fifo2_cons_prod_lock_0"}
// CHECK:           %[[IF2_CONS:.*]] = aie.lock(%[[T4]], 1) {init = 0 : i32, sym_name = "input_fifo2_cons_cons_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 4) {init = 0 : i32, sym_name = "input_fifo2_prod_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 5) {init = 0 : i32, sym_name = "input_fifo2_cons_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock_0"}
// CHECK:           %[[OF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK:           %[[OF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK:           %[[OF_PROD:.*]] = aie.lock(%[[T2]], 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:           %[[OF_CONS:.*]] = aie.lock(%[[T2]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:           %[[IF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[IF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[IF_PROD:.*]] = aie.lock(%[[T2]], 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[IF_CONS:.*]] = aie.lock(%[[T2]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[T0]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           aie.flow(%[[T2]], DMA : 0, %[[T0]], DMA : 0)
// CHECK:           aie.flow(%[[T0]], DMA : 1, %[[T4]], DMA : 0)
// CHECK:           aie.flow(%[[T4]], DMA : 0, %[[T0]], DMA : 1)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[IDX0:.*]] = arith.constant 0 : index
// CHECK:             %[[IDX1:.*]] = arith.constant 1 : index
// CHECK:             %[[IDX10:.*]] = arith.constant 10 : index
// CHECK:             %{{.*}}:4 = scf.for %{{.*}} = %[[IDX0]] to %[[IDX10]] step %[[IDX1]] iter_args(%[[A1:.*]] = %[[C0I]], %[[A2:.*]] = %[[C0I]], %[[A3:.*]] = %[[C0I]], %[[A4:.*]] = %[[C0I]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[C1A:.*]] = arith.constant 1 : i32
// CHECK:               %[[C0A:.*]] = arith.constant 0 : i32
// CHECK:               %[[S0:.*]] = arith.subi %[[C1A]], %[[A3]] : i32
// CHECK:               %[[M0:.*]] = arith.maxsi %[[S0]], %[[C0A]] : i32
// CHECK:               aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[M0]])
// CHECK:               %[[OUT:.*]] = arith.addi %[[A3]], %[[M0]] : i32
// CHECK:               %[[ICO:.*]] = arith.index_cast %[[A1]] : i32 to index
// CHECK:               %[[OB:.*]] = scf.index_switch %[[ICO]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[OF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[C1B:.*]] = arith.constant 1 : i32
// CHECK:               %[[C0B:.*]] = arith.constant 0 : i32
// CHECK:               %[[S1:.*]] = arith.subi %[[C1B]], %[[A4]] : i32
// CHECK:               %[[M1:.*]] = arith.maxsi %[[S1]], %[[C0B]] : i32
// CHECK:               aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:               %[[IN:.*]] = arith.addi %[[A4]], %[[M1]] : i32
// CHECK:               %[[ICA:.*]] = arith.index_cast %[[A2]] : i32 to index
// CHECK:               %[[IB:.*]] = scf.index_switch %[[ICA]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[IB]], %[[OB]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[C1C:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[IF_PROD]], Release, %[[C1C]])
// CHECK:               %[[IND:.*]] = arith.subi %[[IN]], %[[C1C]] : i32
// CHECK:               %[[C2A:.*]] = arith.constant 2 : i32
// CHECK:               %[[C1D:.*]] = arith.constant 1 : i32
// CHECK:               %[[II:.*]] = arith.addi %[[A2]], %[[C1D]] : i32
// CHECK:               %[[ICMP:.*]] = arith.cmpi sge, %[[II]], %[[C2A]] : i32
// CHECK:               %[[IW:.*]] = arith.subi %[[II]], %[[C2A]] : i32
// CHECK:               %[[ISEL:.*]] = arith.select %[[ICMP]], %[[IW]], %[[II]] : i32
// CHECK:               %[[C1E:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[OF_CONS]], Release, %[[C1E]])
// CHECK:               %[[OUTD:.*]] = arith.subi %[[OUT]], %[[C1E]] : i32
// CHECK:               %[[C2B:.*]] = arith.constant 2 : i32
// CHECK:               %[[C1F:.*]] = arith.constant 1 : i32
// CHECK:               %[[OI:.*]] = arith.addi %[[A1]], %[[C1F]] : i32
// CHECK:               %[[OCMP:.*]] = arith.cmpi sge, %[[OI]], %[[C2B]] : i32
// CHECK:               %[[OW:.*]] = arith.subi %[[OI]], %[[C2B]] : i32
// CHECK:               %[[OSEL:.*]] = arith.select %[[OCMP]], %[[OW]], %[[OI]] : i32
// CHECK:               scf.yield %[[OSEL]], %[[ISEL]], %[[OUTD]], %[[IND]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = false}
// CHECK:           %{{.*}} = aie.core(%[[T4]]) {
// CHECK:             %[[E_C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[E_IDX0:.*]] = arith.constant 0 : index
// CHECK:             %[[E_IDX1:.*]] = arith.constant 1 : index
// CHECK:             %[[E_IDX10:.*]] = arith.constant 10 : index
// CHECK:             %{{.*}}:4 = scf.for %{{.*}} = %[[E_IDX0]] to %[[E_IDX10]] step %[[E_IDX1]] iter_args(%[[E_A1:.*]] = %[[E_C0I]], %[[E_A2:.*]] = %[[E_C0I]], %[[E_A3:.*]] = %[[E_C0I]], %[[E_A4:.*]] = %[[E_C0I]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[E_C1A:.*]] = arith.constant 1 : i32
// CHECK:               %[[E_C0A:.*]] = arith.constant 0 : i32
// CHECK:               %[[E_S0:.*]] = arith.subi %[[E_C1A]], %[[E_A3]] : i32
// CHECK:               %[[E_M0:.*]] = arith.maxsi %[[E_S0]], %[[E_C0A]] : i32
// CHECK:               aie.use_lock(%[[OF2_PROD]], AcquireGreaterEqual, %[[E_M0]])
// CHECK:               %[[E_OUT:.*]] = arith.addi %[[E_A3]], %[[E_M0]] : i32
// CHECK:               %[[E_ICO:.*]] = arith.index_cast %[[E_A1]] : i32 to index
// CHECK:               %[[E_OB:.*]] = scf.index_switch %[[E_ICO]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[OF2_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[OF2_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[OF2_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[E_C1B:.*]] = arith.constant 1 : i32
// CHECK:               %[[E_C0B:.*]] = arith.constant 0 : i32
// CHECK:               %[[E_S1:.*]] = arith.subi %[[E_C1B]], %[[E_A4]] : i32
// CHECK:               %[[E_M1:.*]] = arith.maxsi %[[E_S1]], %[[E_C0B]] : i32
// CHECK:               aie.use_lock(%[[IF2_CONS]], AcquireGreaterEqual, %[[E_M1]])
// CHECK:               %[[E_IN:.*]] = arith.addi %[[E_A4]], %[[E_M1]] : i32
// CHECK:               %[[E_ICA:.*]] = arith.index_cast %[[E_A2]] : i32 to index
// CHECK:               %[[E_IB:.*]] = scf.index_switch %[[E_ICA]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF2_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF2_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF2_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[E_IB]], %[[E_OB]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[E_C1C:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[IF2_PROD]], Release, %[[E_C1C]])
// CHECK:               %[[E_IND:.*]] = arith.subi %[[E_IN]], %[[E_C1C]] : i32
// CHECK:               %[[E_C2A:.*]] = arith.constant 2 : i32
// CHECK:               %[[E_C1D:.*]] = arith.constant 1 : i32
// CHECK:               %[[E_II:.*]] = arith.addi %[[E_A2]], %[[E_C1D]] : i32
// CHECK:               %[[E_ICMP:.*]] = arith.cmpi sge, %[[E_II]], %[[E_C2A]] : i32
// CHECK:               %[[E_IW:.*]] = arith.subi %[[E_II]], %[[E_C2A]] : i32
// CHECK:               %[[E_ISEL:.*]] = arith.select %[[E_ICMP]], %[[E_IW]], %[[E_II]] : i32
// CHECK:               %[[E_C1E:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[OF2_CONS]], Release, %[[E_C1E]])
// CHECK:               %[[E_OUTD:.*]] = arith.subi %[[E_OUT]], %[[E_C1E]] : i32
// CHECK:               %[[E_C2B:.*]] = arith.constant 2 : i32
// CHECK:               %[[E_C1F:.*]] = arith.constant 1 : i32
// CHECK:               %[[E_OI:.*]] = arith.addi %[[E_A1]], %[[E_C1F]] : i32
// CHECK:               %[[E_OCMP:.*]] = arith.cmpi sge, %[[E_OI]], %[[E_C2B]] : i32
// CHECK:               %[[E_OW:.*]] = arith.subi %[[E_OI]], %[[E_C2B]] : i32
// CHECK:               %[[E_OSEL:.*]] = arith.select %[[E_OCMP]], %[[E_OW]], %[[E_OI]] : i32
// CHECK:               scf.yield %[[E_OSEL]], %[[E_ISEL]], %[[E_OUTD]], %[[E_IND]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[T0]], MM2S, 0)
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[OF_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[OF_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[T0]], S2MM, 0)
// CHECK:           aie.shim_dma_allocation @input_fifo2_shim_alloc(%[[T0]], MM2S, 1)
// CHECK:           %{{.*}} = aie.mem(%[[T4]]) {
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IF2_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF2_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF2_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IF2_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF2_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF2_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF2_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[OF2_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF2_PROD]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF2_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[OF2_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF2_PROD]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo2_shim_alloc(%[[T0]], S2MM, 1)

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
        %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
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
        %0 = aie.objectfifo.acquire @output_fifo2(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %2 = aie.objectfifo.acquire @input_fifo2(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo2(Consume, 1)
        aie.objectfifo.release @output_fifo2(Produce, 1)
      }

      aie.end
    }
  }
}
