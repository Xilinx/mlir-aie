//===- different_depth_objectfifos_test.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="dynamic-objFifos=true" %s | FileCheck %s

// The output fifo has depth [2, 2] and the input fifo depth [2, 3]. The consumer
// slides a 2-element window (acquire 2, release 1); the runtime lowering peels
// the first and last iterations and threads buffer indices / held counts through
// a 4-way iter_args loop. The input index selects among three buffers and wraps
// at 3, while the output index wraps at 2.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @add_10_i32(%{{.*}}: memref<10xi32>, %{{.*}}: memref<10xi32>, %{{.*}}: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[T0:.*]] = aie.tile(0, 0)
// CHECK:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock_0"}
// CHECK:           %[[OF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK:           %[[OF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK:           %[[OF_PROD:.*]] = aie.lock(%[[T2]], 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:           %[[OF_CONS:.*]] = aie.lock(%[[T2]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:           %[[IF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[IF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[IF_B2:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_2"} : memref<10xi32>
// CHECK:           %[[IF_PROD:.*]] = aie.lock(%[[T2]], 0) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[IF_CONS:.*]] = aie.lock(%[[T2]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %{{.*}} = aie.lock(%[[T0]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[T0]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           aie.flow(%[[T2]], DMA : 0, %[[T0]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[IDX0:.*]] = arith.constant 0 : index
// CHECK:             %[[IDX1:.*]] = arith.constant 1 : index
// CHECK:             %[[IDX9:.*]] = arith.constant 9 : index
// CHECK:             %[[PC1A:.*]] = arith.constant 1 : i32
// CHECK:             %[[PC0A:.*]] = arith.constant 0 : i32
// CHECK:             %[[PS0:.*]] = arith.subi %[[PC1A]], %[[C0I]] : i32
// CHECK:             %[[PM0:.*]] = arith.maxsi %[[PS0]], %[[PC0A]] : i32
// CHECK:             aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[PM0]])
// CHECK:             %[[POUT:.*]] = arith.addi %[[C0I]], %[[PM0]] : i32
// CHECK:             %[[PIC0:.*]] = arith.index_cast %[[C0I]] : i32 to index
// CHECK:             %[[POB:.*]] = scf.index_switch %[[PIC0]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[OF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[PC1B:.*]] = arith.constant 1 : i32
// CHECK:             %[[PC0B:.*]] = arith.constant 0 : i32
// CHECK:             %[[PS1:.*]] = arith.subi %[[PC1B]], %[[C0I]] : i32
// CHECK:             %[[PM1:.*]] = arith.maxsi %[[PS1]], %[[PC0B]] : i32
// CHECK:             aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[PM1]])
// CHECK:             %[[PIN:.*]] = arith.addi %[[C0I]], %[[PM1]] : i32
// CHECK:             %[[PIC1:.*]] = arith.index_cast %[[C0I]] : i32 to index
// CHECK:             %[[PIB:.*]] = scf.index_switch %[[PIC1]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[PIB]], %[[PIB]], %[[POB]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[PC1C:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[OF_CONS]], Release, %[[PC1C]])
// CHECK:             %[[POUTD:.*]] = arith.subi %[[POUT]], %[[PC1C]] : i32
// CHECK:             %[[PC2:.*]] = arith.constant 2 : i32
// CHECK:             %[[PC1D:.*]] = arith.constant 1 : i32
// CHECK:             %[[POI:.*]] = arith.addi %[[C0I]], %[[PC1D]] : i32
// CHECK:             %[[POCMP:.*]] = arith.cmpi sge, %[[POI]], %[[PC2]] : i32
// CHECK:             %[[POW:.*]] = arith.subi %[[POI]], %[[PC2]] : i32
// CHECK:             %[[POSEL:.*]] = arith.select %[[POCMP]], %[[POW]], %[[POI]] : i32
// CHECK:             %[[LOOP:.*]]:4 = scf.for %{{.*}} = %[[IDX0]] to %[[IDX9]] step %[[IDX1]] iter_args(%[[A1:.*]] = %[[POSEL]], %[[A2:.*]] = %[[C0I]], %[[A3:.*]] = %[[POUTD]], %[[A4:.*]] = %[[PIN]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[LC1A:.*]] = arith.constant 1 : i32
// CHECK:               %[[LC0A:.*]] = arith.constant 0 : i32
// CHECK:               %[[LS0:.*]] = arith.subi %[[LC1A]], %[[A3]] : i32
// CHECK:               %[[LM0:.*]] = arith.maxsi %[[LS0]], %[[LC0A]] : i32
// CHECK:               aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[LM0]])
// CHECK:               %[[LOUT:.*]] = arith.addi %[[A3]], %[[LM0]] : i32
// CHECK:               %[[LICO:.*]] = arith.index_cast %[[A1]] : i32 to index
// CHECK:               %[[LOB:.*]] = scf.index_switch %[[LICO]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[OF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[LC2A:.*]] = arith.constant 2 : i32
// CHECK:               %[[LC0B:.*]] = arith.constant 0 : i32
// CHECK:               %[[LS1:.*]] = arith.subi %[[LC2A]], %[[A4]] : i32
// CHECK:               %[[LM1:.*]] = arith.maxsi %[[LS1]], %[[LC0B]] : i32
// CHECK:               aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[LM1]])
// CHECK:               %[[LIN:.*]] = arith.addi %[[A4]], %[[LM1]] : i32
// CHECK:               %[[LICA:.*]] = arith.index_cast %[[A2]] : i32 to index
// CHECK:               %[[LIB0:.*]] = scf.index_switch %[[LICA]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[LICB:.*]] = arith.index_cast %[[A2]] : i32 to index
// CHECK:               %[[LIB1:.*]] = scf.index_switch %[[LICB]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @add_10_i32(%[[LIB0]], %[[LIB1]], %[[LOB]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[LC1B:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[IF_PROD]], Release, %[[LC1B]])
// CHECK:               %[[LIND:.*]] = arith.subi %[[LIN]], %[[LC1B]] : i32
// CHECK:               %[[LC3:.*]] = arith.constant 3 : i32
// CHECK:               %[[LC1C:.*]] = arith.constant 1 : i32
// CHECK:               %[[LII:.*]] = arith.addi %[[A2]], %[[LC1C]] : i32
// CHECK:               %[[LICMP:.*]] = arith.cmpi sge, %[[LII]], %[[LC3]] : i32
// CHECK:               %[[LIW:.*]] = arith.subi %[[LII]], %[[LC3]] : i32
// CHECK:               %[[LISEL:.*]] = arith.select %[[LICMP]], %[[LIW]], %[[LII]] : i32
// CHECK:               %[[LC1D:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[OF_CONS]], Release, %[[LC1D]])
// CHECK:               %[[LOUTD:.*]] = arith.subi %[[LOUT]], %[[LC1D]] : i32
// CHECK:               %[[LC2C:.*]] = arith.constant 2 : i32
// CHECK:               %[[LC1E:.*]] = arith.constant 1 : i32
// CHECK:               %[[LOI:.*]] = arith.addi %[[A1]], %[[LC1E]] : i32
// CHECK:               %[[LOCMP:.*]] = arith.cmpi sge, %[[LOI]], %[[LC2C]] : i32
// CHECK:               %[[LOW:.*]] = arith.subi %[[LOI]], %[[LC2C]] : i32
// CHECK:               %[[LOSEL:.*]] = arith.select %[[LOCMP]], %[[LOW]], %[[LOI]] : i32
// CHECK:               scf.yield %[[LOSEL]], %[[LISEL]], %[[LOUTD]], %[[LIND]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             %[[EC1A:.*]] = arith.constant 1 : i32
// CHECK:             %[[EC0A:.*]] = arith.constant 0 : i32
// CHECK:             %[[ES0:.*]] = arith.subi %[[EC1A]], %[[LOOP]]#2 : i32
// CHECK:             %[[EM0:.*]] = arith.maxsi %[[ES0]], %[[EC0A]] : i32
// CHECK:             aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[EM0]])
// CHECK:             %[[EOUT:.*]] = arith.addi %[[LOOP]]#2, %[[EM0]] : i32
// CHECK:             %[[EICO:.*]] = arith.index_cast %[[LOOP]]#0 : i32 to index
// CHECK:             %[[EOB:.*]] = scf.index_switch %[[EICO]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[OF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[EC2A:.*]] = arith.constant 2 : i32
// CHECK:             %[[EC0B:.*]] = arith.constant 0 : i32
// CHECK:             %[[ES1:.*]] = arith.subi %[[EC2A]], %[[LOOP]]#3 : i32
// CHECK:             %[[EM1:.*]] = arith.maxsi %[[ES1]], %[[EC0B]] : i32
// CHECK:             aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[EM1]])
// CHECK:             %[[EIN:.*]] = arith.addi %[[LOOP]]#3, %[[EM1]] : i32
// CHECK:             %[[EICA:.*]] = arith.index_cast %[[LOOP]]#1 : i32 to index
// CHECK:             %[[EIB0:.*]] = scf.index_switch %[[EICA]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[EICB:.*]] = arith.index_cast %[[LOOP]]#1 : i32 to index
// CHECK:             %[[EIB1:.*]] = scf.index_switch %[[EICB]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[EIB0]], %[[EIB1]], %[[EOB]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[EC2B:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[IF_PROD]], Release, %[[EC2B]])
// CHECK:             %{{.*}} = arith.subi %[[EIN]], %[[EC2B]] : i32
// CHECK:             %[[EC1B:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[OF_CONS]], Release, %[[EC1B]])
// CHECK:             %{{.*}} = arith.subi %[[EOUT]], %[[EC1B]] : i32
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[T0]], MM2S, 0)
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[IF_B2]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[OF_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[OF_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[T0]], S2MM, 0)

module {
  aie.device(npu1_1col) {
    func.func @add_10_i32(%line_in1: memref<10xi32>, %line_in2: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, [2, 3]) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, [2, 2]) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 9 : index

      %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
      %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      func.call @add_10_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @output_fifo(Produce, 1)

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %4 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %6 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @add_10_i32(%7, %8, %5) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      %9 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
      %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %11 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
      %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      func.call @add_10_i32(%12, %13, %10) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @input_fifo(Consume, 2)
      aie.objectfifo.release @output_fifo(Produce, 1)

      aie.end
    }
  }
}
