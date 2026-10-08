//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Three scf.while loops nested inside each other. The core replaces every
// element by its Collatz total stopping time: the number of steps that
// x <- x/2 (x even) or x <- 3x+1 (x odd) needs to reach 1.
//
// Each loop level stresses a different property of the lowering:
//
//   outer   constant-true condition, object fifo acquire/release in the body,
//           so the fifo locks pace the core
//   middle  walks the elements of one tile, carrying the index
//   inner   data-dependent trip count, since the step count differs per element
//
// The inner loop picks the next value with arith.select so that the loop body
// stays branch-free, which keeps the nesting itself as the thing under test.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %shim = aie.tile(0, 0)
    %core_tile = aie.tile(0, 2)

    aie.objectfifo @of_in(%shim, {%core_tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%core_tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core_tile) {
      %true = arith.constant true
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32

      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %in = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
        %out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>

        %end = scf.while (%i = %c0) : (index) -> index {
          %more = arith.cmpi slt, %i, %c16 : index
          scf.condition(%more) %i : index
        } do {
        ^bb0(%i : index):
          %seed = memref.load %in[%i] : memref<16xi32>
          %collatz:2 = scf.while (%x = %seed, %n = %c0_i32) : (i32, i32) -> (i32, i32) {
            %running = arith.cmpi ne, %x, %c1_i32 : i32
            scf.condition(%running) %x, %n : i32, i32
          } do {
          ^bb0(%x : i32, %n : i32):
            %low_bit = arith.andi %x, %c1_i32 : i32
            %odd = arith.cmpi eq, %low_bit, %c1_i32 : i32
            %tripled = arith.muli %x, %c3_i32 : i32
            %climb = arith.addi %tripled, %c1_i32 : i32
            %halved = arith.shrui %x, %c1_i32 : i32
            %next_x = arith.select %odd, %climb, %halved : i32
            %next_n = arith.addi %n, %c1_i32 : i32
            scf.yield %next_x, %next_n : i32, i32
          }
          memref.store %collatz#1, %out[%i] : memref<16xi32>
          %next_i = arith.addi %i, %c1 : index
          scf.yield %next_i : index
        }

        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
        scf.yield
      }

      aie.end
    }

    aie.runtime_sequence(%in : memref<128xi32>, %out : memref<128xi32>) {
      %t_out = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%out : memref<128xi32> offset = 0 len = 128)
        aie.end
      } {issue_token = true}
      %t_in = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<128xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%t_out)
      aiex.dma_start_task(%t_in)
      aiex.dma_await_task(%t_out)
    }
  }
}
