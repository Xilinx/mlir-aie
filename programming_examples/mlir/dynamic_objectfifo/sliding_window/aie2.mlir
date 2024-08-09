//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf 

module {
  aie.device(npu1_1col) {
    func.func private @passthrough_64_i32(memref<10xi32>, memref<10xi32>, memref<10xi32>)

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>

        %is_first_iter = arith.cmpi eq, %arg0, %c0 : index
        %last_iter = arith.subi %c4294967295, %c1 : index
        %is_last_iter = arith.cmpi eq, %arg0, %last_iter : index

        scf.if %is_first_iter {
          %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
          func.call @sum_64_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()

        } else {
          scf.if %is_last_iter {
            %2 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
            %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
            func.call @sum_64_i32(%3, %4, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
            aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
            
          } else {
            %2 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
            %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
            func.call @sum_64_i32(%3, %4, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
            aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
          }
        }

        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      aie.end
    } {link_with = "kernel.o"}

    aiex.runtime_sequence(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<10xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<10xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
