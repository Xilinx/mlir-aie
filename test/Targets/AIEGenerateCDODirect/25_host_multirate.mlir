// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin
// CHECK: Generating: {{.*}}aie_cdo_init.bin

module {
  aie.device(ipu) {
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_0 = aie.tile(1, 0)
    %hostLock = aie.lock(%tile_0_4, 0) {sym_name = "hostLock"}
    func.func @evaluate_condition(%arg0: i32) -> i1 {
      %true = arith.constant true
      return %true : i1
    }
    func.func @payload(%arg0: i32) -> i32 {
      %c1_i32 = arith.constant 1 : i32
      return %c1_i32 : i32
    }
    %ddr_test_buffer_in = aie.external_buffer {sym_name = "ddr_test_buffer_in"} : memref<256xi32>
    %ddr_test_buffer_out = aie.external_buffer {sym_name = "ddr_test_buffer_out"} : memref<64xi32>
    aie.objectfifo @of_in(%tile_1_0, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%tile_0_4, {%tile_1_0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.register_external_buffers @of_in(%tile_1_0, {%ddr_test_buffer_in}) : (memref<256xi32>)
    aie.objectfifo.register_external_buffers @of_out(%tile_1_0, {%ddr_test_buffer_out}) : (memref<64xi32>)
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c1_i32 = arith.constant 1 : i32
      %0 = scf.while (%arg0 = %c1_i32) : (i32) -> i32 {
        %1 = func.call @evaluate_condition(%arg0) : (i32) -> i1
        scf.condition(%1) %arg0 : i32
      } do {
      ^bb0(%arg0: i32):
        %1 = func.call @payload(%arg0) : (i32) -> i32
        aie.use_lock(%hostLock, Acquire, 1)
        %2 = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
        %3 = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %4 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        %5 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        scf.for %arg1 = %c0 to %c64 step %c1 {
          %6 = memref.load %4[%arg1] : memref<64xi32>
          memref.store %6, %5[%arg1] : memref<64xi32>
        }
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
        aie.use_lock(%hostLock, Release, 0)
        scf.yield %1 : i32
      }
      aie.end
    }
  }
}

