// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin

module @dmaChannels {
  aie.device(ipu) {
    %tile_1_2 = aie.tile(1, 2)
    %tile_3_3 = aie.tile(3, 3)
    %out = aie.buffer(%tile_3_3) {sym_name = "out"} : memref<10x16xi32>
    %lock_out = aie.lock(%tile_3_3, 0) {sym_name = "lock_out"}
    aie.objectfifo @of_in0(%tile_3_3, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_in1(%tile_3_3, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out0(%tile_1_2, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out1(%tile_1_2, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @copy(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg2 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg2] : memref<16xi32>
        memref.store %0, %arg1[%arg2] : memref<16xi32>
      }
      return
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @of_in0(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %2 = aie.objectfifo.acquire @of_in1(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %4 = aie.objectfifo.acquire @of_out0(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %6 = aie.objectfifo.acquire @of_out1(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @copy(%1, %5) : (memref<16xi32>, memref<16xi32>) -> ()
        // func.call @copy(%3, %7) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @of_in0(Consume, 1)
        aie.objectfifo.release @of_in1(Consume, 1)
        aie.objectfifo.release @of_out0(Produce, 1)
        aie.objectfifo.release @of_out1(Produce, 1)
      }
      aie.end
    }
    func.func @generateLineScalar(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %0 = arith.index_cast %arg1 : index to i32
        memref.store %0, %arg0[%arg1] : memref<16xi32>
      }
      return
    }
    func.func @addAndStore(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: index, %arg3: memref<10x16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg4] : memref<16xi32>
        %1 = memref.load %arg1[%arg4] : memref<16xi32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg3[%arg2, %arg4] : memref<10x16xi32>
      }
      return
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      aie.use_lock(%lock_out, Acquire, 0)
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @of_in0(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %2 = aie.objectfifo.acquire @of_in1(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @generateLineScalar(%1) : (memref<16xi32>) -> ()
        // func.call @generateLineScalar(%3) : (memref<16xi32>) -> ()
        aie.objectfifo.release @of_in0(Produce, 1)
        aie.objectfifo.release @of_in1(Produce, 1)
        %4 = aie.objectfifo.acquire @of_out0(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %6 = aie.objectfifo.acquire @of_out1(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @addAndStore(%5, %7, %arg0, %out) : (memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()
        aie.objectfifo.release @of_out0(Consume, 1)
        aie.objectfifo.release @of_out1(Consume, 1)
      }
      aie.use_lock(%lock_out, Release, 1)
      aie.end
    }
  }
}

