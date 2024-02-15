// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @multi_depth {
  aie.device(ipu) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_5 = aie.tile(2, 5)
    %lock_pc = aie.lock(%tile_2_5, 0) {sym_name = "lock_pc"}
    %lock_out = aie.lock(%tile_2_5, 1) {sym_name = "lock_out"}
    %buff_out = aie.buffer(%tile_2_5) {sym_name = "buff_out"} : memref<4x32xi32>
    aie.objectfifo @of_in(%tile_2_0, {%tile_2_3, %tile_2_5}, [2, 2, 3]) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of_inter(%tile_2_3, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    %ext_buffer_in_0 = aie.external_buffer {sym_name = "ext_buffer_in_0"} : memref<32xi32>
    %ext_buffer_in_1 = aie.external_buffer {sym_name = "ext_buffer_in_1"} : memref<32xi32>
    aie.objectfifo.register_external_buffers @of_in(%tile_2_0, {%ext_buffer_in_0, %ext_buffer_in_1}) : (memref<32xi32>, memref<32xi32>)
    func.func @add_one(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c32 = arith.constant 32 : index
      scf.for %arg2 = %c0 to %c32 step %c1 {
        %0 = memref.load %arg0[%arg2] : memref<32xi32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %arg1[%arg2] : memref<32xi32>
      }
      return
    }
    func.func @add_store(%arg0: memref<32xi32>, %arg1: memref<32xi32>, %arg2: memref<4x32xi32>, %arg3: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      scf.for %arg4 = %c0 to %c32 step %c1 {
        %0 = memref.load %arg0[%arg4] : memref<32xi32>
        %1 = memref.load %arg1[%arg4] : memref<32xi32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg2[%arg3, %arg4] : memref<4x32xi32>
      }
      return
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %2 = aie.objectfifo.acquire @of_inter(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        // func.call @add_one(%1, %3) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_inter(Produce, 1)
      }
      aie.end
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      aie.use_lock(%lock_pc, Acquire, 0)
      aie.use_lock(%lock_out, Acquire, 0)
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        %2 = aie.objectfifo.acquire @of_inter(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        // func.call @add_store(%1, %3, %buff_out, %arg0) : (memref<32xi32>, memref<32xi32>, memref<4x32xi32>, index) -> ()
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_inter(Consume, 1)
      }
      aie.use_lock(%lock_out, Release, 1)
      aie.use_lock(%lock_pc, Release, 1)
      aie.end
    }
  }
}

