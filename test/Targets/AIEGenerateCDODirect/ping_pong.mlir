// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @ping_pong {
  aie.device(ipu) {
    %tile_1_2 = aie.tile(1, 2)
    %tile_3_3 = aie.tile(3, 3)
    %out = aie.buffer(%tile_3_3) {sym_name = "out"} : memref<10x16xi32>
    %lock_out = aie.lock(%tile_3_3, 0) {sym_name = "lock_out"}
    aie.objectfifo @objfifo(%tile_1_2, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @generateLineScalar(%arg0: index, %arg1: memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg2 = %c0 to %c16 step %c1 {
        %1 = arith.index_cast %arg2 : index to i32
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg1[%arg2] : memref<16xi32>
      }
      return
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @generateLineScalar(%arg0, %1) : (index, memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo(Produce, 1)
      }
      aie.end
    }
    func.func @storeLineScalar(%arg0: memref<16xi32>, %arg1: index, %arg2: memref<10x16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg3] : memref<16xi32>
        memref.store %0, %arg2[%arg1, %arg3] : memref<10x16xi32>
      }
      return
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      aie.use_lock(%lock_out, Acquire, 0)
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @objfifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @storeLineScalar(%1, %arg0, %out) : (memref<16xi32>, index, memref<10x16xi32>) -> ()
        aie.objectfifo.release @objfifo(Consume, 1)
      }
      aie.use_lock(%lock_out, Release, 1)
      aie.end
    }
  }
}

