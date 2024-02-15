// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @broadcast {
  aie.device(ipu) {
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_3_3 = aie.tile(3, 3)
    %out12 = aie.buffer(%tile_1_2) {sym_name = "out12"} : memref<4x16xi32>
    %lock_out12 = aie.lock(%tile_1_2, 0) {sym_name = "lock_out12"}
    %out14 = aie.buffer(%tile_1_4) {sym_name = "out14"} : memref<4x16xi32>
    %lock_out14 = aie.lock(%tile_1_4, 0) {sym_name = "lock_out14"}
    %out33 = aie.buffer(%tile_3_3) {sym_name = "out33"} : memref<4x16xi32>
    %lock_out33 = aie.lock(%tile_3_3, 0) {sym_name = "lock_out33"}
    aie.objectfifo @objfifo(%tile_1_3, {%tile_1_2, %tile_1_4, %tile_3_3}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
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
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @generateLineScalar(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @objfifo(Produce, 1)
      }
      aie.end
    }
    func.func @storeLineScalar(%arg0: memref<16xi32>, %arg1: index, %arg2: memref<4x16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg3] : memref<16xi32>
        memref.store %0, %arg2[%arg1, %arg3] : memref<4x16xi32>
      }
      return
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      aie.use_lock(%lock_out12, Acquire, 0)
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @objfifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @storeLineScalar(%1, %arg0, %out12) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
        aie.objectfifo.release @objfifo(Consume, 1)
      }
      aie.use_lock(%lock_out12, Release, 1)
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      aie.use_lock(%lock_out14, Acquire, 0)
      scf.for %arg0 = %c0 to %c4 step %c2 {
        %0 = aie.objectfifo.acquire @objfifo(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @storeLineScalar(%1, %arg0, %out14) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
        %3 = arith.addi %arg0, %c1 : index
        // func.call @storeLineScalar(%2, %3, %out14) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
        aie.objectfifo.release @objfifo(Consume, 2)
      }
      aie.use_lock(%lock_out14, Release, 1)
      aie.end
    }
    func.func @addAndStore(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: index, %arg3: memref<4x16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg4] : memref<16xi32>
        %1 = memref.load %arg1[%arg4] : memref<16xi32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg3[%arg2, %arg4] : memref<4x16xi32>
      }
      return
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c3_0 = arith.constant 3 : index
      aie.use_lock(%lock_out33, Acquire, 0)
      scf.for %arg0 = %c0 to %c3_0 step %c1 {
        %2 = aie.objectfifo.acquire @objfifo(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @addAndStore(%3, %4, %arg0, %out33) : (memref<16xi32>, memref<16xi32>, index, memref<4x16xi32>) -> ()
        aie.objectfifo.release @objfifo(Consume, 1)
      }
      %0 = aie.objectfifo.acquire @objfifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      // func.call @storeLineScalar(%1, %c3, %out33) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
      aie.objectfifo.release @objfifo(Consume, 1)
      aie.use_lock(%lock_out33, Release, 1)
      aie.end
    }
  }
}

