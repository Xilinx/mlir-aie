// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @twoFilter2D {
  aie.device(ipu) {
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %out = aie.buffer(%tile_1_4) {sym_name = "out"} : memref<10x16xi32>
    %lock_out = aie.lock(%tile_1_4, 0) {sym_name = "lock_out"}
    aie.objectfifo @of1(%tile_1_2, {%tile_1_3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2(%tile_1_3, {%tile_1_4}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
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
    func.func @firstFilterTwoLines(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg3] : memref<16xi32>
        %1 = memref.load %arg1[%arg3] : memref<16xi32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg2[%arg3] : memref<16xi32>
      }
      return
    }
    func.func @firstFilterThreeLines(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>, %arg3: memref<16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg4 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg4] : memref<16xi32>
        %1 = memref.load %arg1[%arg4] : memref<16xi32>
        %2 = memref.load %arg2[%arg4] : memref<16xi32>
        %3 = arith.addi %0, %1 : i32
        %4 = arith.addi %3, %2 : i32
        memref.store %4, %arg3[%arg4] : memref<16xi32>
      }
      return
    }
    func.func @secondFilterTwoLines(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: index, %arg3: memref<10x16xi32>) {
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
    func.func @secondFilterThreeLines(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>, %arg3: index, %arg4: memref<10x16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg5 = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%arg5] : memref<16xi32>
        %1 = memref.load %arg1[%arg5] : memref<16xi32>
        %2 = memref.load %arg2[%arg5] : memref<16xi32>
        %3 = arith.addi %0, %1 : i32
        %4 = arith.addi %3, %2 : i32
        memref.store %4, %arg4[%arg3, %arg5] : memref<10x16xi32>
      }
      return
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @of1(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @generateLineScalar(%arg0, %1) : (index, memref<16xi32>) -> ()
        aie.objectfifo.release @of1(Produce, 1)
      }
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c9 = arith.constant 9 : index
      %0 = aie.objectfifo.acquire @of1(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %3 = aie.objectfifo.acquire @of2(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      // func.call @firstFilterTwoLines(%1, %2, %4) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.objectfifo.release @of2(Produce, 1)
      scf.for %arg0 = %c1 to %c9 step %c1 {
        %10 = aie.objectfifo.acquire @of1(Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %12 = aie.objectfifo.subview.access %10[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %13 = aie.objectfifo.subview.access %10[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %14 = aie.objectfifo.acquire @of2(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @firstFilterThreeLines(%11, %12, %13, %15) : (memref<16xi32>, memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @of1(Consume, 1)
        aie.objectfifo.release @of2(Produce, 1)
      }
      %5 = aie.objectfifo.acquire @of1(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
      %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %8 = aie.objectfifo.acquire @of2(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      // func.call @firstFilterTwoLines(%6, %7, %9) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
      aie.objectfifo.release @of1(Consume, 2)
      aie.objectfifo.release @of2(Produce, 1)
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c9 = arith.constant 9 : index
      aie.use_lock(%lock_out, Acquire, 0)
      %0 = aie.objectfifo.acquire @of2(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      // func.call @secondFilterTwoLines(%1, %2, %c0, %out) : (memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()
      scf.for %arg0 = %c1 to %c9 step %c1 {
        %6 = aie.objectfifo.acquire @of2(Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %9 = aie.objectfifo.subview.access %6[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        // func.call @secondFilterThreeLines(%7, %8, %9, %arg0, %out) : (memref<16xi32>, memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()
        aie.objectfifo.release @of2(Consume, 1)
      }
      %3 = aie.objectfifo.acquire @of2(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
      %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %5 = aie.objectfifo.subview.access %3[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      // func.call @secondFilterTwoLines(%4, %5, %c9, %out) : (memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()
      aie.objectfifo.release @of2(Consume, 2)
      aie.use_lock(%lock_out, Release, 1)
      aie.end
    }
  }
}

