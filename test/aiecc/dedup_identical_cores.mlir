//===- dedup_identical_cores.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Cores running one program over their own buffers compile it once. Cores 0
// and 1 copy; core 2 adds one. Opt and llc run on two canonical modules, and
// every core's object carries its own names again before it links.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --verbose --tmpdir %t %s 2>&1 | FileCheck %s --check-prefix=LOG
// RUN: ls %t/canonical_*.ll | count 2
// RUN: llvm-readelf -s %t/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s --check-prefix=CORE02 --implicit-check-not=__aiecc_canon_
// RUN: llvm-readelf -s %t/elfs_main_core_1_2/elfs_main_core_1_2.elf | FileCheck %s --check-prefix=CORE12 --implicit-check-not=__aiecc_canon_
// RUN: llvm-readelf -s %t/elfs_main_core_2_2/elfs_main_core_2_2.elf | FileCheck %s --check-prefix=CORE22 --implicit-check-not=__aiecc_canon_

// LOG-DAG: aiecc: main_core_0_2: {{.*}}canonical_{{[0-9A-F]+}}.o renamed to {{.*}}objects_main_core_0_2.o
// LOG-DAG: aiecc: main_core_1_2: {{.*}}canonical_{{[0-9A-F]+}}.o renamed to {{.*}}objects_main_core_1_2.o
// LOG-DAG: aiecc: main_core_2_2: {{.*}}canonical_{{[0-9A-F]+}}.o renamed to {{.*}}objects_main_core_2_2.o

// CORE02-DAG: FUNC GLOBAL DEFAULT {{.*}} core_0_2
// CORE02-DAG: in0_cons_buff_0
// CORE02-DAG: out0_buff_0
// CORE12-DAG: FUNC GLOBAL DEFAULT {{.*}} core_1_2
// CORE12-DAG: in1_cons_buff_0
// CORE12-DAG: out1_buff_0
// CORE22-DAG: FUNC GLOBAL DEFAULT {{.*}} core_2_2
// CORE22-DAG: in2_cons_buff_0
// CORE22-DAG: out2_buff_0

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)

    aie.objectfifo @in0(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out0(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%tile_1_2, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2(%tile_2_0, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%tile_2_2, {%tile_2_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %elem_in = aie.objectfifo.acquire @in0(Consume, 1) : memref<16xi32>
      %elem_out = aie.objectfifo.acquire @out0(Produce, 1) : memref<16xi32>
      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        memref.store %val, %elem_out[%i] : memref<16xi32>
      }
      aie.objectfifo.release @in0(Consume, 1)
      aie.objectfifo.release @out0(Produce, 1)
      aie.end
    }

    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %elem_in = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
      %elem_out = aie.objectfifo.acquire @out1(Produce, 1) : memref<16xi32>
      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        memref.store %val, %elem_out[%i] : memref<16xi32>
      }
      aie.objectfifo.release @in1(Consume, 1)
      aie.objectfifo.release @out1(Produce, 1)
      aie.end
    }

    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %one = arith.constant 1 : i32
      %elem_in = aie.objectfifo.acquire @in2(Consume, 1) : memref<16xi32>
      %elem_out = aie.objectfifo.acquire @out2(Produce, 1) : memref<16xi32>
      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %sum = arith.addi %val, %one : i32
        memref.store %sum, %elem_out[%i] : memref<16xi32>
      }
      aie.objectfifo.release @in2(Consume, 1)
      aie.objectfifo.release @out2(Produce, 1)
      aie.end
    }
  }
}
