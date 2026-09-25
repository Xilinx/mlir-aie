//===- only_insts_sequence_cores.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An instruction-only build compiles just the cores whose tiles hold a buffer
// the runtime sequence names, and its instructions match a full build's.
//
// Placement measures each compiled core before placing its tile's buffers, so
// @rtp's address depends on core (0, 2)'s static data: the last run places it
// without that measurement and must produce different instructions, which is
// what keeps the `cmp` checks from passing vacuously. Cores (1, 2) and
// (2, 2) hold nothing the sequence names and are not compiled at all.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d/seq %t.d/unified %t.d/full %t.d/unmeasured

// RUN: cd %t.d/seq && aiecc -v --get-npu-insts --npu-insts-name=insts.bin %s 2>&1 | FileCheck %s --check-prefix=SEQ --implicit-check-not=main_core_1_2 --implicit-check-not=main_core_2_2
// RUN: cd %t.d/unified && aiecc -v --unified --get-npu-insts --npu-insts-name=insts.bin %s 2>&1 | FileCheck %s --check-prefix=SEQ --implicit-check-not=main_core_1_2 --implicit-check-not=main_core_2_2
// RUN: cd %t.d/full && aiecc -v --get-npu-insts --get-core-elfs --npu-insts-name=insts.bin %s 2>&1 | FileCheck %s --check-prefix=FULL
// RUN: cd %t.d/unmeasured && aiecc --no-measure-data-size --get-npu-insts --get-core-elfs --npu-insts-name=insts.bin %s

// RUN: cmp %t.d/seq/insts.bin %t.d/full/insts.bin
// RUN: cmp %t.d/unified/insts.bin %t.d/full/insts.bin
// RUN: not cmp -s %t.d/unmeasured/insts.bin %t.d/full/insts.bin

// SEQ: probeElfs_main_core_0_2

// FULL-DAG: probeElfs_main_core_0_2
// FULL-DAG: probeElfs_main_core_1_2
// FULL-DAG: probeElfs_main_core_2_2

module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)
    %rtp = aie.buffer(%t02) {sym_name = "rtp"} : memref<4xi32>
    %out = aie.buffer(%t02) {sym_name = "out"} : memref<64xi32>
    %scratch = aie.buffer(%t12) {sym_name = "scratch"} : memref<64xi32>
    memref.global "private" constant @table : memref<64xi32> = dense<[
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
      33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
      49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]>
    %core_0_2 = aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %table = memref.get_global @table : memref<64xi32>
      %k = memref.load %rtp[%c0] : memref<4xi32>
      scf.for %i = %c0 to %c64 step %c1 {
        %v = memref.load %table[%i] : memref<64xi32>
        %s = arith.addi %v, %k : i32
        memref.store %s, %out[%i] : memref<64xi32>
      }
      aie.end
    }
    %core_1_2 = aie.core(%t12) {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : i32
      memref.store %c7, %scratch[%c0] : memref<64xi32>
      aie.end
    }
    %core_2_2 = aie.core(%t22) {
      aie.end
    }
    aie.runtime_sequence(%a : memref<64xi32>) {
      %c42 = arith.constant 42 : i32
      aiex.npu.rtp_write(@rtp, 0, %c42) : i32
    }
  }
}
