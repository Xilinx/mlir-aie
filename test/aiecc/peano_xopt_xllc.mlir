//===- peano_xopt_xllc.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// -Xopt and -Xllc pass one argument each, unsplit at commas, to the opt and
// llc that compile a core, after aiecc's own arguments. With --xchesscc they
// are an error.

// REQUIRES: peano

// RUN: aiecc -v --output-dir=%t --tmpdir=%t.prj --get-core-elfs -Xopt -opt-disable=licm,loop-unroll-full -Xopt=-opt-disable-enable-verbosity -Xllc=--enable-aie-zol-without-minitercount=false %s 2>&1 | FileCheck %s
// CHECK: exec:{{.*}}opt --passes=default<O1> -inline-threshold=10 -S {{.*}} -opt-disable=licm,loop-unroll-full -opt-disable-enable-verbosity{{$}}
// CHECK-DAG: OptDisable: NOT running pass licm
// CHECK-DAG: OptDisable: NOT running pass loop-unroll-full
// CHECK: exec:{{.*}}llc {{.*}} --enable-aie-zol-without-minitercount=false{{$}}

// The arguments reach llc, which rejects one it does not know.
// RUN: not aiecc --output-dir=%t.bad --tmpdir=%t.bad.prj --get-core-elfs -Xllc=--no-such-llc-option %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// UNKNOWN: Unknown command line argument '--no-such-llc-option'

// RUN: not aiecc --xchesscc -Xopt=-opt-disable=licm %s 2>&1 | FileCheck %s --check-prefix=CHESS
// RUN: not aiecc --xchesscc -Xllc=-O1 %s 2>&1 | FileCheck %s --check-prefix=CHESS
// CHESS: aiecc: -Xopt and -Xllc require Peano compilation

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<16xi32>
    %n = aie.buffer(%tile_0_2) {sym_name = "n"} : memref<1xindex>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %bound = memref.load %n[%c0] : memref<1xindex>
      scf.for %i = %c0 to %bound step %c1 {
        %v = memref.load %buf[%c0] : memref<16xi32>
        %w = arith.addi %v, %v : i32
        memref.store %w, %buf[%i] : memref<16xi32>
      }
      aie.end
    }
  }
}
