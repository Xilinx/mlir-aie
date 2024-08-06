// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" -aieml=true --aie-vectorize="shift=10 zero-offset=4" | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: xchesscc -f -g +s -p me -P %aietools/data/aie_ml/lib/ +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc -f -g +s -p me -P %aietools/data/aie_ml/lib/ +w work +o work -I%S -I. %S/testbench.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out"

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<?x288xi16>, %arg1: memref<?xi16>, %arg2: memref<?x256xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 256 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<?x288xi16>
        %1 = arith.extsi %0 : i16 to i32
        %2 = affine.load %arg1[0] : memref<?xi16>
        %3 = arith.extsi %2 : i16 to i32
        %4 = arith.muli %1, %3 : i32
        %5 = affine.load %arg0[%arg3, %arg4 + 1] : memref<?x288xi16>
        %6 = arith.extsi %5 : i16 to i32
        %7 = affine.load %arg1[1] : memref<?xi16>
        %8 = arith.extsi %7 : i16 to i32
        %9 = arith.muli %6, %8 : i32
        %10 = arith.addi %4, %9 : i32
        %11 = affine.load %arg0[%arg3, %arg4 + 2] : memref<?x288xi16>
        %12 = arith.extsi %11 : i16 to i32
        %13 = affine.load %arg1[2] : memref<?xi16>
        %14 = arith.extsi %13 : i16 to i32
        %15 = arith.muli %12, %14 : i32
        %16 = arith.addi %10, %15 : i32
        %17 = arith.trunci %16 : i32 to i16
        affine.store %17, %arg2[%arg3, %arg4] : memref<?x256xi16>
      }
    }
    return
  }
}
