// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=10 zero-offset=4" -aieml=true -canonicalize -split-input-file | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<?x288xi16>, %arg1: memref<?xi16>, %arg2: memref<?x256xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
        affine.store %16, %arg2[%arg3, %arg4] : memref<?x256xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: @conv2d
// CHECK-SAME: %[[A0:[0-9a-zA-Z]*]]: memref<?x288xi16>
// CHECK-SAME: %[[A1:[0-9a-zA-Z]*]]: memref<?xi16>
// CHECK-SAME: %[[A2:[0-9a-zA-Z]*]]: memref<?x256xi32>
//      CHECK:    %[[C256:.*]] = arith.constant 256 : index
//      CHECK:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK:    %[[C16:.*]] = arith.constant 16 : index
//      CHECK:    %[[C10:.*]] = arith.constant 10 : i32
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[T0:.*]] = aievec.upd %arg1[%[[C0:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
//      CHECK:    %[[T1:.*]] = aievec.concat %[[T0:.*]], %[[T0:.*]] : vector<16xi16>, vector<32xi16>
//      CHECK:    scf.for %[[A3:.*]] = %[[C0:.*]] to %[[C16:.*]] step %[[C1:.*]] {
//      CHECK:      scf.for %[[A4:.*]] = %[[C0:.*]] to %[[C256:.*]] step %[[C16:.*]] {
//      CHECK:        %[[T2:.*]] = aievec.upd %[[A0]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x288xi16>, vector<32xi16>
//      CHECK:        %[[T3:.*]] = aievec.mul_conv %[[T2:.*]], %[[T1:.*]] {M = 16 : i32, N = 4 : i32} : vector<32xi16>, vector<32xi16>, vector<16xi64>
//      CHECK:        %[[T4:.*]] = aievec.srs %[[T3:.*]], %[[C10]] : vector<16xi64>, i32, vector<16xi32>
//      CHECK:        vector.transfer_write %[[T4:.*]], %[[A2]][%[[A3:.*]], %[[A4:.*]]] : vector<16xi32>, memref<?x256xi32>
