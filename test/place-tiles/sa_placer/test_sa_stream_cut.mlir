//===- test_sa_stream_cut.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Columns 3 and 4 send 26 circuit streams to 13 consumers, one from each
// column per consumer. West of them are 12 free cores, east only 3, and the
// boundary between columns 2 and 3 carries 20 channels west, so at most 10
// consumers may sit west. Both placers count the streams each row and column
// boundary must carry, and the design routes.

// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages{placer=sa_placer sa-seed=1})' %s -o /dev/null
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages{placer=sequential_placer})' %s -o /dev/null
// RUN: aie-opt --split-input-file --aie-place-tiles=placer=sequential_placer %s | FileCheck %s

// The same design with its ops reversed and renamed places alike.
// CHECK-LABEL: module @stream_cut
// CHECK-DAG: aie.tile(5, 5)
// CHECK-DAG: aie.tile(6, 5)
// CHECK-DAG: aie.tile(7, 5)
// CHECK-DAG: aie.tile(2, 3)
// CHECK-LABEL: module @stream_cut_permuted
// CHECK-DAG: aie.tile(5, 5)
// CHECK-DAG: aie.tile(6, 5)
// CHECK-DAG: aie.tile(7, 5)
// CHECK-DAG: aie.tile(2, 3)

module @stream_cut {
  aie.device(npu2) {
    %m3 = aie.tile(3, 1)
    %p3_2 = aie.tile(3, 2)
    %p3_3 = aie.tile(3, 3)
    %p3_4 = aie.tile(3, 4)
    %p3_5 = aie.tile(3, 5)
    %m4 = aie.tile(4, 1)
    %p4_2 = aie.tile(4, 2)
    %p4_3 = aie.tile(4, 3)
    %p4_4 = aie.tile(4, 4)
    %p4_5 = aie.tile(4, 5)
    %b5_2 = aie.tile(5, 2)
    aie.core(%b5_2) { aie.end }
    %b5_3 = aie.tile(5, 3)
    aie.core(%b5_3) { aie.end }
    %b5_4 = aie.tile(5, 4)
    aie.core(%b5_4) { aie.end }
    %b6_2 = aie.tile(6, 2)
    aie.core(%b6_2) { aie.end }
    %b6_3 = aie.tile(6, 3)
    aie.core(%b6_3) { aie.end }
    %b6_4 = aie.tile(6, 4)
    aie.core(%b6_4) { aie.end }
    %b7_2 = aie.tile(7, 2)
    aie.core(%b7_2) { aie.end }
    %b7_3 = aie.tile(7, 3)
    aie.core(%b7_3) { aie.end }
    %b7_4 = aie.tile(7, 4)
    aie.core(%b7_4) { aie.end }
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    %c5 = aie.logical_tile<CoreTile>(?, ?)
    %c6 = aie.logical_tile<CoreTile>(?, ?)
    %c7 = aie.logical_tile<CoreTile>(?, ?)
    %c8 = aie.logical_tile<CoreTile>(?, ?)
    %c9 = aie.logical_tile<CoreTile>(?, ?)
    %c10 = aie.logical_tile<CoreTile>(?, ?)
    %c11 = aie.logical_tile<CoreTile>(?, ?)
    %c12 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in3(%p3_2, {%m3}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo @f3_0(%p3_2, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_1(%p3_3, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_2(%p3_3, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_3(%p3_4, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_4(%p3_4, {%c4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_5(%p3_5, {%c5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_6(%p3_5, {%c6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_7(%m3, {%c7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_8(%m3, {%c8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_9(%m3, {%c9}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_10(%m3, {%c10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_11(%m3, {%c11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3_12(%m3, {%c12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in3] -> [@f3_7, @f3_8, @f3_9, @f3_10, @f3_11, @f3_12] ([] [0, 16, 32, 48, 64, 80])
    aie.objectfifo @in4(%p4_2, {%m4}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo @f4_0(%p4_2, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_1(%p4_3, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_2(%p4_3, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_3(%p4_4, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_4(%p4_4, {%c4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_5(%p4_5, {%c5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_6(%p4_5, {%c6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_7(%m4, {%c7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_8(%m4, {%c8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_9(%m4, {%c9}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_10(%m4, {%c10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_11(%m4, {%c11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4_12(%m4, {%c12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in4] -> [@f4_7, @f4_8, @f4_9, @f4_10, @f4_11, @f4_12] ([] [0, 16, 32, 48, 64, 80])
    aie.core(%c0) {
      %a3 = aie.objectfifo.acquire @f3_0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_0(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_0(Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %a3 = aie.objectfifo.acquire @f3_1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_1(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_1(Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %a3 = aie.objectfifo.acquire @f3_2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_2(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_2(Consume, 1)
      aie.end
    }
    aie.core(%c3) {
      %a3 = aie.objectfifo.acquire @f3_3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_3(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_3(Consume, 1)
      aie.end
    }
    aie.core(%c4) {
      %a3 = aie.objectfifo.acquire @f3_4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_4(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_4(Consume, 1)
      aie.end
    }
    aie.core(%c5) {
      %a3 = aie.objectfifo.acquire @f3_5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_5(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_5(Consume, 1)
      aie.end
    }
    aie.core(%c6) {
      %a3 = aie.objectfifo.acquire @f3_6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_6(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_6(Consume, 1)
      aie.end
    }
    aie.core(%c7) {
      %a3 = aie.objectfifo.acquire @f3_7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_7(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_7(Consume, 1)
      aie.end
    }
    aie.core(%c8) {
      %a3 = aie.objectfifo.acquire @f3_8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_8(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_8(Consume, 1)
      aie.end
    }
    aie.core(%c9) {
      %a3 = aie.objectfifo.acquire @f3_9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_9(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_9(Consume, 1)
      aie.end
    }
    aie.core(%c10) {
      %a3 = aie.objectfifo.acquire @f3_10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_10(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_10(Consume, 1)
      aie.end
    }
    aie.core(%c11) {
      %a3 = aie.objectfifo.acquire @f3_11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_11(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_11(Consume, 1)
      aie.end
    }
    aie.core(%c12) {
      %a3 = aie.objectfifo.acquire @f3_12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3_12(Consume, 1)
      %a4 = aie.objectfifo.acquire @f4_12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4_12(Consume, 1)
      aie.end
    }
    aie.core(%p3_2) {
      %o0 = aie.objectfifo.acquire @in3(Produce, 1) : memref<96xi32>
      aie.objectfifo.release @in3(Produce, 1)
      %o1 = aie.objectfifo.acquire @f3_0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_0(Produce, 1)
      aie.end
    }
    aie.core(%p3_3) {
      %o0 = aie.objectfifo.acquire @f3_1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_1(Produce, 1)
      %o1 = aie.objectfifo.acquire @f3_2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_2(Produce, 1)
      aie.end
    }
    aie.core(%p3_4) {
      %o0 = aie.objectfifo.acquire @f3_3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_3(Produce, 1)
      %o1 = aie.objectfifo.acquire @f3_4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_4(Produce, 1)
      aie.end
    }
    aie.core(%p3_5) {
      %o0 = aie.objectfifo.acquire @f3_5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_5(Produce, 1)
      %o1 = aie.objectfifo.acquire @f3_6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3_6(Produce, 1)
      aie.end
    }
    aie.core(%p4_2) {
      %o0 = aie.objectfifo.acquire @in4(Produce, 1) : memref<96xi32>
      aie.objectfifo.release @in4(Produce, 1)
      %o1 = aie.objectfifo.acquire @f4_0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_0(Produce, 1)
      aie.end
    }
    aie.core(%p4_3) {
      %o0 = aie.objectfifo.acquire @f4_1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_1(Produce, 1)
      %o1 = aie.objectfifo.acquire @f4_2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_2(Produce, 1)
      aie.end
    }
    aie.core(%p4_4) {
      %o0 = aie.objectfifo.acquire @f4_3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_3(Produce, 1)
      %o1 = aie.objectfifo.acquire @f4_4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_4(Produce, 1)
      aie.end
    }
    aie.core(%p4_5) {
      %o0 = aie.objectfifo.acquire @f4_5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_5(Produce, 1)
      %o1 = aie.objectfifo.acquire @f4_6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4_6(Produce, 1)
      aie.end
    }
  }
}
// -----

module @stream_cut_permuted {
  aie.device(npu2) {
    %consumer12 = aie.logical_tile<CoreTile>(?, ?)
    %consumer11 = aie.logical_tile<CoreTile>(?, ?)
    %consumer10 = aie.logical_tile<CoreTile>(?, ?)
    %consumer9 = aie.logical_tile<CoreTile>(?, ?)
    %consumer8 = aie.logical_tile<CoreTile>(?, ?)
    %consumer7 = aie.logical_tile<CoreTile>(?, ?)
    %consumer6 = aie.logical_tile<CoreTile>(?, ?)
    %consumer5 = aie.logical_tile<CoreTile>(?, ?)
    %consumer4 = aie.logical_tile<CoreTile>(?, ?)
    %consumer3 = aie.logical_tile<CoreTile>(?, ?)
    %consumer2 = aie.logical_tile<CoreTile>(?, ?)
    %consumer1 = aie.logical_tile<CoreTile>(?, ?)
    %consumer0 = aie.logical_tile<CoreTile>(?, ?)
    %b7_4 = aie.tile(7, 4)
    %b7_3 = aie.tile(7, 3)
    %b7_2 = aie.tile(7, 2)
    %b6_4 = aie.tile(6, 4)
    %b6_3 = aie.tile(6, 3)
    %b6_2 = aie.tile(6, 2)
    %b5_4 = aie.tile(5, 4)
    %b5_3 = aie.tile(5, 3)
    %b5_2 = aie.tile(5, 2)
    %p4_5 = aie.tile(4, 5)
    %p4_4 = aie.tile(4, 4)
    %p4_3 = aie.tile(4, 3)
    %p4_2 = aie.tile(4, 2)
    %m4 = aie.tile(4, 1)
    %p3_5 = aie.tile(3, 5)
    %p3_4 = aie.tile(3, 4)
    %p3_3 = aie.tile(3, 3)
    %p3_2 = aie.tile(3, 2)
    %m3 = aie.tile(3, 1)
    aie.objectfifo @stream_12_from4(%m4, {%consumer12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_11_from4(%m4, {%consumer11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_10_from4(%m4, {%consumer10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_9_from4(%m4, {%consumer9}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_8_from4(%m4, {%consumer8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_7_from4(%m4, {%consumer7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_6_from4(%p4_5, {%consumer6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_5_from4(%p4_5, {%consumer5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_4_from4(%p4_4, {%consumer4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_3_from4(%p4_4, {%consumer3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_2_from4(%p4_3, {%consumer2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_1_from4(%p4_3, {%consumer1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_0_from4(%p4_2, {%consumer0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @feed4(%p4_2, {%m4}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo @stream_12_from3(%m3, {%consumer12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_11_from3(%m3, {%consumer11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_10_from3(%m3, {%consumer10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_9_from3(%m3, {%consumer9}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_8_from3(%m3, {%consumer8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_7_from3(%m3, {%consumer7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_6_from3(%p3_5, {%consumer6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_5_from3(%p3_5, {%consumer5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_4_from3(%p3_4, {%consumer4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_3_from3(%p3_4, {%consumer3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_2_from3(%p3_3, {%consumer2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_1_from3(%p3_3, {%consumer1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_0_from3(%p3_2, {%consumer0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @feed3(%p3_2, {%m3}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@feed3] -> [@stream_7_from3, @stream_8_from3, @stream_9_from3, @stream_10_from3, @stream_11_from3, @stream_12_from3] ([] [0, 16, 32, 48, 64, 80])
    aie.objectfifo.link [@feed4] -> [@stream_7_from4, @stream_8_from4, @stream_9_from4, @stream_10_from4, @stream_11_from4, @stream_12_from4] ([] [0, 16, 32, 48, 64, 80])
    aie.core(%p4_5) {
      %o0 = aie.objectfifo.acquire @stream_5_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_5_from4(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_6_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_6_from4(Produce, 1)
      aie.end
    }
    aie.core(%p4_4) {
      %o0 = aie.objectfifo.acquire @stream_3_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_3_from4(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_4_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_4_from4(Produce, 1)
      aie.end
    }
    aie.core(%p4_3) {
      %o0 = aie.objectfifo.acquire @stream_1_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_1_from4(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_2_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_2_from4(Produce, 1)
      aie.end
    }
    aie.core(%p4_2) {
      %o0 = aie.objectfifo.acquire @feed4(Produce, 1) : memref<96xi32>
      aie.objectfifo.release @feed4(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_0_from4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_0_from4(Produce, 1)
      aie.end
    }
    aie.core(%p3_5) {
      %o0 = aie.objectfifo.acquire @stream_5_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_5_from3(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_6_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_6_from3(Produce, 1)
      aie.end
    }
    aie.core(%p3_4) {
      %o0 = aie.objectfifo.acquire @stream_3_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_3_from3(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_4_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_4_from3(Produce, 1)
      aie.end
    }
    aie.core(%p3_3) {
      %o0 = aie.objectfifo.acquire @stream_1_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_1_from3(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_2_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_2_from3(Produce, 1)
      aie.end
    }
    aie.core(%p3_2) {
      %o0 = aie.objectfifo.acquire @feed3(Produce, 1) : memref<96xi32>
      aie.objectfifo.release @feed3(Produce, 1)
      %o1 = aie.objectfifo.acquire @stream_0_from3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stream_0_from3(Produce, 1)
      aie.end
    }
    aie.core(%consumer12) {
      %a3 = aie.objectfifo.acquire @stream_12_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_12_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_12_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_12_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer11) {
      %a3 = aie.objectfifo.acquire @stream_11_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_11_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_11_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_11_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer10) {
      %a3 = aie.objectfifo.acquire @stream_10_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_10_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_10_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_10_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer9) {
      %a3 = aie.objectfifo.acquire @stream_9_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_9_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_9_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_9_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer8) {
      %a3 = aie.objectfifo.acquire @stream_8_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_8_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_8_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_8_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer7) {
      %a3 = aie.objectfifo.acquire @stream_7_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_7_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_7_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_7_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer6) {
      %a3 = aie.objectfifo.acquire @stream_6_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_6_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_6_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_6_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer5) {
      %a3 = aie.objectfifo.acquire @stream_5_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_5_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_5_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_5_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer4) {
      %a3 = aie.objectfifo.acquire @stream_4_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_4_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_4_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_4_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer3) {
      %a3 = aie.objectfifo.acquire @stream_3_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_3_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_3_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_3_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer2) {
      %a3 = aie.objectfifo.acquire @stream_2_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_2_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_2_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_2_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer1) {
      %a3 = aie.objectfifo.acquire @stream_1_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_1_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_1_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_1_from4(Consume, 1)
      aie.end
    }
    aie.core(%consumer0) {
      %a3 = aie.objectfifo.acquire @stream_0_from3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_0_from3(Consume, 1)
      %a4 = aie.objectfifo.acquire @stream_0_from4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @stream_0_from4(Consume, 1)
      aie.end
    }
    aie.core(%b7_4) { aie.end }
    aie.core(%b7_3) { aie.end }
    aie.core(%b7_2) { aie.end }
    aie.core(%b6_4) { aie.end }
    aie.core(%b6_3) { aie.end }
    aie.core(%b6_2) { aie.end }
    aie.core(%b5_4) { aie.end }
    aie.core(%b5_3) { aie.end }
    aie.core(%b5_2) { aie.end }
  }
}
