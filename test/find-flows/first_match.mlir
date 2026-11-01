//===- first_match.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A slave port hands a packet to the first rule that matches its id, so where
// two rules overlap, rule order decides where an id goes.

// RUN: aie-opt --aie-find-flows --split-input-file %s | FileCheck %s

// Id 5 matches both rules and takes the first, to DMA : 1. The other ids of
// 0 through 7 go to DMA : 0.
// CHECK-LABEL: module
// CHECK-NOT: aie.switchbox
// CHECK: aie.packet_flow(0, mask = 25) {
// CHECK-NEXT: aie.packet_source<%{{.*}}tile_0_2, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}tile_0_3, DMA : 0>
// CHECK: aie.packet_flow(3, mask = 27) {
// CHECK-NEXT: aie.packet_source<%{{.*}}tile_0_2, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}tile_0_3, DMA : 0>
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT: aie.packet_source<%{{.*}}tile_0_2, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}tile_0_3, DMA : 0>
// CHECK: aie.packet_flow(5) {
// CHECK-NEXT: aie.packet_source<%{{.*}}tile_0_2, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}tile_0_3, DMA : 1>
// CHECK-NOT: aie.packet_flow

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a = aie.amsel<0> (0)
      %b = aie.amsel<1> (0)
      %m0 = aie.masterset(North : 0, %a)
      %m1 = aie.masterset(North : 1, %b)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 5, %b)
        aie.rule(24, 0, %a)
      }
    }
    %sb03 = aie.switchbox(%t03) {
      %a = aie.amsel<0> (0)
      %b = aie.amsel<1> (0)
      %m0 = aie.masterset(DMA : 0, %a)
      %m1 = aie.masterset(DMA : 1, %b)
      aie.packet_rules(South : 0) {
        aie.rule(0, 0, %a)
      }
      aie.packet_rules(South : 1) {
        aie.rule(0, 0, %b)
      }
    }
    aie.wire(%t02 : DMA, %sb02 : DMA)
    aie.wire(%sb02 : North, %sb03 : South)
    aie.wire(%t03 : DMA, %sb03 : DMA)
  }
}

// -----

// With the rules swapped, the wide rule takes id 5 too and the narrow one
// matches nothing. No flow reaches DMA : 1, and the route to it stays behind
// as switchboxes.
// CHECK-LABEL: module
// CHECK: aie.switchbox
// CHECK:   aie.rule(31, 5
// CHECK: aie.switchbox
// CHECK: aie.packet_flow(0, mask = 24) {
// CHECK-NEXT: aie.packet_source<%{{.*}}tile_0_2, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}tile_0_3, DMA : 0>
// CHECK-NOT: aie.packet_flow

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a = aie.amsel<0> (0)
      %b = aie.amsel<1> (0)
      %m0 = aie.masterset(North : 0, %a)
      %m1 = aie.masterset(North : 1, %b)
      aie.packet_rules(DMA : 0) {
        aie.rule(24, 0, %a)
        aie.rule(31, 5, %b)
      }
    }
    %sb03 = aie.switchbox(%t03) {
      %a = aie.amsel<0> (0)
      %b = aie.amsel<1> (0)
      %m0 = aie.masterset(DMA : 0, %a)
      %m1 = aie.masterset(DMA : 1, %b)
      aie.packet_rules(South : 0) {
        aie.rule(0, 0, %a)
      }
      aie.packet_rules(South : 1) {
        aie.rule(0, 0, %b)
      }
    }
    aie.wire(%t02 : DMA, %sb02 : DMA)
    aie.wire(%sb02 : North, %sb03 : South)
    aie.wire(%t03 : DMA, %sb03 : DMA)
  }
}
