//===- packet_flow_dest_before_source.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Regression test for issue #2583:
// AIEPathfinderPass assumed packet_source always precedes packet_dest in a
// block. MLIR provides no such ordering guarantee. This test verifies that
// placing aie.packet_dest before aie.packet_source produces correct output.

// Test 1: Single dest before source (basic reproducer from issue #2583)
// CHECK-LABEL: aie.device(xcvc1902)
// CHECK:         %[[T11:.*]] = aie.tile(1, 1)
// CHECK:         %[[SW11:.*]] = aie.switchbox(%[[T11]]) {
// CHECK:           %[[AMSEL0:.*]] = aie.amsel<0> (0)
// CHECK:           aie.masterset(Core : 0, %[[AMSEL0]])
// CHECK:           aie.packet_rules(West : 0) {
// CHECK:             aie.rule(31, 0, %[[AMSEL0]])
// CHECK:           }
// CHECK:         }

// Test 2: Multiple dests before source (fanout with dest-first ordering)
// CHECK:         %[[T12:.*]] = aie.tile(1, 2)
// CHECK:         %[[SW12:.*]] = aie.switchbox(%[[T12]]) {
// CHECK:           aie.masterset(Core : 0,
// CHECK:           aie.masterset(Core : 1,
// CHECK:           aie.packet_rules(West : 0) {

// Test 3: keep_pkt_header with dest-first ordering
// CHECK:         %[[T13:.*]] = aie.tile(1, 3)
// CHECK:         %[[SW13:.*]] = aie.switchbox(%[[T13]]) {
// CHECK:           aie.masterset(Core : 0,
// CHECK:           aie.packet_rules(West : 0) {

module @packet_flow_dest_before_source {
 aie.device(xcvc1902) {

  // Test 1: single dest before source
  %t11 = aie.tile(1, 1)
  aie.packet_flow(0x0) {
    aie.packet_dest<%t11, Core : 0>
    aie.packet_source<%t11, West : 0>
  }

  // Test 2: multiple dests before source
  %t12 = aie.tile(1, 2)
  aie.packet_flow(0x0) {
    aie.packet_dest<%t12, Core : 0>
    aie.packet_dest<%t12, Core : 1>
    aie.packet_source<%t12, West : 0>
  }

  // Test 3: keep_pkt_header with dest-first ordering
  %t13 = aie.tile(1, 3)
  aie.packet_flow(0x0) {
    aie.packet_dest<%t13, Core : 0>
    aie.packet_source<%t13, West : 0>
  } {keep_pkt_header = true}

 }
}
