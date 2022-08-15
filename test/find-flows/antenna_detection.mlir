//===- find_flows.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-find-flows %s 2>&1 | FileCheck %s

///////////////////////Antenna1////////////////////////////////
// CHECK: antenna_detection.mlir:{{.*}}: warning: Antenna
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} AIE.connect<South:1, North:2>
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} %sb0 = AIE.switchbox(%tile23) {
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} AIE.connect<Core :1, North:1>
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}}  %sb1 = AIE.switchbox(%tile22) {
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} %tile22 = AIE.tile(2, 2)
///////////////////////Antenna2////////////////////////////////
// CHECK: antenna_detection.mlir:{{.*}}: warning: Antenna
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} %21 = AIE.masterset(East : 1, %20)
// CHECK: antenna_detection.mlir:{{.*}}: warning: Antenna
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} AIE.rule(31, 0, %20)
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} AIE.packetrules(DMA : 0) {
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} %sb1 = AIE.switchbox(%tile22) {
// CHECK: antenna_detection.mlir:{{.*}}: remark: Path Traceback
// CHECK-NEXT: {{[[:space:]].*[[:space:]].*}} %tile22 = AIE.tile(2, 2)
///////////////////////danglingAntenna1////////////////////////////////
// CHECK: antenna_detection.mlir:{{.*}}: warning: Dangling Island Antenna
// CHECK-NEXT: {{[[:space:]].*}} AIE.connect<South:3, Core :0>
///////////////////////danglingAntenna2////////////////////////////////
// CHECK: antenna_detection.mlir:{{.*}}: warning: Dangling Island Antenna
// CHECK-NEXT: {{[[:space:]].*}} AIE.connect<East :1, North :3>
///////////////////////danglingAntenna3////////////////////////////////
// CHECK: antenna_detection.mlir:{{.*}}: warning: Dangling Island Antenna
// CHECK-NEXT: {{[[:space:]].*}} %23 = AIE.masterset(East : 2, %22)
///////////////////////danglingAntenna4////////////////////////////////
// CHECK: antenna_detection.mlir:{{.*}}: warning: Dangling Island Antenna
// CHECK-NEXT: {{[[:space:]].*}} AIE.packetrules(West : 0) {

///////////////////////Flows////////////////////////////////
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T22]], Core : 0)
// CHECK: AIE.flow(%[[T22]], Core : 1, %[[T23]], Core : 1)
// CHECK: AIE.packet_flow(0) {
// CHECK:   AIE.packet_source<%[[T22]], DMA : 0>
// CHECK:   AIE.packet_dest<%[[T23]], DMA : 1>
// CHECK: }
// CHECK: AIE.flow(%2, Core : 0, %9, DMA : 0)

module {
  %tile23 = AIE.tile(2, 3)
  %tile22 = AIE.tile(2, 2)
  %tile21 = AIE.tile(2, 1)
  %tile20 = AIE.tile(2, 0)

  %sb0 = AIE.switchbox(%tile23) {
    AIE.connect<Core :0, South:1>
    AIE.connect<South:1, Core :1>
    AIE.connect<South:1, North:2> //endpoint antenna
    AIE.connect<South:3, Core :0> //dangling island antenna 
    %16 = AIE.amsel<0> (0)
    %17 = AIE.masterset(DMA : 1, %16)
    AIE.packetrules(South : 0) {
      AIE.rule(31, 0, %16)
    }
  }
  %sb1 = AIE.switchbox(%tile22) {
    AIE.connect<North:1, Core :1>
    AIE.connect<Core :1, North:1>
    AIE.connect<Core :0, Core :0>
    AIE.connect<East :1, North :3> //dangling island antenna 
    %18 = AIE.amsel<0> (0)
    %19 = AIE.masterset(North : 0, %18)
    %20 = AIE.amsel<0> (1) // packet antennas
    %21 = AIE.masterset(East : 1, %20) // packet antennas    
    AIE.packetrules(DMA : 0) {
      AIE.rule(31, 0, %18)
      AIE.rule(31, 0, %20)
    }
    // dangling packet antennas
    %22 = AIE.amsel<1> (2)
    %23 = AIE.masterset(East : 2, %22)
    AIE.packetrules(West : 0) {
      AIE.rule(31, 0, %22)
    }
  }

  %sb21 = AIE.switchbox(%tile21)  {
    AIE.connect<Core : 0, South : 0>
  }
  %sb20 = AIE.switchbox(%tile20)  {
    AIE.connect<North : 0, South : 2>
  }
  %mux20 = AIE.shimmux(%tile20)  {
    AIE.connect<North : 2, DMA : 0>
    // AIE.connect<North : 2, East : 1> //endpoint antenna
    // AIE.connect<West : 3, East : 3> //dangling island antenna 
  }
  %dma20 = AIE.shimDMA(%tile20)  {
    AIE.end
  }

  AIE.wire(%sb0: Core, %tile23: Core)
  AIE.wire(%sb1: Core, %tile22: Core)
  AIE.wire(%sb0: DMA, %tile23: DMA)
  AIE.wire(%sb1: DMA, %tile22: DMA)
  AIE.wire(%sb0: South, %sb1: North)
  
  AIE.wire(%sb21 : Core, %tile21 : Core)
  AIE.wire(%sb21 : South, %sb20 : North)
  AIE.wire(%sb20 : South, %mux20 : North)
  AIE.wire(%mux20 : DMA, %dma20 : DMA)
}
  