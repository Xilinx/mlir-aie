//===- pathfinder_input.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-canonicalize-device %s | aie-opt --aie-create-pathfinder-flows | FileCheck %s
// CHECK: %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK: }
// CHECK: %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL_3:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK: }
// CHECK: %[[VAL_4:.*]] = aie.tile(0, 3)
// CHECK: %[[VAL_5:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK: }
// CHECK: %[[VAL_6:.*]] = aie.tile(1, 1)
// CHECK: %[[VAL_7:.*]] = aie.tile(1, 2)
// CHECK: %[[VAL_8:.*]] = aie.tile(1, 3)
// CHECK: %[[VAL_9:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK: }
// CHECK: %[[VAL_10:.*]] = aie.tile(2, 1)
// CHECK: %[[VAL_11:.*]] = aie.tile(2, 2)
// CHECK: %[[VAL_12:.*]] = aie.tile(2, 3)
// CHECK: %[[VAL_13:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK: }
// CHECK: %[[VAL_14:.*]] = aie.tile(3, 1)
// CHECK: %[[VAL_15:.*]] = aie.tile(3, 2)
// CHECK: %[[VAL_16:.*]] = aie.tile(3, 3)
// CHECK: %[[VAL_17:.*]] = aie.tile(4, 1)
// CHECK: %[[VAL_18:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK: }
// CHECK: %[[VAL_19:.*]] = aie.tile(4, 2)
// CHECK: %[[VAL_20:.*]] = aie.tile(4, 3)
// CHECK: %[[VAL_21:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:   aie.connect<DMA : 0, North : 1>
// CHECK:   aie.connect<East : 2, DMA : 0>
// CHECK: }
// CHECK: %[[VAL_22:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:   aie.connect<South : 1, East : 1>
// CHECK: }
// CHECK: %[[VAL_23:.*]] = aie.switchbox(%[[VAL_11]]) {
// CHECK:   aie.connect<West : 1, East : 1>
// CHECK:   aie.connect<East : 2, South : 2>
// CHECK: }
// CHECK: %[[VAL_24:.*]] = aie.switchbox(%[[VAL_15]]) {
// CHECK:   aie.connect<West : 1, East : 1>
// CHECK:   aie.connect<East : 1, West : 2>
// CHECK:   aie.connect<South : 1, East : 3>
// CHECK:   aie.connect<North : 2, South : 2>
// CHECK: }
// CHECK: %[[VAL_25:.*]] = aie.switchbox(%[[VAL_19]]) {
// CHECK:   aie.connect<West : 1, DMA : 0>
// CHECK:   aie.connect<DMA : 0, West : 1>
// CHECK:   aie.connect<West : 3, North : 0>
// CHECK: }
// CHECK: %[[VAL_26:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:   aie.connect<North : 2, West : 2>
// CHECK: }
// CHECK: %[[VAL_27:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:   aie.connect<DMA : 0, North : 1>
// CHECK:   aie.connect<North : 2, DMA : 0>
// CHECK: }
// CHECK: %[[VAL_28:.*]] = aie.switchbox(%[[VAL_20]]) {
// CHECK:   aie.connect<South : 0, DMA : 0>
// CHECK:   aie.connect<DMA : 0, West : 0>
// CHECK: }
// CHECK: %[[VAL_29:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:   aie.connect<East : 0, South : 2>
// CHECK: }


module @pathfinder{
%t01 = aie.tile(0, 1)
%t02 = aie.tile(0, 2)
%t03 = aie.tile(0, 3)
%t11 = aie.tile(1, 1)
%t12 = aie.tile(1, 2)
%t13 = aie.tile(1, 3)
%t21 = aie.tile(2, 1)
%t22 = aie.tile(2, 2)
%t23 = aie.tile(2, 3)
%t31 = aie.tile(3, 1)
%t32 = aie.tile(3, 2)
%t33 = aie.tile(3, 3)
%t41 = aie.tile(4, 1)
%t42 = aie.tile(4, 2)
%t43 = aie.tile(4, 3)

aie.flow(%t11, DMA : 0, %t42, DMA : 0)
aie.flow(%t42, DMA : 0, %t11, DMA : 0)
aie.flow(%t31, DMA : 0, %t43, DMA : 0)
aie.flow(%t43, DMA : 0, %t31, DMA : 0)

//aie.flow(%t03, DMA : 0, %t41, DMA : 0)
}

