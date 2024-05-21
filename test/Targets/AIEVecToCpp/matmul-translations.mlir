//===- matmul-translations.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate %s -aieml -aievec-to-cpp | FileCheck %s

// CHECK-LABEL: v16float matmul_nopad(
// CHECK-SAME:             v32bfloat16 [[A:[a-zA-Z0-9]+]],
// CHECK-SAME:             v32bfloat16 [[B:[a-zA-Z0-9]+]],
// CHECK-SAME:             v16float [[C:[a-zA-Z0-9]+]]) {
// CHECK:           v16accfloat [[CACC:.*]] = v16accfloat([[C]]);
// CHECK:           [[CACC]] = mac_4x8_8x4([[A]], [[B]], [[CACC]]);
// CHECK:           v16float [[R:.*]] = v16float([[CACC]]);
// CHECK:           return [[R]];
// CHECK:        }
func.func @matmul_nopad(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                        %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %acc = aievec.cast %C {isResAcc = true} : vector<4x4xf32>, vector<4x4xf32>
  %r = aievec.matmul %A, %B, %acc : vector<4x8xbf16>, vector<8x4xbf16>
                                    into vector<4x4xf32>
  %0 = aievec.cast %r {isResAcc = false} : vector<4x4xf32>, vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: v32int32 matmul_pad1(
// CHECK-SAME:                 v16int16 [[A:[a-zA-Z0-9]+]],
// CHECK-SAME:                 v32int8 [[B:[a-zA-Z0-9]+]],
// CHECK-SAME:                 v32int32 [[C:[a-zA-Z0-9]+]]) {
// CHECK:           v32acc32 [[CACC:.*]] = v32acc32([[C]]);
// CHECK:           v32int16 [[PA:.*]] = concat([[A]], undef_v16int16());
// CHECK:           v64int8 [[PB:.*]] = concat([[B]], undef_v32int8());
// CHECK:           [[CACC]] = mac_4x4_4x8([[PA]], [[PB]], [[CACC]]);
// CHECK:           v32int32 [[R:.*]] = v32int32([[CACC]]);
// CHECK:           return [[R]];
// CHECK:        }
func.func @matmul_pad1(%A : vector<4x4xi16>, %B : vector<4x8xi8>,
                       %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %acc = aievec.cast %C {isResAcc = true} : vector<4x8xi32>, vector<4x8xi32>
  %r = aievec.matmul %A, %B, %acc : vector<4x4xi16>, vector<4x8xi8>
                                    into vector<4x8xi32>
  %0 = aievec.cast %r {isResAcc = false} : vector<4x8xi32>, vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// CHECK-LABEL: v16acc64 matmul_pad2(
// CHECK-SAME:             v8int32 [[A:[a-zA-Z0-9]+]],
// CHECK-SAME:             v8int16 [[B:[a-zA-Z0-9]+]],
// CHECK-SAME:             v16acc64 [[C:[a-zA-Z0-9]+]]) {
// CHECK:           v16int32 [[PA:.*]] = concat([[A]], undef_v8int32());
// CHECK:           v32int16 [[PB:.*]] = concat(concat([[B]], undef_v8int16()),
// CHECK-SAME:                                         undef_v16int16());
// CHECK:           [[C]] = mac_4x2_2x4([[PA]], [[PB]], [[C]]);
// CHECK:           return [[C]];
// CHECK:        }
func.func @matmul_pad2(%A : vector<4x2xi32>, %B : vector<2x4xi16>,
                       %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %acc = aievec.cast %C {isResAcc = true} : vector<4x4xi64>, vector<4x4xi64>
  %r = aievec.matmul %A, %B, %acc : vector<4x2xi32>, vector<2x4xi16>
                                    into vector<4x4xi64>
  %0 = aievec.cast %r {isResAcc = false} : vector<4x4xi64>, vector<4x4xi64>
  return %0 : vector<4x4xi64>
}
