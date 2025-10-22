//===- lower_event.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL: test_aie1
// CHECK: call @llvm.aie.event0()
// CHECK: call @llvm.aie.event1()
module @test_aie1 {
 aie.device(xcvc1902) {
  %tile11 = aie.tile(1, 1)
  %core11 = aie.core(%tile11) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: test_aie2
// CHECK: call @llvm.aie2.event(%{{.*}})
// CHECK: call @llvm.aie2.event(%{{.*}})
module @test_aie2 {
 aie.device(npu1) {
  %tile02 = aie.tile(0, 2)
  %core02 = aie.core(%tile02) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}

// -----

// CHECK-LABEL: test_aie2p
// CHECK: call @llvm.aie2p.event(%{{.*}})
// CHECK: call @llvm.aie2p.event(%{{.*}})
module @test_aie2p {
 aie.device(npu2) {
  %tile02 = aie.tile(0, 2)
  %core02 = aie.core(%tile02) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}
