//===- bad_cascade.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.cascade_flow' op tiles must be adjacent

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  %t33 = aie.tile(3, 3)
  aie.cascade_flow(%t13, %t33)
}

// -----

// CHECK: error{{.*}}'aie.cascade_switchbox' op cannot have more than one ConnectOp in CascadeSwitchboxOp

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  aie.cascade_switchbox(%t13) {
    aie.connect<West: 0, East: 0>
    aie.connect<North: 0, South: 0>
  }
}

// -----

// CHECK: error{{.*}}'aie.connect' op portIndex of ConnectOp is out-of-bounds

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  aie.cascade_switchbox(%t13) {
    aie.connect<West: 0, East: 1>
  }
}

// -----

// CHECK: error{{.*}}'aie.connect' op source port of ConnectOp in CascadeSwitchboxOp must be West or North

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  aie.cascade_switchbox(%t13) {
    aie.connect<East: 0, South: 0>
  }
}

// -----

// CHECK: error{{.*}}'aie.connect' op dest port of ConnectOp in CascadeSwitchboxOp must be East or South

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  aie.cascade_switchbox(%t13) {
    aie.connect<West: 0, North: 0>
  }
}
