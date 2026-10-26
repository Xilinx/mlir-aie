// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file %s | FileCheck %s

// With stack_size absent, a measurement above the default sizes the stack,
// aligned up to npu2's 64-byte stack alignment.
// CHECK-LABEL: @grows
// CHECK: aie.buffer(%{{.*}}) {address = 4160 : i32
module @grows {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<16xi32>
    aie.core(%t) {
      aie.end
    } {measured_stack_size = 4100 : i32}
  }
}

// -----

// A measurement below the default keeps the default.
// CHECK-LABEL: @floor
// CHECK: aie.buffer(%{{.*}}) {address = 1024 : i32
module @floor {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<16xi32>
    aie.core(%t) {
      aie.end
    } {measured_stack_size = 100 : i32}
  }
}

// -----

// A declared stack_size wins over the measurement.
// CHECK-LABEL: @declared
// CHECK: aie.buffer(%{{.*}}) {address = 2048 : i32
module @declared {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<16xi32>
    aie.core(%t) {
      aie.end
    } {measured_stack_size = 4100 : i32, stack_size = 2048 : i32}
  }
}
