//===- bad_data_size.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc writes measured_data_size from the linked ELF. The reservation has to
// cover it, because the buffer allocator gave the rest of the tile to buffers.

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}op data_size 4096 is smaller than the 8192 bytes this core's linked sections occupy
module @reservation_too_small {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_size = 4096 : i32, measured_data_size = 8192 : i32}
  }
}

// -----

// CHECK: error{{.*}}op data_size 0 is smaller than the 64 bytes this core's linked sections occupy
module @nothing_reserved {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_size = 0 : i32, measured_data_size = 64 : i32}
  }
}
