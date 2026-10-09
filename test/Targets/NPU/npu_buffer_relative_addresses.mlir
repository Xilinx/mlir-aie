//===- npu_buffer_relative_addresses.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

// A blockwrite or update_from_scratchpad naming a buffer is encoded at the
// buffer's tile and address plus its word offset, whichever of the device's
// named ops the buffer is (npu2: column << 25, row << 20).

// CHECK: 02200808
// CHECK-NEXT: 00000018
// CHECK-NEXT: 00000005
// CHECK-NEXT: 00000006
// CHECK: 04300414
module {
  aie.device(npu2) {
    %t12 = aie.tile(1, 2)
    %t23 = aie.tile(2, 3)
    %lock = aie.lock(%t12, 0) {sym_name = "lock"}
    %a = aie.buffer(%t12) {address = 1024 : i32, sym_name = "a"} : memref<64xi32>
    %b = aie.buffer(%t12) {address = 2048 : i32, sym_name = "b"} : memref<64xi32>
    %c = aie.buffer(%t23) {address = 1024 : i32, sym_name = "c"} : memref<64xi32>
    memref.global "private" constant @payload : memref<2xi32> = dense<[5, 6]>
    aie.runtime_sequence() {
      %data = memref.get_global @payload : memref<2xi32>
      aiex.npu.blockwrite(%data) {address = 2 : ui32, buffer = @b} : memref<2xi32>
      aiex.npu.create_scratchpad {size = 4 : ui32}
      aiex.npu.update_from_scratchpad {address = 5 : ui32, buffer = @c, state_table_idx = 0 : ui8}
    }
  }
}
