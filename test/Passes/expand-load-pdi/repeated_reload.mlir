//===- repeated_reload.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// Every reload of a device writes the same configuration, in each sequence and
// each enclosing device, and names the payload global of the device it is in.

// CHECK-LABEL: aie.device(npu2_1col) {
// CHECK: memref.global "private" constant @loadpdi_0 : memref<4xi32> = dense<[1, 2, 3, 4]>
// CHECK: aie.runtime_sequence @first
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NEXT: %[[A0:.*]] = arith.constant 2224128 : i32
// CHECK-NEXT: %[[V0:.*]] = arith.constant 1 : i32
// CHECK-NEXT: aiex.npu.write32(%[[A0]], %[[V0]])
// CHECK-NEXT: %[[G0:.*]] = memref.get_global @loadpdi_0
// CHECK-NEXT: aiex.npu.blockwrite(%[[G0]]) {address = 2098176 : ui32}
// CHECK: aie.runtime_sequence @second
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_1
// CHECK-NEXT: %[[A1:.*]] = arith.constant 2224128 : i32
// CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: aiex.npu.write32(%[[A1]], %[[V1]])
// CHECK-NEXT: %[[G1:.*]] = memref.get_global @loadpdi_0
// CHECK-NEXT: aiex.npu.blockwrite(%[[G1]]) {address = 2098176 : ui32}

// CHECK-LABEL: aie.device(npu2_1col) @other {
// CHECK: memref.global "private" constant @loadpdi_0 : memref<4xi32> = dense<[1, 2, 3, 4]>
// CHECK: aie.runtime_sequence
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NEXT: %[[A2:.*]] = arith.constant 2224128 : i32
// CHECK-NEXT: %[[V2:.*]] = arith.constant 1 : i32
// CHECK-NEXT: aiex.npu.write32(%[[A2]], %[[V2]])
// CHECK-NEXT: %[[G2:.*]] = memref.get_global @loadpdi_0
// CHECK-NEXT: aiex.npu.blockwrite(%[[G2]]) {address = 2098176 : ui32}

module {
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
    %b = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<4xi32> = dense<[1, 2, 3, 4]>
  }
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    aie.runtime_sequence @first(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @init}
    }
    aie.runtime_sequence @second(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @init}
    }
  }
  aie.device(npu2_1col) @other {
    %t = aie.tile(0, 2)
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @init}
    }
  }
}
