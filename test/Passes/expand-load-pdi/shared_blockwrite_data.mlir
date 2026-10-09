//===- shared_blockwrite_data.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="inline-config=true" %s | FileCheck %s

// Every reload of a device writes the same payloads, so the reloads share one
// global for each, and a sequence's reloads one get_global of it. Its name is
// past every loadpdi_<n> symbol the device holds, global or not.

// CHECK-LABEL: aie.device(npu2_1col) {
// CHECK: memref.global "private" constant @loadpdi_4 : memref<4xi32> = dense<[1, 2, 3, 4]>
// CHECK-NOT: memref.global
// CHECK: aie.buffer({{.*}}) {{.*}}sym_name = "loadpdi_3"
// CHECK: memref.global "private" constant @loadpdi_1 : memref<2xi32> = dense<[7, 8]>
// CHECK-NOT: memref.global
// CHECK: %[[G:.*]] = memref.get_global @loadpdi_4
// CHECK-NOT: memref.get_global
// CHECK: aiex.npu.load_pdi {device_ref = @empty_0
// CHECK: aiex.npu.blockwrite(%[[G]])
// CHECK: aiex.npu.load_pdi {device_ref = @empty_1
// CHECK: aiex.npu.blockwrite(%[[G]])

module {
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<4xi32> = dense<[1, 2, 3, 4]>
  }
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {address = 1024 : i32, sym_name = "loadpdi_3"} : memref<4xi32>
    memref.global "private" constant @loadpdi_1 : memref<2xi32> = dense<[7, 8]>
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @init}
      aiex.npu.load_pdi {device_ref = @init}
    }
  }
}
