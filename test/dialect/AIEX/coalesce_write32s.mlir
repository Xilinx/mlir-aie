//===- coalesce_write32s.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-coalesce-write32s --split-input-file %s | FileCheck %s

// Four writes at contiguous addresses become one blockwrite over the four
// values.
// CHECK: memref.global "private" constant @[[G:.*]] : memref<4xi32> = dense<[10, 11, 12, 13]>
// CHECK: aie.runtime_sequence @contiguous
// CHECK: %[[D:.*]] = memref.get_global @[[G]]
// CHECK: aiex.npu.blockwrite(%[[D]]) {address = 1024 : ui32}
// CHECK-NOT: aiex.npu.write32
aie.device(npu2) {
  aie.runtime_sequence @contiguous() {
    %a0 = arith.constant 1024 : i32
    %v0 = arith.constant 10 : i32
    aiex.npu.write32(%a0, %v0) : i32, i32
    %a1 = arith.constant 1028 : i32
    %v1 = arith.constant 11 : i32
    aiex.npu.write32(%a1, %v1) : i32, i32
    %a2 = arith.constant 1032 : i32
    %v2 = arith.constant 12 : i32
    aiex.npu.write32(%a2, %v2) : i32, i32
    %a3 = arith.constant 1036 : i32
    %v3 = arith.constant 13 : i32
    aiex.npu.write32(%a3, %v3) : i32, i32
  }
}

// -----

// A gap in the addresses ends the run, and a run shorter than the threshold of
// two stays a write32.
// CHECK: memref.global "private" constant @[[G:.*]] : memref<2xi32> = dense<[10, 11]>
// CHECK: aie.runtime_sequence @gap
// CHECK: %[[D:.*]] = memref.get_global @[[G]]
// CHECK: aiex.npu.blockwrite(%[[D]]) {address = 1024 : ui32}
// CHECK: aiex.npu.write32
aie.device(npu2) {
  aie.runtime_sequence @gap() {
    %a0 = arith.constant 1024 : i32
    %v0 = arith.constant 10 : i32
    aiex.npu.write32(%a0, %v0) : i32, i32
    %a1 = arith.constant 1028 : i32
    %v1 = arith.constant 11 : i32
    aiex.npu.write32(%a1, %v1) : i32, i32
    %a2 = arith.constant 2048 : i32
    %v2 = arith.constant 12 : i32
    aiex.npu.write32(%a2, %v2) : i32, i32
  }
}

// -----

// A later write to the same address supersedes the earlier one, so the
// blockwrite carries the later value.
// CHECK: memref.global "private" constant @{{.*}} : memref<2xi32> = dense<[99, 11]>
// CHECK: aie.runtime_sequence @duplicate
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 1024 : ui32}
// CHECK-NOT: aiex.npu.write32
aie.device(npu2) {
  aie.runtime_sequence @duplicate() {
    %a0 = arith.constant 1024 : i32
    %v0 = arith.constant 10 : i32
    aiex.npu.write32(%a0, %v0) : i32, i32
    %a1 = arith.constant 1028 : i32
    %v1 = arith.constant 11 : i32
    aiex.npu.write32(%a1, %v1) : i32, i32
    %a2 = arith.constant 1024 : i32
    %v2 = arith.constant 99 : i32
    aiex.npu.write32(%a2, %v2) : i32, i32
  }
}

// -----

// A write to Core_Control of tile (0, 2) ends the slice, so the two writes
// around it stay in place.
// CHECK: aie.runtime_sequence @special_register
// CHECK-NOT: aiex.npu.blockwrite
aie.device(npu2) {
  aie.runtime_sequence @special_register() {
    %a0 = arith.constant 1024 : i32
    %v0 = arith.constant 10 : i32
    aiex.npu.write32(%a0, %v0) : i32, i32
    // Core_Control of tile (0, 2): (2 << 20) | 0x32000.
    %a1 = arith.constant 2301952 : i32
    %v1 = arith.constant 1 : i32
    aiex.npu.write32(%a1, %v1) : i32, i32
    %a2 = arith.constant 1028 : i32
    %v2 = arith.constant 11 : i32
    aiex.npu.write32(%a2, %v2) : i32, i32
  }
}

// -----

// A masked write reads the register it writes, so it ends the run and stays a
// maskwrite32.
// CHECK: aie.runtime_sequence @masked
// CHECK-NOT: aiex.npu.blockwrite
// CHECK: aiex.npu.maskwrite32
aie.device(npu2) {
  aie.runtime_sequence @masked() {
    %a0 = arith.constant 1024 : i32
    %v0 = arith.constant 10 : i32
    aiex.npu.write32(%a0, %v0) : i32, i32
    %a1 = arith.constant 1028 : i32
    %v1 = arith.constant 11 : i32
    %m1 = arith.constant 255 : i32
    aiex.npu.maskwrite32(%a1, %v1, %m1) : i32, i32, i32
  }
}
