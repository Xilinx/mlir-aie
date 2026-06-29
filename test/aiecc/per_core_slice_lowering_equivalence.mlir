//===- per_core_slice_lowering_equivalence.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Per-core compile slicing must not change what a core lowers to. compileCores
// builds each per-core compile input by stripping the runtime sequences, the
// sibling devices, and the other cores, on the premise that the per-core
// lowering (aie-standard-lowering with device + tilecol/tilerow) already
// extracts only the target core and discards the rest. This test pins that
// premise: lowering core (0, 2) of @device1 must yield the same function from
// the full multi-device module (this file: two devices, two cores in @device1,
// a runtime sequence) and from the stripped slice (Inputs/...slice.mlir:
// @device1 with the other core's aie.core op, the sequence, and @device2
// removed, exactly the way compileCores strips, keeping the other core's tile,
// buffer, and lock). Both inputs are checked against the same expected lowering,
// so if a future change makes per-core lowering depend on the stripped content
// the two diverge and this fails. Pure aie-opt, no hardware.

// The full module and the stripped slice must lower the target core the same:
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="device=device1 tilecol=0 tilerow=2" %s | FileCheck %s
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="device=device1 tilecol=0 tilerow=2" %S/Inputs/per_core_slice_lowering_equivalence.slice.mlir | FileCheck %s

// Negative guards: the equivalence check has teeth. A slice that diverges in the
// target core, or that keeps the wrong core, must NOT match the same golden.
// These keep the positive checks above from silently going vacuous.
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="device=device1 tilecol=0 tilerow=2" %S/Inputs/per_core_slice_lowering_equivalence.divergent.mlir | not FileCheck %s
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="device=device1 tilecol=0 tilerow=2" %S/Inputs/per_core_slice_lowering_equivalence.wrongcore.mlir | not FileCheck %s

// CHECK-DAG: memref.global "public" @a12 : memref<16xi32>
// CHECK-DAG: memref.global "public" @a02 : memref<16xi32>
// CHECK: func.func @core_0_2() {
// CHECK: call @llvm.aie2.acquire
// CHECK: %[[VAL:.*]] = arith.constant 7 : i32
// CHECK: %[[BUF:.*]] = memref.get_global @a02 : memref<16xi32>
// CHECK: memref.store %[[VAL]], %[[BUF]][%{{.*}}] : memref<16xi32>
// CHECK: call @llvm.aie2.release
// CHECK: return

module @full {
 aie.device(npu1_2col) @device1 {
  %t02 = aie.tile(0, 2)
  %t12 = aie.tile(1, 2)
  %l02 = aie.lock(%t02, 0)
  %b02 = aie.buffer(%t02) { sym_name = "a02" } : memref<16xi32>
  %l12 = aie.lock(%t12, 0)
  %b12 = aie.buffer(%t12) { sym_name = "a12" } : memref<16xi32>
  %c02 = aie.core(%t02) {
    aie.use_lock(%l02, Acquire, 0)
    %v = arith.constant 7 : i32
    %i = arith.constant 3 : index
    memref.store %v, %b02[%i] : memref<16xi32>
    aie.use_lock(%l02, Release, 1)
    aie.end
  }
  %c12 = aie.core(%t12) {
    aie.use_lock(%l12, Acquire, 0)
    %v = arith.constant 99 : i32
    %i = arith.constant 5 : index
    memref.store %v, %b12[%i] : memref<16xi32>
    aie.use_lock(%l12, Release, 1)
    aie.end
  }
  aie.runtime_sequence @seq(%arg0 : memref<16xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    aiex.npu.dma_memcpy_nd(%arg0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @fifo, id = 0 : i64} : memref<16xi32>
  }
 }
 aie.device(npu1_2col) @device2 {
  %t12 = aie.tile(1, 2)
  %l12 = aie.lock(%t12, 0)
  %b22 = aie.buffer(%t12) { sym_name = "a22" } : memref<16xi32>
  %c12 = aie.core(%t12) {
    aie.use_lock(%l12, Acquire, 0)
    %v = arith.constant 42 : i32
    %i = arith.constant 1 : index
    memref.store %v, %b22[%i] : memref<16xi32>
    aie.use_lock(%l12, Release, 1)
    aie.end
  }
 }
}
