//===- duplicate_sym_name.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Named ops that define an SSA value are not `Symbol` ops, so upstream's
// symbol table verifier does not compare their names.

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}redefinition of symbol named 'dup_buf'
// CHECK: note{{.*}}see existing symbol definition here
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %b0 = aie.buffer(%t) {sym_name = "dup_buf"} : memref<4xi32>
  %b1 = aie.buffer(%t) {sym_name = "dup_buf"} : memref<4xi32>
}

// -----

// CHECK: error{{.*}}redefinition of symbol named 'dup_lock'
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %l0 = aie.lock(%t, 0) {sym_name = "dup_lock"}
  %l1 = aie.lock(%t, 1) {sym_name = "dup_lock"}
}

// -----

// A buffer and a lock collide just as two buffers do.
// CHECK: error{{.*}}redefinition of symbol named 'shared'
aie.device(npu1) {
  %t = aie.tile(0, 2)
  %b = aie.buffer(%t) {sym_name = "shared"} : memref<4xi32>
  %l = aie.lock(%t, 0) {sym_name = "shared"}
}

// -----

// Upstream's verifier compares symbol names only, so it never sees this pair.
// CHECK: error{{.*}}redefinition of symbol named 'shared_with_fifo'
aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %core = aie.tile(0, 2)
  aie.objectfifo @shared_with_fifo(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
  %b = aie.buffer(%core) {sym_name = "shared_with_fifo"} : memref<4xi32>
}
