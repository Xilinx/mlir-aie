//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aiex-to-emitc | FileCheck %s

// Each straight-line npu op lowers to an emitc.call_opaque naming the matching
// aie_runtime::txn_append_* function, wrapped in generate_txn_<device>_<seq>.

// CHECK-LABEL: emitc.func @generate_txn_main_seq_write32
// CHECK: call_opaque "aie_runtime::txn_init"
// CHECK: call_opaque "aie_runtime::txn_append_write32"
// CHECK: verbatim "aie_runtime::txn_prepend_header(txn, 1u
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_write32(%arg0: memref<8xi32>) {
      %addr = arith.constant 119300 : i32
      %val = arith.constant 42 : i32
      aiex.npu.write32(%addr, %val) : i32, i32
    }
  }
}

// -----

// CHECK-LABEL: emitc.func @generate_txn_main_seq_maskwrite32
// CHECK: call_opaque "aie_runtime::txn_append_maskwrite32"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_maskwrite32(%arg0: memref<8xi32>) {
      %addr = arith.constant 119300 : i32
      %val = arith.constant 42 : i32
      %mask = arith.constant 255 : i32
      aiex.npu.maskwrite32(%addr, %val, %mask) : i32, i32, i32
    }
  }
}

// -----

// CHECK-LABEL: emitc.func @generate_txn_main_seq_sync
// CHECK: call_opaque "aie_runtime::txn_append_sync"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_sync(%arg0: memref<8xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// CHECK-LABEL: emitc.func @generate_txn_main_seq_address_patch
// CHECK: call_opaque "aie_runtime::txn_append_address_patch"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_address_patch(%arg0: memref<8xi32>) {
      %plus = arith.constant 16 : i32
      aiex.npu.address_patch(%plus : i32) {addr = 119300 : ui32, arg_idx = 0 : i32}
    }
  }
}

// -----

// The blockwrite payload is emitted as a typed EmitC array variable (not a
// verbatim C++ string), and the feeding memref.get_global is dropped.
// CHECK-LABEL: emitc.func @generate_txn_main_seq_blockwrite
// CHECK-NOT: memref.get_global
// CHECK: emitc.variable{{.*}}!emitc.array<4x
// CHECK: call_opaque "aie_runtime::txn_append_blockwrite"
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockdata : memref<4xi32> = dense<[1, 2, 3, 4]>
    aie.runtime_sequence @seq_blockwrite(%arg0: memref<8xi32>) {
      %d = memref.get_global @blockdata : memref<4xi32>
      aiex.npu.blockwrite(%d) {address = 119300 : ui32} : memref<4xi32>
    }
  }
}

// -----

// Multiple runtime sequences in one device each get their own function.
// CHECK-DAG: emitc.func @generate_txn_main_seqA
// CHECK-DAG: emitc.func @generate_txn_main_seqB
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seqA(%arg0: memref<8xi32>) {
      %a = arith.constant 100 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seqB(%arg0: memref<8xi32>) {
      %a = arith.constant 200 : i32
      %v = arith.constant 2 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// A runtime-bound scf.for (the dynamic BD pool's rolled loop) is preserved: the
// body's npu ops convert in place and the loop lowers to emitc.for, so the C++
// stays rolled with a runtime op-count (++__opcount inside the loop).
// CHECK-LABEL: emitc.func @generate_txn_main_seq_scf_for
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : !emitc.size_t {
// CHECK: call_opaque "aie_runtime::txn_append_write32"
// CHECK: verbatim "++{};"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_scf_for(%arg0: memref<8xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %addr = arith.constant 100 : i32
        %val = arith.constant 7 : i32
        aiex.npu.write32(%addr, %val) : i32, i32
      }
    }
  }
}

// -----

// A runtime-bound scf.if lowers to emitc.if with its npu body converted in place.
// CHECK-LABEL: emitc.func @generate_txn_main_seq_scf_if
// CHECK: if %{{.*}} {
// CHECK: call_opaque "aie_runtime::txn_append_write32"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_scf_if(%arg0: memref<8xi32>, %cond: i1) {
      scf.if %cond {
        %addr = arith.constant 100 : i32
        %val = arith.constant 7 : i32
        aiex.npu.write32(%addr, %val) : i32, i32
      }
    }
  }
}

// -----

// npu2 device: the header carries devGen=4, rows=6, cols=8, memtilerows=1
// (vs npu1's devGen=3). Exercises the BaseNPU2TargetModel branch.
// CHECK-LABEL: emitc.func @generate_txn_main_seq_npu2
// CHECK: verbatim "aie_runtime::txn_prepend_header(txn, 1u, {0, 1, 4, 6, 8, 1});"
module {
  aie.device(npu2) {
    aie.runtime_sequence @seq_npu2(%arg0: memref<8xi32>) {
      %a = arith.constant 100 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// An scf.if nested inside an scf.for: the converter recurses into both regions
// and converts the innermost npu op; convert-scf-to-emitc later produces nested
// emitc.for / emitc.if. A loop is present, so the header uses the runtime
// __opcount.
// CHECK-LABEL: emitc.func @generate_txn_main_seq_if_in_for
// CHECK: for
// CHECK: if %{{.*}} {
// CHECK: call_opaque "aie_runtime::txn_append_write32"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_if_in_for(%arg0: memref<8xi32>, %n: index, %cond: i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        scf.if %cond {
          %addr = arith.constant 100 : i32
          %val = arith.constant 7 : i32
          aiex.npu.write32(%addr, %val) : i32, i32
        }
      }
    }
  }
}

// -----

// scf.if with an else branch: each branch converts its own npu op to an
// emitc.if / else with the write32 call in each arm.
// CHECK-LABEL: emitc.func @generate_txn_main_seq_if_else
// CHECK: if %{{.*}} {
// CHECK: call_opaque "aie_runtime::txn_append_write32"
// CHECK: } else {
// CHECK: call_opaque "aie_runtime::txn_append_write32"
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_if_else(%arg0: memref<8xi32>, %cond: i1) {
      scf.if %cond {
        %a0 = arith.constant 100 : i32
        %v0 = arith.constant 7 : i32
        aiex.npu.write32(%a0, %v0) : i32, i32
      } else {
        %a1 = arith.constant 104 : i32
        %v1 = arith.constant 9 : i32
        aiex.npu.write32(%a1, %v1) : i32, i32
      }
    }
  }
}
