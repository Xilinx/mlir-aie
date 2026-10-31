//===- design_named_errors.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An error located at an object IRON named (loc("iron:<name>"(...))) opens with
// that name and points at the Python statement, with its callers as notes;
// the dump of the op it was emitted on is left out, except with --verbose: a
// pass may have generated that op from the object. Any other location keeps
// the dump.

// RUN: not %aiecc -n --tmpdir %t.iron --get=input_physical.mlir %s 2>&1 | FileCheck %s --check-prefix=IRON
// RUN: not %aiecc -n --verbose --tmpdir %t.verbose --get=input_physical.mlir %s 2>&1 | FileCheck %s --check-prefix=VERBOSE
// RUN: sed 's/iron:act_in/act_in/' %s > %t.plain.mlir
// RUN: not %aiecc -n --tmpdir %t.plain --get=input_physical.mlir %t.plain.mlir 2>&1 | FileCheck %s --check-prefix=PLAIN

// IRON: design.py:12:4: error: act_in: 'aie.objectfifo.pool' op segment 0 has no drainer
// IRON-NEXT: design.py:30:2: note: called from
// IRON-NOT: see current operation

// VERBOSE: design.py:12:4: error: act_in: 'aie.objectfifo.pool' op segment 0 has no drainer
// VERBOSE: note: see current operation:
// VERBOSE-NEXT: "aie.objectfifo.pool"{{.*}}sym_name = "act_in__pool"

// PLAIN: design.py:12:4: error: 'aie.objectfifo.pool' op segment 0 has no drainer
// PLAIN: note: see current operation

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @act_in(%t02, {%t03}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc(#act_in)
  }
}
#site = loc("design.py":12:4)
#caller = loc("design.py":30:2)
#act_in = loc("iron:act_in"(callsite(#site at #caller)))
