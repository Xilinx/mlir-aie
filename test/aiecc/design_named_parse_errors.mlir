//===- design_named_parse_errors.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An op the verifier rejects while aiecc parses its input is reported as one
// a pass rejects: opening with the IRON object's name, its callers as notes,
// and without the op's dump.

// RUN: not %aiecc -n --no-progress --tmpdir %t --get=input_physical.mlir %s 2>&1 | FileCheck %s

// CHECK: {{^}}design.py:12:4: error: act_in: 'aie.tile' op column index (9) must be less than the number of columns in the device (1)
// CHECK-NEXT: {{^}}design.py:30:2: note: called from
// CHECK-NOT: see current operation

module {
  aie.device(npu1_1col) {
    %t = aie.tile(9, 2) loc(#act_in)
  }
}
#site = loc("design.py":12:4)
#caller = loc("design.py":30:2)
#act_in = loc("iron:act_in"(callsite(#site at #caller)))
