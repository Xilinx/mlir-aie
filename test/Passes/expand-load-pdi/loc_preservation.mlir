//===- loc_preservation.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi --mlir-print-debuginfo --mlir-print-op-generic %s | FileCheck %s --implicit-check-not='loc(unknown)'

// The initial reset device and its terminator inherit the load's location.
// The trailing reset device and its terminator inherit the sequence's location.
// Generic printing exposes the otherwise hidden device terminators.

module {
  aie.device(npu2_1col) @configuration {
    aie.end loc("user_design.py":3:2)
  } loc("user_design.py":2:0)
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @configuration} loc("user_design.py":12:4)
    } loc("user_design.py":10:2)
  } loc("user_design.py":8:0)
} loc("user_design.py":1:0)

// CHECK: "aie.device"() ({
// CHECK-NEXT: "aie.end"() : () -> () loc(#[[SEQLOC:loc[0-9]*]])
// CHECK-NEXT: }) {{.*}}sym_name = "empty_1"{{.*}} loc(#[[SEQLOC]])
// CHECK-NEXT: "aie.device"() ({
// CHECK-NEXT: "aie.end"() : () -> () loc(#[[LOADLOC:loc[0-9]*]])
// CHECK-NEXT: }) {{.*}}sym_name = "empty_0"{{.*}} loc(#[[LOADLOC]])
// CHECK: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_0{{.*}} loc(#[[LOADLOC]])
// CHECK-NEXT: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_1{{.*}} loc(#[[SEQLOC]])
// CHECK-DAG: #[[LOADLOC]] = loc("user_design.py":12:4)
// CHECK-DAG: #[[SEQLOC]] = loc("user_design.py":10:2)
