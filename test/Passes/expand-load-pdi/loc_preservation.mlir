//===- loc_preservation.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi --split-input-file --mlir-print-debuginfo --mlir-print-op-generic %s | FileCheck %s --implicit-check-not='loc(unknown)'

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

// CHECK: "aie.device"() <{{.*}}sym_name = "empty_1"{{.*}}> ({
// CHECK-NEXT: "aie.end"() : () -> () loc(#[[SEQLOC:loc[0-9]*]])
// CHECK-NEXT: }) : () -> () loc(#[[SEQLOC]])
// CHECK-NEXT: "aie.device"() <{{.*}}sym_name = "empty_0"{{.*}}> ({
// CHECK-NEXT: "aie.end"() : () -> () loc(#[[LOADLOC:loc[0-9]*]])
// CHECK-NEXT: }) : () -> () loc(#[[LOADLOC]])
// CHECK: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_0{{.*}} loc(#[[LOADLOC]])
// CHECK-NEXT: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_1{{.*}} loc(#[[SEQLOC]])
// CHECK-DAG: #[[LOADLOC]] = loc("user_design.py":12:4)
// CHECK-DAG: #[[SEQLOC]] = loc("user_design.py":10:2)

// -----

// Three loads reuse both parity devices. The trailing reset reuses empty_1,
// which must retain both the second load's and the sequence's provenance.
// Each load operation still has only its own origin.
module {
  aie.device(npu2_1col) @configuration {
    aie.end loc("user_design.py":3:2)
  } loc("user_design.py":2:0)
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @configuration} loc("user_design.py":12:4)
      aiex.npu.load_pdi {device_ref = @configuration} loc("user_design.py":13:4)
      aiex.npu.load_pdi {device_ref = @configuration} loc("user_design.py":14:4)
    } loc("user_design.py":10:2)
  } loc("user_design.py":8:0)
} loc("user_design.py":1:0)

// CHECK: "aie.device"() <{{.*}}sym_name = "empty_1"{{.*}}> ({
// CHECK-NEXT: "aie.end"() : () -> () loc(#[[ODDLOC:loc[0-9]*]])
// CHECK-NEXT: }) : () -> () loc(#[[ODDLOC]])
// CHECK-NEXT: "aie.device"() <{{.*}}sym_name = "empty_0"{{.*}}> ({
// CHECK-NEXT: "aie.end"() : () -> () loc(#[[EVENLOC:loc[0-9]*]])
// CHECK-NEXT: }) : () -> () loc(#[[EVENLOC]])
// CHECK: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_0{{.*}} loc(#[[FIRSTLOC:loc[0-9]*]])
// CHECK: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_1{{.*}} loc(#[[SECONDLOC:loc[0-9]*]])
// CHECK: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_0{{.*}} loc(#[[THIRDLOC:loc[0-9]*]])
// CHECK: "aiex.npu.load_pdi"() {{.*}}device_ref = @empty_1{{.*}} loc(#[[TRAILINGLOC:loc[0-9]*]])
// CHECK-DAG: #[[FIRSTLOC]] = loc("user_design.py":12:4)
// CHECK-DAG: #[[SECONDLOC]] = loc("user_design.py":13:4)
// CHECK-DAG: #[[THIRDLOC]] = loc("user_design.py":14:4)
// CHECK-DAG: #[[TRAILINGLOC]] = loc("user_design.py":10:2)
// CHECK-DAG: #[[ODDLOC]] = loc(fused[#[[SECONDLOC]], #[[TRAILINGLOC]]])
// CHECK-DAG: #[[EVENLOC]] = loc(fused[#[[FIRSTLOC]], #[[THIRDLOC]]])
