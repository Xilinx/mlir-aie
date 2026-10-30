// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-expand-load-pdi='ctrl-pkt=true teardown=true' %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence
// CHECK: aiex.npu.load_pdi {device_ref = @transport_a,
// CHECK: aiex.npu.load_pdi {device_ref = @transport_b_copy_{{[0-9]+}},
// CHECK-NOT: aiex.npu.load_pdi
// CHECK: {{^    }}}

module {
  aie.device(npu2_1col) @transport_a {} {aie.is_ctrl_overlay}
  aie.device(npu2_1col) @transport_b {} {aie.is_ctrl_overlay}
  aie.device(npu2_1col) @transport_b_copy {}
  aie.device(npu2_1col) @a {} {overlay = @transport_a}
  aie.device(npu2_1col) @b {} {overlay = @transport_b}
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @a}
      aiex.npu.load_pdi {device_ref = @b}
      aiex.npu.load_pdi {device_ref = @b}
    }
  }
}
