// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A control packet whose address targets a tile the device never declares must
// fail loudly with the controller_id diagnostic -- and must NOT be "repaired"
// by inserting a fresh tile (the translator is read-only over its input module).
// RUN: not aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %s 2>&1 | FileCheck %s

// CHECK: error: {{.*}}has no controller_id
module {
  aie.device(npu1) {
    // NOTE: no aie.tile(0,0) declared, so the packet's target tile is absent.
    aie.runtime_sequence() {
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
