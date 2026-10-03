// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-lower-scratchpad-parameters="output-params-file=%t.params.txt" \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=PARAMS < %t.params.txt

// A length_parameter adds its scratchpad value to the shim BD's buffer
// length register, the BD's first word, after the block write and address
// patch and before the queue push. Both BD forms lower the same way.

// PARAMS: 1
// PARAMS-NEXT: extra 0 i32 len

// CHECK: aiex.scratchpad_parameter @extra : i32 {kind = 2 : i32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.create_scratchpad
// Task BD: bd 3 on tile (0, 0) is at 0x1D060.
// CHECK: aiex.npu.blockwrite({{.*}}) {address = 118880 : ui32}
// CHECK-NEXT: arith.constant
// CHECK-NEXT: aiex.npu.address_patch({{.*}}) {addr = 118884 : ui32, arg_idx = 0 : i32}
// CHECK-NEXT: aiex.npu.update_from_scratchpad {address = 118880 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.write32
// dma_memcpy_nd: bd 5 on tile (1, 0) is at 0x201D0A0.
// CHECK: aiex.npu.blockwrite({{.*}}) {address = 33673376 : ui32}
// CHECK-NEXT: arith.constant
// CHECK-NEXT: aiex.npu.address_patch({{.*}}) {addr = 33673380 : ui32, arg_idx = 1 : i32}
// CHECK-NEXT: aiex.npu.update_from_scratchpad {address = 33673376 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.write32

module {
  aiex.scratchpad_parameter @extra : i32
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    aie.shim_dma_allocation @out0 (%t10, S2MM, 0)
    aie.runtime_sequence @seq(%in : memref<256xbf16>, %out : memref<256xbf16>) {
      %t = aiex.dma_configure_task(%t00, MM2S, 0) {
        aie.dma_bd(%in : memref<256xbf16> offset = 0 len = 32) {bd_id = 3 : i32, length_parameter = @extra}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1]) {id = 5 : i64, metadata = @out0, length_parameter = @extra, issue_token = true} : memref<256xbf16>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
