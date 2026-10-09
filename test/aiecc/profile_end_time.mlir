// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --profile reports when each edge finished, so the critical path reads off
// the table.
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --get-npu-insts --no-progress --profile --output-dir=%t.d --tmpdir=%t.d/work %s 2>&1 | FileCheck %s

// CHECK: aiecc: profile (per-edge time and resident memory):
// CHECK-NEXT: {{^ +ms +end ms +dRSS MiB +peak MiB +edge$}}
// CHECK: {{^ +[0-9]+ +[0-9]+ +[-0-9.]+ +[0-9.]+ +npu_lowered.mlir$}}
// CHECK: {{^ +[0-9]+ +[0-9.]+ +total$}}

module {
  aie.device(npu2) @main {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @input (%tile, MM2S, 0)
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @input} : memref<64xi32>
    }
  }
}
