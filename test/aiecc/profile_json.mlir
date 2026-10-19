// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --profile-json writes --profile's per-edge table as JSON, without --profile's
// stderr table.
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --get-npu-insts --no-progress --profile-json=%t.d/profile.json --output-dir=%t.d --tmpdir=%t.d/work %s 2>&1 | FileCheck %s --check-prefix=STDERR --allow-empty
// RUN: FileCheck %s --input-file=%t.d/profile.json

// STDERR-NOT: aiecc: profile

// CHECK: "edges": [
// CHECK: "name": "placed.mlir",
// CHECK-NEXT: "peak_mib": {{[0-9.]+|null}},
// CHECK-NEXT: "rss_after_mib": {{[0-9.]+|null}},
// CHECK-NEXT: "start_ms": {{[0-9]+}}
// CHECK: "name": "npu_lowered.mlir",
// CHECK: "version": 1,
// CHECK-NEXT: "wall_ms": {{[0-9]+}}

module {
  aie.device(npu2) @main {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @input (%tile, MM2S, 0)
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @input} : memref<64xi32>
    }
  }
}
