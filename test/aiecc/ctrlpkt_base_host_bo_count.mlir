//===- ctrlpkt_base_host_bo_count.mlir --------------------------*- MLIR -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The ctrl-packet reconfig "base" device has no runtime_sequence of its own; its
// host-buffer ABI is defined by the sibling "main" device's sequence. aiecc must
// derive base's kernels.json boN count from main (main's data args + 1 control-
// packet buffer), not from the kNoSequenceHostBOs fallback. Here main has 6 data
// args, so base declares bo0..bo6 (6 + 1 ctrlpkt = 7); the old fallback would
// have wrongly declared only 5 and under-provisioned the host ABI.

// RUN: aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" %s -o %t_overlay.mlir
// RUN: %aiecc -n --device-name=base --aie-generate-xclbin --tmpdir=%t.prj %t_overlay.mlir
// RUN: FileCheck %s --input-file=%t.prj/kernels_base.json

// CHECK: "name": "bo0"
// CHECK: "name": "bo1"
// CHECK: "name": "bo2"
// CHECK: "name": "bo3"
// CHECK: "name": "bo4"
// CHECK: "name": "bo5"
// CHECK: "name": "bo6"
// CHECK-NOT: "name": "bo7"

module {
  aie.device(npu1) @base {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
  }
  aie.device(npu1) @main {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.shim_dma_allocation @objFifo_in0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objFifo_out0 (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @run(%arg1: memref<64x64xi8>, %arg2: memref<64x64xi8>, %arg3: memref<64x64xi8>, %arg4: memref<64x64xi8>, %arg5: memref<64x64xi8>, %arg6: memref<64x64xi8>) {
      aiex.configure @main {
        aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1], packet = <pkt_id = 3, pkt_type = 0>) {id = 0 : i64, metadata = @objFifo_in0} : memref<64x64xi8>
        aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
        aiex.npu.dma_wait { symbol = @objFifo_out0 }
      }
    }
  }
}
