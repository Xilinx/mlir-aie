//===- test_place_attention_routing.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform='skip-verify=true' --aie-create-pathfinder-flows %s -o /dev/null
// RUN: sed -E 's/(%[abc]([0-7]) = aie.logical_tile<CoreTile>)\(\?, \?\)/\1(\2, ?)/' %s | aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform='skip-verify=true' --aie-create-pathfinder-flows -o /dev/null

// Eight three-core pipelines with two memtile forwards each, shared K/V
// broadcasts, and four-way Q splits/O joins. Stream demand should prevent
// packing all 24 cores into six columns. Kernel bodies are omitted.
// CHECK-DAG: aie.tile(6, {{[2-5]}})
// CHECK-NOT: aie.logical_tile
module {
  aie.device(npu2) {
    %sk = aie.logical_tile<ShimNOCTile>(?, ?)
    %sv = aie.logical_tile<ShimNOCTile>(?, ?)
    %sq0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %sq1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %so0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %so1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %k = aie.logical_tile<MemTile>(?, ?)
    %v = aie.logical_tile<MemTile>(?, ?)
    %q0 = aie.logical_tile<MemTile>(?, ?)
    %q1 = aie.logical_tile<MemTile>(?, ?)
    %o0 = aie.logical_tile<MemTile>(?, ?)
    %o1 = aie.logical_tile<MemTile>(?, ?)
    %a0 = aie.logical_tile<CoreTile>(?, ?)
    %b0 = aie.logical_tile<CoreTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %ab0 = aie.logical_tile<MemTile>(?, ?)
    %bc0 = aie.logical_tile<MemTile>(?, ?)
    %a1 = aie.logical_tile<CoreTile>(?, ?)
    %b1 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %ab1 = aie.logical_tile<MemTile>(?, ?)
    %bc1 = aie.logical_tile<MemTile>(?, ?)
    %a2 = aie.logical_tile<CoreTile>(?, ?)
    %b2 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %ab2 = aie.logical_tile<MemTile>(?, ?)
    %bc2 = aie.logical_tile<MemTile>(?, ?)
    %a3 = aie.logical_tile<CoreTile>(?, ?)
    %b3 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %ab3 = aie.logical_tile<MemTile>(?, ?)
    %bc3 = aie.logical_tile<MemTile>(?, ?)
    %a4 = aie.logical_tile<CoreTile>(?, ?)
    %b4 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    %ab4 = aie.logical_tile<MemTile>(?, ?)
    %bc4 = aie.logical_tile<MemTile>(?, ?)
    %a5 = aie.logical_tile<CoreTile>(?, ?)
    %b5 = aie.logical_tile<CoreTile>(?, ?)
    %c5 = aie.logical_tile<CoreTile>(?, ?)
    %ab5 = aie.logical_tile<MemTile>(?, ?)
    %bc5 = aie.logical_tile<MemTile>(?, ?)
    %a6 = aie.logical_tile<CoreTile>(?, ?)
    %b6 = aie.logical_tile<CoreTile>(?, ?)
    %c6 = aie.logical_tile<CoreTile>(?, ?)
    %ab6 = aie.logical_tile<MemTile>(?, ?)
    %bc6 = aie.logical_tile<MemTile>(?, ?)
    %a7 = aie.logical_tile<CoreTile>(?, ?)
    %b7 = aie.logical_tile<CoreTile>(?, ?)
    %c7 = aie.logical_tile<CoreTile>(?, ?)
    %ab7 = aie.logical_tile<MemTile>(?, ?)
    %bc7 = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @k(%k, {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @v(%v, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @k_in(%sk, {%k}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @v_in(%sv, {%v}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@k_in] -> [@k] ([] [])
    aie.objectfifo.link [@v_in] -> [@v] ([] [])
    aie.objectfifo @q0(%q0, {%a0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q1(%q0, {%a1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q2(%q0, {%a2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q3(%q0, {%a3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q4(%q1, {%a4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q5(%q1, {%a5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q6(%q1, {%a6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q7(%q1, {%a7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q_in0(%sq0, {%q0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @q_in1(%sq1, {%q1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@q_in0] -> [@q0, @q1, @q2, @q3] ([] [0, 16, 32, 48])
    aie.objectfifo.link [@q_in1] -> [@q4, @q5, @q6, @q7] ([] [0, 16, 32, 48])
    aie.objectfifo @o0(%c0, {%o0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o1(%c1, {%o0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o2(%c2, {%o0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o3(%c3, {%o0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o4(%c4, {%o1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o5(%c5, {%o1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o6(%c6, {%o1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o7(%c7, {%o1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o_out0(%o0, {%so0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @o_out1(%o1, {%so1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@o0, @o1, @o2, @o3] -> [@o_out0] ([0, 16, 32, 48] [])
    aie.objectfifo.link [@o4, @o5, @o6, @o7] -> [@o_out1] ([0, 16, 32, 48] [])
    aie.objectfifo @a0(%a0, {%ab0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab0(%ab0, {%b0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%b0, {%bc0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc0(%bc0, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a0] -> [@ab0] ([] [])
    aie.objectfifo.link [@b0] -> [@bc0] ([] [])
    aie.objectfifo @a1(%a1, {%ab1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab1(%ab1, {%b1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%b1, {%bc1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc1(%bc1, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a1] -> [@ab1] ([] [])
    aie.objectfifo.link [@b1] -> [@bc1] ([] [])
    aie.objectfifo @a2(%a2, {%ab2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab2(%ab2, {%b2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%b2, {%bc2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc2(%bc2, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a2] -> [@ab2] ([] [])
    aie.objectfifo.link [@b2] -> [@bc2] ([] [])
    aie.objectfifo @a3(%a3, {%ab3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab3(%ab3, {%b3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%b3, {%bc3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc3(%bc3, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a3] -> [@ab3] ([] [])
    aie.objectfifo.link [@b3] -> [@bc3] ([] [])
    aie.objectfifo @a4(%a4, {%ab4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab4(%ab4, {%b4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b4(%b4, {%bc4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc4(%bc4, {%c4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a4] -> [@ab4] ([] [])
    aie.objectfifo.link [@b4] -> [@bc4] ([] [])
    aie.objectfifo @a5(%a5, {%ab5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab5(%ab5, {%b5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b5(%b5, {%bc5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc5(%bc5, {%c5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a5] -> [@ab5] ([] [])
    aie.objectfifo.link [@b5] -> [@bc5] ([] [])
    aie.objectfifo @a6(%a6, {%ab6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab6(%ab6, {%b6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b6(%b6, {%bc6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc6(%bc6, {%c6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a6] -> [@ab6] ([] [])
    aie.objectfifo.link [@b6] -> [@bc6] ([] [])
    aie.objectfifo @a7(%a7, {%ab7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab7(%ab7, {%b7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b7(%b7, {%bc7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc7(%bc7, {%c7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a7] -> [@ab7] ([] [])
    aie.objectfifo.link [@b7] -> [@bc7] ([] [])
  }
}
