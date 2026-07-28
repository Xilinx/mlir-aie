//===- push_to_queue.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s
// CHECK-DAG: %[[V0:.*]] = arith.constant -2147483645 : i32
// CHECK-DAG: %[[A0:.*]] = arith.constant 119308 : i32
// CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
// CHECK-DAG: %[[V1:.*]] = arith.constant 196610 : i32
// CHECK-DAG: %[[A1:.*]] = arith.constant 67228180 : i32
// CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32

module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      %rc0 = arith.constant 0 : i32
      %bd0 = arith.constant 3 : i32
      aiex.npu.push_queue (0, 0, S2MM:1) bd_id %bd0 repeat %rc0 {issue_token = true} : i32, i32
      %rc1 = arith.constant 3 : i32
      %bd1 = arith.constant 2 : i32
      aiex.npu.push_queue (2, 0, MM2S:0) bd_id %bd1 repeat %rc1 {issue_token = false} : i32, i32
    }
  }
}

// -----

// A runtime (SSA) bd_id is no longer rejected: the command word is assembled
// with arith (bd_id & 0xF, or'd with the issue-token bit and the shifted
// repeat_count) instead of folded to a constant.
// CHECK-LABEL: @rt_bd_id
// CHECK: %[[BD:.*]] = arith.andi %arg0, %{{.*}} : i32
// CHECK: %[[CMD:.*]] = arith.ori %{{.*}}, %[[BD]] : i32
// CHECK: %[[CMD2:.*]] = arith.ori %[[CMD]], %{{.*}} : i32
// CHECK: aiex.npu.write32(%{{.*}}, %[[CMD2]])
module {
  aie.device(npu1) {
    aie.runtime_sequence @rt_bd_id(%arg0: i32) {
      %rc0 = arith.constant 0 : i32
      aiex.npu.push_queue (0, 0, S2MM:1) bd_id %arg0 repeat %rc0 {issue_token = true} : i32, i32
    }
  }
}

// -----

// A mem tile has a 6-bit START_BD_ID field (48 BDs), so a head bd_id >= 16 must
// survive the push: the command word keeps bd_id 20 (0x14); a flat 4-bit mask
// would give 20 & 0xF = 4. (The bd_id operand and the command word are both the
// constant 20; match the second, which feeds the write32.)
// CHECK-LABEL: @mem_tile_bd_id
// CHECK: arith.constant 20 : i32
// CHECK: %[[CMD:.*]] = arith.constant 20 : i32
// CHECK: aiex.npu.write32(%{{.*}}, %[[CMD]])
module {
  aie.device(npu1) {
    aie.runtime_sequence @mem_tile_bd_id() {
      %rc = arith.constant 0 : i32
      %bd = arith.constant 20 : i32
      aiex.npu.push_queue (0, 1, S2MM:0) bd_id %bd repeat %rc {issue_token = false} : i32, i32
    }
  }
}
