//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.
//
//===----------------------------------------------------------------------===//

// Verifies that:
//  - --aie-create-cores forwards the source func.call op's location to the
//    synthesized aie.tile / aie.buffer / aie.mem / aie.core / aie.end ops.
//  - --aie-lower-memcpy forwards the source aiex.memcpy op's location to
//    the synthesized aie.flow / aie.dma_start / aie.dma_bd / aiex.useToken
//    / aie.next_bd ops.

// RUN: aie-opt --aie-create-cores --aie-lower-memcpy --mlir-print-debuginfo %s | FileCheck %s

#call_loc = loc("user_design.py":42:4)
#memcpy_loc = loc("user_design.py":50:4)

module @loc_test {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)
  %t22 = aie.tile(2, 2)

  %buf0 = memref.alloc() : memref<256xi32>
  %buf1 = memref.alloc() : memref<256xi32>
  %buf2 = memref.alloc() : memref<256xi32>

  aiex.token(0) { sym_name = "token0" }

  func.func @task0(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    return
  }
  func.func @task1(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    return
  }
  func.func @task2(%arg0: memref<256xi32>) -> () {
    return
  }

  func.call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> () loc(#call_loc)
  aiex.memcpy @token0(1, 2) (%t11 : <%buf0, 0, 256>, %t22 : <%buf1, 0, 256>) : (memref<256xi32>, memref<256xi32>) loc(#memcpy_loc)
  func.call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
  func.call @task2(%buf2) { aie.x = 3, aie.y = 3 } : (memref<256xi32>) -> () loc(#call_loc)
 }
}

// AIECreateCores: synthesized tile / core / end / mem / buffer for the call
// op carry
// the call's loc.
// CHECK-DAG: aie.tile({{.*}}) loc(#[[CALLLOC:loc[0-9]*]])
// CHECK-DAG: aie.mem({{.*}})
// CHECK-DAG: aie.core({{.*}}) {{[{][[:space:]]*$}}
// CHECK-DAG: aie.end loc(#[[CALLLOC]])
// CHECK-DAG: } loc(#[[CALLLOC]])
// CHECK-DAG: aie.buffer({{.*}}) : memref<256xi32> loc(#[[CALLLOC]])

// AIELowerMemcpy: synthesized aie.flow / dma_start / dma_bd / use_token /
// next_bd carry the memcpy's loc.
// CHECK-DAG: aie.flow({{.*}}) loc(#[[MCLOC:loc[0-9]*]])
// CHECK-DAG: aie.dma_start({{.*}}) loc(#[[MCLOC]])
// CHECK-DAG: aie.dma_bd({{.*}}) loc(#[[MCLOC]])
// CHECK-DAG: aie.next_bd {{.*}} loc(#[[MCLOC]])
// CHECK-DAG: aiex.useToken {{.*}} loc(#[[MCLOC]])

// CHECK-DAG: #[[CALLLOC]] = loc("user_design.py":42:4)
// CHECK-DAG: #[[MCLOC]] = loc("user_design.py":50:4)
