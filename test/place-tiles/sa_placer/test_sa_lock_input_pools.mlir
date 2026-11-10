//===- test_sa_lock_input_pools.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s | aie-opt --aie-objectfifo-allocate | FileCheck %s --check-prefix=ALLOC

// Pools already in the input take their locks as aie-objectfifo-allocate
// gives them: one group per segment, used by the endpoints selecting it.
// CHECK: 3 sa-lock-groups
// ALLOC-COUNT-3: __prodlock{{[0-9]+}} = aie.lock
// ALLOC-NOT: __prodlock{{[0-9]+}} = aie.lock

module @input_pools {
  aie.device(npu1) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo.pool @p(%core) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @q(%mem) {depth = 2 : i32} : memref<32xi32> {
      aie.objectfifo.segment @t0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @t1 {offset = 16 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @p_dma(%core) fills @p
    aie.objectfifo.core_endpoint @p_core(%core) drains @p
    aie.objectfifo.dma_endpoint @q_in(%mem) fills @q {segments = [@t0, @t1]}
    aie.objectfifo.dma_endpoint @q_out0(%mem) drains @q {segments = [@t0]}
    aie.objectfifo.dma_endpoint @q_out1(%mem) drains @q {segments = [@t1]}
    aie.route from @q_out0 to [@p_dma]
  }
}
