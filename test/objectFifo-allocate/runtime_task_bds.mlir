//===- runtime_task_bds.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-lock-ids,aie-reserve-runtime-bd-ids,aie-assign-bd-ids))' %s | FileCheck %s --check-prefix=IDS

// A runtime task on mem tile channel S2MM 2 pins BD ids 0-7, in the even
// channels' half of 24. The link's two ends need 10 BDs each. Channels are
// chosen with the runtime tasks' BDs counted, so the second end takes channel
// 1, in the other half, and aie-assign-bd-ids keeps clear of the pinned ids.
// Without them both ends took channel 0: 20 BDs in the 16 ids left.
// CHECK: aie.objectfifo.dma_endpoint @in__cons__dma(%{{.*}}) fills @in__cons__pool {channelIndex = 0 : i32
// CHECK: aie.objectfifo.dma_endpoint @out__prod__dma(%{{.*}}) drains @in__cons__pool {channelIndex = 1 : i32
// IDS:   aie.dma_start(S2MM, 0,
// IDS:   aie.dma_start(MM2S, 1,
module @runtime_task_pins_low_half {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %scratch = aie.buffer(%mem) {sym_name = "scratch"} : memref<64xi32>
    aie.objectfifo @in(%shim, {%mem}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%mem, {%core}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<64xi32>) {
      %t = aiex.dma_configure_task(%mem, S2MM, 2) {
        aie.dma_bd(%scratch : memref<64xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%scratch : memref<64xi32> offset = 8 len = 8) {bd_id = 1 : i32}
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%scratch : memref<64xi32> offset = 16 len = 8) {bd_id = 2 : i32}
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%scratch : memref<64xi32> offset = 24 len = 8) {bd_id = 3 : i32}
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%scratch : memref<64xi32> offset = 32 len = 8) {bd_id = 4 : i32}
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%scratch : memref<64xi32> offset = 40 len = 8) {bd_id = 5 : i32}
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%scratch : memref<64xi32> offset = 48 len = 8) {bd_id = 6 : i32}
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%scratch : memref<64xi32> offset = 56 len = 8) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
