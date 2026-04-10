//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// E2E test: Shim MM2S BD reuse with >16 fire-and-forget tasks.
//
// The host fires 20 MM2S tasks on shim tile_0_0 channel 0 (exceeds 16-BD
// limit), alternating between two source buffers (buf_a, buf_b). After
// every 8 tasks, the host awaits the last issued MM2S task to reclaim
// BDs. The core tile has a pre-configured looping S2MM→MM2S passthrough.
// The shim S2MM side receives all 20 transfers into the output buffer.
//
// This matches the kv_cache_prefill pattern where VIn sends hundreds of
// MM2S tasks (LUT + V interleaved) without any S2MM-side await.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    %core_buf = aie.buffer(%tile_0_2) {sym_name = "core_buf"} : memref<256xi32>

    // Input path: shim MM2S ch0 → core S2MM ch0
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    // Output path: core MM2S ch0 → shim S2MM ch0
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    // Packet flows for issue_token on shim MM2S
    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_0, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    }

    // Core tile: continuously looping S2MM→MM2S passthrough
    %lock_in = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "lock_in"}
    %lock_out = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "lock_out"}

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)
    ^s2mm:
      aie.use_lock(%lock_in, AcquireGreaterEqual, 1)
      aie.dma_bd(%core_buf : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_out, Release, 1)
      aie.next_bd ^s2mm
    ^mm2s_entry:
      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)
    ^mm2s:
      aie.use_lock(%lock_out, AcquireGreaterEqual, 1)
      aie.dma_bd(%core_buf : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_in, Release, 1)
      aie.next_bd ^mm2s
    ^end:
      aie.end
    }

    // buf_a: 2560 i32 (10 slices of 256 — even-numbered transfers)
    // buf_b: 2560 i32 (10 slices of 256 — odd-numbered transfers)
    // output: 5120 i32 (20 slices of 256)
    aie.runtime_sequence(%buf_a: memref<2560xi32>, %buf_b: memref<2560xi32>, %output: memref<5120xi32>) {

      // Pre-configure shim S2MM ch0: looping BD that receives all 20
      // transfers into output[0..5119]. Uses repeat_count to avoid
      // consuming shim BDs for the receive side.
      %recv = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%output : memref<5120xi32>, 0, 5120, [<size = 20, stride = 256>, <size = 256, stride = 1>]) {bd_id = 8 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%recv)

      // ===== Batch 0: MM2S tasks 0-7, BD IDs 0-7 =====
      // Pattern: a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]

      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 0, 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)

      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 0, 256) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)

      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 256, 256) {bd_id = 2 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)

      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 256, 256) {bd_id = 3 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)

      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 512, 256) {bd_id = 4 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)

      %t5 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 512, 256) {bd_id = 5 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)

      %t6 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 768, 256) {bd_id = 6 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)

      %t7 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 768, 256) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t7)

      // Await last MM2S task of batch 0, then free all to reclaim BD IDs
      aiex.dma_await_task(%t7)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
      aiex.dma_free_task(%t4)
      aiex.dma_free_task(%t5)
      aiex.dma_free_task(%t6)

      // ===== Batch 1: MM2S tasks 8-15, BD IDs 0-7 reused =====

      %t8 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 1024, 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t8)

      %t9 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 1024, 256) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t9)

      %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 1280, 256) {bd_id = 2 : i32}
        aie.end
      }
      aiex.dma_start_task(%t10)

      %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 1280, 256) {bd_id = 3 : i32}
        aie.end
      }
      aiex.dma_start_task(%t11)

      %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 1536, 256) {bd_id = 4 : i32}
        aie.end
      }
      aiex.dma_start_task(%t12)

      %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 1536, 256) {bd_id = 5 : i32}
        aie.end
      }
      aiex.dma_start_task(%t13)

      %t14 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 1792, 256) {bd_id = 6 : i32}
        aie.end
      }
      aiex.dma_start_task(%t14)

      %t15 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 1792, 256) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t15)

      // Await + free batch 1
      aiex.dma_await_task(%t15)
      aiex.dma_free_task(%t8)
      aiex.dma_free_task(%t9)
      aiex.dma_free_task(%t10)
      aiex.dma_free_task(%t11)
      aiex.dma_free_task(%t12)
      aiex.dma_free_task(%t13)
      aiex.dma_free_task(%t14)

      // ===== Batch 2: MM2S tasks 16-19, BD IDs 0-3 reused again =====

      %t16 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 2048, 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t16)

      %t17 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 2048, 256) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t17)

      %t18 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<2560xi32>, 2304, 256) {bd_id = 2 : i32}
        aie.end
      }
      aiex.dma_start_task(%t18)

      %t19 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<2560xi32>, 2304, 256) {bd_id = 3 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t19)

      // Wait for all MM2S to complete, then wait for S2MM output
      aiex.dma_await_task(%t19)
      aiex.dma_await_task(%recv)
    }
  }
}
