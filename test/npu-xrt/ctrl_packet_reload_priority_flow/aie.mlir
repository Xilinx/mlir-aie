//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Loads by control packets a design whose data travels on prioritized packet
// flows, one of them from the shim DMA channel the control overlay also sends
// control packets from. The reload keeps only the overlay's switch settings,
// so it must configure these flows itself (#3837). Core (0,2) adds 2.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg : memref<16xi32>) {
      aiex.configure @pd {
        aiex.run @pd_sequence (%arg) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu2) @pd {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.buffer @bin(%t02) : memref<16xi32>
    aie.buffer @bout(%t02) : memref<16xi32>
    aie.lock @in_prod(%t02, 0) {init = 1 : i32}
    aie.lock @in_cons(%t02, 1) {init = 0 : i32}
    aie.lock @out_prod(%t02, 2) {init = 1 : i32}
    aie.lock @out_cons(%t02, 3) {init = 0 : i32}
    aie.packet_flow(5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    } {priority_route = true}
    aie.packet_flow(6) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    } {priority_route = true}
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %k = arith.constant 2 : i32
      %one = arith.constant 1 : i32
      aie.use_lock(@in_cons, AcquireGreaterEqual, %one)
      aie.use_lock(@out_prod, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c16 step %c1 {
        %bin = aie.get_buffer @bin : memref<16xi32>
        %x = memref.load %bin[%i] : memref<16xi32>
        %y = arith.addi %x, %k : i32
        %bout = aie.get_buffer @bout : memref<16xi32>
        memref.store %y, %bout[%i] : memref<16xi32>
      }
      aie.use_lock(@in_prod, Release, %one)
      aie.use_lock(@out_cons, Release, %one)
      aie.end
    }
    aie.mem(%t02) {
      aie.dma(S2MM, 0) [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(@in_prod, AcquireGreaterEqual, %c1)
        %bin_1 = aie.get_buffer @bin : memref<16xi32>
        aie.dma_bd(%bin_1 : memref<16xi32>)
        aie.use_lock(@in_cons, Release, %c1)
      }]
      aie.dma(MM2S, 0) [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(@out_cons, AcquireGreaterEqual, %c1)
        %bout_1 = aie.get_buffer @bout : memref<16xi32>
        aie.dma_bd(%bout_1 : memref<16xi32>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
        aie.use_lock(@out_prod, Release, %c1)
      }]
      aie.end
    }
    aie.shim_dma_allocation @pin (%t00, MM2S, 0)
    aie.shim_dma_allocation @pout (%t00, S2MM, 0)
    aie.runtime_sequence @pd_sequence(%a : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd (%a[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1], packet = <pkt_id = 5, pkt_type = 0>) {id = 0 : i64, metadata = @pin} : memref<16xi32>
      aiex.npu.dma_memcpy_nd (%a[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1]) {id = 1 : i64, metadata = @pout, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @pout}
    }
  }
}
