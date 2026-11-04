// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module {
  aie.device(NPUDEVICE) @main {
    aie.runtime_sequence @sequence(%readback : memref<8xi32>, %output : memref<8xi32>) {
      aiex.configure @design {
        aiex.run @run(%readback, %output) : (memref<8xi32>, memref<8xi32>)
      }
    }
  }
  aie.device(NPUDEVICE) @design {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2)

    %input_lock0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "input_lock0"}
    %input_lock2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "input_lock2"}
    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %input_buffer = aie.buffer(%tile_0_2) {sym_name = "input_buffer"} : memref<8xi32>
    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>
    // Control read requests address this buffer at tile-relative address 0x440.
    %other_buffer = aie.buffer(%tile_0_2) {sym_name = "other_buffer", address = 0x440 : i32} : memref<8xi32>

    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, TileControl : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_buffer[%arg1] : memref<8xi32>
        memref.store %c1_i32, %other_buffer[%arg1] : memref<8xi32>
      }
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%input_lock0, AcquireGreaterEqual, %c1_ul1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%input_lock0, AcquireGreaterEqual, %c1_ul2)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%input_lock2, AcquireGreaterEqual, %c1_ul3)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%input_lock2, AcquireGreaterEqual, %c1_ul4)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%output_lock5, AcquireGreaterEqual, %c1_ul5)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_buffer[%arg1] : memref<8xi32>
            %2 = arith.addi %1, %c1_i32 : i32
            memref.store %2, %other_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%output_lock4, Release, %c1_ul6)
      }
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%output_lock4, AcquireGreaterEqual, %c1_ul7)
      aie.dma_bd(%output_buffer : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%output_lock5, Release, %c1_ul8)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)

    aie.runtime_sequence @run(%readback : memref<8xi32>, %output : memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%readback[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<8xi32>

      // Tile (0, 2), lock 0 and lock 2: two increments each.
      aiex.control_packet {address = 0x21f000 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 0x21f020 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_wait {symbol = @out0}

      // Response ID 2 selects the manual TileControl-to-shim DMA route.
      aiex.control_packet {address = 0x200440 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
      aiex.control_packet {address = 0x200450 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
      aiex.npu.dma_wait {symbol = @ctrl0}
    }
  }
}
