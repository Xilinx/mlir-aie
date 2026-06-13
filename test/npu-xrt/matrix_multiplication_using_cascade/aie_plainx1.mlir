module {
  aie.device(npu1) {
    func.func private @matmul_scalar_4x2x4_4x8x4_i32_i32(memref<2x4x4x8xi32, 2 : i32>, memref<4x2x8x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) attributes {link_with = "mm.o"}
    // <trace>
    func.func private @event_0() attributes {link_with = "mm.o"}
    func.func private @event_1() attributes {link_with = "mm.o"}
    func.func private @flush_trace() attributes {link_with = "mm.o"}
    // </trace>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %lock_1_1 = aie.lock(%tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_0 = aie.lock(%tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_1 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_2 = aie.lock(%tile_2_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf5 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf5"} : memref<16x16xi32, 1 : i32> 
    %buf4 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf4"} : memref<16x16xi32, 1 : i32> 
    %buf3 = aie.buffer(%tile_2_1) {mem_bank = 0 : i32, sym_name = "buf3"} : memref<16x16xi32, 1 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf2"} : memref<2x4x4x8xi32, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf1"} : memref<4x2x8x4xi32, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf0"} : memref<4x4x4x4xi32, 2 : i32> 
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<2x4x4x8xi32, 2 : i32>, 0, 256)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<4x2x8x4xi32, 2 : i32>, 0, 256)
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<4x4x4x4xi32, 2 : i32>, 0, 256, [<size = 16, stride = 4>, <size = 4, stride = 64>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      // <trace>
      func.call @event_0() : () -> ()
      // </trace>
      scf.for %arg0 = %c0 to %c4 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf0[%arg0, %arg1, %arg2, %arg3] : memref<4x4x4x4xi32, 2 : i32>
            }
          }
        }
      }
      // <trace>
      func.call @event_1() : () -> ()
      // </trace>
      func.call @matmul_scalar_4x2x4_4x8x4_i32_i32(%buf2, %buf1, %buf0) : (memref<2x4x4x8xi32, 2 : i32>, memref<4x2x8x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_7, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      // <trace>
      func.call @flush_trace() : () -> ()
      // </trace>
      cf.br ^bb1
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_1_1, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_0_0, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_2_1, DMA : 0)
    // <trace>
    aie.packet_flow(0) { 
      aie.packet_source<%tile_0_2, Trace : 0> 
      aie.packet_dest<%tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(4) { 
      aie.packet_source<%tile_0_1, Trace : 0> 
      aie.packet_dest<%tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    // </trace>
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<16x16xi32, 1 : i32>, 0, 256)
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<16x16xi32, 1 : i32>, 0, 256)
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<16x16xi32, 1 : i32>, 0, 256)
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<16x16xi32, 1 : i32>, 0, 256, [<size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<16x16xi32, 1 : i32>, 0, 256)
      aie.use_lock(%lock_1_1_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<16x16xi32, 1 : i32>, 0, 256, [<size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>])
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    }
    aie.shim_dma_allocation @airMemcpyId12 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId4 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId5 (%tile_0_0, MM2S, 1)
    aie.runtime_sequence(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>, %arg2: memref<16x16xi32>) {     
      // <trace>
      %w32_addr_c0 = arith.constant 212992 : i32
      %w32_val_c0 = arith.constant 31232 : i32
      aiex.npu.write32(%w32_addr_c0, %w32_val_c0) {column = 0 : i32, row = 2 : i32} : i32, i32 // [14:8] reset event: 122(BROADCAST_15)
      %w32_addr_c1 = arith.constant 213200 : i32
      %w32_val_c1 = arith.constant 7995392 : i32
      aiex.npu.write32(%w32_addr_c1, %w32_val_c1) {column = 0 : i32, row = 2 : i32} : i32, i32 // [22:16] start event: 122(BROADCAST_15)
      %w32_addr_c2 = arith.constant 213204 : i32
      %w32_val_c2 = arith.constant 0 : i32
      aiex.npu.write32(%w32_addr_c2, %w32_val_c2) {column = 0 : i32, row = 2 : i32} : i32, i32 // packet_type: 0(core), packet_id: 0
      %w32_addr_c3 = arith.constant 213216 : i32
      %w32_val_c3 = arith.constant 1260527873 : i32
      aiex.npu.write32(%w32_addr_c3, %w32_val_c3) {column = 0 : i32, row = 2 : i32} : i32, i32 // events: 0x4B(port0 run) 22(event1) 21(event0) 01(true)
      %w32_addr_c4 = arith.constant 213220 : i32
      %w32_val_c4 = arith.constant 757865039 : i32
      aiex.npu.write32(%w32_addr_c4, %w32_val_c4) {column = 0 : i32, row = 2 : i32} : i32, i32 // events: 0x2D(lock release) 2C(lock acquire) 1A(lock stall) 4F(port1 run)
      %w32_addr_c5 = arith.constant 261888 : i32
      %w32_val_c5 = arith.constant 289 : i32
      aiex.npu.write32(%w32_addr_c5, %w32_val_c5) {column = 0 : i32, row = 2 : i32} : i32, i32 // [13:8] port1 MM2S-0+1, [5:0] port0 S2MM-0+1
      %w32_addr = arith.constant 261892 : i32
      %w32_val = arith.constant 0 : i32
      aiex.npu.write32(%w32_addr, %w32_val) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0: i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %w32_addr_1 = arith.constant 119308 : i32
      %w32_val_1 = arith.constant 12 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 0 : i32, row = 0 : i32} : i32, i32

      %w32_addr_c6 = arith.constant 606208 : i32
      %w32_val_c6 = arith.constant 40192 : i32
      aiex.npu.write32(%w32_addr_c6, %w32_val_c6) {column = 0 : i32, row = 1 : i32} : i32, i32 // [15:8] reset event: 157(BROADCAST_15)
      %w32_addr_c7 = arith.constant 606416 : i32
      %w32_val_c7 = arith.constant 10289152 : i32
      aiex.npu.write32(%w32_addr_c7, %w32_val_c7) {column = 0 : i32, row = 1 : i32} : i32, i32 // [23:16] start event: 157(BROADCAST_15)
      %w32_addr_c8 = arith.constant 606420 : i32
      %w32_val_c8 = arith.constant 12292 : i32
      aiex.npu.write32(%w32_addr_c8, %w32_val_c8) {column = 0 : i32, row = 1 : i32} : i32, i32 // [14:12] packet_type: 3(mem_tile), [4:0] packet_id: 4
      %w32_addr_c9 = arith.constant 606432 : i32
      %w32_val_c9 = arith.constant 760239192 : i32
      aiex.npu.write32(%w32_addr_c9, %w32_val_c9) {column = 0 : i32, row = 1 : i32} : i32, i32 // events: 0x2D(lock release) 50(port0 run) 0x54(port1 run) 58(port2 run)
      %w32_addr_c10 = arith.constant 606436 : i32
      %w32_val_c10 = arith.constant 1549821032 : i32
      aiex.npu.write32(%w32_addr_c10, %w32_val_c10) {column = 0 : i32, row = 1 : i32} : i32, i32 // events: 5C(port3 run) 60(port4 run) 64(port5 run) 68(port6 run)
      %w32_addr_c11 = arith.constant 724736 : i32
      %w32_val_c11 = arith.constant 33620000 : i32
      aiex.npu.write32(%w32_addr_c11, %w32_val_c11) {column = 0 : i32, row = 1 : i32} : i32, i32 // [29:24] port3 MM2S-2, [21:16] port2 MM2S-1, [13:8] port1 MM2S-0, [5:0] port0 S2MM-0
      %w32_addr_c12 = arith.constant 724740 : i32
      %w32_val_c12 = arith.constant 270595 : i32
      aiex.npu.write32(%w32_addr_c12, %w32_val_c12) {column = 0 : i32, row = 1 : i32} : i32, i32 // [21:16] port6 MM2S-4, [13:8] port5 S2MM-1, [5:0] port4 MM2S-3
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 4: i32, packet_type = 3 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %w32_addr_2 = arith.constant 119308 : i32
      %w32_val_2 = arith.constant 10 : i32
      aiex.npu.write32(%w32_addr_2, %w32_val_2) {column = 0 : i32, row = 0 : i32} : i32, i32
     
      %w32_addr_c13 = arith.constant 212992 : i32
      %w32_val_c13 = arith.constant 32512 : i32
      aiex.npu.write32(%w32_addr_c13, %w32_val_c13) {column = 0 : i32, row = 0 : i32} : i32, i32 // [14:8] reset event: 127(USER_EVENT_1)
      %w32_addr_c14 = arith.constant 213068 : i32
      %w32_val_c14 = arith.constant 127 : i32
      aiex.npu.write32(%w32_addr_c14, %w32_val_c14) {column = 0 : i32, row = 0 : i32} : i32, i32 // [6:0] broadcast 15: 127(USER_EVENT_1)
      %w32_addr_c15 = arith.constant 213000 : i32
      %w32_val_c15 = arith.constant 127 : i32
      aiex.npu.write32(%w32_addr_c15, %w32_val_c15) {column = 0 : i32, row = 0 : i32} : i32, i32 // event generate [6:0]: 127(USER_EVENT_1)
      // </trace>
      memref.assume_alignment %arg0, 64 : memref<16x16xi32>
      memref.assume_alignment %arg1, 64 : memref<16x16xi32>
      memref.assume_alignment %arg2, 64 : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 2 : i64, metadata = @airMemcpyId12, issue_token = true} : memref<16x16xi32>
      aiex.npu.dma_wait { symbol = @airMemcpyId12}
    }
  } {sym_name = "segment_0"}
}
