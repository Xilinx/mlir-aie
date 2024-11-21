module {
  aie.device(npu1_4col) {
    func.func private @matmul_scalar_4x2x4_4x8x4_i32_i32(memref<2x4x4x8xi32, 2 : i32>, memref<4x2x8x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>)
    // <trace>
    func.func private @event_0()
    func.func private @event_1()
    func.func private @flush_trace()
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
    } {elf_file = "segment_0_core_0_2.elf", link_with = "mm.o"}
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
    aie.shim_dma_allocation @airMemcpyId12(S2MM, 0, 0)
    memref.global "public" @airMemcpyId12 : memref<16x16xi32, 1 : i32>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<16x16xi32, 1 : i32>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
    memref.global "public" @airMemcpyId5 : memref<16x16xi32, 1 : i32>
    aiex.runtime_sequence(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>, %arg2: memref<16x16xi32>) {     
      // <trace>
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32} // [14:8] reset event: 122(BROADCAST_15)
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 7995392 : ui32} // [22:16] start event: 122(BROADCAST_15)
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32} // packet_type: 0(core), packet_id: 0
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1260527873 : ui32} // events: 0x4B(port0 run) 22(event1) 21(event0) 01(true)
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 757865039 : ui32} // events: 0x2D(lock release) 2C(lock acquire) 1A(lock stall) 4F(port1 run)
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32} // [13:8] port1 MM2S-0+1, [5:0] port0 S2MM-0+1
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0: i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 12 : ui32} 

      aiex.npu.write32 {address = 606208 : ui32, column = 0 : i32, row = 1 : i32, value = 40192 : ui32} // [15:8] reset event: 157(BROADCAST_15)
      aiex.npu.write32 {address = 606416 : ui32, column = 0 : i32, row = 1 : i32, value = 10289152 : ui32} // [23:16] start event: 157(BROADCAST_15)
      aiex.npu.write32 {address = 606420 : ui32, column = 0 : i32, row = 1 : i32, value = 12292 : ui32} // [14:12] packet_type: 3(mem_tile), [4:0] packet_id: 4
      aiex.npu.write32 {address = 606432 : ui32, column = 0 : i32, row = 1 : i32, value = 760239192 : ui32} // events: 0x2D(lock release) 50(port0 run) 0x54(port1 run) 58(port2 run)
      aiex.npu.write32 {address = 606436 : ui32, column = 0 : i32, row = 1 : i32, value = 1549821032 : ui32} // events: 5C(port3 run) 60(port4 run) 64(port5 run) 68(port6 run)
      aiex.npu.write32 {address = 724736 : ui32, column = 0 : i32, row = 1 : i32, value = 33620000 : ui32} // [29:24] port3 MM2S-2, [21:16] port2 MM2S-1, [13:8] port1 MM2S-0, [5:0] port0 S2MM-0
      aiex.npu.write32 {address = 724740: ui32, column = 0 : i32, row = 1 : i32, value = 270595 : ui32} // [21:16] port6 MM2S-4, [13:8] port5 S2MM-1, [5:0] port4 MM2S-3
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 4: i32, packet_type = 3 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 10 : ui32} 
     
      aiex.npu.write32 {address = 212992: ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32} // [14:8] reset event: 127(USER_EVENT_1)
      aiex.npu.write32 {address = 213068: ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32} // [6:0] broadcast 15: 127(USER_EVENT_1)
      aiex.npu.write32 {address = 213000: ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32} // event generate [6:0]: 127(USER_EVENT_1)
      // </trace>
      memref.assume_alignment %arg0, 64 : memref<16x16xi32>
      memref.assume_alignment %arg1, 64 : memref<16x16xi32>
      memref.assume_alignment %arg2, 64 : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (0, 0, %arg1[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (0, 0, %arg2[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 2 : i64, metadata = @airMemcpyId12, issue_token = true} : memref<16x16xi32>
      aiex.npu.dma_wait { symbol = @airMemcpyId12}
    }
  } {sym_name = "segment_0"}
}
