module {
  aie.device(npu1_1col) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in_fwd(%mem_tile_0_1, {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 1 : i32) {runtimeDMAs = 0 : i32} : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])

    
    aie.packet_flow(0x1) {
      aie.packet_source<%mem_tile_0_1, "TileControl" : 0>
      aie.packet_dest<%shim_noc_tile_0_0, "South" : 0>
    }
    
    aiex.runtime_sequence @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in_fwd {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)

      // BD0, DMA_S2MM_0_Start_Queue
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 1 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.maskwrite32 {address = 0xa0600 : ui32, column = 0 : i32, row = 1 : i32, mask = 0x00000F00 : ui32, value = 0x100 : ui32}
      aiex.npu.write32 {address = 0xa0604 : ui32, column = 0 : i32, row = 1 : i32, value = 0x80000000 : ui32}

      // sync with the copy into memtile before starting copy out of memtile
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 1 : i32, row_num = 1 : i32}

      // BD1, DMA_MM2S_0_Start_Queue
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 0xa0634 : ui32, column = 0 : i32, row = 1 : i32, value = 1 : ui32}

      // sync with the copy out via shimdma
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
 

