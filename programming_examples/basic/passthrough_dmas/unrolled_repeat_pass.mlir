module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) {runtimeDmas = true} : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo @in_fwd(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) {runtimeDmas = true} : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])
    aiex.runtime_sequence @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 1024 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 64 : i32, lock_acq_val = 127 : i32, lock_rel_id = 65 : i32, lock_rel_val = 1 : i32, next_bd = 1 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 656900 : ui32, column = 0 : i32, row = 1 : i32, value = 0 : ui32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 1024 : i32, buffer_offset = 1024 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 64 : i32, lock_acq_val = 127 : i32, lock_rel_id = 65 : i32, lock_rel_val = 1 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 656904 : ui32, column = 0 : i32, row = 1 : i32, value = 0 : ui32}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 1024 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 64 : i32, lock_acq_val = 127 : i32, lock_rel_id = 65 : i32, lock_rel_val = 1 : i32, next_bd = 1 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 656900 : ui32, column = 0 : i32, row = 1 : i32, value = 0 : ui32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 1024 : i32, buffer_offset = 1024 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 64 : i32, lock_acq_val = 127 : i32, lock_rel_id = 65 : i32, lock_rel_val = 1 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 656904 : ui32, column = 0 : i32, row = 1 : i32, value = 0 : ui32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 1024 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 65 : i32, lock_acq_val = 127 : i32, lock_rel_id = 64 : i32, lock_rel_val = 1 : i32, next_bd = 3 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 656948 : ui32, column = 0 : i32, row = 1 : i32, value = 2 : ui32}
      aiex.npu.writebd {bd_id = 3 : i32, buffer_length = 1024 : i32, buffer_offset = 1024 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 65 : i32, lock_acq_val = 127 : i32, lock_rel_id = 64 : i32, lock_rel_val = 1 : i32, next_bd = 2 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      aiex.npu.write32 {address = 656948 : ui32, column = 0 : i32, row = 1 : i32, value = 3 : ui32}
      %1 = aiex.dma_configure_task_for @in_fwd {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

