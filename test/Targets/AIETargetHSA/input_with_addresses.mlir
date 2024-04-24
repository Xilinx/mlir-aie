
// RUN: aie-translate --aie-generate-hsa %s | FileCheck %s

//CHECK: void invoke_data_movement(hsa_queue_t *q, hsa_agent_t *a, void *buf2, void *buf0) {
//CHECK: 	uint64_t wr_idx = 0;
//CHECK: 	uint64_t packet_id = 0;
//CHECK: 	hsa_agent_dispatch_packet_t pkt0 ;
//CHECK: 	wr_idx  = hsa_queue_add_write_index_relaxed(q, 1);
//CHECK: 	packet_id  = wr_idx % q->size;
//CHECK: 	mlir_aie_packet_nd_memcpy(&pkt0, 0 /* herd_id */, 6 /* col */, 0 /* dir */, 0/* channel */, 4 /* Burst length */, 2 /* Memory space */, (uint64_t)buf2 + 0 /* Address */, 256 /* 1d_length */, 1 /* 2d_length */, 0 /* 2d_stride */, 1 /* 3d_length */, 0 /* 3d_stride */ , 1 /* 4d_length */, 0 /* 4d_stride */);
//CHECK: 	hsa_amd_signal_create_on_agent(1, 0, nullptr, a, 0, &pkt0.completion_signal);
//CHECK: 	mlir_aie_write_pkt<hsa_agent_dispatch_packet_t>(q, packet_id, &pkt0);
//CHECK: 	hsa_agent_dispatch_packet_t pkt1 ;
//CHECK: 	wr_idx  = hsa_queue_add_write_index_relaxed(q, 1);
//CHECK: 	packet_id  = wr_idx % q->size;
//CHECK: 	mlir_aie_packet_nd_memcpy(&pkt1, 0 /* herd_id */, 6 /* col */, 1 /* dir */, 0/* channel */, 4 /* Burst length */, 2 /* Memory space */, (uint64_t)buf0 + 0 /* Address */, 256 /* 1d_length */, 1 /* 2d_length */, 0 /* 2d_stride */, 1 /* 3d_length */, 0 /* 3d_stride */ , 1 /* 4d_length */, 0 /* 4d_stride */);
//CHECK: 	mlir_aie_queue_dispatch_and_wait(a, q, packet_id, wr_idx, &pkt1, false);
//CHECK: 	while (hsa_signal_wait_scacquire(pkt0.completion_signal,
//CHECK: 	HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
//CHECK: 	HSA_WAIT_STATE_ACTIVE) != 0);
//CHECK: 	while (hsa_signal_wait_scacquire(pkt1.completion_signal,
//CHECK: 	HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
//CHECK: 	HSA_WAIT_STATE_ACTIVE) != 0);
//CHECK: 	hsa_signal_destroy(pkt0.completion_signal);
//CHECK: 	hsa_signal_destroy(pkt1.completion_signal);
//CHECK: }

module {
  aie.device(xcvc1902) {
    memref.global "public" @out0 : memref<16xi32>
    memref.global "public" @in0 : memref<16xi32>
    %tile_6_0 = aie.tile(6, 0)
    %switchbox_6_0 = aie.switchbox(%tile_6_0) {
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
    }

    aie.flow(%tile_6_0, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %tile_6_0, DMA : 0)
    %core_6_2 = aie.core(%tile_6_2) {
      aie.end
    }

    aie.shim_dma_allocation @in0(MM2S, 0, 6)
    aie.shim_dma_allocation @out0(S2MM, 0, 6)

    func.func @sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 0 : i64, metadata = @out0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 1 : i64, metadata = @in0} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}
