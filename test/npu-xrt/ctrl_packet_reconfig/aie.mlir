module {
  aie.device(npu1_1col) {
    memref.global "public" @ctrlpkt0 : memref<1024xi32>
    memref.global "public" @objFifo_out0 : memref<64x64xi8>
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}

    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 1>
    }
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 0>
    }
    aie.packet_flow(4) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true}
    // TODO: make shim tile ctrl packet flow part of the column control overlay
    // aie.packet_flow(4) {
    //   aie.packet_source<%tile_0_0, DMA : 0>
    //   aie.packet_dest<%tile_0_0, Ctrl : 0>
    // } {keep_pkt_header = true}
    aie.packet_flow(5) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, Ctrl : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, Ctrl : 0>
    } {keep_pkt_header = true}
    aie.shim_dma_allocation @ctrlpkt0(MM2S, 0, 0)
    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)
    aiex.runtime_sequence(%arg0: memref<64x64xi8>, %arg2: memref<64x64xi8>, %arg3: memref<1024xi32>) {

      // Reset core (0,2)
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 0][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Reset DMA channels (leads to deadlock)
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 2][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 4][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 6][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 8][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // Load core tile (0,2) program memory
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 10][1, 1, 1, 4][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 14][1, 1, 1, 4][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 18][1, 1, 1, 4][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 22][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 27][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 32][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 37][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 42][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 47][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 52][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 57][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 62][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 67][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 72][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 77][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 82][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 87][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 92][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 97][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 102][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 107][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 112][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 117][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 122][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 127][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 132][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 137][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 142][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 147][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 152][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 157][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 162][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 167][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 172][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 177][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 182][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 187][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 192][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 197][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 202][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 207][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 212][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 217][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 222][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 227][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 232][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 237][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 242][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 247][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 252][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 257][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 262][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 267][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 272][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 277][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 282][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 287][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 292][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 297][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 302][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 307][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 312][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 317][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 322][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 327][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 332][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 337][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 339][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 341][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 343][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 345][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 347][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      
      // Core tile (0,2) locks
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 349][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 351][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 353][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 355][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 357][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 359][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 361][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 363][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 365][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 367][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 369][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 371][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 373][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 375][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 377][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 379][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 381][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 383][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 385][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 387][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // memtile (0,1) locks
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 389][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 391][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 393][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 395][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // core tile bds
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 397][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 402][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 405][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 410][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 413][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 415][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 417][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 419][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // memtile bds
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 421][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 426][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 431][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 436][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 441][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 446][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 451][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 456][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 461][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 463][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 465][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 467][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 469][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 471][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 473][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 475][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // shim tile (0,0) bds
      // TODO: make shim tile ctrl packet flow part of the column control overlay
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 477][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 479][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 481][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 483][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 485][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 487][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 489][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 491][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 493][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // memtile stream switches
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 495][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 497][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 499][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 501][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 503][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 505][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 507][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 509][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 511][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 513][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 515][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 517][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 519][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 521][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 523][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 525][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 527][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 529][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // core tile stream switches
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 531][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 533][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 535][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 537][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 539][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 541][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 543][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 545][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 547][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // shim tile stream switches
      // TODO: make shim tile ctrl packet flow part of the column control overlay
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 549][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      // aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 551][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 4>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      // aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 553][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 6>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // AIE design's instructions
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c56_i64 = arith.constant 56 : i64
      %c61_i64 = arith.constant 61 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64], packet = <pkt_id = 3, pkt_type = 1>) {id = 0 : i64, metadata = @ctrlpkt0} : memref<64x64xi8>
      aiex.npu.dma_memcpy_nd (0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }

    }
    
  }
}

