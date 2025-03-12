module {
  aie.device(npu1_1col) {
    memref.global "public" @outC_cons : memref<64x96xi16>
    memref.global "public" @outC : memref<64x96xi16>
    memref.global "public" @memC_cons : memref<64x96xi16>
    memref.global "public" @memC : memref<64x96xi16>
    memref.global "public" @memB_cons : memref<120x96xi8>
    memref.global "public" @memB : memref<120x96xi8>
    memref.global "public" @inB_cons : memref<120x96xi8>
    memref.global "public" @inB : memref<120x96xi8>
    memref.global "public" @memA_cons : memref<64x120xi8>
    memref.global "public" @memA : memref<64x120xi8>
    memref.global "public" @inA_cons : memref<64x120xi8>
    memref.global "public" @inA : memref<64x120xi8>
    func.func private @zero_i16(memref<64x96xi16>)
    func.func private @matmul_i8_i16(memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %outC_cons_prod_lock = aie.lock(%shim_noc_tile_0_0, 4) {init = 1 : i32, sym_name = "outC_cons_prod_lock"}
    %outC_cons_cons_lock = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock"}
    %memC_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "memC_cons_buff_0"} : memref<64x96xi16> 
    %memC_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 12288 : i32, sym_name = "memC_cons_buff_1"} : memref<64x96xi16> 
    %memC_cons_prod_lock = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "memC_cons_prod_lock"}
    %memC_cons_cons_lock = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "memC_cons_cons_lock"}
    %memC_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "memC_buff_0"} : memref<64x96xi16> 
    %memC_buff_1 = aie.buffer(%tile_0_2) {address = 13312 : i32, sym_name = "memC_buff_1"} : memref<64x96xi16> 
    %memC_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "memC_prod_lock"}
    %memC_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "memC_cons_lock"}
    %memB_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "memB_cons_buff_0"} : memref<120x96xi8> 
    %memB_cons_buff_1 = aie.buffer(%tile_0_2) {address = 37120 : i32, sym_name = "memB_cons_buff_1"} : memref<120x96xi8> 
    %memB_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memB_cons_prod_lock"}
    %memB_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memB_cons_cons_lock"}
    %inB_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 24576 : i32, sym_name = "inB_cons_buff_0"} : memref<120x96xi8> 
    %inB_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 36096 : i32, sym_name = "inB_cons_buff_1"} : memref<120x96xi8> 
    %inB_cons_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
    %inB_cons_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
    %inB_prod_lock = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "inB_prod_lock"}
    %inB_cons_lock = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "inB_cons_lock"}
    %memA_cons_buff_0 = aie.buffer(%tile_0_2) {address = 48640 : i32, sym_name = "memA_cons_buff_0"} : memref<64x120xi8> 
    %memA_cons_buff_1 = aie.buffer(%tile_0_2) {address = 56320 : i32, sym_name = "memA_cons_buff_1"} : memref<64x120xi8> 
    %memA_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memA_cons_prod_lock"}
    %memA_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memA_cons_cons_lock"}
    %inA_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 47616 : i32, sym_name = "inA_cons_buff_0"} : memref<64x120xi8> 
    %inA_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 55296 : i32, sym_name = "inA_cons_buff_1"} : memref<64x120xi8> 
    %inA_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
    %inA_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
    %inA_prod_lock = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "inA_prod_lock"}
    %inA_cons_lock = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "inA_cons_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb11
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c40 = arith.constant 40 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c40 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memC_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC_buff_0) : (memref<64x96xi16>) -> ()
      %c0_2 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_3 = arith.constant 1 : index
      %c2_4 = arith.constant 2 : index
      cf.br ^bb5(%c0_2 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c4 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i16(%memA_cons_buff_0, %memB_cons_buff_0, %memC_buff_0) : (memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i16(%memA_cons_buff_1, %memB_cons_buff_1, %memC_buff_0) : (memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      %6 = arith.addi %4, %c2_4 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%memC_cons_lock, Release, 1)
      aie.use_lock(%memC_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC_buff_1) : (memref<64x96xi16>) -> ()
      %c0_5 = arith.constant 0 : index
      %c4_6 = arith.constant 4 : index
      %c1_7 = arith.constant 1 : index
      %c2_8 = arith.constant 2 : index
      cf.br ^bb8(%c0_5 : index)
    ^bb8(%7: index):  // 2 preds: ^bb7, ^bb9
      %8 = arith.cmpi slt, %7, %c4_6 : index
      cf.cond_br %8, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i16(%memA_cons_buff_0, %memB_cons_buff_0, %memC_buff_1) : (memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i16(%memA_cons_buff_1, %memB_cons_buff_1, %memC_buff_1) : (memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      %9 = arith.addi %7, %c2_8 : index
      cf.br ^bb8(%9 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%memC_cons_lock, Release, 1)
      %10 = arith.addi %2, %c2 : index
      cf.br ^bb3(%10 : index)
    ^bb11:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb12:  // pred: ^bb1
      aie.end
    } {link_with = "mm_64x120x96.o"}
    aiex.runtime_sequence(%arg0: memref<245760xi8>, %arg1: memref<230400xi8>, %arg2: memref<245760xi16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 7995392 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 491520 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 2 : i32, arg_plus = 491520 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 15 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][2, 5, 64, 96][30720, 96, 480, 1]) {id = 0 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][5, 4, 64, 120][0, 120, 480, 1]) {id = 1 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 2 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 30720][5, 4, 64, 120][0, 120, 480, 1]) {id = 3 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 4 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 61440][2, 5, 64, 96][30720, 96, 480, 1]) {id = 8 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 61440][5, 4, 64, 120][0, 120, 480, 1]) {id = 9 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 10 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 92160][5, 4, 64, 120][0, 120, 480, 1]) {id = 11 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 12 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 122880][2, 5, 64, 96][30720, 96, 480, 1]) {id = 0 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 122880][5, 4, 64, 120][0, 120, 480, 1]) {id = 1 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 2 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 153600][5, 4, 64, 120][0, 120, 480, 1]) {id = 3 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 4 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 184320][2, 5, 64, 96][30720, 96, 480, 1]) {id = 8 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 184320][5, 4, 64, 120][0, 120, 480, 1]) {id = 9 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 10 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 215040][5, 4, 64, 120][0, 120, 480, 1]) {id = 11 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 12 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_wait {symbol = @outC}
    }
    aie.shim_dma_allocation @inA(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<64x120xi8>, 0, 7680) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<64x120xi8>, 0, 7680) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<64x120xi8>, 0, 7680, [<size = 16, stride = 480>, <size = 15, stride = 8>, <size = 4, stride = 120>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<64x120xi8>, 0, 7680, [<size = 16, stride = 480>, <size = 15, stride = 8>, <size = 4, stride = 120>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<120x96xi8>, 0, 11520) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<120x96xi8>, 0, 11520) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<120x96xi8>, 0, 11520, [<size = 15, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inB_cons_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<120x96xi8>, 0, 11520, [<size = 15, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inB_cons_prod_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%memC_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_0 : memref<64x96xi16>, 0, 6144) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memC_cons_cons_lock, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%memC_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_1 : memref<64x96xi16>, 0, 6144) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memC_cons_cons_lock, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%memC_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_0 : memref<64x96xi16>, 0, 6144, [<size = 16, stride = 384>, <size = 4, stride = 8>, <size = 12, stride = 32>, <size = 8, stride = 1>]) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%memC_cons_prod_lock, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%memC_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_1 : memref<64x96xi16>, 0, 6144, [<size = 16, stride = 384>, <size = 4, stride = 8>, <size = 12, stride = 32>, <size = 8, stride = 1>]) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%memC_cons_prod_lock, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA_cons_buff_0 : memref<64x120xi8>, 0, 7680) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA_cons_buff_1 : memref<64x120xi8>, 0, 7680) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_cons_buff_0 : memref<120x96xi8>, 0, 11520) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_cons_buff_1 : memref<120x96xi8>, 0, 11520) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_buff_0 : memref<64x96xi16>, 0, 6144) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memC_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_buff_1 : memref<64x96xi16>, 0, 6144) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memC_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @inB(MM2S, 1, 0)
    aie.shim_dma_allocation @outC(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
