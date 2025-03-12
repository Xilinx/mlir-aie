module {
  aie.device(npu1_1col) {
    memref.global "public" @outC_cons : memref<32x32xi32>
    memref.global "public" @outC : memref<32x32xi32>
    memref.global "public" @memC_cons : memref<32x32xi32>
    memref.global "public" @memC : memref<32x32xi32>
    memref.global "public" @memB_cons : memref<32x32xi8>
    memref.global "public" @memB : memref<32x32xi8>
    memref.global "public" @inB_cons : memref<32x32xi8>
    memref.global "public" @inB : memref<32x32xi8>
    memref.global "public" @memA_cons : memref<32x32xi8>
    memref.global "public" @memA : memref<32x32xi8>
    memref.global "public" @inA_cons : memref<32x32xi8>
    memref.global "public" @inA : memref<32x32xi8>
    func.func private @zero_i32(memref<32x32xi32>)
    func.func private @matmul_i8_i32(memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %memC_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "memC_cons_buff_0"} : memref<32x32xi32> 
    %memC_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 4096 : i32, sym_name = "memC_cons_buff_1"} : memref<32x32xi32> 
    %memC_cons_prod_lock = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "memC_cons_prod_lock"}
    %memC_cons_cons_lock = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "memC_cons_cons_lock"}
    %memC_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "memC_buff_0"} : memref<32x32xi32> 
    %memC_buff_1 = aie.buffer(%tile_0_2) {address = 5120 : i32, sym_name = "memC_buff_1"} : memref<32x32xi32> 
    %memC_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "memC_prod_lock"}
    %memC_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "memC_cons_lock"}
    %memB_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "memB_cons_buff_0"} : memref<32x32xi8> 
    %memB_cons_buff_1 = aie.buffer(%tile_0_2) {address = 10240 : i32, sym_name = "memB_cons_buff_1"} : memref<32x32xi8> 
    %memB_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memB_cons_prod_lock"}
    %memB_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memB_cons_cons_lock"}
    %inB_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 8192 : i32, sym_name = "inB_cons_buff_0"} : memref<32x32xi8> 
    %inB_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 9216 : i32, sym_name = "inB_cons_buff_1"} : memref<32x32xi8> 
    %inB_cons_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
    %inB_cons_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
    %memA_cons_buff_0 = aie.buffer(%tile_0_2) {address = 11264 : i32, sym_name = "memA_cons_buff_0"} : memref<32x32xi8> 
    %memA_cons_buff_1 = aie.buffer(%tile_0_2) {address = 12288 : i32, sym_name = "memA_cons_buff_1"} : memref<32x32xi8> 
    %memA_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memA_cons_prod_lock"}
    %memA_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memA_cons_cons_lock"}
    %inA_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 10240 : i32, sym_name = "inA_cons_buff_0"} : memref<32x32xi8> 
    %inA_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 11264 : i32, sym_name = "inA_cons_buff_1"} : memref<32x32xi8> 
    %inA_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
    %inA_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb11
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%memC_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i32(%memC_buff_0) : (memref<32x32xi32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i32(%memA_cons_buff_0, %memB_cons_buff_0, %memC_buff_0) : (memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i32(%memA_cons_buff_1, %memB_cons_buff_1, %memC_buff_0) : (memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%memC_cons_lock, Release, 1)
      aie.use_lock(%memC_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i32(%memC_buff_1) : (memref<32x32xi32>) -> ()
      cf.br ^bb8(%c0 : index)
    ^bb8(%7: index):  // 2 preds: ^bb7, ^bb9
      %8 = arith.cmpi slt, %7, %c8 : index
      cf.cond_br %8, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i32(%memA_cons_buff_0, %memB_cons_buff_0, %memC_buff_1) : (memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      aie.use_lock(%memA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @matmul_i8_i32(%memA_cons_buff_1, %memB_cons_buff_1, %memC_buff_1) : (memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>) -> ()
      aie.use_lock(%memA_cons_prod_lock, Release, 1)
      aie.use_lock(%memB_cons_prod_lock, Release, 1)
      %9 = arith.addi %7, %c2 : index
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
    } {link_with = "mm_32x32x32.o"}
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[8192, 0, 0, 33554432, -2113928961, 31, 1056767, 33554432]>
    memref.global "private" constant @blockwrite_data_1 : memref<8xi32> = dense<[2048, 0, 0, 8388608, -2113929153, 7, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_2 : memref<8xi32> = dense<[2048, 0, 0, 8388608, -2113929153, 2047, 7340039, 33554432]>
    aiex.runtime_sequence(%arg0: memref<65536xi8>, %arg1: memref<65536xi8>, %arg2: memref<65536xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147549184 : ui32}
      %1 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458753 : ui32}
      %2 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%2) {address = 118848 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118852 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458754 : ui32}
      %3 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 118880 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118884 : ui32, arg_idx = 0 : i32, arg_plus = 8192 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458755 : ui32}
      %4 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%4) {address = 118912 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118916 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458756 : ui32}
      %5 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%5) {address = 119040 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119044 : ui32, arg_idx = 2 : i32, arg_plus = 65536 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147549192 : ui32}
      %6 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%6) {address = 119072 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119076 : ui32, arg_idx = 0 : i32, arg_plus = 16384 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458761 : ui32}
      %7 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%7) {address = 119104 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119108 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458762 : ui32}
      %8 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%8) {address = 119136 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119140 : ui32, arg_idx = 0 : i32, arg_plus = 24576 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458763 : ui32}
      %9 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%9) {address = 119168 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119172 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458764 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      %10 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%10) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 131072 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147549184 : ui32}
      %11 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%11) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 32768 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458753 : ui32}
      %12 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%12) {address = 118848 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118852 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458754 : ui32}
      %13 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%13) {address = 118880 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118884 : ui32, arg_idx = 0 : i32, arg_plus = 40960 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458755 : ui32}
      %14 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%14) {address = 118912 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118916 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458756 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      %15 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%15) {address = 119040 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119044 : ui32, arg_idx = 2 : i32, arg_plus = 196608 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147549192 : ui32}
      %16 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%16) {address = 119072 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119076 : ui32, arg_idx = 0 : i32, arg_plus = 49152 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458761 : ui32}
      %17 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%17) {address = 119104 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119108 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458762 : ui32}
      %18 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%18) {address = 119136 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119140 : ui32, arg_idx = 0 : i32, arg_plus = 57344 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 458763 : ui32}
      %19 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%19) {address = 119168 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119172 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 458764 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @inA(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<32x32xi8>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<32x32xi8>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<32x32xi8>, 0, 1024, [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<32x32xi8>, 0, 1024, [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<32x32xi8>, 0, 1024) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<32x32xi8>, 0, 1024) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<32x32xi8>, 0, 1024, [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inB_cons_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<32x32xi8>, 0, 1024, [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inB_cons_prod_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%memC_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_0 : memref<32x32xi32>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memC_cons_cons_lock, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%memC_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_1 : memref<32x32xi32>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%memC_cons_cons_lock, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%memC_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_0 : memref<32x32xi32>, 0, 1024, [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%memC_cons_prod_lock, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%memC_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_cons_buff_1 : memref<32x32xi32>, 0, 1024, [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%memC_cons_prod_lock, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA_cons_buff_0 : memref<32x32xi8>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA_cons_buff_1 : memref<32x32xi8>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_cons_buff_0 : memref<32x32xi8>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_cons_buff_1 : memref<32x32xi8>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_buff_0 : memref<32x32xi32>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%memC_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC_buff_1 : memref<32x32xi32>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
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

