module {
  aie.device(npu1_4col) {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    memref.global "public" @out_cons : memref<1xi32>
    memref.global "public" @out : memref<1xi32>
    memref.global "public" @outC_cons : memref<4xi32>
    memref.global "public" @outC : memref<4xi32>
    memref.global "public" @memC3_cons : memref<1xi32>
    memref.global "public" @memC3 : memref<1xi32>
    memref.global "public" @memA3_cons : memref<512xi32>
    memref.global "public" @memA3 : memref<512xi32>
    memref.global "public" @memC2_cons : memref<1xi32>
    memref.global "public" @memC2 : memref<1xi32>
    memref.global "public" @memA2_cons : memref<512xi32>
    memref.global "public" @memA2 : memref<512xi32>
    memref.global "public" @memC1_cons : memref<1xi32>
    memref.global "public" @memC1 : memref<1xi32>
    memref.global "public" @memA1_cons : memref<512xi32>
    memref.global "public" @memA1 : memref<512xi32>
    memref.global "public" @memC0_cons : memref<1xi32>
    memref.global "public" @memC0 : memref<1xi32>
    memref.global "public" @memA0_cons : memref<512xi32>
    memref.global "public" @memA0 : memref<512xi32>
    memref.global "public" @inA_cons : memref<2048xi32>
    memref.global "public" @inA : memref<2048xi32>
    func.func private @reduce_max_vector(memref<512xi32>, memref<1xi32>, i32)
    func.func private @reduce_max_scalar(memref<4xi32>, memref<1xi32>, i32)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "out_cons_prod_lock_0"}
    %out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock_0"}
    %out_buff_0 = aie.buffer(%tile_0_2) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "out_buff_0"} : memref<1xi32> 
    %out_buff_1 = aie.buffer(%tile_0_2) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "out_buff_1"} : memref<1xi32> 
    %out_prod_lock_0 = aie.lock(%tile_0_2, 6) {init = 2 : i32, sym_name = "out_prod_lock_0"}
    %out_cons_lock_0 = aie.lock(%tile_0_2, 7) {init = 0 : i32, sym_name = "out_cons_lock_0"}
    %outC_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "outC_cons_buff_0"} : memref<4xi32> 
    %outC_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "outC_cons_buff_1"} : memref<4xi32> 
    %outC_cons_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "outC_cons_prod_lock_0"}
    %outC_cons_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock_0"}
    %outC_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "outC_buff_0"} : memref<4xi32> 
    %outC_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "outC_buff_1"} : memref<4xi32> 
    %outC_prod_lock_0 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "outC_prod_lock_0"}
    %outC_cons_lock_0 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "outC_cons_lock_0"}
    %outC_prod_lock_1 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "outC_prod_lock_1"}
    %outC_cons_lock_1 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "outC_cons_lock_1"}
    %outC_prod_lock_2 = aie.lock(%mem_tile_0_1, 12) {init = 2 : i32, sym_name = "outC_prod_lock_2"}
    %outC_cons_lock_2 = aie.lock(%mem_tile_0_1, 13) {init = 0 : i32, sym_name = "outC_cons_lock_2"}
    %outC_prod_lock_3 = aie.lock(%mem_tile_0_1, 14) {init = 2 : i32, sym_name = "outC_prod_lock_3"}
    %outC_cons_lock_3 = aie.lock(%mem_tile_0_1, 15) {init = 0 : i32, sym_name = "outC_cons_lock_3"}
    %memC3_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memC3_buff_0"} : memref<1xi32> 
    %memC3_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memC3_buff_1"} : memref<1xi32> 
    %memC3_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "memC3_prod_lock_0"}
    %memC3_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "memC3_cons_lock_0"}
    %memA3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "memA3_cons_buff_0"} : memref<512xi32> 
    %memA3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memA3_cons_buff_1"} : memref<512xi32> 
    %memA3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "memA3_cons_prod_lock_0"}
    %memA3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "memA3_cons_cons_lock_0"}
    %memC2_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memC2_buff_0"} : memref<1xi32> 
    %memC2_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memC2_buff_1"} : memref<1xi32> 
    %memC2_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "memC2_prod_lock_0"}
    %memC2_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "memC2_cons_lock_0"}
    %memA2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "memA2_cons_buff_0"} : memref<512xi32> 
    %memA2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memA2_cons_buff_1"} : memref<512xi32> 
    %memA2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "memA2_cons_prod_lock_0"}
    %memA2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "memA2_cons_cons_lock_0"}
    %memC1_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memC1_buff_0"} : memref<1xi32> 
    %memC1_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memC1_buff_1"} : memref<1xi32> 
    %memC1_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "memC1_prod_lock_0"}
    %memC1_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "memC1_cons_lock_0"}
    %memA1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "memA1_cons_buff_0"} : memref<512xi32> 
    %memA1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memA1_cons_buff_1"} : memref<512xi32> 
    %memA1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "memA1_cons_prod_lock_0"}
    %memA1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "memA1_cons_cons_lock_0"}
    %memC0_buff_0 = aie.buffer(%tile_0_2) {address = 32784 : i32, mem_bank = 2 : i32, sym_name = "memC0_buff_0"} : memref<1xi32> 
    %memC0_buff_1 = aie.buffer(%tile_0_2) {address = 49168 : i32, mem_bank = 3 : i32, sym_name = "memC0_buff_1"} : memref<1xi32> 
    %memC0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memC0_prod_lock_0"}
    %memC0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memC0_cons_lock_0"}
    %memA0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "memA0_cons_buff_0"} : memref<512xi32> 
    %memA0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memA0_cons_buff_1"} : memref<512xi32> 
    %memA0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memA0_cons_prod_lock_0"}
    %memA0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memA0_cons_cons_lock_0"}
    %inA_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inA_cons_buff_0"} : memref<2048xi32> 
    %inA_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "inA_cons_buff_1"} : memref<2048xi32> 
    %inA_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock_0"}
    %inA_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock_0"}
    %inA_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "inA_cons_prod_lock_1"}
    %inA_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "inA_cons_cons_lock_1"}
    %inA_cons_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "inA_cons_prod_lock_2"}
    %inA_cons_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "inA_cons_cons_lock_2"}
    %inA_cons_prod_lock_3 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "inA_cons_prod_lock_3"}
    %inA_cons_cons_lock_3 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "inA_cons_cons_lock_3"}
    %inA_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "inA_prod_lock_0"}
    %inA_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "inA_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%mem_tile_0_1, DMA : 4, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 1, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memC0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA0_cons_buff_0, %memC0_buff_0, %c512_i32) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC0_cons_lock_0, Release, 1)
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%outC_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c4_i32 = arith.constant 4 : i32
      func.call @reduce_max_scalar(%outC_cons_buff_0, %out_buff_0, %c4_i32) : (memref<4xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%outC_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      aie.use_lock(%memC0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_0 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA0_cons_buff_1, %memC0_buff_1, %c512_i32_0) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC0_cons_lock_0, Release, 1)
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%outC_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c4_i32_1 = arith.constant 4 : i32
      func.call @reduce_max_scalar(%outC_cons_buff_1, %out_buff_1, %c4_i32_1) : (memref<4xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%outC_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%memC0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_2 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA0_cons_buff_0, %memC0_buff_0, %c512_i32_2) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC0_cons_lock_0, Release, 1)
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%outC_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c4_i32_3 = arith.constant 4 : i32
      func.call @reduce_max_scalar(%outC_cons_buff_0, %out_buff_0, %c4_i32_3) : (memref<4xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%outC_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "reduce_max.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memC1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA1_cons_buff_0, %memC1_buff_0, %c512_i32) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC1_cons_lock_0, Release, 1)
      aie.use_lock(%memC1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_0 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA1_cons_buff_1, %memC1_buff_1, %c512_i32_0) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%memC1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_1 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA1_cons_buff_0, %memC1_buff_0, %c512_i32_1) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "reduce_max.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memC2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA2_cons_buff_0, %memC2_buff_0, %c512_i32) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC2_cons_lock_0, Release, 1)
      aie.use_lock(%memC2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_0 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA2_cons_buff_1, %memC2_buff_1, %c512_i32_0) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC2_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%memC2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_1 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA2_cons_buff_0, %memC2_buff_0, %c512_i32_1) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC2_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "reduce_max.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memC3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA3_cons_buff_0, %memC3_buff_0, %c512_i32) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC3_cons_lock_0, Release, 1)
      aie.use_lock(%memC3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_0 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA3_cons_buff_1, %memC3_buff_1, %c512_i32_0) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC3_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%memC3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memA3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c512_i32_1 = arith.constant 512 : i32
      func.call @reduce_max_vector(%memA3_cons_buff_0, %memC3_buff_0, %c512_i32_1) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%memA3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memC3_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "reduce_max.cc.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<2048xi32>, %arg1: memref<1xi32>) {
      %0 = aiex.dma_configure_task_for @inA {
        aie.dma_bd(%arg0 : memref<2048xi32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
    aie.shim_dma_allocation @inA(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 2 preds: ^bb0, ^bb8
      aie.use_lock(%inA_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inA_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inA_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 512, 512) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inA_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inA_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 1024, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inA_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inA_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 1536, 512) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%inA_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inA_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 0, 512) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%inA_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%inA_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 512, 512) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%inA_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inA_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 1024, 512) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%inA_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inA_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 1536, 512) {bd_id = 7 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inA_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inA_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 0, 512) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%inA_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inA_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 0, 512) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%inA_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %2 = aie.dma_start(S2MM, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%outC_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 0, 1) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%outC_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%outC_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 0, 1) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%outC_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inA_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 512, 512) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inA_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inA_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 512, 512) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inA_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%outC_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 1, 1) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%outC_cons_lock_1, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%outC_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 1, 1) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%outC_cons_lock_1, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(MM2S, 2, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%inA_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 1024, 512) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%inA_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%inA_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 1024, 512) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%inA_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 3, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%outC_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 2, 1) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%outC_cons_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%outC_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 2, 1) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%outC_cons_lock_2, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %7 = aie.dma_start(MM2S, 3, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%inA_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<2048xi32>, 1536, 512) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%inA_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%inA_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<2048xi32>, 1536, 512) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%inA_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %8 = aie.dma_start(S2MM, 4, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%outC_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 3, 1) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%outC_cons_lock_3, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%outC_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 3, 1) {bd_id = 15 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%outC_cons_lock_3, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %9 = aie.dma_start(MM2S, 4, ^bb34, ^bb42)
    ^bb34:  // 2 preds: ^bb33, ^bb41
      aie.use_lock(%outC_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 0, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%outC_prod_lock_0, Release, 1)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%outC_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 1, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.use_lock(%outC_prod_lock_1, Release, 1)
      aie.next_bd ^bb36
    ^bb36:  // pred: ^bb35
      aie.use_lock(%outC_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 2, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.use_lock(%outC_prod_lock_2, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%outC_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<4xi32>, 3, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.use_lock(%outC_prod_lock_3, Release, 1)
      aie.next_bd ^bb38
    ^bb38:  // pred: ^bb37
      aie.use_lock(%outC_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 0, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.use_lock(%outC_prod_lock_0, Release, 1)
      aie.next_bd ^bb39
    ^bb39:  // pred: ^bb38
      aie.use_lock(%outC_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 1, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.use_lock(%outC_prod_lock_1, Release, 1)
      aie.next_bd ^bb40
    ^bb40:  // pred: ^bb39
      aie.use_lock(%outC_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 2, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.use_lock(%outC_prod_lock_2, Release, 1)
      aie.next_bd ^bb41
    ^bb41:  // pred: ^bb40
      aie.use_lock(%outC_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<4xi32>, 3, 1, [<size = 1, stride = 1>, <size = 1, stride = 1>]) {bd_id = 23 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%outC_prod_lock_3, Release, 1)
      aie.next_bd ^bb34
    ^bb42:  // pred: ^bb33
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memC0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC0_buff_0 : memref<1xi32>, 0, 1) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memC0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memC0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC0_buff_1 : memref<1xi32>, 0, 1) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memC0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%outC_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_cons_buff_0 : memref<4xi32>, 0, 4) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%outC_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%outC_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_cons_buff_1 : memref<4xi32>, 0, 4) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%outC_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<1xi32>, 0, 1) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<1xi32>, 0, 1) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memC1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC1_buff_0 : memref<1xi32>, 0, 1) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memC1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memC1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC1_buff_1 : memref<1xi32>, 0, 1) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memC1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memC2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC2_buff_0 : memref<1xi32>, 0, 1) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memC2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memC2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC2_buff_1 : memref<1xi32>, 0, 1) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memC2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memA3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memA3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memC3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC3_buff_0 : memref<1xi32>, 0, 1) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memC3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memC3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC3_buff_1 : memref<1xi32>, 0, 1) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memC3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
