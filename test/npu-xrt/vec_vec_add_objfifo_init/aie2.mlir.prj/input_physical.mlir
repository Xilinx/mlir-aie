module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<16xi32>
    memref.global "public" @out : memref<16xi32>
    memref.global "public" @in2_cons : memref<16xi32>
    memref.global "public" @in2 : memref<16xi32>
    memref.global "public" @in1_cons : memref<256xi32>
    memref.global "public" @in1 : memref<256xi32>
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "out_cons_prod_lock_0"}
    %out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock_0"}
    %out_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_buff_0"} : memref<16xi32> 
    %out_buff_1 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out_buff_1"} : memref<16xi32> 
    %out_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "out_prod_lock_0"}
    %out_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "out_cons_lock_0"}
    %in2_cons_buff_0 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_cons_buff_0"} : memref<16xi32> 
    %in2_cons_buff_1 = aie.buffer(%tile_0_2) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "in2_cons_buff_1"} : memref<16xi32> 
    %in2_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "in2_cons_prod_lock_0"}
    %in2_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "in2_cons_cons_lock_0"}
    %in2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "in2_prod_lock_0"}
    %in2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in2_cons_lock_0"}
    %in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in1_cons_buff_0"} : memref<256xi32> 
    %in1_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in1_cons_prod_lock_0"}
    %in1_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_cons_cons_lock_0"}
    %in1_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "in1_buff_0"} : memref<256xi32> = dense<"0x0100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF00000000010000">
    %in1_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32, sym_name = "in1_prod_lock_0"}
    %in1_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32, sym_name = "in1_cons_lock_0"}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 4, North : 4>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 4, DMA : 1>
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 4>
      aie.connect<North : 1, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb11
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %c0_0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%in2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      %c0_2 = arith.constant 0 : index
      %c16_3 = arith.constant 16 : index
      %c1_4 = arith.constant 1 : index
      cf.br ^bb5(%c0_2 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16_3 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %c16_5 = arith.constant 16 : index
      %6 = arith.muli %2, %c16_5 : index
      %7 = arith.addi %6, %4 : index
      %8 = memref.load %in1_cons_buff_0[%7] : memref<256xi32>
      %9 = memref.load %in2_cons_buff_0[%4] : memref<16xi32>
      %10 = arith.addi %8, %9 : i32
      memref.store %10, %out_buff_0[%4] : memref<16xi32>
      %11 = arith.addi %4, %c1_4 : index
      cf.br ^bb5(%11 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%in2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      %c1_6 = arith.constant 1 : index
      %12 = arith.addi %2, %c1_1 : index
      aie.use_lock(%in2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      %c0_7 = arith.constant 0 : index
      %c16_8 = arith.constant 16 : index
      %c1_9 = arith.constant 1 : index
      cf.br ^bb8(%c0_7 : index)
    ^bb8(%13: index):  // 2 preds: ^bb7, ^bb9
      %14 = arith.cmpi slt, %13, %c16_8 : index
      cf.cond_br %14, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %c16_10 = arith.constant 16 : index
      %15 = arith.muli %12, %c16_10 : index
      %16 = arith.addi %15, %13 : index
      %17 = memref.load %in1_cons_buff_0[%16] : memref<256xi32>
      %18 = memref.load %in2_cons_buff_1[%13] : memref<16xi32>
      %19 = arith.addi %17, %18 : i32
      memref.store %19, %out_buff_1[%13] : memref<16xi32>
      %20 = arith.addi %13, %c1_9 : index
      cf.br ^bb8(%20 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%in2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      %21 = arith.addi %2, %c2 : index
      cf.br ^bb3(%21 : index)
    ^bb11:  // pred: ^bb3
      aie.use_lock(%in1_cons_prod_lock_0, Release, 1)
      %22 = arith.addi %0, %c1 : index
      cf.br ^bb1(%22 : index)
    ^bb12:  // pred: ^bb1
      aie.end
    }
    aiex.runtime_sequence @sequence(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @in2} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_buff_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_cons_buff_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%in2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%in2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<16xi32>, 0, 16) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @in2(MM2S, 0, 0)
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}

