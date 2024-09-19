module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    memref.global "private" constant @blockwrite_data : memref<9xi32> = dense<[608, 0, 0, 0, 0, 0, 0, 0, 8]>
    memref.global "private" constant @blockwrite_data_0 : memref<236xi32> = dense<"0x43100038C30100005500000800005500000C0000990731169501404000C00100010001000100010055F87FFCFFFF5976301F2F780000B20462000000000000004D768CDCF0DF01000100010001000100010019140010010001000100010001009908B1149501402000C00100010001000100010059763E1803204B24000000001501007000000100010001000100010015010088010099063018010001000100C00300880300000000000000000000000100010001000100010001001908001001000100010001000100010019000000950000680000010001000100010001001920033899C2FF0F15010010010043C86300E8173E000100010001001900000015010010010043486700E8173E00010001000100370100000000000000000000BB10B03800000008E6075580E3080000550068000700BB10009AC80100000000C0030010A818010000000000000000001DFD0178A60201000100010001000100C00300880300000000000000A0220000C00300384F06000000000078A6020000C0030088030000000000000000000000C0030088030000000000000000000000C0030088030000000000000000000000C0030088030000000000000000000000C0030088030000000000000000000000198C30166D9E0C00A02201000100010001001501002001004348630028100000010001000100BB8E0300000000000000D9C2FF0795000020010001000100010001007F000000790004CD00E0FF07000019121210191800100100010001000100C0030088030000000000000000000000010001000100010001000100191210101918001001000100010001001900000059960B185500000C000059763E185500800B000043280B9E3F8C2F009501407001C01920033819ECFF0F9942FE0F19C0FE0F2F780000380040008046FF070000D98603070100010001000100010001001914001059768E1D01000100010001009968F1159501405001C001000100010001002F78000038004000000000000000D942FE0759EEFE0759ECFF07D97EFF070100010001001918001001000100010001007F0000007100000000E0FF070000BB10101AD00100480001BB100042D00100C850003B299B24AA173E4050005936061C59F6841C55A0660C00000100010079F6C1189920C41043288B98F7300700D98E0307D986FB07010001000100010001001954001001000100010019000000C00300280B8002000000000000000000198CE71459160A1801000100010001005916791A1918001043280B8C01212300010001000100BB8E0300000000000000">
    memref.global "private" constant @blockwrite_data_1 : memref<6xi32> = dense<[4195328, 0, 0, 0, 0, 100941792]>
    memref.global "private" constant @blockwrite_data_2 : memref<6xi32> = dense<[37749760, 1074266112, 0, 0, 0, 235167715]>
    memref.global "private" constant @blockwrite_data_3 : memref<8xi32> = dense<[1024, 655360, 0, 0, 0, 0, 0, -2126381248]>
    memref.global "private" constant @blockwrite_data_4 : memref<8xi32> = dense<[-2147482624, 1703936, 0, 0, 0, 0, 0, -2126446783]>
    memref.global "private" constant @blockwrite_data_5 : memref<8xi32> = dense<[-2130705408, 25823232, 0, 0, 0, 0, 0, -2126315709]>
    memref.global "private" constant @blockwrite_data_6 : memref<8xi32> = dense<[1024, 26871808, 0, 0, 0, 0, 0, -2126250174]>
    aiex.runtime_sequence(%arg0: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 2][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 4][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 6][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 10][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 15][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 20][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 22][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 27][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 32][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 37][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 42][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 47][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 52][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 57][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 62][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 67][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 72][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 77][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 82][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 87][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 92][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 97][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 102][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 107][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 112][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 117][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 122][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 127][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 132][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 137][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 142][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 147][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 152][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 157][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 162][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 167][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 172][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 177][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 182][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 187][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 192][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 197][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 202][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 207][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 212][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 217][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 222][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 227][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 232][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 237][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 242][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 247][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 252][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 257][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 262][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 267][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 272][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 277][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 282][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 287][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 292][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 297][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 302][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 307][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 312][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 317][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 319][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 321][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 323][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 325][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 327][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 329][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 331][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 333][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 335][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 337][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 339][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 341][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 343][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 345][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 347][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 349][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 351][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 353][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 355][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 357][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 359][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 361][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 363][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 365][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 367][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 369][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 371][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 373][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 375][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 377][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 382][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 385][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 390][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 393][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 395][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 397][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 399][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 401][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 406][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 411][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 416][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 421][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 426][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 431][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 436][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 441][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 443][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 445][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 447][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 449][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 451][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 453][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 455][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 457][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 459][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 461][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 463][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 465][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 467][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 469][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 471][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 473][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 475][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 477][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 479][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 481][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 483][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 485][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 487][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 489][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 491][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 493][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 495][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 497][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 499][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 501][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 503][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 505][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 507][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 509][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 511][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 513][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 515][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 517][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 519][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 521][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 523][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 525][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 527][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 529][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 531][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 533][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 535][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 537][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 539][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 541][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 543][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 545][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_0, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
    memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
    aie.packet_flow(26) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

