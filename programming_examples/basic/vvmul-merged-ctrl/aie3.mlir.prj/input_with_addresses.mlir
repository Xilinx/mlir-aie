module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    memref.global "private" constant @blockwrite_data : memref<9xi32> = dense<[1440, 0, 0, 0, 0, 0, 0, 0, 8]>
    memref.global "private" constant @blockwrite_data_0 : memref<444xi32> = dense<"0x43100038C30100005500000800005500000C0000990731169501404000C00100010001000100010055F87FFCFFFF5976301F2F780000B20462000000000000004D768CDCF0DF01000100010001000100010019140010010001000100010001009908B1149501402000C00100010001000100010059763E1803204B24000000001501007000000100010001000100010015010028030099063018010001000100C00300880300000000000000000000000100010001000100010001001908001001000100010001000100010019000000950000680000010001000100010001003B288B6508900140030003204B0500D08EFF3B290B25E8473F403EFF0320CB4500404EFF3B110002C30100D07FFF3B11A0A2C201008098FF3B1160C2C20100C0A8FF3B11B81803000000B8FF1960FE0F19C2FE0F1966FF0F19E4FF0F1962FB0FC00300880300000080C2FA070000000077108022C2010070F90FE40F5500090907007F00000002447338000000000000150100B0020043C8FF2728060000010001000100370100000000000000000000150100B0020043C8FF2768060000010001000100370100000000000000000000150100B0020043C8FF27880600000100010001003701000000000000000000004D9638D928FF4D16390A762059BA030259B4030159BE030259B8030159BC030259B2030159B603029D8F7B4F744059B00301FBAFF50700BA03C8774059BA0301FBCFB50700B40308772059BC0302FB9FF30600B40388762059B60302FB8F630400B20308762059B20302FBDFFB0700A203C8772059A20302FBCFBB0700BA0388774059B80301FBAFF50600BA0388744059BA0301FB8F710600B40388762059B60302FB1FFF0700B00308762059B20302FBCFBD0700BE0308772059BE0302FB2F650700BC0348773C59BCE302BDAFF58674000100998F71160100150100C00200BDCFF1877600BDEF7107760003203B2102007600F7203B4202B0E30000000000C0030048630028108046F90700000000150100C0020043C8032048060000010001000100370100000000000000000000150100C0020043C80320A8060000010001000100370100000000000000000000150100B0020043C8FF2728060000010001000100370100000000000000000000150100B0020043C8FF2768060000010001000100370100000000000000000000150100B0020043C8FF278806000001000100010037010000000000000000000059163B184D963A09760059B8030159B2030159BC030059B6030159BE030059B4030159BA03009D8F39CF740059B00301FBEF730600B8038F770059B80301FBBFF30700B203CF762059BE0300FBAF730700B2034F770059B40301FB8FE70400B2030F762059B20300FBCFB90700A6038F772059A60300FBBFF70700B8030F772059BE0300FBAF750700B6034F750059BA0301FB8F710600B4038F762059B60300FBEFFD0400B0030F762059B20300FBCFFD0700BC030F772059BE0300FB5F670700BC034F773C59BCE30099AFF51601000100150100C00200BD8F71C674E0BDCFF18776E0BDEF710776E003203BC0020076E07F0076FC652754C1000000482000150100C0020043C8032048060000010001000100370100000000000000000000150100C0020043C80320A8060000010001000100370100000000000000000000198C2114010001000100010001001D0132884EFFBB90FF67FFFF1F083FFF010001000100010001001907B416BDC273864EFFBD98F3463EFF9948B51699A57316950140A800C85500080C070055808A0A07005580090B070001000100D9C2FA075962FB0759E4FF075966FF0759E8FE075960FE0759EAFD07596CFD0759EEFC07D976FC07D9FEFB0719180010010001000100010037010000000000C0FF07000019121210191800100100010001000100C0030088030000000000000000000000010001000100010001000100191210101918001001000100010001001900000059960B185500000C000059763E185500800B000043280B9E3F8C2F009501401003C01920033819ECFF0F9942FE0F19C0FE0F2F780000380040008046FF070000D98603070100010001000100010001001914001059768E1D01000100010001009968F115950140F002C001000100010001002F78000038004000000000000000D942FE0759EEFE0759ECFF07D97EFF070100010001001918001001000100010001007F0000007100000000E0FF070000BB10D01AC00100480001BB10C042C00100C850003B299B24AA173E4050005936061C59F6841C55206D0C00000100010079F6C1189920C41043288B98F7300700D98E0307D986FB07010001000100010001001954001001000100010019000000C00300280B8002000000000000000000198CE71459160A1801000100010001005916791A1918001043280B8C01212300010001000100BB8E0300000000000000">
    memref.global "private" constant @blockwrite_data_1 : memref<6xi32> = dense<[5242896, 0, 0, 0, 0, 235159520]>
    memref.global "private" constant @blockwrite_data_2 : memref<6xi32> = dense<[5505040, 0, 0, 0, 0, 100941792]>
    memref.global "private" constant @blockwrite_data_3 : memref<6xi32> = dense<[4718608, 0, 0, 0, 0, 503611362]>
    memref.global "private" constant @blockwrite_data_4 : memref<6xi32> = dense<[4980752, 0, 0, 0, 0, 369393634]>
    memref.global "private" constant @blockwrite_data_5 : memref<6xi32> = dense<[4194320, 0, 0, 0, 0, 772055013]>
    memref.global "private" constant @blockwrite_data_6 : memref<6xi32> = dense<[4456464, 0, 0, 0, 0, 637837285]>
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
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 317][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 322][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 327][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 332][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 337][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 342][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 347][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 352][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 357][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 362][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 367][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 372][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 377][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 382][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 387][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 392][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 397][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 402][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 407][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 412][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 417][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 422][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 427][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 432][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 437][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 442][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 447][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 452][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 457][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 462][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 467][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 472][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 477][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 482][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 487][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 492][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 497][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 502][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 507][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 512][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 517][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 522][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 527][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 532][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 537][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 542][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 547][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 552][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 557][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 562][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 567][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 572][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 577][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 579][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 581][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 583][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 585][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 587][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 589][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 591][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 593][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 595][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 597][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 599][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 601][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 603][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 605][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 607][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 609][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 611][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 613][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 615][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 617][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 619][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 621][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 623][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 625][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 627][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 629][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 631][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 633][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 635][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 637][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 639][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 641][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 643][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 645][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 650][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 653][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 658][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 661][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 666][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 669][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 674][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 677][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 682][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 685][1, 1, 1, 5][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 690][1, 1, 1, 3][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 693][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 695][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 697][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 699][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 701][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 703][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 705][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 707][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 709][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 711][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 713][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 715][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 717][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 719][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 721][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 723][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 725][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 727][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 729][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 731][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 733][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 735][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 737][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 26>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 739][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 741][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 743][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 745][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 747][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 749][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 751][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 753][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 755][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 757][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 759][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 15>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 761][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 27>) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(26) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
    memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_0, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(26) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_0, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
