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
    memref.global "private" constant @blockwrite_data_7 : memref<8xi32> = dense<[2, 0, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_8 : memref<8xi32> = dense<[2, 8, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_9 : memref<8xi32> = dense<[2, 16, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_10 : memref<8xi32> = dense<[2, 24, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_11 : memref<8xi32> = dense<[2, 32, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_12 : memref<8xi32> = dense<[5, 40, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_13 : memref<8xi32> = dense<[5, 60, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_14 : memref<8xi32> = dense<[2, 80, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_15 : memref<8xi32> = dense<[5, 88, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_16 : memref<8xi32> = dense<[5, 108, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_17 : memref<8xi32> = dense<[5, 128, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_18 : memref<8xi32> = dense<[5, 148, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_19 : memref<8xi32> = dense<[5, 168, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_20 : memref<8xi32> = dense<[5, 188, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_21 : memref<8xi32> = dense<[5, 208, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_22 : memref<8xi32> = dense<[5, 228, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_23 : memref<8xi32> = dense<[5, 248, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_24 : memref<8xi32> = dense<[5, 268, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_25 : memref<8xi32> = dense<[5, 288, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_26 : memref<8xi32> = dense<[5, 308, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_27 : memref<8xi32> = dense<[5, 328, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_28 : memref<8xi32> = dense<[5, 348, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_29 : memref<8xi32> = dense<[5, 368, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_30 : memref<8xi32> = dense<[5, 388, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_31 : memref<8xi32> = dense<[5, 408, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_32 : memref<8xi32> = dense<[5, 428, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_33 : memref<8xi32> = dense<[5, 448, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_34 : memref<8xi32> = dense<[5, 468, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_35 : memref<8xi32> = dense<[5, 488, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_36 : memref<8xi32> = dense<[5, 508, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_37 : memref<8xi32> = dense<[5, 528, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_38 : memref<8xi32> = dense<[5, 548, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_39 : memref<8xi32> = dense<[5, 568, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_40 : memref<8xi32> = dense<[5, 588, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_41 : memref<8xi32> = dense<[5, 608, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_42 : memref<8xi32> = dense<[5, 628, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_43 : memref<8xi32> = dense<[5, 648, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_44 : memref<8xi32> = dense<[5, 668, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_45 : memref<8xi32> = dense<[5, 688, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_46 : memref<8xi32> = dense<[5, 708, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_47 : memref<8xi32> = dense<[5, 728, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_48 : memref<8xi32> = dense<[5, 748, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_49 : memref<8xi32> = dense<[5, 768, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_50 : memref<8xi32> = dense<[5, 788, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_51 : memref<8xi32> = dense<[5, 808, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_52 : memref<8xi32> = dense<[5, 828, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_53 : memref<8xi32> = dense<[5, 848, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_54 : memref<8xi32> = dense<[5, 868, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_55 : memref<8xi32> = dense<[5, 888, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_56 : memref<8xi32> = dense<[5, 908, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_57 : memref<8xi32> = dense<[5, 928, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_58 : memref<8xi32> = dense<[5, 948, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_59 : memref<8xi32> = dense<[5, 968, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_60 : memref<8xi32> = dense<[5, 988, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_61 : memref<8xi32> = dense<[5, 1008, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_62 : memref<8xi32> = dense<[5, 1028, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_63 : memref<8xi32> = dense<[5, 1048, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_64 : memref<8xi32> = dense<[5, 1068, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_65 : memref<8xi32> = dense<[5, 1088, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_66 : memref<8xi32> = dense<[5, 1108, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_67 : memref<8xi32> = dense<[5, 1128, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_68 : memref<8xi32> = dense<[5, 1148, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_69 : memref<8xi32> = dense<[5, 1168, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_70 : memref<8xi32> = dense<[5, 1188, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_71 : memref<8xi32> = dense<[5, 1208, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_72 : memref<8xi32> = dense<[5, 1228, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_73 : memref<8xi32> = dense<[5, 1248, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_74 : memref<8xi32> = dense<[5, 1268, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_75 : memref<8xi32> = dense<[5, 1288, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_76 : memref<8xi32> = dense<[5, 1308, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_77 : memref<8xi32> = dense<[5, 1328, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_78 : memref<8xi32> = dense<[5, 1348, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_79 : memref<8xi32> = dense<[5, 1368, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_80 : memref<8xi32> = dense<[5, 1388, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_81 : memref<8xi32> = dense<[5, 1408, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_82 : memref<8xi32> = dense<[5, 1428, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_83 : memref<8xi32> = dense<[5, 1448, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_84 : memref<8xi32> = dense<[5, 1468, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_85 : memref<8xi32> = dense<[5, 1488, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_86 : memref<8xi32> = dense<[5, 1508, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_87 : memref<8xi32> = dense<[5, 1528, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_88 : memref<8xi32> = dense<[5, 1548, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_89 : memref<8xi32> = dense<[5, 1568, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_90 : memref<8xi32> = dense<[5, 1588, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_91 : memref<8xi32> = dense<[5, 1608, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_92 : memref<8xi32> = dense<[5, 1628, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_93 : memref<8xi32> = dense<[5, 1648, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_94 : memref<8xi32> = dense<[5, 1668, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_95 : memref<8xi32> = dense<[5, 1688, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_96 : memref<8xi32> = dense<[5, 1708, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_97 : memref<8xi32> = dense<[5, 1728, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_98 : memref<8xi32> = dense<[5, 1748, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_99 : memref<8xi32> = dense<[5, 1768, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_100 : memref<8xi32> = dense<[5, 1788, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_101 : memref<8xi32> = dense<[5, 1808, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_102 : memref<8xi32> = dense<[5, 1828, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_103 : memref<8xi32> = dense<[5, 1848, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_104 : memref<8xi32> = dense<[5, 1868, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_105 : memref<8xi32> = dense<[5, 1888, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_106 : memref<8xi32> = dense<[5, 1908, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_107 : memref<8xi32> = dense<[5, 1928, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_108 : memref<8xi32> = dense<[5, 1948, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_109 : memref<8xi32> = dense<[5, 1968, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_110 : memref<8xi32> = dense<[5, 1988, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_111 : memref<8xi32> = dense<[5, 2008, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_112 : memref<8xi32> = dense<[5, 2028, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_113 : memref<8xi32> = dense<[5, 2048, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_114 : memref<8xi32> = dense<[5, 2068, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_115 : memref<8xi32> = dense<[5, 2088, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_116 : memref<8xi32> = dense<[5, 2108, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_117 : memref<8xi32> = dense<[5, 2128, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_118 : memref<8xi32> = dense<[5, 2148, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_119 : memref<8xi32> = dense<[5, 2168, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_120 : memref<8xi32> = dense<[5, 2188, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_121 : memref<8xi32> = dense<[5, 2208, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_122 : memref<8xi32> = dense<[5, 2228, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_123 : memref<8xi32> = dense<[5, 2248, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_124 : memref<8xi32> = dense<[5, 2268, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_125 : memref<8xi32> = dense<[5, 2288, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_126 : memref<8xi32> = dense<[2, 2308, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_127 : memref<8xi32> = dense<[2, 2316, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_128 : memref<8xi32> = dense<[2, 2324, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_129 : memref<8xi32> = dense<[2, 2332, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_130 : memref<8xi32> = dense<[2, 2340, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_131 : memref<8xi32> = dense<[2, 2348, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_132 : memref<8xi32> = dense<[2, 2356, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_133 : memref<8xi32> = dense<[2, 2364, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_134 : memref<8xi32> = dense<[2, 2372, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_135 : memref<8xi32> = dense<[2, 2380, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_136 : memref<8xi32> = dense<[2, 2388, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_137 : memref<8xi32> = dense<[2, 2396, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_138 : memref<8xi32> = dense<[2, 2404, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_139 : memref<8xi32> = dense<[2, 2412, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_140 : memref<8xi32> = dense<[2, 2420, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_141 : memref<8xi32> = dense<[2, 2428, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_142 : memref<8xi32> = dense<[2, 2436, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_143 : memref<8xi32> = dense<[2, 2444, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_144 : memref<8xi32> = dense<[2, 2452, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_145 : memref<8xi32> = dense<[2, 2460, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_146 : memref<8xi32> = dense<[2, 2468, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_147 : memref<8xi32> = dense<[2, 2476, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_148 : memref<8xi32> = dense<[2, 2484, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_149 : memref<8xi32> = dense<[2, 2492, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_150 : memref<8xi32> = dense<[2, 2500, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_151 : memref<8xi32> = dense<[2, 2508, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_152 : memref<8xi32> = dense<[2, 2516, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_153 : memref<8xi32> = dense<[2, 2524, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_154 : memref<8xi32> = dense<[2, 2532, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_155 : memref<8xi32> = dense<[2, 2540, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_156 : memref<8xi32> = dense<[2, 2548, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_157 : memref<8xi32> = dense<[2, 2556, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_158 : memref<8xi32> = dense<[2, 2564, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_159 : memref<8xi32> = dense<[2, 2572, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_160 : memref<8xi32> = dense<[5, 2580, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_161 : memref<8xi32> = dense<[3, 2600, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_162 : memref<8xi32> = dense<[5, 2612, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_163 : memref<8xi32> = dense<[3, 2632, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_164 : memref<8xi32> = dense<[5, 2644, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_165 : memref<8xi32> = dense<[3, 2664, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_166 : memref<8xi32> = dense<[5, 2676, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_167 : memref<8xi32> = dense<[3, 2696, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_168 : memref<8xi32> = dense<[5, 2708, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_169 : memref<8xi32> = dense<[3, 2728, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_170 : memref<8xi32> = dense<[5, 2740, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_171 : memref<8xi32> = dense<[3, 2760, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_172 : memref<8xi32> = dense<[2, 2772, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_173 : memref<8xi32> = dense<[2, 2780, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_174 : memref<8xi32> = dense<[2, 2788, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_175 : memref<8xi32> = dense<[2, 2796, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_176 : memref<8xi32> = dense<[2, 2804, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_177 : memref<8xi32> = dense<[2, 2812, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_178 : memref<8xi32> = dense<[2, 2820, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_179 : memref<8xi32> = dense<[2, 2828, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_180 : memref<8xi32> = dense<[2, 2836, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_181 : memref<8xi32> = dense<[2, 2844, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_182 : memref<8xi32> = dense<[2, 2852, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_183 : memref<8xi32> = dense<[2, 2860, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_184 : memref<8xi32> = dense<[2, 2868, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_185 : memref<8xi32> = dense<[2, 2876, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_186 : memref<8xi32> = dense<[2, 2884, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_187 : memref<8xi32> = dense<[2, 2892, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_188 : memref<8xi32> = dense<[2, 2900, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_189 : memref<8xi32> = dense<[2, 2908, 1087373312, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_190 : memref<8xi32> = dense<[2, 2916, 1087373312, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_191 : memref<8xi32> = dense<[2, 2924, 1087373312, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_192 : memref<8xi32> = dense<[2, 2932, 1087373312, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_193 : memref<8xi32> = dense<[2, 2940, 1087373312, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_194 : memref<8xi32> = dense<[2, 2948, 1087373312, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_195 : memref<8xi32> = dense<[2, 2956, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_196 : memref<8xi32> = dense<[2, 2964, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_197 : memref<8xi32> = dense<[2, 2972, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_198 : memref<8xi32> = dense<[2, 2980, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_199 : memref<8xi32> = dense<[2, 2988, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_200 : memref<8xi32> = dense<[2, 2996, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_201 : memref<8xi32> = dense<[2, 3004, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_202 : memref<8xi32> = dense<[2, 3012, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_203 : memref<8xi32> = dense<[2, 3020, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_204 : memref<8xi32> = dense<[2, 3028, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_205 : memref<8xi32> = dense<[2, 3036, 1081606144, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_206 : memref<8xi32> = dense<[2, 3044, 1087897600, 0, -2147483648, 0, 0, 33554432]>
    aiex.runtime_sequence(%arg0: memref<1024xi32>) {
      %0 = memref.get_global @blockwrite_data_7 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %1 = memref.get_global @blockwrite_data_8 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 8 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %2 = memref.get_global @blockwrite_data_9 : memref<8xi32>
      aiex.npu.blockwrite(%2) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 16 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %3 = memref.get_global @blockwrite_data_10 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 24 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %4 = memref.get_global @blockwrite_data_11 : memref<8xi32>
      aiex.npu.blockwrite(%4) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 32 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %5 = memref.get_global @blockwrite_data_12 : memref<8xi32>
      aiex.npu.blockwrite(%5) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 40 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %6 = memref.get_global @blockwrite_data_13 : memref<8xi32>
      aiex.npu.blockwrite(%6) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 60 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %7 = memref.get_global @blockwrite_data_14 : memref<8xi32>
      aiex.npu.blockwrite(%7) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 80 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %8 = memref.get_global @blockwrite_data_15 : memref<8xi32>
      aiex.npu.blockwrite(%8) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 88 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %9 = memref.get_global @blockwrite_data_16 : memref<8xi32>
      aiex.npu.blockwrite(%9) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 108 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %10 = memref.get_global @blockwrite_data_17 : memref<8xi32>
      aiex.npu.blockwrite(%10) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 128 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %11 = memref.get_global @blockwrite_data_18 : memref<8xi32>
      aiex.npu.blockwrite(%11) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 148 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %12 = memref.get_global @blockwrite_data_19 : memref<8xi32>
      aiex.npu.blockwrite(%12) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 168 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %13 = memref.get_global @blockwrite_data_20 : memref<8xi32>
      aiex.npu.blockwrite(%13) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 188 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %14 = memref.get_global @blockwrite_data_21 : memref<8xi32>
      aiex.npu.blockwrite(%14) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 208 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %15 = memref.get_global @blockwrite_data_22 : memref<8xi32>
      aiex.npu.blockwrite(%15) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 228 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %16 = memref.get_global @blockwrite_data_23 : memref<8xi32>
      aiex.npu.blockwrite(%16) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 248 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %17 = memref.get_global @blockwrite_data_24 : memref<8xi32>
      aiex.npu.blockwrite(%17) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 268 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %18 = memref.get_global @blockwrite_data_25 : memref<8xi32>
      aiex.npu.blockwrite(%18) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 288 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %19 = memref.get_global @blockwrite_data_26 : memref<8xi32>
      aiex.npu.blockwrite(%19) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 308 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %20 = memref.get_global @blockwrite_data_27 : memref<8xi32>
      aiex.npu.blockwrite(%20) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 328 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %21 = memref.get_global @blockwrite_data_28 : memref<8xi32>
      aiex.npu.blockwrite(%21) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 348 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %22 = memref.get_global @blockwrite_data_29 : memref<8xi32>
      aiex.npu.blockwrite(%22) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 368 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %23 = memref.get_global @blockwrite_data_30 : memref<8xi32>
      aiex.npu.blockwrite(%23) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 388 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %24 = memref.get_global @blockwrite_data_31 : memref<8xi32>
      aiex.npu.blockwrite(%24) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 408 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %25 = memref.get_global @blockwrite_data_32 : memref<8xi32>
      aiex.npu.blockwrite(%25) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 428 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %26 = memref.get_global @blockwrite_data_33 : memref<8xi32>
      aiex.npu.blockwrite(%26) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 448 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %27 = memref.get_global @blockwrite_data_34 : memref<8xi32>
      aiex.npu.blockwrite(%27) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 468 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %28 = memref.get_global @blockwrite_data_35 : memref<8xi32>
      aiex.npu.blockwrite(%28) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 488 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %29 = memref.get_global @blockwrite_data_36 : memref<8xi32>
      aiex.npu.blockwrite(%29) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 508 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %30 = memref.get_global @blockwrite_data_37 : memref<8xi32>
      aiex.npu.blockwrite(%30) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 528 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %31 = memref.get_global @blockwrite_data_38 : memref<8xi32>
      aiex.npu.blockwrite(%31) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 548 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %32 = memref.get_global @blockwrite_data_39 : memref<8xi32>
      aiex.npu.blockwrite(%32) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 568 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %33 = memref.get_global @blockwrite_data_40 : memref<8xi32>
      aiex.npu.blockwrite(%33) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 588 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %34 = memref.get_global @blockwrite_data_41 : memref<8xi32>
      aiex.npu.blockwrite(%34) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 608 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %35 = memref.get_global @blockwrite_data_42 : memref<8xi32>
      aiex.npu.blockwrite(%35) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 628 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %36 = memref.get_global @blockwrite_data_43 : memref<8xi32>
      aiex.npu.blockwrite(%36) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 648 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %37 = memref.get_global @blockwrite_data_44 : memref<8xi32>
      aiex.npu.blockwrite(%37) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 668 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %38 = memref.get_global @blockwrite_data_45 : memref<8xi32>
      aiex.npu.blockwrite(%38) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 688 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %39 = memref.get_global @blockwrite_data_46 : memref<8xi32>
      aiex.npu.blockwrite(%39) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 708 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %40 = memref.get_global @blockwrite_data_47 : memref<8xi32>
      aiex.npu.blockwrite(%40) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 728 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %41 = memref.get_global @blockwrite_data_48 : memref<8xi32>
      aiex.npu.blockwrite(%41) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 748 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %42 = memref.get_global @blockwrite_data_49 : memref<8xi32>
      aiex.npu.blockwrite(%42) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 768 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %43 = memref.get_global @blockwrite_data_50 : memref<8xi32>
      aiex.npu.blockwrite(%43) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 788 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %44 = memref.get_global @blockwrite_data_51 : memref<8xi32>
      aiex.npu.blockwrite(%44) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 808 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %45 = memref.get_global @blockwrite_data_52 : memref<8xi32>
      aiex.npu.blockwrite(%45) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 828 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %46 = memref.get_global @blockwrite_data_53 : memref<8xi32>
      aiex.npu.blockwrite(%46) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 848 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %47 = memref.get_global @blockwrite_data_54 : memref<8xi32>
      aiex.npu.blockwrite(%47) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 868 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %48 = memref.get_global @blockwrite_data_55 : memref<8xi32>
      aiex.npu.blockwrite(%48) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 888 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %49 = memref.get_global @blockwrite_data_56 : memref<8xi32>
      aiex.npu.blockwrite(%49) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 908 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %50 = memref.get_global @blockwrite_data_57 : memref<8xi32>
      aiex.npu.blockwrite(%50) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 928 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %51 = memref.get_global @blockwrite_data_58 : memref<8xi32>
      aiex.npu.blockwrite(%51) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 948 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %52 = memref.get_global @blockwrite_data_59 : memref<8xi32>
      aiex.npu.blockwrite(%52) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 968 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %53 = memref.get_global @blockwrite_data_60 : memref<8xi32>
      aiex.npu.blockwrite(%53) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 988 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %54 = memref.get_global @blockwrite_data_61 : memref<8xi32>
      aiex.npu.blockwrite(%54) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1008 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %55 = memref.get_global @blockwrite_data_62 : memref<8xi32>
      aiex.npu.blockwrite(%55) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1028 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %56 = memref.get_global @blockwrite_data_63 : memref<8xi32>
      aiex.npu.blockwrite(%56) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1048 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %57 = memref.get_global @blockwrite_data_64 : memref<8xi32>
      aiex.npu.blockwrite(%57) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1068 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %58 = memref.get_global @blockwrite_data_65 : memref<8xi32>
      aiex.npu.blockwrite(%58) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1088 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %59 = memref.get_global @blockwrite_data_66 : memref<8xi32>
      aiex.npu.blockwrite(%59) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1108 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %60 = memref.get_global @blockwrite_data_67 : memref<8xi32>
      aiex.npu.blockwrite(%60) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1128 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %61 = memref.get_global @blockwrite_data_68 : memref<8xi32>
      aiex.npu.blockwrite(%61) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1148 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %62 = memref.get_global @blockwrite_data_69 : memref<8xi32>
      aiex.npu.blockwrite(%62) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1168 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %63 = memref.get_global @blockwrite_data_70 : memref<8xi32>
      aiex.npu.blockwrite(%63) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1188 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %64 = memref.get_global @blockwrite_data_71 : memref<8xi32>
      aiex.npu.blockwrite(%64) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1208 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %65 = memref.get_global @blockwrite_data_72 : memref<8xi32>
      aiex.npu.blockwrite(%65) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1228 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %66 = memref.get_global @blockwrite_data_73 : memref<8xi32>
      aiex.npu.blockwrite(%66) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1248 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %67 = memref.get_global @blockwrite_data_74 : memref<8xi32>
      aiex.npu.blockwrite(%67) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1268 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %68 = memref.get_global @blockwrite_data_75 : memref<8xi32>
      aiex.npu.blockwrite(%68) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1288 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %69 = memref.get_global @blockwrite_data_76 : memref<8xi32>
      aiex.npu.blockwrite(%69) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1308 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %70 = memref.get_global @blockwrite_data_77 : memref<8xi32>
      aiex.npu.blockwrite(%70) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1328 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %71 = memref.get_global @blockwrite_data_78 : memref<8xi32>
      aiex.npu.blockwrite(%71) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1348 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %72 = memref.get_global @blockwrite_data_79 : memref<8xi32>
      aiex.npu.blockwrite(%72) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1368 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %73 = memref.get_global @blockwrite_data_80 : memref<8xi32>
      aiex.npu.blockwrite(%73) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1388 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %74 = memref.get_global @blockwrite_data_81 : memref<8xi32>
      aiex.npu.blockwrite(%74) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1408 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %75 = memref.get_global @blockwrite_data_82 : memref<8xi32>
      aiex.npu.blockwrite(%75) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1428 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %76 = memref.get_global @blockwrite_data_83 : memref<8xi32>
      aiex.npu.blockwrite(%76) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1448 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %77 = memref.get_global @blockwrite_data_84 : memref<8xi32>
      aiex.npu.blockwrite(%77) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1468 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %78 = memref.get_global @blockwrite_data_85 : memref<8xi32>
      aiex.npu.blockwrite(%78) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1488 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %79 = memref.get_global @blockwrite_data_86 : memref<8xi32>
      aiex.npu.blockwrite(%79) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1508 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %80 = memref.get_global @blockwrite_data_87 : memref<8xi32>
      aiex.npu.blockwrite(%80) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1528 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %81 = memref.get_global @blockwrite_data_88 : memref<8xi32>
      aiex.npu.blockwrite(%81) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1548 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %82 = memref.get_global @blockwrite_data_89 : memref<8xi32>
      aiex.npu.blockwrite(%82) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1568 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %83 = memref.get_global @blockwrite_data_90 : memref<8xi32>
      aiex.npu.blockwrite(%83) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1588 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %84 = memref.get_global @blockwrite_data_91 : memref<8xi32>
      aiex.npu.blockwrite(%84) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1608 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %85 = memref.get_global @blockwrite_data_92 : memref<8xi32>
      aiex.npu.blockwrite(%85) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1628 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %86 = memref.get_global @blockwrite_data_93 : memref<8xi32>
      aiex.npu.blockwrite(%86) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1648 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %87 = memref.get_global @blockwrite_data_94 : memref<8xi32>
      aiex.npu.blockwrite(%87) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1668 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %88 = memref.get_global @blockwrite_data_95 : memref<8xi32>
      aiex.npu.blockwrite(%88) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1688 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %89 = memref.get_global @blockwrite_data_96 : memref<8xi32>
      aiex.npu.blockwrite(%89) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1708 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %90 = memref.get_global @blockwrite_data_97 : memref<8xi32>
      aiex.npu.blockwrite(%90) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1728 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %91 = memref.get_global @blockwrite_data_98 : memref<8xi32>
      aiex.npu.blockwrite(%91) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1748 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %92 = memref.get_global @blockwrite_data_99 : memref<8xi32>
      aiex.npu.blockwrite(%92) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1768 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %93 = memref.get_global @blockwrite_data_100 : memref<8xi32>
      aiex.npu.blockwrite(%93) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1788 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %94 = memref.get_global @blockwrite_data_101 : memref<8xi32>
      aiex.npu.blockwrite(%94) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1808 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %95 = memref.get_global @blockwrite_data_102 : memref<8xi32>
      aiex.npu.blockwrite(%95) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1828 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %96 = memref.get_global @blockwrite_data_103 : memref<8xi32>
      aiex.npu.blockwrite(%96) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1848 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %97 = memref.get_global @blockwrite_data_104 : memref<8xi32>
      aiex.npu.blockwrite(%97) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1868 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %98 = memref.get_global @blockwrite_data_105 : memref<8xi32>
      aiex.npu.blockwrite(%98) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1888 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %99 = memref.get_global @blockwrite_data_106 : memref<8xi32>
      aiex.npu.blockwrite(%99) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1908 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %100 = memref.get_global @blockwrite_data_107 : memref<8xi32>
      aiex.npu.blockwrite(%100) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1928 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %101 = memref.get_global @blockwrite_data_108 : memref<8xi32>
      aiex.npu.blockwrite(%101) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1948 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %102 = memref.get_global @blockwrite_data_109 : memref<8xi32>
      aiex.npu.blockwrite(%102) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1968 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %103 = memref.get_global @blockwrite_data_110 : memref<8xi32>
      aiex.npu.blockwrite(%103) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 1988 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %104 = memref.get_global @blockwrite_data_111 : memref<8xi32>
      aiex.npu.blockwrite(%104) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2008 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %105 = memref.get_global @blockwrite_data_112 : memref<8xi32>
      aiex.npu.blockwrite(%105) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2028 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %106 = memref.get_global @blockwrite_data_113 : memref<8xi32>
      aiex.npu.blockwrite(%106) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2048 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %107 = memref.get_global @blockwrite_data_114 : memref<8xi32>
      aiex.npu.blockwrite(%107) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2068 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %108 = memref.get_global @blockwrite_data_115 : memref<8xi32>
      aiex.npu.blockwrite(%108) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2088 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %109 = memref.get_global @blockwrite_data_116 : memref<8xi32>
      aiex.npu.blockwrite(%109) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2108 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %110 = memref.get_global @blockwrite_data_117 : memref<8xi32>
      aiex.npu.blockwrite(%110) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2128 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %111 = memref.get_global @blockwrite_data_118 : memref<8xi32>
      aiex.npu.blockwrite(%111) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2148 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %112 = memref.get_global @blockwrite_data_119 : memref<8xi32>
      aiex.npu.blockwrite(%112) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2168 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %113 = memref.get_global @blockwrite_data_120 : memref<8xi32>
      aiex.npu.blockwrite(%113) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2188 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %114 = memref.get_global @blockwrite_data_121 : memref<8xi32>
      aiex.npu.blockwrite(%114) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2208 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %115 = memref.get_global @blockwrite_data_122 : memref<8xi32>
      aiex.npu.blockwrite(%115) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2228 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %116 = memref.get_global @blockwrite_data_123 : memref<8xi32>
      aiex.npu.blockwrite(%116) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2248 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %117 = memref.get_global @blockwrite_data_124 : memref<8xi32>
      aiex.npu.blockwrite(%117) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2268 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %118 = memref.get_global @blockwrite_data_125 : memref<8xi32>
      aiex.npu.blockwrite(%118) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2288 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %119 = memref.get_global @blockwrite_data_126 : memref<8xi32>
      aiex.npu.blockwrite(%119) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2308 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %120 = memref.get_global @blockwrite_data_127 : memref<8xi32>
      aiex.npu.blockwrite(%120) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2316 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %121 = memref.get_global @blockwrite_data_128 : memref<8xi32>
      aiex.npu.blockwrite(%121) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2324 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %122 = memref.get_global @blockwrite_data_129 : memref<8xi32>
      aiex.npu.blockwrite(%122) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2332 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %123 = memref.get_global @blockwrite_data_130 : memref<8xi32>
      aiex.npu.blockwrite(%123) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2340 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %124 = memref.get_global @blockwrite_data_131 : memref<8xi32>
      aiex.npu.blockwrite(%124) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2348 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %125 = memref.get_global @blockwrite_data_132 : memref<8xi32>
      aiex.npu.blockwrite(%125) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2356 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %126 = memref.get_global @blockwrite_data_133 : memref<8xi32>
      aiex.npu.blockwrite(%126) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2364 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %127 = memref.get_global @blockwrite_data_134 : memref<8xi32>
      aiex.npu.blockwrite(%127) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2372 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %128 = memref.get_global @blockwrite_data_135 : memref<8xi32>
      aiex.npu.blockwrite(%128) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2380 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %129 = memref.get_global @blockwrite_data_136 : memref<8xi32>
      aiex.npu.blockwrite(%129) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2388 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %130 = memref.get_global @blockwrite_data_137 : memref<8xi32>
      aiex.npu.blockwrite(%130) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2396 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %131 = memref.get_global @blockwrite_data_138 : memref<8xi32>
      aiex.npu.blockwrite(%131) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2404 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %132 = memref.get_global @blockwrite_data_139 : memref<8xi32>
      aiex.npu.blockwrite(%132) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2412 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %133 = memref.get_global @blockwrite_data_140 : memref<8xi32>
      aiex.npu.blockwrite(%133) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2420 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %134 = memref.get_global @blockwrite_data_141 : memref<8xi32>
      aiex.npu.blockwrite(%134) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2428 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %135 = memref.get_global @blockwrite_data_142 : memref<8xi32>
      aiex.npu.blockwrite(%135) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2436 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %136 = memref.get_global @blockwrite_data_143 : memref<8xi32>
      aiex.npu.blockwrite(%136) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2444 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %137 = memref.get_global @blockwrite_data_144 : memref<8xi32>
      aiex.npu.blockwrite(%137) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2452 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %138 = memref.get_global @blockwrite_data_145 : memref<8xi32>
      aiex.npu.blockwrite(%138) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2460 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %139 = memref.get_global @blockwrite_data_146 : memref<8xi32>
      aiex.npu.blockwrite(%139) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2468 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %140 = memref.get_global @blockwrite_data_147 : memref<8xi32>
      aiex.npu.blockwrite(%140) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2476 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %141 = memref.get_global @blockwrite_data_148 : memref<8xi32>
      aiex.npu.blockwrite(%141) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2484 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %142 = memref.get_global @blockwrite_data_149 : memref<8xi32>
      aiex.npu.blockwrite(%142) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2492 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %143 = memref.get_global @blockwrite_data_150 : memref<8xi32>
      aiex.npu.blockwrite(%143) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2500 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %144 = memref.get_global @blockwrite_data_151 : memref<8xi32>
      aiex.npu.blockwrite(%144) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2508 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %145 = memref.get_global @blockwrite_data_152 : memref<8xi32>
      aiex.npu.blockwrite(%145) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2516 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %146 = memref.get_global @blockwrite_data_153 : memref<8xi32>
      aiex.npu.blockwrite(%146) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2524 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %147 = memref.get_global @blockwrite_data_154 : memref<8xi32>
      aiex.npu.blockwrite(%147) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2532 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %148 = memref.get_global @blockwrite_data_155 : memref<8xi32>
      aiex.npu.blockwrite(%148) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2540 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %149 = memref.get_global @blockwrite_data_156 : memref<8xi32>
      aiex.npu.blockwrite(%149) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2548 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %150 = memref.get_global @blockwrite_data_157 : memref<8xi32>
      aiex.npu.blockwrite(%150) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2556 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %151 = memref.get_global @blockwrite_data_158 : memref<8xi32>
      aiex.npu.blockwrite(%151) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2564 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %152 = memref.get_global @blockwrite_data_159 : memref<8xi32>
      aiex.npu.blockwrite(%152) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2572 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %153 = memref.get_global @blockwrite_data_160 : memref<8xi32>
      aiex.npu.blockwrite(%153) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2580 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %154 = memref.get_global @blockwrite_data_161 : memref<8xi32>
      aiex.npu.blockwrite(%154) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2600 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %155 = memref.get_global @blockwrite_data_162 : memref<8xi32>
      aiex.npu.blockwrite(%155) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2612 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %156 = memref.get_global @blockwrite_data_163 : memref<8xi32>
      aiex.npu.blockwrite(%156) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2632 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %157 = memref.get_global @blockwrite_data_164 : memref<8xi32>
      aiex.npu.blockwrite(%157) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2644 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %158 = memref.get_global @blockwrite_data_165 : memref<8xi32>
      aiex.npu.blockwrite(%158) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2664 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %159 = memref.get_global @blockwrite_data_166 : memref<8xi32>
      aiex.npu.blockwrite(%159) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2676 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %160 = memref.get_global @blockwrite_data_167 : memref<8xi32>
      aiex.npu.blockwrite(%160) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2696 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %161 = memref.get_global @blockwrite_data_168 : memref<8xi32>
      aiex.npu.blockwrite(%161) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2708 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %162 = memref.get_global @blockwrite_data_169 : memref<8xi32>
      aiex.npu.blockwrite(%162) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2728 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %163 = memref.get_global @blockwrite_data_170 : memref<8xi32>
      aiex.npu.blockwrite(%163) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2740 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %164 = memref.get_global @blockwrite_data_171 : memref<8xi32>
      aiex.npu.blockwrite(%164) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2760 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %165 = memref.get_global @blockwrite_data_172 : memref<8xi32>
      aiex.npu.blockwrite(%165) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2772 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %166 = memref.get_global @blockwrite_data_173 : memref<8xi32>
      aiex.npu.blockwrite(%166) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2780 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %167 = memref.get_global @blockwrite_data_174 : memref<8xi32>
      aiex.npu.blockwrite(%167) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2788 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %168 = memref.get_global @blockwrite_data_175 : memref<8xi32>
      aiex.npu.blockwrite(%168) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2796 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %169 = memref.get_global @blockwrite_data_176 : memref<8xi32>
      aiex.npu.blockwrite(%169) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2804 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %170 = memref.get_global @blockwrite_data_177 : memref<8xi32>
      aiex.npu.blockwrite(%170) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2812 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %171 = memref.get_global @blockwrite_data_178 : memref<8xi32>
      aiex.npu.blockwrite(%171) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2820 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %172 = memref.get_global @blockwrite_data_179 : memref<8xi32>
      aiex.npu.blockwrite(%172) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2828 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %173 = memref.get_global @blockwrite_data_180 : memref<8xi32>
      aiex.npu.blockwrite(%173) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2836 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %174 = memref.get_global @blockwrite_data_181 : memref<8xi32>
      aiex.npu.blockwrite(%174) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2844 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %175 = memref.get_global @blockwrite_data_182 : memref<8xi32>
      aiex.npu.blockwrite(%175) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2852 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %176 = memref.get_global @blockwrite_data_183 : memref<8xi32>
      aiex.npu.blockwrite(%176) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2860 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %177 = memref.get_global @blockwrite_data_184 : memref<8xi32>
      aiex.npu.blockwrite(%177) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2868 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %178 = memref.get_global @blockwrite_data_185 : memref<8xi32>
      aiex.npu.blockwrite(%178) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2876 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %179 = memref.get_global @blockwrite_data_186 : memref<8xi32>
      aiex.npu.blockwrite(%179) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2884 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %180 = memref.get_global @blockwrite_data_187 : memref<8xi32>
      aiex.npu.blockwrite(%180) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2892 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %181 = memref.get_global @blockwrite_data_188 : memref<8xi32>
      aiex.npu.blockwrite(%181) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2900 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %182 = memref.get_global @blockwrite_data_189 : memref<8xi32>
      aiex.npu.blockwrite(%182) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2908 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %183 = memref.get_global @blockwrite_data_190 : memref<8xi32>
      aiex.npu.blockwrite(%183) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2916 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %184 = memref.get_global @blockwrite_data_191 : memref<8xi32>
      aiex.npu.blockwrite(%184) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2924 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %185 = memref.get_global @blockwrite_data_192 : memref<8xi32>
      aiex.npu.blockwrite(%185) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2932 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %186 = memref.get_global @blockwrite_data_193 : memref<8xi32>
      aiex.npu.blockwrite(%186) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2940 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %187 = memref.get_global @blockwrite_data_194 : memref<8xi32>
      aiex.npu.blockwrite(%187) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2948 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %188 = memref.get_global @blockwrite_data_195 : memref<8xi32>
      aiex.npu.blockwrite(%188) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2956 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %189 = memref.get_global @blockwrite_data_196 : memref<8xi32>
      aiex.npu.blockwrite(%189) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2964 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %190 = memref.get_global @blockwrite_data_197 : memref<8xi32>
      aiex.npu.blockwrite(%190) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2972 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %191 = memref.get_global @blockwrite_data_198 : memref<8xi32>
      aiex.npu.blockwrite(%191) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2980 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %192 = memref.get_global @blockwrite_data_199 : memref<8xi32>
      aiex.npu.blockwrite(%192) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2988 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %193 = memref.get_global @blockwrite_data_200 : memref<8xi32>
      aiex.npu.blockwrite(%193) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 2996 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %194 = memref.get_global @blockwrite_data_201 : memref<8xi32>
      aiex.npu.blockwrite(%194) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 3004 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %195 = memref.get_global @blockwrite_data_202 : memref<8xi32>
      aiex.npu.blockwrite(%195) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 3012 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %196 = memref.get_global @blockwrite_data_203 : memref<8xi32>
      aiex.npu.blockwrite(%196) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 3020 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %197 = memref.get_global @blockwrite_data_204 : memref<8xi32>
      aiex.npu.blockwrite(%197) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 3028 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %198 = memref.get_global @blockwrite_data_205 : memref<8xi32>
      aiex.npu.blockwrite(%198) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 3036 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %199 = memref.get_global @blockwrite_data_206 : memref<8xi32>
      aiex.npu.blockwrite(%199) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 3044 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
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

