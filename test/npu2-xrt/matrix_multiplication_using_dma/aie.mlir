//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %lock_3_1 = aie.lock(%tile_3_1, 9) {init = 4 : i32}
    %lock_3_1_0 = aie.lock(%tile_3_1, 8) {init = 0 : i32}
    %lock_3_1_1 = aie.lock(%tile_3_1, 7) {init = 1 : i32}
    %lock_3_1_2 = aie.lock(%tile_3_1, 6) {init = 0 : i32}
    %lock_3_1_3 = aie.lock(%tile_3_1, 5) {init = 1 : i32}
    %lock_3_1_4 = aie.lock(%tile_3_1, 4) {init = 0 : i32}
    %lock_3_1_5 = aie.lock(%tile_3_1, 3) {init = 1 : i32}
    %lock_3_1_6 = aie.lock(%tile_3_1, 2) {init = 0 : i32}
    %lock_3_1_7 = aie.lock(%tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_8 = aie.lock(%tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%tile_2_1, 9) {init = 4 : i32}
    %lock_2_1_9 = aie.lock(%tile_2_1, 8) {init = 0 : i32}
    %lock_2_1_10 = aie.lock(%tile_2_1, 7) {init = 1 : i32}
    %lock_2_1_11 = aie.lock(%tile_2_1, 6) {init = 0 : i32}
    %lock_2_1_12 = aie.lock(%tile_2_1, 5) {init = 1 : i32}
    %lock_2_1_13 = aie.lock(%tile_2_1, 4) {init = 0 : i32}
    %lock_2_1_14 = aie.lock(%tile_2_1, 3) {init = 1 : i32}
    %lock_2_1_15 = aie.lock(%tile_2_1, 2) {init = 0 : i32}
    %lock_2_1_16 = aie.lock(%tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_17 = aie.lock(%tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%tile_1_1, 9) {init = 4 : i32}
    %lock_1_1_18 = aie.lock(%tile_1_1, 8) {init = 0 : i32}
    %lock_1_1_19 = aie.lock(%tile_1_1, 7) {init = 1 : i32}
    %lock_1_1_20 = aie.lock(%tile_1_1, 6) {init = 0 : i32}
    %lock_1_1_21 = aie.lock(%tile_1_1, 5) {init = 1 : i32}
    %lock_1_1_22 = aie.lock(%tile_1_1, 4) {init = 0 : i32}
    %lock_1_1_23 = aie.lock(%tile_1_1, 3) {init = 1 : i32}
    %lock_1_1_24 = aie.lock(%tile_1_1, 2) {init = 0 : i32}
    %lock_1_1_25 = aie.lock(%tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_26 = aie.lock(%tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%tile_0_1, 9) {init = 4 : i32}
    %lock_0_1_27 = aie.lock(%tile_0_1, 8) {init = 0 : i32}
    %lock_0_1_28 = aie.lock(%tile_0_1, 7) {init = 1 : i32}
    %lock_0_1_29 = aie.lock(%tile_0_1, 6) {init = 0 : i32}
    %lock_0_1_30 = aie.lock(%tile_0_1, 5) {init = 1 : i32}
    %lock_0_1_31 = aie.lock(%tile_0_1, 4) {init = 0 : i32}
    %lock_0_1_32 = aie.lock(%tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_33 = aie.lock(%tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_34 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_35 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 2 : i32}
    %lock_0_2_36 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_37 = aie.lock(%tile_0_2, 3) {init = 2 : i32}
    %lock_0_2_38 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_39 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_40 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 2 : i32}
    %lock_1_2_41 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_42 = aie.lock(%tile_1_2, 3) {init = 2 : i32}
    %lock_1_2_43 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_44 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_45 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 2 : i32}
    %lock_2_2_46 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_47 = aie.lock(%tile_2_2, 3) {init = 2 : i32}
    %lock_2_2_48 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_49 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_50 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 2 : i32}
    %lock_3_2_51 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_52 = aie.lock(%tile_3_2, 3) {init = 2 : i32}
    %lock_3_2_53 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_54 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_55 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 2 : i32}
    %lock_0_3_56 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_57 = aie.lock(%tile_0_3, 3) {init = 2 : i32}
    %lock_0_3_58 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_59 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_60 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 5) {init = 2 : i32}
    %lock_1_3_61 = aie.lock(%tile_1_3, 4) {init = 0 : i32}
    %lock_1_3_62 = aie.lock(%tile_1_3, 3) {init = 2 : i32}
    %lock_1_3_63 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_64 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_65 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 5) {init = 2 : i32}
    %lock_2_3_66 = aie.lock(%tile_2_3, 4) {init = 0 : i32}
    %lock_2_3_67 = aie.lock(%tile_2_3, 3) {init = 2 : i32}
    %lock_2_3_68 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_69 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_70 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 5) {init = 2 : i32}
    %lock_3_3_71 = aie.lock(%tile_3_3, 4) {init = 0 : i32}
    %lock_3_3_72 = aie.lock(%tile_3_3, 3) {init = 2 : i32}
    %lock_3_3_73 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_74 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_75 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 2 : i32}
    %lock_0_4_76 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_77 = aie.lock(%tile_0_4, 3) {init = 2 : i32}
    %lock_0_4_78 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_79 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_80 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 5) {init = 2 : i32}
    %lock_1_4_81 = aie.lock(%tile_1_4, 4) {init = 0 : i32}
    %lock_1_4_82 = aie.lock(%tile_1_4, 3) {init = 2 : i32}
    %lock_1_4_83 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_84 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_85 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 5) {init = 2 : i32}
    %lock_2_4_86 = aie.lock(%tile_2_4, 4) {init = 0 : i32}
    %lock_2_4_87 = aie.lock(%tile_2_4, 3) {init = 2 : i32}
    %lock_2_4_88 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_89 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_90 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 5) {init = 2 : i32}
    %lock_3_4_91 = aie.lock(%tile_3_4, 4) {init = 0 : i32}
    %lock_3_4_92 = aie.lock(%tile_3_4, 3) {init = 2 : i32}
    %lock_3_4_93 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_94 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_95 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 5) {init = 2 : i32}
    %lock_0_5_96 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_97 = aie.lock(%tile_0_5, 3) {init = 2 : i32}
    %lock_0_5_98 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_99 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_100 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 5) {init = 2 : i32}
    %lock_1_5_101 = aie.lock(%tile_1_5, 4) {init = 0 : i32}
    %lock_1_5_102 = aie.lock(%tile_1_5, 3) {init = 2 : i32}
    %lock_1_5_103 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_104 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_105 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 5) {init = 2 : i32}
    %lock_2_5_106 = aie.lock(%tile_2_5, 4) {init = 0 : i32}
    %lock_2_5_107 = aie.lock(%tile_2_5, 3) {init = 2 : i32}
    %lock_2_5_108 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_109 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_110 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 5) {init = 2 : i32}
    %lock_3_5_111 = aie.lock(%tile_3_5, 4) {init = 0 : i32}
    %lock_3_5_112 = aie.lock(%tile_3_5, 3) {init = 2 : i32}
    %lock_3_5_113 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_114 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_115 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf99 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf99"} : memref<32x256xbf16, 1> 
    %buf98 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf98"} : memref<32x256xbf16, 1> 
    %buf97 = aie.buffer(%tile_2_1) {mem_bank = 0 : i32, sym_name = "buf97"} : memref<32x256xbf16, 1> 
    %buf96 = aie.buffer(%tile_3_1) {mem_bank = 0 : i32, sym_name = "buf96"} : memref<32x256xbf16, 1> 
    %buf95 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf95"} : memref<256x32xbf16, 1> 
    %buf94 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf94"} : memref<256x32xbf16, 1> 
    %buf93 = aie.buffer(%tile_2_1) {mem_bank = 0 : i32, sym_name = "buf93"} : memref<256x32xbf16, 1> 
    %buf92 = aie.buffer(%tile_3_1) {mem_bank = 0 : i32, sym_name = "buf92"} : memref<256x32xbf16, 1> 
    %buf91 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf91"} : memref<32x256xbf16, 1> 
    %buf90 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf90"} : memref<32x256xbf16, 1> 
    %buf89 = aie.buffer(%tile_2_1) {mem_bank = 0 : i32, sym_name = "buf89"} : memref<32x256xbf16, 1> 
    %buf88 = aie.buffer(%tile_3_1) {mem_bank = 0 : i32, sym_name = "buf88"} : memref<32x256xbf16, 1> 
    %buf87 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf87"} : memref<256x32xbf16, 1> 
    %buf86 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf86"} : memref<256x32xbf16, 1> 
    %buf85 = aie.buffer(%tile_2_1) {mem_bank = 0 : i32, sym_name = "buf85"} : memref<256x32xbf16, 1> 
    %buf84 = aie.buffer(%tile_3_1) {mem_bank = 0 : i32, sym_name = "buf84"} : memref<256x32xbf16, 1> 
    %buf83 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf83"} : memref<32x128xf32, 1> 
    %buf82 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf82"} : memref<32x128xf32, 1> 
    %buf81 = aie.buffer(%tile_2_1) {mem_bank = 0 : i32, sym_name = "buf81"} : memref<32x128xf32, 1> 
    %buf80 = aie.buffer(%tile_3_1) {mem_bank = 0 : i32, sym_name = "buf80"} : memref<32x128xf32, 1> 
    %buf79 = aie.buffer(%tile_3_5) {mem_bank = 0 : i32, sym_name = "buf79"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf78 = aie.buffer(%tile_3_5) {mem_bank = 0 : i32, sym_name = "buf78"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf77 = aie.buffer(%tile_3_5) {mem_bank = 0 : i32, sym_name = "buf77"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf76 = aie.buffer(%tile_3_5) {mem_bank = 0 : i32, sym_name = "buf76"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf75 = aie.buffer(%tile_3_5) {mem_bank = 0 : i32, sym_name = "buf75"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf74 = aie.buffer(%tile_2_5) {mem_bank = 0 : i32, sym_name = "buf74"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf73 = aie.buffer(%tile_2_5) {mem_bank = 0 : i32, sym_name = "buf73"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf72 = aie.buffer(%tile_2_5) {mem_bank = 0 : i32, sym_name = "buf72"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf71 = aie.buffer(%tile_2_5) {mem_bank = 0 : i32, sym_name = "buf71"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_2_5) {mem_bank = 0 : i32, sym_name = "buf70"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_1_5) {mem_bank = 0 : i32, sym_name = "buf69"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf68 = aie.buffer(%tile_1_5) {mem_bank = 0 : i32, sym_name = "buf68"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf67 = aie.buffer(%tile_1_5) {mem_bank = 0 : i32, sym_name = "buf67"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_1_5) {mem_bank = 0 : i32, sym_name = "buf66"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_1_5) {mem_bank = 0 : i32, sym_name = "buf65"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_0_5) {mem_bank = 0 : i32, sym_name = "buf64"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf63 = aie.buffer(%tile_0_5) {mem_bank = 0 : i32, sym_name = "buf63"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_0_5) {mem_bank = 0 : i32, sym_name = "buf62"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_0_5) {mem_bank = 0 : i32, sym_name = "buf61"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_0_5) {mem_bank = 0 : i32, sym_name = "buf60"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_3_4) {mem_bank = 0 : i32, sym_name = "buf59"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf58 = aie.buffer(%tile_3_4) {mem_bank = 0 : i32, sym_name = "buf58"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_3_4) {mem_bank = 0 : i32, sym_name = "buf57"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_3_4) {mem_bank = 0 : i32, sym_name = "buf56"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_3_4) {mem_bank = 0 : i32, sym_name = "buf55"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_2_4) {mem_bank = 0 : i32, sym_name = "buf54"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf53 = aie.buffer(%tile_2_4) {mem_bank = 0 : i32, sym_name = "buf53"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_2_4) {mem_bank = 0 : i32, sym_name = "buf52"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_2_4) {mem_bank = 0 : i32, sym_name = "buf51"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_2_4) {mem_bank = 0 : i32, sym_name = "buf50"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_1_4) {mem_bank = 0 : i32, sym_name = "buf49"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf48 = aie.buffer(%tile_1_4) {mem_bank = 0 : i32, sym_name = "buf48"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_1_4) {mem_bank = 0 : i32, sym_name = "buf47"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_1_4) {mem_bank = 0 : i32, sym_name = "buf46"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_1_4) {mem_bank = 0 : i32, sym_name = "buf45"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_0_4) {mem_bank = 0 : i32, sym_name = "buf44"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf43 = aie.buffer(%tile_0_4) {mem_bank = 0 : i32, sym_name = "buf43"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_0_4) {mem_bank = 0 : i32, sym_name = "buf42"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_0_4) {mem_bank = 0 : i32, sym_name = "buf41"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_0_4) {mem_bank = 0 : i32, sym_name = "buf40"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf39 = aie.buffer(%tile_3_3) {mem_bank = 0 : i32, sym_name = "buf39"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf38 = aie.buffer(%tile_3_3) {mem_bank = 0 : i32, sym_name = "buf38"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf37 = aie.buffer(%tile_3_3) {mem_bank = 0 : i32, sym_name = "buf37"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf36 = aie.buffer(%tile_3_3) {mem_bank = 0 : i32, sym_name = "buf36"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf35 = aie.buffer(%tile_3_3) {mem_bank = 0 : i32, sym_name = "buf35"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf34 = aie.buffer(%tile_2_3) {mem_bank = 0 : i32, sym_name = "buf34"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf33 = aie.buffer(%tile_2_3) {mem_bank = 0 : i32, sym_name = "buf33"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf32 = aie.buffer(%tile_2_3) {mem_bank = 0 : i32, sym_name = "buf32"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf31 = aie.buffer(%tile_2_3) {mem_bank = 0 : i32, sym_name = "buf31"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf30 = aie.buffer(%tile_2_3) {mem_bank = 0 : i32, sym_name = "buf30"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf29 = aie.buffer(%tile_1_3) {mem_bank = 0 : i32, sym_name = "buf29"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf28 = aie.buffer(%tile_1_3) {mem_bank = 0 : i32, sym_name = "buf28"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf27 = aie.buffer(%tile_1_3) {mem_bank = 0 : i32, sym_name = "buf27"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_1_3) {mem_bank = 0 : i32, sym_name = "buf26"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_1_3) {mem_bank = 0 : i32, sym_name = "buf25"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_0_3) {mem_bank = 0 : i32, sym_name = "buf24"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf23 = aie.buffer(%tile_0_3) {mem_bank = 0 : i32, sym_name = "buf23"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_0_3) {mem_bank = 0 : i32, sym_name = "buf22"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_0_3) {mem_bank = 0 : i32, sym_name = "buf21"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_0_3) {mem_bank = 0 : i32, sym_name = "buf20"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf19"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf18 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf18"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf17"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf16"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf15"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf14"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf13 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf13"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf12"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf11"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf10 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf10"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf9 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf9"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf8 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf8"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf7 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf7"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf6 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf6"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf5 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf5"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf4"} : memref<4x4x8x8xf32, 2 : i32> 
    %buf3 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf3"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf2"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf1"} : memref<4x4x8x8xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf0"} : memref<4x4x8x8xbf16, 2 : i32> 
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_5_112, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_5_113, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_5_112, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf77 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_5_113, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_5_111, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf78 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_5_111, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_5_115, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf79 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_5_114, Release, 1)
      aie.next_bd ^bb8
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_114, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf79) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_3_5_113, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf76, %buf75, %buf79) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_112, Release, 1)
        aie.use_lock(%lock_3_5, Release, 1)
        aie.use_lock(%lock_3_5_113, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5_111, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf77, %buf78, %buf79) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_112, Release, 1)
        aie.use_lock(%lock_3_5, Release, 1)
      }
      aie.use_lock(%lock_3_5_115, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_3_5.elf", link_with = "mm.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_5_107, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf71 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_5_108, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_5_107, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf72 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_5_108, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf70 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_5_106, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf73 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_5_106, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_5_110, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_5_109, Release, 1)
      aie.next_bd ^bb8
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_109, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf74) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_2_5_108, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5_106, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf71, %buf70, %buf74) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_107, Release, 1)
        aie.use_lock(%lock_2_5, Release, 1)
        aie.use_lock(%lock_2_5_108, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5_106, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf72, %buf73, %buf74) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_107, Release, 1)
        aie.use_lock(%lock_2_5, Release, 1)
      }
      aie.use_lock(%lock_2_5_110, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_2_5.elf", link_with = "mm.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_5_102, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf66 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_5_103, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_5_102, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf67 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_5_103, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf65 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_5_101, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf68 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_5_101, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_5_105, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf69 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_5_104, Release, 1)
      aie.next_bd ^bb8
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_104, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf69) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_1_5_103, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5_101, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf66, %buf65, %buf69) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_102, Release, 1)
        aie.use_lock(%lock_1_5, Release, 1)
        aie.use_lock(%lock_1_5_103, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5_101, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf67, %buf68, %buf69) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_102, Release, 1)
        aie.use_lock(%lock_1_5, Release, 1)
      }
      aie.use_lock(%lock_1_5_105, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_1_5.elf", link_with = "mm.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_5_97, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf61 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_98, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_5_97, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf62 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_98, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf60 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_96, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf63 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_96, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_5_100, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf64 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_5_99, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_99, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf64) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_0_5_98, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5_96, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf61, %buf60, %buf64) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_97, Release, 1)
        aie.use_lock(%lock_0_5, Release, 1)
        aie.use_lock(%lock_0_5_98, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5_96, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf62, %buf63, %buf64) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_97, Release, 1)
        aie.use_lock(%lock_0_5, Release, 1)
      }
      aie.use_lock(%lock_0_5_100, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_5.elf", link_with = "mm.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_4_92, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf56 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_4_93, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_4_92, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf57 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_4_93, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf55 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_4_91, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf58 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_4_91, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_4_95, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf59 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_4_94, Release, 1)
      aie.next_bd ^bb8
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_94, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf59) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_3_4_93, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4_91, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf56, %buf55, %buf59) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_92, Release, 1)
        aie.use_lock(%lock_3_4, Release, 1)
        aie.use_lock(%lock_3_4_93, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4_91, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf57, %buf58, %buf59) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_92, Release, 1)
        aie.use_lock(%lock_3_4, Release, 1)
      }
      aie.use_lock(%lock_3_4_95, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_3_4.elf", link_with = "mm.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_4_87, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf51 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_4_88, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_4_87, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf52 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_4_88, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf50 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_4_86, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf53 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_4_86, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_4_90, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_4_89, Release, 1)
      aie.next_bd ^bb8
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_89, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf54) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_2_4_88, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4_86, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf51, %buf50, %buf54) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_87, Release, 1)
        aie.use_lock(%lock_2_4, Release, 1)
        aie.use_lock(%lock_2_4_88, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4_86, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf52, %buf53, %buf54) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_87, Release, 1)
        aie.use_lock(%lock_2_4, Release, 1)
      }
      aie.use_lock(%lock_2_4_90, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_2_4.elf", link_with = "mm.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_4_82, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf46 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_4_83, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_4_82, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf47 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_4_83, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf45 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_4_81, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf48 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_4_81, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_4_85, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf49 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_4_84, Release, 1)
      aie.next_bd ^bb8
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_84, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf49) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_1_4_83, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4_81, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf46, %buf45, %buf49) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_82, Release, 1)
        aie.use_lock(%lock_1_4, Release, 1)
        aie.use_lock(%lock_1_4_83, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4_81, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf47, %buf48, %buf49) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_82, Release, 1)
        aie.use_lock(%lock_1_4, Release, 1)
      }
      aie.use_lock(%lock_1_4_85, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_1_4.elf", link_with = "mm.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_4_77, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf41 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_78, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_4_77, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_78, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf40 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_76, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf43 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_76, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_4_80, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_4_79, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_79, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf44) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_0_4_78, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4_76, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf41, %buf40, %buf44) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_77, Release, 1)
        aie.use_lock(%lock_0_4, Release, 1)
        aie.use_lock(%lock_0_4_78, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4_76, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf42, %buf43, %buf44) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_77, Release, 1)
        aie.use_lock(%lock_0_4, Release, 1)
      }
      aie.use_lock(%lock_0_4_80, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_4.elf", link_with = "mm.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_3_72, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_3_73, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_3_72, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf37 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_3_73, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf35 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_3_71, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_3_71, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_3_75, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf39 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_3_74, Release, 1)
      aie.next_bd ^bb8
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_74, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf39) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_3_3_73, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_71, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf36, %buf35, %buf39) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_72, Release, 1)
        aie.use_lock(%lock_3_3, Release, 1)
        aie.use_lock(%lock_3_3_73, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_71, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf37, %buf38, %buf39) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_72, Release, 1)
        aie.use_lock(%lock_3_3, Release, 1)
      }
      aie.use_lock(%lock_3_3_75, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_3_3.elf", link_with = "mm.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_3_67, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf31 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_3_68, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_3_67, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf32 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_3_68, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_3_66, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf33 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_3_66, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_3_70, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_3_69, Release, 1)
      aie.next_bd ^bb8
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_69, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf34) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_2_3_68, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_66, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf31, %buf30, %buf34) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_67, Release, 1)
        aie.use_lock(%lock_2_3, Release, 1)
        aie.use_lock(%lock_2_3_68, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_66, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf32, %buf33, %buf34) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_67, Release, 1)
        aie.use_lock(%lock_2_3, Release, 1)
      }
      aie.use_lock(%lock_2_3_70, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_2_3.elf", link_with = "mm.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_3_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_3_63, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_3_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf27 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_3_63, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf25 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_3_61, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_3_61, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_3_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf29 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_3_64, Release, 1)
      aie.next_bd ^bb8
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_64, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf29) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_1_3_63, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_61, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf26, %buf25, %buf29) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_62, Release, 1)
        aie.use_lock(%lock_1_3, Release, 1)
        aie.use_lock(%lock_1_3_63, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_61, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf27, %buf28, %buf29) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_62, Release, 1)
        aie.use_lock(%lock_1_3, Release, 1)
      }
      aie.use_lock(%lock_1_3_65, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_1_3.elf", link_with = "mm.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_3_57, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf21 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_58, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_3_57, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_58, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_56, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf23 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_56, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_3_60, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf24 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_3_59, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_59, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf24) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_0_3_58, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_56, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf21, %buf20, %buf24) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_57, Release, 1)
        aie.use_lock(%lock_0_3, Release, 1)
        aie.use_lock(%lock_0_3_58, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_56, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf22, %buf23, %buf24) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_57, Release, 1)
        aie.use_lock(%lock_0_3, Release, 1)
      }
      aie.use_lock(%lock_0_3_60, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_3.elf", link_with = "mm.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_2_52, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf16 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_2_53, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_2_52, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf17 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_2_53, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf15 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_2_51, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf18 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_3_2_51, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_2_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf19 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_2_54, Release, 1)
      aie.next_bd ^bb8
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_54, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf19) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_3_2_53, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_51, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf16, %buf15, %buf19) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_52, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
        aie.use_lock(%lock_3_2_53, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_51, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf17, %buf18, %buf19) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_52, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      }
      aie.use_lock(%lock_3_2_55, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_3_2.elf", link_with = "mm.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_2_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_2_48, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_2_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_2_48, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_2_46, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_2_2_46, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_2_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_2_49, Release, 1)
      aie.next_bd ^bb8
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_49, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf14) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_2_2_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_46, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf11, %buf10, %buf14) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_47, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
        aie.use_lock(%lock_2_2_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_46, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf12, %buf13, %buf14) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_47, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      }
      aie.use_lock(%lock_2_2_50, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_2_2.elf", link_with = "mm.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2_42, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_2_43, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_2_42, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_2_43, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_2_41, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_1_2_41, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_2_45, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_2_44, Release, 1)
      aie.next_bd ^bb8
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_44, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf9) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_1_2_43, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_41, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf6, %buf5, %buf9) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_42, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
        aie.use_lock(%lock_1_2_43, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_41, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf7, %buf8, %buf9) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_42, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      }
      aie.use_lock(%lock_1_2_45, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_1_2.elf", link_with = "mm.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_37, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_38, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_37, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_36, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<4x4x8x8xbf16, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_36, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_2_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<4x4x8x8xf32, 2 : i32>, 0, 1024, [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_2_39, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_39, AcquireGreaterEqual, 1)
      func.call @linalg_fill_f32_view4x4x8x8xf32as2(%cst, %buf4) : (f32, memref<4x4x8x8xf32, 2 : i32>) -> ()
      scf.for %arg0 = %c0 to %c64 step %c8 {
        aie.use_lock(%lock_0_2_38, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_36, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf1, %buf0, %buf4) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_37, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
        aie.use_lock(%lock_0_2_38, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_36, AcquireGreaterEqual, 1)
        func.call @matmul_bf16_f32(%buf2, %buf3, %buf4) : (memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_37, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      }
      aie.use_lock(%lock_0_2_40, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_2.elf", link_with = "mm.o"}
    func.func private @linalg_fill_f32_view4x4x8x8xf32as2(f32, memref<4x4x8x8xf32, 2 : i32>)
    func.func private @matmul_bf16_f32(memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xbf16, 2 : i32>, memref<4x4x8x8xf32, 2 : i32>)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_1_1, DMA : 0)
    aie.flow(%tile_1_0, DMA : 0, %tile_2_1, DMA : 0)
    aie.flow(%tile_1_0, DMA : 1, %tile_3_1, DMA : 0)
    aie.flow(%tile_2_0, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_2_0, DMA : 1, %tile_1_1, DMA : 1)
    aie.flow(%tile_3_0, DMA : 0, %tile_2_1, DMA : 1)
    aie.flow(%tile_3_0, DMA : 1, %tile_3_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_0_0, DMA : 1)
    aie.flow(%tile_2_1, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_3_1, DMA : 0, %tile_1_0, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_4, DMA : 0)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%tile_2_1, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%tile_2_1, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%tile_2_1, DMA : 1, %tile_2_4, DMA : 0)
    aie.flow(%tile_2_1, DMA : 1, %tile_2_5, DMA : 0)
    aie.flow(%tile_3_1, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%tile_3_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%tile_3_1, DMA : 1, %tile_3_4, DMA : 0)
    aie.flow(%tile_3_1, DMA : 1, %tile_3_5, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_1, DMA : 2, %tile_1_2, DMA : 1)
    aie.flow(%tile_0_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%tile_0_1, DMA : 2, %tile_3_2, DMA : 1)
    aie.flow(%tile_1_1, DMA : 2, %tile_0_3, DMA : 1)
    aie.flow(%tile_1_1, DMA : 2, %tile_1_3, DMA : 1)
    aie.flow(%tile_1_1, DMA : 2, %tile_2_3, DMA : 1)
    aie.flow(%tile_1_1, DMA : 2, %tile_3_3, DMA : 1)
    aie.flow(%tile_2_1, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%tile_2_1, DMA : 2, %tile_1_4, DMA : 1)
    aie.flow(%tile_2_1, DMA : 2, %tile_2_4, DMA : 1)
    aie.flow(%tile_2_1, DMA : 2, %tile_3_4, DMA : 1)
    aie.flow(%tile_3_1, DMA : 2, %tile_0_5, DMA : 1)
    aie.flow(%tile_3_1, DMA : 2, %tile_1_5, DMA : 1)
    aie.flow(%tile_3_1, DMA : 2, %tile_2_5, DMA : 1)
    aie.flow(%tile_3_1, DMA : 2, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_1, DMA : 2)
    aie.flow(%tile_2_2, DMA : 0, %tile_2_1, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %tile_3_1, DMA : 2)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 3)
    aie.flow(%tile_1_3, DMA : 0, %tile_1_1, DMA : 3)
    aie.flow(%tile_2_3, DMA : 0, %tile_2_1, DMA : 3)
    aie.flow(%tile_3_3, DMA : 0, %tile_3_1, DMA : 3)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 4)
    aie.flow(%tile_1_4, DMA : 0, %tile_1_1, DMA : 4)
    aie.flow(%tile_2_4, DMA : 0, %tile_2_1, DMA : 4)
    aie.flow(%tile_3_4, DMA : 0, %tile_3_1, DMA : 4)
    aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 5)
    aie.flow(%tile_1_5, DMA : 0, %tile_1_1, DMA : 5)
    aie.flow(%tile_2_5, DMA : 0, %tile_2_1, DMA : 5)
    aie.flow(%tile_3_5, DMA : 0, %tile_3_1, DMA : 5)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb20, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_1_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf99 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_0_1_35, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_1_32, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf91 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_0_1_33, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_1_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_0_1_31, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_1_28, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf87 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_0_1_29, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb9
      %2 = aie.dma_start(S2MM, 2, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<32x128xf32, 1>, 0, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_27, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb11
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<32x128xf32, 1>, 32, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_27, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb13
      %4 = aie.dma_start(S2MM, 4, ^bb12, ^bb9, repeat_count = 1)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<32x128xf32, 1>, 64, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_27, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb15
      %5 = aie.dma_start(S2MM, 5, ^bb14, ^bb11, repeat_count = 1)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf83 : memref<32x128xf32, 1>, 96, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_27, Release, 1)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb17
      %6 = aie.dma_start(MM2S, 0, ^bb16, ^bb13, repeat_count = 1)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_0_1_27, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf83 : memref<32x128xf32, 1>, 0, 4096)
      aie.use_lock(%lock_0_1, Release, 4)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb20
      %7 = aie.dma_start(MM2S, 1, ^bb18, ^bb15, repeat_count = 1)
    ^bb18:  // 2 preds: ^bb17, ^bb19
      aie.use_lock(%lock_0_1_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf99 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_34, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%lock_0_1_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf91 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_32, Release, 1)
      aie.next_bd ^bb18
    ^bb20:  // pred: ^bb0
      %8 = aie.dma_start(MM2S, 2, ^bb21, ^bb17, repeat_count = 1)
    ^bb21:  // 2 preds: ^bb20, ^bb22
      aie.use_lock(%lock_0_1_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf95 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_30, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%lock_0_1_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf87 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_28, Release, 1)
      aie.next_bd ^bb21
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb20, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_1_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_1_1_26, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_1_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_1_1_24, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_1_1_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_1_1_22, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_1_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf86 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_1_1_20, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb9
      %2 = aie.dma_start(S2MM, 2, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<32x128xf32, 1>, 0, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb11
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<32x128xf32, 1>, 32, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb13
      %4 = aie.dma_start(S2MM, 4, ^bb12, ^bb9, repeat_count = 1)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<32x128xf32, 1>, 64, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb15
      %5 = aie.dma_start(S2MM, 5, ^bb14, ^bb11, repeat_count = 1)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf82 : memref<32x128xf32, 1>, 96, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_1_1_18, Release, 1)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb17
      %6 = aie.dma_start(MM2S, 0, ^bb16, ^bb13, repeat_count = 1)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_1_1_18, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf82 : memref<32x128xf32, 1>, 0, 4096)
      aie.use_lock(%lock_1_1, Release, 4)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb20
      %7 = aie.dma_start(MM2S, 1, ^bb18, ^bb15, repeat_count = 1)
    ^bb18:  // 2 preds: ^bb17, ^bb19
      aie.use_lock(%lock_1_1_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf98 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_1_25, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%lock_1_1_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_1_23, Release, 1)
      aie.next_bd ^bb18
    ^bb20:  // pred: ^bb0
      %8 = aie.dma_start(MM2S, 2, ^bb21, ^bb17, repeat_count = 1)
    ^bb21:  // 2 preds: ^bb20, ^bb22
      aie.use_lock(%lock_1_1_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_1_21, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%lock_1_1_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf86 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_1_1_19, Release, 1)
      aie.next_bd ^bb21
    }
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb20, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_2_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf97 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_2_1_17, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf89 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_2_1_15, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_2_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf93 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_2_1_13, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_2_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf85 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_2_1_11, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb9
      %2 = aie.dma_start(S2MM, 2, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<32x128xf32, 1>, 0, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb11
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<32x128xf32, 1>, 32, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb13
      %4 = aie.dma_start(S2MM, 4, ^bb12, ^bb9, repeat_count = 1)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<32x128xf32, 1>, 64, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb15
      %5 = aie.dma_start(S2MM, 5, ^bb14, ^bb11, repeat_count = 1)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf81 : memref<32x128xf32, 1>, 96, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb17
      %6 = aie.dma_start(MM2S, 0, ^bb16, ^bb13, repeat_count = 1)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf81 : memref<32x128xf32, 1>, 0, 4096)
      aie.use_lock(%lock_2_1, Release, 4)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb20
      %7 = aie.dma_start(MM2S, 1, ^bb18, ^bb15, repeat_count = 1)
    ^bb18:  // 2 preds: ^bb17, ^bb19
      aie.use_lock(%lock_2_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf97 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_1_16, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%lock_2_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf89 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_1_14, Release, 1)
      aie.next_bd ^bb18
    ^bb20:  // pred: ^bb0
      %8 = aie.dma_start(MM2S, 2, ^bb21, ^bb17, repeat_count = 1)
    ^bb21:  // 2 preds: ^bb20, ^bb22
      aie.use_lock(%lock_2_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf93 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_1_12, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%lock_2_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf85 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_2_1_10, Release, 1)
      aie.next_bd ^bb21
    }
    %memtile_dma_3_1 = aie.memtile_dma(%tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb20, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_3_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_3_1_8, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf88 : memref<32x256xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_3_1_6, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_3_1_4, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf84 : memref<256x32xbf16, 1>, 0, 8192)
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb9
      %2 = aie.dma_start(S2MM, 2, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<32x128xf32, 1>, 0, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb11
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<32x128xf32, 1>, 32, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb13
      %4 = aie.dma_start(S2MM, 4, ^bb12, ^bb9, repeat_count = 1)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<32x128xf32, 1>, 64, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb15
      %5 = aie.dma_start(S2MM, 5, ^bb14, ^bb11, repeat_count = 1)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf80 : memref<32x128xf32, 1>, 96, 1024, [<size = 32, stride = 128>, <size = 32, stride = 1>])
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb17
      %6 = aie.dma_start(MM2S, 0, ^bb16, ^bb13, repeat_count = 1)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf80 : memref<32x128xf32, 1>, 0, 4096)
      aie.use_lock(%lock_3_1, Release, 4)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb20
      %7 = aie.dma_start(MM2S, 1, ^bb18, ^bb15, repeat_count = 1)
    ^bb18:  // 2 preds: ^bb17, ^bb19
      aie.use_lock(%lock_3_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_1_7, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%lock_3_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf88 : memref<32x256xbf16, 1>, 0, 8192, [<size = 32, stride = 8>, <size = 32, stride = 256>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_1_5, Release, 1)
      aie.next_bd ^bb18
    ^bb20:  // pred: ^bb0
      %8 = aie.dma_start(MM2S, 2, ^bb21, ^bb17, repeat_count = 1)
    ^bb21:  // 2 preds: ^bb20, ^bb22
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf84 : memref<256x32xbf16, 1>, 0, 8192, [<size = 8, stride = 1024>, <size = 4, stride = 8>, <size = 32, stride = 32>, <size = 8, stride = 1>])
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb21
    }
    aie.shim_dma_allocation @airMemcpyId78(S2MM, 0, 0)
    memref.global "public" @airMemcpyId78 : memref<32x128xf32, 1>
    aie.shim_dma_allocation @airMemcpyId79(S2MM, 1, 0)
    memref.global "public" @airMemcpyId79 : memref<32x128xf32, 1>
    aie.shim_dma_allocation @airMemcpyId80(S2MM, 0, 1)
    memref.global "public" @airMemcpyId80 : memref<32x128xf32, 1>
    aie.shim_dma_allocation @airMemcpyId81(S2MM, 1, 1)
    memref.global "public" @airMemcpyId81 : memref<32x128xf32, 1>
    aie.shim_dma_allocation @airMemcpyId13(MM2S, 0, 0)
    memref.global "public" @airMemcpyId13 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId29(MM2S, 0, 0)
    memref.global "public" @airMemcpyId29 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId15(MM2S, 1, 0)
    memref.global "public" @airMemcpyId15 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId31(MM2S, 1, 0)
    memref.global "public" @airMemcpyId31 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId17(MM2S, 0, 1)
    memref.global "public" @airMemcpyId17 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId33(MM2S, 0, 1)
    memref.global "public" @airMemcpyId33 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId19(MM2S, 1, 1)
    memref.global "public" @airMemcpyId19 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId35(MM2S, 1, 1)
    memref.global "public" @airMemcpyId35 : memref<32x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId21(MM2S, 0, 2)
    memref.global "public" @airMemcpyId21 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId37(MM2S, 0, 2)
    memref.global "public" @airMemcpyId37 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId23(MM2S, 1, 2)
    memref.global "public" @airMemcpyId23 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId39(MM2S, 1, 2)
    memref.global "public" @airMemcpyId39 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId25(MM2S, 0, 3)
    memref.global "public" @airMemcpyId25 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId41(MM2S, 0, 3)
    memref.global "public" @airMemcpyId41 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId27(MM2S, 1, 3)
    memref.global "public" @airMemcpyId27 : memref<256x32xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId43(MM2S, 1, 3)
    memref.global "public" @airMemcpyId43 : memref<256x32xbf16, 1>
    func.func @matmul_512x512_512xbf16__dispatch_0_matmul_512x512x512_bf16xbf16xf32(%arg0: memref<131072xi32>, %arg1: memref<131072xi32>, %arg2: memref<512x512xf32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][4, 2, 32, 128][0, 128, 256, 1]) {id = 0 : i64, metadata = @airMemcpyId13} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 32768][4, 2, 32, 128][0, 128, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId13} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 65536][4, 2, 32, 128][0, 128, 256, 1]) {id = 2 : i64, metadata = @airMemcpyId13} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 98304][4, 2, 32, 128][0, 128, 256, 1]) {id = 3 : i64, metadata = @airMemcpyId13} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][4, 2, 32, 128][0, 128, 256, 1]) {id = 4 : i64, metadata = @airMemcpyId15} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 40960][4, 2, 32, 128][0, 128, 256, 1]) {id = 5 : i64, metadata = @airMemcpyId15} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 73728][4, 2, 32, 128][0, 128, 256, 1]) {id = 6 : i64, metadata = @airMemcpyId15} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 106496][4, 2, 32, 128][0, 128, 256, 1]) {id = 7 : i64, metadata = @airMemcpyId15} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][4, 2, 32, 128][0, 128, 256, 1]) {id = 0 : i64, metadata = @airMemcpyId17} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 49152][4, 2, 32, 128][0, 128, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId17} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 81920][4, 2, 32, 128][0, 128, 256, 1]) {id = 2 : i64, metadata = @airMemcpyId17} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 114688][4, 2, 32, 128][0, 128, 256, 1]) {id = 3 : i64, metadata = @airMemcpyId17} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][4, 2, 32, 128][0, 128, 256, 1]) {id = 4 : i64, metadata = @airMemcpyId19} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 57344][4, 2, 32, 128][0, 128, 256, 1]) {id = 5 : i64, metadata = @airMemcpyId19} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 90112][4, 2, 32, 128][0, 128, 256, 1]) {id = 6 : i64, metadata = @airMemcpyId19} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 122880][4, 2, 32, 128][0, 128, 256, 1]) {id = 7 : i64, metadata = @airMemcpyId19} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 4, 512, 16][0, 64, 256, 1]) {id = 0 : i64, metadata = @airMemcpyId21} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][4, 4, 512, 16][0, 64, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId23} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 32][4, 4, 512, 16][0, 64, 256, 1]) {id = 0 : i64, metadata = @airMemcpyId25} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][4, 4, 512, 16][0, 64, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId27} : memref<131072xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][4, 4, 32, 128][65536, 128, 512, 1]) {id = 8 : i64, metadata = @airMemcpyId78} : memref<512x512xf32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 32, 0][4, 4, 32, 128][65536, 128, 512, 1]) {id = 9 : i64, metadata = @airMemcpyId79} : memref<512x512xf32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 64, 0][4, 4, 32, 128][65536, 128, 512, 1]) {id = 8 : i64, metadata = @airMemcpyId80} : memref<512x512xf32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 96, 0][4, 4, 32, 128][65536, 128, 512, 1]) {id = 9 : i64, metadata = @airMemcpyId81} : memref<512x512xf32>
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 1 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  } {sym_name = "segment_0"}
}
