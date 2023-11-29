// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
  
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


module @hdiff_large_0 {
//---col 0---*-
  %tile0_2 = AIE.tile(0, 2)
//---col 1---*-
  %tile1_2 = AIE.tile(1, 2)
//---col 2---*-
  %tile2_2 = AIE.tile(2, 2)
//---col 3---*-
  %tile3_2 = AIE.tile(3, 2)
//---col 4---*-
  %tile4_2 = AIE.tile(4, 2)
//---col 5---*-
  %tile5_2 = AIE.tile(5, 2)
//---col 6---*-
  %tile6_2 = AIE.tile(6, 2)
//---col 7---*-
  %tile7_2 = AIE.tile(7, 2)
//---col 8---*-
  %tile8_2 = AIE.tile(8, 2)
//---col 9---*-
  %tile9_2 = AIE.tile(9, 2)
//---col 10---*-
  %tile10_2 = AIE.tile(10, 2)
//---col 11---*-
  %tile11_2 = AIE.tile(11, 2)
//---col 12---*-
  %tile12_2 = AIE.tile(12, 2)
//---col 13---*-
  %tile13_2 = AIE.tile(13, 2)
//---col 14---*-
  %tile14_2 = AIE.tile(14, 2)
//---col 15---*-
  %tile15_2 = AIE.tile(15, 2)
//---col 16---*-
  %tile16_2 = AIE.tile(16, 2)
//---col 17---*-
  %tile17_2 = AIE.tile(17, 2)
//---col 18---*-
  %tile18_2 = AIE.tile(18, 2)
//---col 19---*-
  %tile19_2 = AIE.tile(19, 2)
//---col 20---*-
  %tile20_2 = AIE.tile(20, 2)
//---col 21---*-
  %tile21_2 = AIE.tile(21, 2)
//---col 22---*-
  %tile22_2 = AIE.tile(22, 2)
//---col 23---*-
  %tile23_2 = AIE.tile(23, 2)
//---col 24---*-
  %tile24_2 = AIE.tile(24, 2)
//---col 25---*-
  %tile25_2 = AIE.tile(25, 2)
//---col 26---*-
  %tile26_2 = AIE.tile(26, 2)
//---col 27---*-
  %tile27_2 = AIE.tile(27, 2)
//---col 28---*-
  %tile28_2 = AIE.tile(28, 2)
//---col 29---*-
  %tile29_2 = AIE.tile(29, 2)
//---col 30---*-
  %tile30_2 = AIE.tile(30, 2)
//---col 31---*-
  %tile31_2 = AIE.tile(31, 2)

//---NOC TILE 2---*-
  %tile2_0 = AIE.tile(2, 0)
//---NOC TILE 3---*-
  %tile3_0 = AIE.tile(3, 0)
//---NOC TILE 6---*-
  %tile6_0 = AIE.tile(6, 0)
//---NOC TILE 7---*-
  %tile7_0 = AIE.tile(7, 0)
//---NOC TILE 10---*-
  %tile10_0 = AIE.tile(10, 0)
//---NOC TILE 11---*-
  %tile11_0 = AIE.tile(11, 0)
//---NOC TILE 18---*-
  %tile18_0 = AIE.tile(18, 0)
//---NOC TILE 19---*-
  %tile19_0 = AIE.tile(19, 0)
//---NOC TILE 26---*-
  %tile26_0 = AIE.tile(26, 0)
//---NOC TILE 27---*-
  %tile27_0 = AIE.tile(27, 0)
//---NOC TILE 34---*-
  %tile34_0 = AIE.tile(34, 0)
//---NOC TILE 35---*-
  %tile35_0 = AIE.tile(35, 0)
//---NOC TILE 42---*-
  %tile42_0 = AIE.tile(42, 0)
//---NOC TILE 43---*-
  %tile43_0 = AIE.tile(43, 0)
//---NOC TILE 46---*-
  %tile46_0 = AIE.tile(46, 0)
//---NOC TILE 47---*-
  %tile47_0 = AIE.tile(47, 0)

  %buf_in_0_shim_2 = AIE.objectfifo.createObjectFifo(%tile2_0,{%tile0_2},6) { sym_name = "obj_in_0" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_0_2_shim_2 = AIE.objectfifo.createObjectFifo(%tile0_2,{%tile2_0},2) { sym_name = "obj_out_0_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_1_shim_2 = AIE.objectfifo.createObjectFifo(%tile2_0,{%tile1_2},6) { sym_name = "obj_in_1" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_1_2_shim_2 = AIE.objectfifo.createObjectFifo(%tile1_2,{%tile2_0},2) { sym_name = "obj_out_1_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_2_shim_3 = AIE.objectfifo.createObjectFifo(%tile3_0,{%tile2_2},6) { sym_name = "obj_in_2" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_2_2_shim_3 = AIE.objectfifo.createObjectFifo(%tile2_2,{%tile3_0},2) { sym_name = "obj_out_2_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_3_shim_3 = AIE.objectfifo.createObjectFifo(%tile3_0,{%tile3_2},6) { sym_name = "obj_in_3" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_3_2_shim_3 = AIE.objectfifo.createObjectFifo(%tile3_2,{%tile3_0},2) { sym_name = "obj_out_3_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_4_shim_6 = AIE.objectfifo.createObjectFifo(%tile6_0,{%tile4_2},6) { sym_name = "obj_in_4" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_4_2_shim_6 = AIE.objectfifo.createObjectFifo(%tile4_2,{%tile6_0},2) { sym_name = "obj_out_4_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_5_shim_6 = AIE.objectfifo.createObjectFifo(%tile6_0,{%tile5_2},6) { sym_name = "obj_in_5" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_5_2_shim_6 = AIE.objectfifo.createObjectFifo(%tile5_2,{%tile6_0},2) { sym_name = "obj_out_5_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_6_shim_7 = AIE.objectfifo.createObjectFifo(%tile7_0,{%tile6_2},6) { sym_name = "obj_in_6" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_6_2_shim_7 = AIE.objectfifo.createObjectFifo(%tile6_2,{%tile7_0},2) { sym_name = "obj_out_6_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_7_shim_7 = AIE.objectfifo.createObjectFifo(%tile7_0,{%tile7_2},6) { sym_name = "obj_in_7" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_7_2_shim_7 = AIE.objectfifo.createObjectFifo(%tile7_2,{%tile7_0},2) { sym_name = "obj_out_7_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_8_shim_10 = AIE.objectfifo.createObjectFifo(%tile10_0,{%tile8_2},6) { sym_name = "obj_in_8" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_8_2_shim_10 = AIE.objectfifo.createObjectFifo(%tile8_2,{%tile10_0},2) { sym_name = "obj_out_8_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_9_shim_10 = AIE.objectfifo.createObjectFifo(%tile10_0,{%tile9_2},6) { sym_name = "obj_in_9" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_9_2_shim_10 = AIE.objectfifo.createObjectFifo(%tile9_2,{%tile10_0},2) { sym_name = "obj_out_9_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_10_shim_11 = AIE.objectfifo.createObjectFifo(%tile11_0,{%tile10_2},6) { sym_name = "obj_in_10" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_10_2_shim_11 = AIE.objectfifo.createObjectFifo(%tile10_2,{%tile11_0},2) { sym_name = "obj_out_10_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_11_shim_11 = AIE.objectfifo.createObjectFifo(%tile11_0,{%tile11_2},6) { sym_name = "obj_in_11" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_11_2_shim_11 = AIE.objectfifo.createObjectFifo(%tile11_2,{%tile11_0},2) { sym_name = "obj_out_11_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_12_shim_18 = AIE.objectfifo.createObjectFifo(%tile18_0,{%tile12_2},6) { sym_name = "obj_in_12" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_12_2_shim_18 = AIE.objectfifo.createObjectFifo(%tile12_2,{%tile18_0},2) { sym_name = "obj_out_12_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_13_shim_18 = AIE.objectfifo.createObjectFifo(%tile18_0,{%tile13_2},6) { sym_name = "obj_in_13" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_13_2_shim_18 = AIE.objectfifo.createObjectFifo(%tile13_2,{%tile18_0},2) { sym_name = "obj_out_13_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_14_shim_19 = AIE.objectfifo.createObjectFifo(%tile19_0,{%tile14_2},6) { sym_name = "obj_in_14" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_14_2_shim_19 = AIE.objectfifo.createObjectFifo(%tile14_2,{%tile19_0},2) { sym_name = "obj_out_14_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_15_shim_19 = AIE.objectfifo.createObjectFifo(%tile19_0,{%tile15_2},6) { sym_name = "obj_in_15" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_15_2_shim_19 = AIE.objectfifo.createObjectFifo(%tile15_2,{%tile19_0},2) { sym_name = "obj_out_15_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_16_shim_26 = AIE.objectfifo.createObjectFifo(%tile26_0,{%tile16_2},6) { sym_name = "obj_in_16" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_16_2_shim_26 = AIE.objectfifo.createObjectFifo(%tile16_2,{%tile26_0},2) { sym_name = "obj_out_16_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_17_shim_26 = AIE.objectfifo.createObjectFifo(%tile26_0,{%tile17_2},6) { sym_name = "obj_in_17" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_17_2_shim_26 = AIE.objectfifo.createObjectFifo(%tile17_2,{%tile26_0},2) { sym_name = "obj_out_17_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_18_shim_27 = AIE.objectfifo.createObjectFifo(%tile27_0,{%tile18_2},6) { sym_name = "obj_in_18" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_18_2_shim_27 = AIE.objectfifo.createObjectFifo(%tile18_2,{%tile27_0},2) { sym_name = "obj_out_18_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_19_shim_27 = AIE.objectfifo.createObjectFifo(%tile27_0,{%tile19_2},6) { sym_name = "obj_in_19" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_19_2_shim_27 = AIE.objectfifo.createObjectFifo(%tile19_2,{%tile27_0},2) { sym_name = "obj_out_19_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_20_shim_34 = AIE.objectfifo.createObjectFifo(%tile34_0,{%tile20_2},6) { sym_name = "obj_in_20" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_20_2_shim_34 = AIE.objectfifo.createObjectFifo(%tile20_2,{%tile34_0},2) { sym_name = "obj_out_20_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_21_shim_34 = AIE.objectfifo.createObjectFifo(%tile34_0,{%tile21_2},6) { sym_name = "obj_in_21" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_21_2_shim_34 = AIE.objectfifo.createObjectFifo(%tile21_2,{%tile34_0},2) { sym_name = "obj_out_21_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_22_shim_35 = AIE.objectfifo.createObjectFifo(%tile35_0,{%tile22_2},6) { sym_name = "obj_in_22" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_22_2_shim_35 = AIE.objectfifo.createObjectFifo(%tile22_2,{%tile35_0},2) { sym_name = "obj_out_22_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_23_shim_35 = AIE.objectfifo.createObjectFifo(%tile35_0,{%tile23_2},6) { sym_name = "obj_in_23" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_23_2_shim_35 = AIE.objectfifo.createObjectFifo(%tile23_2,{%tile35_0},2) { sym_name = "obj_out_23_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_24_shim_42 = AIE.objectfifo.createObjectFifo(%tile42_0,{%tile24_2},6) { sym_name = "obj_in_24" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_24_2_shim_42 = AIE.objectfifo.createObjectFifo(%tile24_2,{%tile42_0},2) { sym_name = "obj_out_24_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_25_shim_42 = AIE.objectfifo.createObjectFifo(%tile42_0,{%tile25_2},6) { sym_name = "obj_in_25" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_25_2_shim_42 = AIE.objectfifo.createObjectFifo(%tile25_2,{%tile42_0},2) { sym_name = "obj_out_25_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_26_shim_43 = AIE.objectfifo.createObjectFifo(%tile43_0,{%tile26_2},6) { sym_name = "obj_in_26" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_26_2_shim_43 = AIE.objectfifo.createObjectFifo(%tile26_2,{%tile43_0},2) { sym_name = "obj_out_26_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_27_shim_43 = AIE.objectfifo.createObjectFifo(%tile43_0,{%tile27_2},6) { sym_name = "obj_in_27" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_27_2_shim_43 = AIE.objectfifo.createObjectFifo(%tile27_2,{%tile43_0},2) { sym_name = "obj_out_27_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_28_shim_46 = AIE.objectfifo.createObjectFifo(%tile46_0,{%tile28_2},6) { sym_name = "obj_in_28" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_28_2_shim_46 = AIE.objectfifo.createObjectFifo(%tile28_2,{%tile46_0},2) { sym_name = "obj_out_28_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_29_shim_46 = AIE.objectfifo.createObjectFifo(%tile46_0,{%tile29_2},6) { sym_name = "obj_in_29" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_29_2_shim_46 = AIE.objectfifo.createObjectFifo(%tile29_2,{%tile46_0},2) { sym_name = "obj_out_29_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_30_shim_47 = AIE.objectfifo.createObjectFifo(%tile47_0,{%tile30_2},6) { sym_name = "obj_in_30" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_30_2_shim_47 = AIE.objectfifo.createObjectFifo(%tile30_2,{%tile47_0},2) { sym_name = "obj_out_30_2" } : !AIE.objectfifo<memref<256xi32>>

  %buf_in_31_shim_47 = AIE.objectfifo.createObjectFifo(%tile47_0,{%tile31_2},6) { sym_name = "obj_in_31" } : !AIE.objectfifo<memref<256xi32>>
  %buf_out_31_2_shim_47 = AIE.objectfifo.createObjectFifo(%tile31_2,{%tile47_0},2) { sym_name = "obj_out_31_2" } : !AIE.objectfifo<memref<256xi32>>

  %ext_buffer_in_0 = AIE.external_buffer  {sym_name = "ddr_buffer_in_0"}: memref<1536 x i32>
  %ext_buffer_out_0_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_0_2"}: memref<512 x i32>

  %ext_buffer_in_1 = AIE.external_buffer  {sym_name = "ddr_buffer_in_1"}: memref<1536 x i32>
  %ext_buffer_out_1_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_1_2"}: memref<512 x i32>

  %ext_buffer_in_2 = AIE.external_buffer  {sym_name = "ddr_buffer_in_2"}: memref<1536 x i32>
  %ext_buffer_out_2_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_2_2"}: memref<512 x i32>

  %ext_buffer_in_3 = AIE.external_buffer  {sym_name = "ddr_buffer_in_3"}: memref<1536 x i32>
  %ext_buffer_out_3_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_3_2"}: memref<512 x i32>

  %ext_buffer_in_4 = AIE.external_buffer  {sym_name = "ddr_buffer_in_4"}: memref<1536 x i32>
  %ext_buffer_out_4_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_4_2"}: memref<512 x i32>

  %ext_buffer_in_5 = AIE.external_buffer  {sym_name = "ddr_buffer_in_5"}: memref<1536 x i32>
  %ext_buffer_out_5_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_5_2"}: memref<512 x i32>

  %ext_buffer_in_6 = AIE.external_buffer  {sym_name = "ddr_buffer_in_6"}: memref<1536 x i32>
  %ext_buffer_out_6_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_6_2"}: memref<512 x i32>

  %ext_buffer_in_7 = AIE.external_buffer  {sym_name = "ddr_buffer_in_7"}: memref<1536 x i32>
  %ext_buffer_out_7_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_7_2"}: memref<512 x i32>

  %ext_buffer_in_8 = AIE.external_buffer  {sym_name = "ddr_buffer_in_8"}: memref<1536 x i32>
  %ext_buffer_out_8_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_8_2"}: memref<512 x i32>

  %ext_buffer_in_9 = AIE.external_buffer  {sym_name = "ddr_buffer_in_9"}: memref<1536 x i32>
  %ext_buffer_out_9_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_9_2"}: memref<512 x i32>

  %ext_buffer_in_10 = AIE.external_buffer  {sym_name = "ddr_buffer_in_10"}: memref<1536 x i32>
  %ext_buffer_out_10_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_10_2"}: memref<512 x i32>

  %ext_buffer_in_11 = AIE.external_buffer  {sym_name = "ddr_buffer_in_11"}: memref<1536 x i32>
  %ext_buffer_out_11_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_11_2"}: memref<512 x i32>

  %ext_buffer_in_12 = AIE.external_buffer  {sym_name = "ddr_buffer_in_12"}: memref<1536 x i32>
  %ext_buffer_out_12_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_12_2"}: memref<512 x i32>

  %ext_buffer_in_13 = AIE.external_buffer  {sym_name = "ddr_buffer_in_13"}: memref<1536 x i32>
  %ext_buffer_out_13_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_13_2"}: memref<512 x i32>

  %ext_buffer_in_14 = AIE.external_buffer  {sym_name = "ddr_buffer_in_14"}: memref<1536 x i32>
  %ext_buffer_out_14_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_14_2"}: memref<512 x i32>

  %ext_buffer_in_15 = AIE.external_buffer  {sym_name = "ddr_buffer_in_15"}: memref<1536 x i32>
  %ext_buffer_out_15_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_15_2"}: memref<512 x i32>

  %ext_buffer_in_16 = AIE.external_buffer  {sym_name = "ddr_buffer_in_16"}: memref<1536 x i32>
  %ext_buffer_out_16_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_16_2"}: memref<512 x i32>

  %ext_buffer_in_17 = AIE.external_buffer  {sym_name = "ddr_buffer_in_17"}: memref<1536 x i32>
  %ext_buffer_out_17_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_17_2"}: memref<512 x i32>

  %ext_buffer_in_18 = AIE.external_buffer  {sym_name = "ddr_buffer_in_18"}: memref<1536 x i32>
  %ext_buffer_out_18_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_18_2"}: memref<512 x i32>

  %ext_buffer_in_19 = AIE.external_buffer  {sym_name = "ddr_buffer_in_19"}: memref<1536 x i32>
  %ext_buffer_out_19_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_19_2"}: memref<512 x i32>

  %ext_buffer_in_20 = AIE.external_buffer  {sym_name = "ddr_buffer_in_20"}: memref<1536 x i32>
  %ext_buffer_out_20_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_20_2"}: memref<512 x i32>

  %ext_buffer_in_21 = AIE.external_buffer  {sym_name = "ddr_buffer_in_21"}: memref<1536 x i32>
  %ext_buffer_out_21_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_21_2"}: memref<512 x i32>

  %ext_buffer_in_22 = AIE.external_buffer  {sym_name = "ddr_buffer_in_22"}: memref<1536 x i32>
  %ext_buffer_out_22_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_22_2"}: memref<512 x i32>

  %ext_buffer_in_23 = AIE.external_buffer  {sym_name = "ddr_buffer_in_23"}: memref<1536 x i32>
  %ext_buffer_out_23_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_23_2"}: memref<512 x i32>

  %ext_buffer_in_24 = AIE.external_buffer  {sym_name = "ddr_buffer_in_24"}: memref<1536 x i32>
  %ext_buffer_out_24_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_24_2"}: memref<512 x i32>

  %ext_buffer_in_25 = AIE.external_buffer  {sym_name = "ddr_buffer_in_25"}: memref<1536 x i32>
  %ext_buffer_out_25_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_25_2"}: memref<512 x i32>

  %ext_buffer_in_26 = AIE.external_buffer  {sym_name = "ddr_buffer_in_26"}: memref<1536 x i32>
  %ext_buffer_out_26_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_26_2"}: memref<512 x i32>

  %ext_buffer_in_27 = AIE.external_buffer  {sym_name = "ddr_buffer_in_27"}: memref<1536 x i32>
  %ext_buffer_out_27_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_27_2"}: memref<512 x i32>

  %ext_buffer_in_28 = AIE.external_buffer  {sym_name = "ddr_buffer_in_28"}: memref<1536 x i32>
  %ext_buffer_out_28_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_28_2"}: memref<512 x i32>

  %ext_buffer_in_29 = AIE.external_buffer  {sym_name = "ddr_buffer_in_29"}: memref<1536 x i32>
  %ext_buffer_out_29_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_29_2"}: memref<512 x i32>

  %ext_buffer_in_30 = AIE.external_buffer  {sym_name = "ddr_buffer_in_30"}: memref<1536 x i32>
  %ext_buffer_out_30_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_30_2"}: memref<512 x i32>

  %ext_buffer_in_31 = AIE.external_buffer  {sym_name = "ddr_buffer_in_31"}: memref<1536 x i32>
  %ext_buffer_out_31_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_31_2"}: memref<512 x i32>

//Registering buffers
  AIE.objectfifo.register_external_buffers(%tile2_0, %buf_in_0_shim_2  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_0}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile2_0, %buf_out_0_2_shim_2  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_0_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile2_0, %buf_in_1_shim_2  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_1}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile2_0, %buf_out_1_2_shim_2  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_1_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile3_0, %buf_in_2_shim_3  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_2}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile3_0, %buf_out_2_2_shim_3  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_2_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile3_0, %buf_in_3_shim_3  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_3}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile3_0, %buf_out_3_2_shim_3  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_3_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile6_0, %buf_in_4_shim_6  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_4}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile6_0, %buf_out_4_2_shim_6  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_4_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile6_0, %buf_in_5_shim_6  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_5}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile6_0, %buf_out_5_2_shim_6  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_5_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile7_0, %buf_in_6_shim_7  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_6}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile7_0, %buf_out_6_2_shim_7  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_6_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile7_0, %buf_in_7_shim_7  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_7}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile7_0, %buf_out_7_2_shim_7  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_7_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile10_0, %buf_in_8_shim_10  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_8}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile10_0, %buf_out_8_2_shim_10  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_8_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile10_0, %buf_in_9_shim_10  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_9}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile10_0, %buf_out_9_2_shim_10  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_9_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile11_0, %buf_in_10_shim_11  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_10}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile11_0, %buf_out_10_2_shim_11  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_10_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile11_0, %buf_in_11_shim_11  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_11}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile11_0, %buf_out_11_2_shim_11  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_11_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile18_0, %buf_in_12_shim_18  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_12}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile18_0, %buf_out_12_2_shim_18  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_12_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile18_0, %buf_in_13_shim_18  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_13}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile18_0, %buf_out_13_2_shim_18  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_13_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile19_0, %buf_in_14_shim_19  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_14}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile19_0, %buf_out_14_2_shim_19  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_14_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile19_0, %buf_in_15_shim_19  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_15}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile19_0, %buf_out_15_2_shim_19  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_15_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile26_0, %buf_in_16_shim_26  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_16}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile26_0, %buf_out_16_2_shim_26  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_16_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile26_0, %buf_in_17_shim_26  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_17}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile26_0, %buf_out_17_2_shim_26  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_17_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile27_0, %buf_in_18_shim_27  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_18}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile27_0, %buf_out_18_2_shim_27  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_18_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile27_0, %buf_in_19_shim_27  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_19}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile27_0, %buf_out_19_2_shim_27  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_19_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile34_0, %buf_in_20_shim_34  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_20}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile34_0, %buf_out_20_2_shim_34  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_20_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile34_0, %buf_in_21_shim_34  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_21}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile34_0, %buf_out_21_2_shim_34  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_21_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile35_0, %buf_in_22_shim_35  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_22}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile35_0, %buf_out_22_2_shim_35  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_22_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile35_0, %buf_in_23_shim_35  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_23}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile35_0, %buf_out_23_2_shim_35  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_23_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile42_0, %buf_in_24_shim_42  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_24}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile42_0, %buf_out_24_2_shim_42  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_24_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile42_0, %buf_in_25_shim_42  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_25}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile42_0, %buf_out_25_2_shim_42  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_25_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile43_0, %buf_in_26_shim_43  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_26}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile43_0, %buf_out_26_2_shim_43  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_26_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile43_0, %buf_in_27_shim_43  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_27}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile43_0, %buf_out_27_2_shim_43  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_27_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile46_0, %buf_in_28_shim_46  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_28}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile46_0, %buf_out_28_2_shim_46  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_28_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile46_0, %buf_in_29_shim_46  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_29}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile46_0, %buf_out_29_2_shim_46  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_29_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile47_0, %buf_in_30_shim_47  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_30}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile47_0, %buf_out_30_2_shim_47  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_30_2}) : (memref<512xi32>)

  AIE.objectfifo.register_external_buffers(%tile47_0, %buf_in_31_shim_47  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_in_31}) : (memref<1536xi32>)
  AIE.objectfifo.register_external_buffers(%tile47_0, %buf_out_31_2_shim_47  : !AIE.objectfifo<memref<256xi32>>, {%ext_buffer_out_31_2}) : (memref<512xi32>)


  func.func private @vec_hdiff(%A: memref<256xi32>,%B: memref<256xi32>, %C:  memref<256xi32>, %D: memref<256xi32>, %E:  memref<256xi32>,  %O: memref<256xi32>) -> ()

  %core0_2 = AIE.core(%tile0_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_0_shim_2: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_0_2_shim_2: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_0_shim_2: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_0_2_shim_2: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_0_shim_2: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core1_2 = AIE.core(%tile1_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_1_shim_2: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_1_2_shim_2: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_1_shim_2: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_1_2_shim_2: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_1_shim_2: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core2_2 = AIE.core(%tile2_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_2_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_2_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core3_2 = AIE.core(%tile3_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_3_shim_3: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_3_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_3_shim_3: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_3_2_shim_3: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_3_shim_3: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core4_2 = AIE.core(%tile4_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_4_shim_6: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_4_2_shim_6: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_4_shim_6: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_4_2_shim_6: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_4_shim_6: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core5_2 = AIE.core(%tile5_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_5_shim_6: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_5_2_shim_6: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_5_shim_6: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_5_2_shim_6: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_5_shim_6: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core6_2 = AIE.core(%tile6_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_6_shim_7: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_6_2_shim_7: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_6_shim_7: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_6_2_shim_7: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_6_shim_7: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core7_2 = AIE.core(%tile7_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_7_shim_7: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_7_2_shim_7: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_7_shim_7: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_7_2_shim_7: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_7_shim_7: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core8_2 = AIE.core(%tile8_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_8_shim_10: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_8_2_shim_10: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_8_shim_10: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_8_2_shim_10: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_8_shim_10: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core9_2 = AIE.core(%tile9_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_9_shim_10: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_9_2_shim_10: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_9_shim_10: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_9_2_shim_10: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_9_shim_10: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core10_2 = AIE.core(%tile10_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_10_shim_11: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_10_2_shim_11: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_10_shim_11: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_10_2_shim_11: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_10_shim_11: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core11_2 = AIE.core(%tile11_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_11_shim_11: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_11_2_shim_11: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_11_shim_11: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_11_2_shim_11: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_11_shim_11: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core12_2 = AIE.core(%tile12_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_12_shim_18: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_12_2_shim_18: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_12_shim_18: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_12_2_shim_18: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_12_shim_18: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core13_2 = AIE.core(%tile13_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_13_shim_18: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_13_2_shim_18: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_13_shim_18: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_13_2_shim_18: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_13_shim_18: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core14_2 = AIE.core(%tile14_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_14_shim_19: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_14_2_shim_19: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_14_shim_19: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_14_2_shim_19: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_14_shim_19: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core15_2 = AIE.core(%tile15_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_15_shim_19: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_15_2_shim_19: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_15_shim_19: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_15_2_shim_19: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_15_shim_19: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core16_2 = AIE.core(%tile16_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_16_shim_26: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_16_2_shim_26: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_16_shim_26: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_16_2_shim_26: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_16_shim_26: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core17_2 = AIE.core(%tile17_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_17_shim_26: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_17_2_shim_26: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_17_shim_26: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_17_2_shim_26: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_17_shim_26: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core18_2 = AIE.core(%tile18_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_18_shim_27: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_18_2_shim_27: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_18_shim_27: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_18_2_shim_27: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_18_shim_27: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core19_2 = AIE.core(%tile19_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_19_shim_27: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_19_2_shim_27: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_19_shim_27: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_19_2_shim_27: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_19_shim_27: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core20_2 = AIE.core(%tile20_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_20_shim_34: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_20_2_shim_34: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_20_shim_34: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_20_2_shim_34: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_20_shim_34: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core21_2 = AIE.core(%tile21_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_21_shim_34: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_21_2_shim_34: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_21_shim_34: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_21_2_shim_34: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_21_shim_34: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core22_2 = AIE.core(%tile22_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_22_shim_35: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_22_2_shim_35: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_22_shim_35: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_22_2_shim_35: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_22_shim_35: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core23_2 = AIE.core(%tile23_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_23_shim_35: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_23_2_shim_35: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_23_shim_35: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_23_2_shim_35: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_23_shim_35: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core24_2 = AIE.core(%tile24_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_24_shim_42: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_24_2_shim_42: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_24_shim_42: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_24_2_shim_42: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_24_shim_42: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core25_2 = AIE.core(%tile25_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_25_shim_42: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_25_2_shim_42: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_25_shim_42: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_25_2_shim_42: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_25_shim_42: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core26_2 = AIE.core(%tile26_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_26_shim_43: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_26_2_shim_43: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_26_shim_43: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_26_2_shim_43: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_26_shim_43: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core27_2 = AIE.core(%tile27_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_27_shim_43: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_27_2_shim_43: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_27_shim_43: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_27_2_shim_43: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_27_shim_43: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core28_2 = AIE.core(%tile28_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_28_shim_46: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_28_2_shim_46: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_28_shim_46: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_28_2_shim_46: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_28_shim_46: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core29_2 = AIE.core(%tile29_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_29_shim_46: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_29_2_shim_46: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_29_shim_46: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_29_2_shim_46: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_29_shim_46: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core30_2 = AIE.core(%tile30_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_30_shim_47: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_30_2_shim_47: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_30_shim_47: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_30_2_shim_47: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_30_shim_47: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

  %core31_2 = AIE.core(%tile31_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire<Consume>(%buf_in_31_shim_47: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = AIE.objectfifo.acquire<Produce>(%buf_out_31_2_shim_47: !AIE.objectfifo<memref<256xi32>>, 5) : !AIE.objectfifosubview<memref<256xi32>>
      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectfifo.release<Consume>(%buf_in_31_shim_47: !AIE.objectfifo<memref<256xi32>>, 1)
      AIE.objectfifo.release<Produce>(%buf_out_31_2_shim_47: !AIE.objectfifo<memref<256xi32>>, 1)
  }

  AIE.objectfifo.release<Consume>(%buf_in_31_shim_47: !AIE.objectfifo<memref<256xi32>>, 4)
  AIE.end
 } { link_with="hdiff.o" }

}
