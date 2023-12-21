//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
// 
//
//===----------------------------------------------------------------------===//


module @hdiff_large_0 {
//---col 0---*-
  %tile0_2 = aie.tile(0, 2)
//---col 1---*-
  %tile1_2 = aie.tile(1, 2)
//---col 2---*-
  %tile2_2 = aie.tile(2, 2)
//---col 3---*-
  %tile3_2 = aie.tile(3, 2)
//---col 4---*-
  %tile4_2 = aie.tile(4, 2)
//---col 5---*-
  %tile5_2 = aie.tile(5, 2)
//---col 6---*-
  %tile6_2 = aie.tile(6, 2)
//---col 7---*-
  %tile7_2 = aie.tile(7, 2)
//---col 8---*-
  %tile8_2 = aie.tile(8, 2)
//---col 9---*-
  %tile9_2 = aie.tile(9, 2)
//---col 10---*-
  %tile10_2 = aie.tile(10, 2)
//---col 11---*-
  %tile11_2 = aie.tile(11, 2)
//---col 12---*-
  %tile12_2 = aie.tile(12, 2)
//---col 13---*-
  %tile13_2 = aie.tile(13, 2)
//---col 14---*-
  %tile14_2 = aie.tile(14, 2)
//---col 15---*-
  %tile15_2 = aie.tile(15, 2)
//---col 16---*-
  %tile16_2 = aie.tile(16, 2)
//---col 17---*-
  %tile17_2 = aie.tile(17, 2)
//---col 18---*-
  %tile18_2 = aie.tile(18, 2)
//---col 19---*-
  %tile19_2 = aie.tile(19, 2)
//---col 20---*-
  %tile20_2 = aie.tile(20, 2)
//---col 21---*-
  %tile21_2 = aie.tile(21, 2)
//---col 22---*-
  %tile22_2 = aie.tile(22, 2)
//---col 23---*-
  %tile23_2 = aie.tile(23, 2)
//---col 24---*-
  %tile24_2 = aie.tile(24, 2)
//---col 25---*-
  %tile25_2 = aie.tile(25, 2)
//---col 26---*-
  %tile26_2 = aie.tile(26, 2)
//---col 27---*-
  %tile27_2 = aie.tile(27, 2)
//---col 28---*-
  %tile28_2 = aie.tile(28, 2)
//---col 29---*-
  %tile29_2 = aie.tile(29, 2)
//---col 30---*-
  %tile30_2 = aie.tile(30, 2)
//---col 31---*-
  %tile31_2 = aie.tile(31, 2)

//---NOC TILE 2---*-
  %tile2_0 = aie.tile(2, 0)
//---NOC TILE 3---*-
  %tile3_0 = aie.tile(3, 0)
//---NOC TILE 6---*-
  %tile6_0 = aie.tile(6, 0)
//---NOC TILE 7---*-
  %tile7_0 = aie.tile(7, 0)
//---NOC TILE 10---*-
  %tile10_0 = aie.tile(10, 0)
//---NOC TILE 11---*-
  %tile11_0 = aie.tile(11, 0)
//---NOC TILE 18---*-
  %tile18_0 = aie.tile(18, 0)
//---NOC TILE 19---*-
  %tile19_0 = aie.tile(19, 0)
//---NOC TILE 26---*-
  %tile26_0 = aie.tile(26, 0)
//---NOC TILE 27---*-
  %tile27_0 = aie.tile(27, 0)
//---NOC TILE 34---*-
  %tile34_0 = aie.tile(34, 0)
//---NOC TILE 35---*-
  %tile35_0 = aie.tile(35, 0)
//---NOC TILE 42---*-
  %tile42_0 = aie.tile(42, 0)
//---NOC TILE 43---*-
  %tile43_0 = aie.tile(43, 0)
//---NOC TILE 46---*-
  %tile46_0 = aie.tile(46, 0)
//---NOC TILE 47---*-
  %tile47_0 = aie.tile(47, 0)

  %buf_in_0_shim_2 = aie.objectfifo.createObjectFifo(%tile2_0,{%tile0_2},6) { sym_name = "obj_in_0" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_0_2_shim_2 = aie.objectfifo.createObjectFifo(%tile0_2,{%tile2_0},2) { sym_name = "obj_out_0_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_1_shim_2 = aie.objectfifo.createObjectFifo(%tile2_0,{%tile1_2},6) { sym_name = "obj_in_1" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_1_2_shim_2 = aie.objectfifo.createObjectFifo(%tile1_2,{%tile2_0},2) { sym_name = "obj_out_1_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_2_shim_3 = aie.objectfifo.createObjectFifo(%tile3_0,{%tile2_2},6) { sym_name = "obj_in_2" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_2_2_shim_3 = aie.objectfifo.createObjectFifo(%tile2_2,{%tile3_0},2) { sym_name = "obj_out_2_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_3_shim_3 = aie.objectfifo.createObjectFifo(%tile3_0,{%tile3_2},6) { sym_name = "obj_in_3" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_3_2_shim_3 = aie.objectfifo.createObjectFifo(%tile3_2,{%tile3_0},2) { sym_name = "obj_out_3_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_4_shim_6 = aie.objectfifo.createObjectFifo(%tile6_0,{%tile4_2},6) { sym_name = "obj_in_4" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_4_2_shim_6 = aie.objectfifo.createObjectFifo(%tile4_2,{%tile6_0},2) { sym_name = "obj_out_4_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_5_shim_6 = aie.objectfifo.createObjectFifo(%tile6_0,{%tile5_2},6) { sym_name = "obj_in_5" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_5_2_shim_6 = aie.objectfifo.createObjectFifo(%tile5_2,{%tile6_0},2) { sym_name = "obj_out_5_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_6_shim_7 = aie.objectfifo.createObjectFifo(%tile7_0,{%tile6_2},6) { sym_name = "obj_in_6" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_6_2_shim_7 = aie.objectfifo.createObjectFifo(%tile6_2,{%tile7_0},2) { sym_name = "obj_out_6_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_7_shim_7 = aie.objectfifo.createObjectFifo(%tile7_0,{%tile7_2},6) { sym_name = "obj_in_7" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_7_2_shim_7 = aie.objectfifo.createObjectFifo(%tile7_2,{%tile7_0},2) { sym_name = "obj_out_7_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_8_shim_10 = aie.objectfifo.createObjectFifo(%tile10_0,{%tile8_2},6) { sym_name = "obj_in_8" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_8_2_shim_10 = aie.objectfifo.createObjectFifo(%tile8_2,{%tile10_0},2) { sym_name = "obj_out_8_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_9_shim_10 = aie.objectfifo.createObjectFifo(%tile10_0,{%tile9_2},6) { sym_name = "obj_in_9" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_9_2_shim_10 = aie.objectfifo.createObjectFifo(%tile9_2,{%tile10_0},2) { sym_name = "obj_out_9_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_10_shim_11 = aie.objectfifo.createObjectFifo(%tile11_0,{%tile10_2},6) { sym_name = "obj_in_10" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_10_2_shim_11 = aie.objectfifo.createObjectFifo(%tile10_2,{%tile11_0},2) { sym_name = "obj_out_10_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_11_shim_11 = aie.objectfifo.createObjectFifo(%tile11_0,{%tile11_2},6) { sym_name = "obj_in_11" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_11_2_shim_11 = aie.objectfifo.createObjectFifo(%tile11_2,{%tile11_0},2) { sym_name = "obj_out_11_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_12_shim_18 = aie.objectfifo.createObjectFifo(%tile18_0,{%tile12_2},6) { sym_name = "obj_in_12" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_12_2_shim_18 = aie.objectfifo.createObjectFifo(%tile12_2,{%tile18_0},2) { sym_name = "obj_out_12_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_13_shim_18 = aie.objectfifo.createObjectFifo(%tile18_0,{%tile13_2},6) { sym_name = "obj_in_13" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_13_2_shim_18 = aie.objectfifo.createObjectFifo(%tile13_2,{%tile18_0},2) { sym_name = "obj_out_13_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_14_shim_19 = aie.objectfifo.createObjectFifo(%tile19_0,{%tile14_2},6) { sym_name = "obj_in_14" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_14_2_shim_19 = aie.objectfifo.createObjectFifo(%tile14_2,{%tile19_0},2) { sym_name = "obj_out_14_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_15_shim_19 = aie.objectfifo.createObjectFifo(%tile19_0,{%tile15_2},6) { sym_name = "obj_in_15" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_15_2_shim_19 = aie.objectfifo.createObjectFifo(%tile15_2,{%tile19_0},2) { sym_name = "obj_out_15_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_16_shim_26 = aie.objectfifo.createObjectFifo(%tile26_0,{%tile16_2},6) { sym_name = "obj_in_16" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_16_2_shim_26 = aie.objectfifo.createObjectFifo(%tile16_2,{%tile26_0},2) { sym_name = "obj_out_16_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_17_shim_26 = aie.objectfifo.createObjectFifo(%tile26_0,{%tile17_2},6) { sym_name = "obj_in_17" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_17_2_shim_26 = aie.objectfifo.createObjectFifo(%tile17_2,{%tile26_0},2) { sym_name = "obj_out_17_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_18_shim_27 = aie.objectfifo.createObjectFifo(%tile27_0,{%tile18_2},6) { sym_name = "obj_in_18" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_18_2_shim_27 = aie.objectfifo.createObjectFifo(%tile18_2,{%tile27_0},2) { sym_name = "obj_out_18_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_19_shim_27 = aie.objectfifo.createObjectFifo(%tile27_0,{%tile19_2},6) { sym_name = "obj_in_19" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_19_2_shim_27 = aie.objectfifo.createObjectFifo(%tile19_2,{%tile27_0},2) { sym_name = "obj_out_19_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_20_shim_34 = aie.objectfifo.createObjectFifo(%tile34_0,{%tile20_2},6) { sym_name = "obj_in_20" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_20_2_shim_34 = aie.objectfifo.createObjectFifo(%tile20_2,{%tile34_0},2) { sym_name = "obj_out_20_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_21_shim_34 = aie.objectfifo.createObjectFifo(%tile34_0,{%tile21_2},6) { sym_name = "obj_in_21" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_21_2_shim_34 = aie.objectfifo.createObjectFifo(%tile21_2,{%tile34_0},2) { sym_name = "obj_out_21_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_22_shim_35 = aie.objectfifo.createObjectFifo(%tile35_0,{%tile22_2},6) { sym_name = "obj_in_22" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_22_2_shim_35 = aie.objectfifo.createObjectFifo(%tile22_2,{%tile35_0},2) { sym_name = "obj_out_22_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_23_shim_35 = aie.objectfifo.createObjectFifo(%tile35_0,{%tile23_2},6) { sym_name = "obj_in_23" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_23_2_shim_35 = aie.objectfifo.createObjectFifo(%tile23_2,{%tile35_0},2) { sym_name = "obj_out_23_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_24_shim_42 = aie.objectfifo.createObjectFifo(%tile42_0,{%tile24_2},6) { sym_name = "obj_in_24" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_24_2_shim_42 = aie.objectfifo.createObjectFifo(%tile24_2,{%tile42_0},2) { sym_name = "obj_out_24_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_25_shim_42 = aie.objectfifo.createObjectFifo(%tile42_0,{%tile25_2},6) { sym_name = "obj_in_25" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_25_2_shim_42 = aie.objectfifo.createObjectFifo(%tile25_2,{%tile42_0},2) { sym_name = "obj_out_25_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_26_shim_43 = aie.objectfifo.createObjectFifo(%tile43_0,{%tile26_2},6) { sym_name = "obj_in_26" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_26_2_shim_43 = aie.objectfifo.createObjectFifo(%tile26_2,{%tile43_0},2) { sym_name = "obj_out_26_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_27_shim_43 = aie.objectfifo.createObjectFifo(%tile43_0,{%tile27_2},6) { sym_name = "obj_in_27" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_27_2_shim_43 = aie.objectfifo.createObjectFifo(%tile27_2,{%tile43_0},2) { sym_name = "obj_out_27_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_28_shim_46 = aie.objectfifo.createObjectFifo(%tile46_0,{%tile28_2},6) { sym_name = "obj_in_28" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_28_2_shim_46 = aie.objectfifo.createObjectFifo(%tile28_2,{%tile46_0},2) { sym_name = "obj_out_28_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_29_shim_46 = aie.objectfifo.createObjectFifo(%tile46_0,{%tile29_2},6) { sym_name = "obj_in_29" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_29_2_shim_46 = aie.objectfifo.createObjectFifo(%tile29_2,{%tile46_0},2) { sym_name = "obj_out_29_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_30_shim_47 = aie.objectfifo.createObjectFifo(%tile47_0,{%tile30_2},6) { sym_name = "obj_in_30" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_30_2_shim_47 = aie.objectfifo.createObjectFifo(%tile30_2,{%tile47_0},2) { sym_name = "obj_out_30_2" } : !aie.objectfifo<memref<256xi32>>

  %buf_in_31_shim_47 = aie.objectfifo.createObjectFifo(%tile47_0,{%tile31_2},6) { sym_name = "obj_in_31" } : !aie.objectfifo<memref<256xi32>>
  %buf_out_31_2_shim_47 = aie.objectfifo.createObjectFifo(%tile31_2,{%tile47_0},2) { sym_name = "obj_out_31_2" } : !aie.objectfifo<memref<256xi32>>

  %ext_buffer_in_0 = aie.external_buffer  {sym_name = "ddr_buffer_in_0"}: memref<1536 x i32>
  %ext_buffer_out_0_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_0_2"}: memref<512 x i32>

  %ext_buffer_in_1 = aie.external_buffer  {sym_name = "ddr_buffer_in_1"}: memref<1536 x i32>
  %ext_buffer_out_1_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_1_2"}: memref<512 x i32>

  %ext_buffer_in_2 = aie.external_buffer  {sym_name = "ddr_buffer_in_2"}: memref<1536 x i32>
  %ext_buffer_out_2_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_2_2"}: memref<512 x i32>

  %ext_buffer_in_3 = aie.external_buffer  {sym_name = "ddr_buffer_in_3"}: memref<1536 x i32>
  %ext_buffer_out_3_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_3_2"}: memref<512 x i32>

  %ext_buffer_in_4 = aie.external_buffer  {sym_name = "ddr_buffer_in_4"}: memref<1536 x i32>
  %ext_buffer_out_4_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_4_2"}: memref<512 x i32>

  %ext_buffer_in_5 = aie.external_buffer  {sym_name = "ddr_buffer_in_5"}: memref<1536 x i32>
  %ext_buffer_out_5_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_5_2"}: memref<512 x i32>

  %ext_buffer_in_6 = aie.external_buffer  {sym_name = "ddr_buffer_in_6"}: memref<1536 x i32>
  %ext_buffer_out_6_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_6_2"}: memref<512 x i32>

  %ext_buffer_in_7 = aie.external_buffer  {sym_name = "ddr_buffer_in_7"}: memref<1536 x i32>
  %ext_buffer_out_7_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_7_2"}: memref<512 x i32>

  %ext_buffer_in_8 = aie.external_buffer  {sym_name = "ddr_buffer_in_8"}: memref<1536 x i32>
  %ext_buffer_out_8_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_8_2"}: memref<512 x i32>

  %ext_buffer_in_9 = aie.external_buffer  {sym_name = "ddr_buffer_in_9"}: memref<1536 x i32>
  %ext_buffer_out_9_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_9_2"}: memref<512 x i32>

  %ext_buffer_in_10 = aie.external_buffer  {sym_name = "ddr_buffer_in_10"}: memref<1536 x i32>
  %ext_buffer_out_10_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_10_2"}: memref<512 x i32>

  %ext_buffer_in_11 = aie.external_buffer  {sym_name = "ddr_buffer_in_11"}: memref<1536 x i32>
  %ext_buffer_out_11_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_11_2"}: memref<512 x i32>

  %ext_buffer_in_12 = aie.external_buffer  {sym_name = "ddr_buffer_in_12"}: memref<1536 x i32>
  %ext_buffer_out_12_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_12_2"}: memref<512 x i32>

  %ext_buffer_in_13 = aie.external_buffer  {sym_name = "ddr_buffer_in_13"}: memref<1536 x i32>
  %ext_buffer_out_13_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_13_2"}: memref<512 x i32>

  %ext_buffer_in_14 = aie.external_buffer  {sym_name = "ddr_buffer_in_14"}: memref<1536 x i32>
  %ext_buffer_out_14_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_14_2"}: memref<512 x i32>

  %ext_buffer_in_15 = aie.external_buffer  {sym_name = "ddr_buffer_in_15"}: memref<1536 x i32>
  %ext_buffer_out_15_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_15_2"}: memref<512 x i32>

  %ext_buffer_in_16 = aie.external_buffer  {sym_name = "ddr_buffer_in_16"}: memref<1536 x i32>
  %ext_buffer_out_16_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_16_2"}: memref<512 x i32>

  %ext_buffer_in_17 = aie.external_buffer  {sym_name = "ddr_buffer_in_17"}: memref<1536 x i32>
  %ext_buffer_out_17_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_17_2"}: memref<512 x i32>

  %ext_buffer_in_18 = aie.external_buffer  {sym_name = "ddr_buffer_in_18"}: memref<1536 x i32>
  %ext_buffer_out_18_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_18_2"}: memref<512 x i32>

  %ext_buffer_in_19 = aie.external_buffer  {sym_name = "ddr_buffer_in_19"}: memref<1536 x i32>
  %ext_buffer_out_19_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_19_2"}: memref<512 x i32>

  %ext_buffer_in_20 = aie.external_buffer  {sym_name = "ddr_buffer_in_20"}: memref<1536 x i32>
  %ext_buffer_out_20_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_20_2"}: memref<512 x i32>

  %ext_buffer_in_21 = aie.external_buffer  {sym_name = "ddr_buffer_in_21"}: memref<1536 x i32>
  %ext_buffer_out_21_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_21_2"}: memref<512 x i32>

  %ext_buffer_in_22 = aie.external_buffer  {sym_name = "ddr_buffer_in_22"}: memref<1536 x i32>
  %ext_buffer_out_22_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_22_2"}: memref<512 x i32>

  %ext_buffer_in_23 = aie.external_buffer  {sym_name = "ddr_buffer_in_23"}: memref<1536 x i32>
  %ext_buffer_out_23_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_23_2"}: memref<512 x i32>

  %ext_buffer_in_24 = aie.external_buffer  {sym_name = "ddr_buffer_in_24"}: memref<1536 x i32>
  %ext_buffer_out_24_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_24_2"}: memref<512 x i32>

  %ext_buffer_in_25 = aie.external_buffer  {sym_name = "ddr_buffer_in_25"}: memref<1536 x i32>
  %ext_buffer_out_25_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_25_2"}: memref<512 x i32>

  %ext_buffer_in_26 = aie.external_buffer  {sym_name = "ddr_buffer_in_26"}: memref<1536 x i32>
  %ext_buffer_out_26_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_26_2"}: memref<512 x i32>

  %ext_buffer_in_27 = aie.external_buffer  {sym_name = "ddr_buffer_in_27"}: memref<1536 x i32>
  %ext_buffer_out_27_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_27_2"}: memref<512 x i32>

  %ext_buffer_in_28 = aie.external_buffer  {sym_name = "ddr_buffer_in_28"}: memref<1536 x i32>
  %ext_buffer_out_28_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_28_2"}: memref<512 x i32>

  %ext_buffer_in_29 = aie.external_buffer  {sym_name = "ddr_buffer_in_29"}: memref<1536 x i32>
  %ext_buffer_out_29_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_29_2"}: memref<512 x i32>

  %ext_buffer_in_30 = aie.external_buffer  {sym_name = "ddr_buffer_in_30"}: memref<1536 x i32>
  %ext_buffer_out_30_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_30_2"}: memref<512 x i32>

  %ext_buffer_in_31 = aie.external_buffer  {sym_name = "ddr_buffer_in_31"}: memref<1536 x i32>
  %ext_buffer_out_31_2 = aie.external_buffer  {sym_name = "ddr_buffer_out_31_2"}: memref<512 x i32>

//Registering buffers
  aie.objectfifo.register_external_buffers(%tile2_0, %buf_in_0_shim_2  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_0}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile2_0, %buf_out_0_2_shim_2  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_0_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile2_0, %buf_in_1_shim_2  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_1}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile2_0, %buf_out_1_2_shim_2  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_1_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile3_0, %buf_in_2_shim_3  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_2}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile3_0, %buf_out_2_2_shim_3  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_2_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile3_0, %buf_in_3_shim_3  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_3}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile3_0, %buf_out_3_2_shim_3  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_3_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile6_0, %buf_in_4_shim_6  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_4}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile6_0, %buf_out_4_2_shim_6  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_4_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile6_0, %buf_in_5_shim_6  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_5}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile6_0, %buf_out_5_2_shim_6  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_5_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile7_0, %buf_in_6_shim_7  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_6}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile7_0, %buf_out_6_2_shim_7  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_6_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile7_0, %buf_in_7_shim_7  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_7}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile7_0, %buf_out_7_2_shim_7  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_7_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile10_0, %buf_in_8_shim_10  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_8}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile10_0, %buf_out_8_2_shim_10  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_8_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile10_0, %buf_in_9_shim_10  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_9}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile10_0, %buf_out_9_2_shim_10  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_9_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile11_0, %buf_in_10_shim_11  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_10}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile11_0, %buf_out_10_2_shim_11  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_10_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile11_0, %buf_in_11_shim_11  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_11}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile11_0, %buf_out_11_2_shim_11  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_11_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile18_0, %buf_in_12_shim_18  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_12}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile18_0, %buf_out_12_2_shim_18  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_12_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile18_0, %buf_in_13_shim_18  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_13}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile18_0, %buf_out_13_2_shim_18  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_13_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile19_0, %buf_in_14_shim_19  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_14}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile19_0, %buf_out_14_2_shim_19  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_14_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile19_0, %buf_in_15_shim_19  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_15}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile19_0, %buf_out_15_2_shim_19  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_15_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile26_0, %buf_in_16_shim_26  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_16}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile26_0, %buf_out_16_2_shim_26  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_16_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile26_0, %buf_in_17_shim_26  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_17}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile26_0, %buf_out_17_2_shim_26  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_17_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile27_0, %buf_in_18_shim_27  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_18}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile27_0, %buf_out_18_2_shim_27  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_18_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile27_0, %buf_in_19_shim_27  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_19}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile27_0, %buf_out_19_2_shim_27  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_19_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile34_0, %buf_in_20_shim_34  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_20}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile34_0, %buf_out_20_2_shim_34  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_20_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile34_0, %buf_in_21_shim_34  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_21}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile34_0, %buf_out_21_2_shim_34  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_21_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile35_0, %buf_in_22_shim_35  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_22}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile35_0, %buf_out_22_2_shim_35  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_22_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile35_0, %buf_in_23_shim_35  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_23}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile35_0, %buf_out_23_2_shim_35  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_23_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile42_0, %buf_in_24_shim_42  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_24}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile42_0, %buf_out_24_2_shim_42  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_24_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile42_0, %buf_in_25_shim_42  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_25}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile42_0, %buf_out_25_2_shim_42  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_25_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile43_0, %buf_in_26_shim_43  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_26}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile43_0, %buf_out_26_2_shim_43  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_26_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile43_0, %buf_in_27_shim_43  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_27}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile43_0, %buf_out_27_2_shim_43  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_27_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile46_0, %buf_in_28_shim_46  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_28}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile46_0, %buf_out_28_2_shim_46  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_28_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile46_0, %buf_in_29_shim_46  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_29}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile46_0, %buf_out_29_2_shim_46  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_29_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile47_0, %buf_in_30_shim_47  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_30}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile47_0, %buf_out_30_2_shim_47  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_30_2}) : (memref<512xi32>)

  aie.objectfifo.register_external_buffers(%tile47_0, %buf_in_31_shim_47  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_in_31}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers(%tile47_0, %buf_out_31_2_shim_47  : !aie.objectfifo<memref<256xi32>>, {%ext_buffer_out_31_2}) : (memref<512xi32>)


  func.func private @vec_hdiff(%A: memref<256xi32>,%B: memref<256xi32>, %C:  memref<256xi32>, %D: memref<256xi32>, %E:  memref<256xi32>,  %O: memref<256xi32>) -> ()

  %core0_2 = aie.core(%tile0_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_0_shim_2: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_0_2_shim_2: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_0_shim_2: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_0_2_shim_2: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_0_shim_2: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core1_2 = aie.core(%tile1_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_1_shim_2: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_1_2_shim_2: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_1_shim_2: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_1_2_shim_2: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_1_shim_2: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core2_2 = aie.core(%tile2_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_2_shim_3: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_2_2_shim_3: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_2_shim_3: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_2_2_shim_3: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_2_shim_3: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core3_2 = aie.core(%tile3_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_3_shim_3: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_3_2_shim_3: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_3_shim_3: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_3_2_shim_3: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_3_shim_3: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core4_2 = aie.core(%tile4_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_4_shim_6: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_4_2_shim_6: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_4_shim_6: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_4_2_shim_6: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_4_shim_6: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core5_2 = aie.core(%tile5_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_5_shim_6: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_5_2_shim_6: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_5_shim_6: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_5_2_shim_6: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_5_shim_6: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core6_2 = aie.core(%tile6_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_6_shim_7: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_6_2_shim_7: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_6_shim_7: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_6_2_shim_7: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_6_shim_7: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core7_2 = aie.core(%tile7_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_7_shim_7: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_7_2_shim_7: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_7_shim_7: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_7_2_shim_7: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_7_shim_7: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core8_2 = aie.core(%tile8_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_8_shim_10: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_8_2_shim_10: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_8_shim_10: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_8_2_shim_10: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_8_shim_10: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core9_2 = aie.core(%tile9_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_9_shim_10: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_9_2_shim_10: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_9_shim_10: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_9_2_shim_10: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_9_shim_10: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core10_2 = aie.core(%tile10_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_10_shim_11: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_10_2_shim_11: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_10_shim_11: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_10_2_shim_11: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_10_shim_11: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core11_2 = aie.core(%tile11_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_11_shim_11: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_11_2_shim_11: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_11_shim_11: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_11_2_shim_11: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_11_shim_11: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core12_2 = aie.core(%tile12_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_12_shim_18: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_12_2_shim_18: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_12_shim_18: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_12_2_shim_18: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_12_shim_18: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core13_2 = aie.core(%tile13_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_13_shim_18: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_13_2_shim_18: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_13_shim_18: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_13_2_shim_18: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_13_shim_18: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core14_2 = aie.core(%tile14_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_14_shim_19: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_14_2_shim_19: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_14_shim_19: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_14_2_shim_19: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_14_shim_19: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core15_2 = aie.core(%tile15_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_15_shim_19: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_15_2_shim_19: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_15_shim_19: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_15_2_shim_19: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_15_shim_19: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core16_2 = aie.core(%tile16_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_16_shim_26: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_16_2_shim_26: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_16_shim_26: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_16_2_shim_26: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_16_shim_26: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core17_2 = aie.core(%tile17_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_17_shim_26: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_17_2_shim_26: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_17_shim_26: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_17_2_shim_26: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_17_shim_26: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core18_2 = aie.core(%tile18_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_18_shim_27: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_18_2_shim_27: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_18_shim_27: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_18_2_shim_27: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_18_shim_27: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core19_2 = aie.core(%tile19_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_19_shim_27: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_19_2_shim_27: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_19_shim_27: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_19_2_shim_27: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_19_shim_27: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core20_2 = aie.core(%tile20_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_20_shim_34: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_20_2_shim_34: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_20_shim_34: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_20_2_shim_34: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_20_shim_34: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core21_2 = aie.core(%tile21_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_21_shim_34: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_21_2_shim_34: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_21_shim_34: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_21_2_shim_34: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_21_shim_34: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core22_2 = aie.core(%tile22_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_22_shim_35: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_22_2_shim_35: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_22_shim_35: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_22_2_shim_35: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_22_shim_35: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core23_2 = aie.core(%tile23_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_23_shim_35: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_23_2_shim_35: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_23_shim_35: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_23_2_shim_35: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_23_shim_35: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core24_2 = aie.core(%tile24_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_24_shim_42: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_24_2_shim_42: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_24_shim_42: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_24_2_shim_42: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_24_shim_42: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core25_2 = aie.core(%tile25_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_25_shim_42: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_25_2_shim_42: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_25_shim_42: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_25_2_shim_42: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_25_shim_42: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core26_2 = aie.core(%tile26_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_26_shim_43: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_26_2_shim_43: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_26_shim_43: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_26_2_shim_43: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_26_shim_43: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core27_2 = aie.core(%tile27_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_27_shim_43: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_27_2_shim_43: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_27_shim_43: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_27_2_shim_43: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_27_shim_43: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core28_2 = aie.core(%tile28_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_28_shim_46: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_28_2_shim_46: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_28_shim_46: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_28_2_shim_46: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_28_shim_46: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core29_2 = aie.core(%tile29_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_29_shim_46: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_29_2_shim_46: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_29_shim_46: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_29_2_shim_46: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_29_shim_46: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core30_2 = aie.core(%tile30_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_30_shim_47: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_30_2_shim_47: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_30_shim_47: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_30_2_shim_47: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_30_shim_47: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

  %core31_2 = aie.core(%tile31_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire<Consume>(%buf_in_31_shim_47: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_subview = aie.objectfifo.acquire<Produce>(%buf_out_31_2_shim_47: !aie.objectfifo<memref<256xi32>>, 5) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out = aie.objectfifo.subview.access %obj_out_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release<Consume>(%buf_in_31_shim_47: !aie.objectfifo<memref<256xi32>>, 1)
      aie.objectfifo.release<Produce>(%buf_out_31_2_shim_47: !aie.objectfifo<memref<256xi32>>, 1)
  }

  aie.objectfifo.release<Consume>(%buf_in_31_shim_47: !aie.objectfifo<memref<256xi32>>, 4)
  aie.end
 } { link_with="hdiff.o" }

}
