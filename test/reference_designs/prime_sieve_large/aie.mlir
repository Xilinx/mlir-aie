//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//
// Note:
// this large prime-sieve pattern is generated from code_gen.py,
// it contains 360 cores in this pattern, user could change the core numbers
// by specifying different rows and cols value in code_gen.py

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf 


module @test16_prime_sieve_large {
  %tile1_1 = AIE.tile(1, 1)
  %tile1_2 = AIE.tile(1, 2)
  %tile1_3 = AIE.tile(1, 3)
  %tile1_4 = AIE.tile(1, 4)
  %tile1_5 = AIE.tile(1, 5)
  %tile1_6 = AIE.tile(1, 6)
  %tile1_7 = AIE.tile(1, 7)
  %tile2_7 = AIE.tile(2, 7)
  %tile2_6 = AIE.tile(2, 6)
  %tile2_5 = AIE.tile(2, 5)
  %tile2_4 = AIE.tile(2, 4)
  %tile2_3 = AIE.tile(2, 3)
  %tile2_2 = AIE.tile(2, 2)
  %tile2_1 = AIE.tile(2, 1)
  %tile3_1 = AIE.tile(3, 1)
  %tile3_2 = AIE.tile(3, 2)
  %tile3_3 = AIE.tile(3, 3)
  %tile3_4 = AIE.tile(3, 4)
  %tile3_5 = AIE.tile(3, 5)
  %tile3_6 = AIE.tile(3, 6)
  %tile3_7 = AIE.tile(3, 7)
  %tile4_7 = AIE.tile(4, 7)
  %tile4_6 = AIE.tile(4, 6)
  %tile4_5 = AIE.tile(4, 5)
  %tile4_4 = AIE.tile(4, 4)
  %tile4_3 = AIE.tile(4, 3)
  %tile4_2 = AIE.tile(4, 2)
  %tile4_1 = AIE.tile(4, 1)
  %tile5_1 = AIE.tile(5, 1)
  %tile5_2 = AIE.tile(5, 2)
  %tile5_3 = AIE.tile(5, 3)
  %tile5_4 = AIE.tile(5, 4)
  %tile5_5 = AIE.tile(5, 5)
  %tile5_6 = AIE.tile(5, 6)
  %tile5_7 = AIE.tile(5, 7)
  %tile6_7 = AIE.tile(6, 7)
  %tile6_6 = AIE.tile(6, 6)
  %tile6_5 = AIE.tile(6, 5)
  %tile6_4 = AIE.tile(6, 4)
  %tile6_3 = AIE.tile(6, 3)
  %tile6_2 = AIE.tile(6, 2)
  %tile6_1 = AIE.tile(6, 1)
  %tile7_1 = AIE.tile(7, 1)
  %tile7_2 = AIE.tile(7, 2)
  %tile7_3 = AIE.tile(7, 3)
  %tile7_4 = AIE.tile(7, 4)
  %tile7_5 = AIE.tile(7, 5)
  %tile7_6 = AIE.tile(7, 6)
  %tile7_7 = AIE.tile(7, 7)
  %tile8_7 = AIE.tile(8, 7)
  %tile8_6 = AIE.tile(8, 6)
  %tile8_5 = AIE.tile(8, 5)
  %tile8_4 = AIE.tile(8, 4)
  %tile8_3 = AIE.tile(8, 3)
  %tile8_2 = AIE.tile(8, 2)
  %tile8_1 = AIE.tile(8, 1)
  %tile9_1 = AIE.tile(9, 1)
  %tile9_2 = AIE.tile(9, 2)
  %tile9_3 = AIE.tile(9, 3)
  %tile9_4 = AIE.tile(9, 4)
  %tile9_5 = AIE.tile(9, 5)
  %tile9_6 = AIE.tile(9, 6)
  %tile9_7 = AIE.tile(9, 7)
  %tile10_7 = AIE.tile(10, 7)
  %tile10_6 = AIE.tile(10, 6)
  %tile10_5 = AIE.tile(10, 5)
  %tile10_4 = AIE.tile(10, 4)
  %tile10_3 = AIE.tile(10, 3)
  %tile10_2 = AIE.tile(10, 2)
  %tile10_1 = AIE.tile(10, 1)
  %tile11_1 = AIE.tile(11, 1)
  %tile11_2 = AIE.tile(11, 2)
  %tile11_3 = AIE.tile(11, 3)
  %tile11_4 = AIE.tile(11, 4)
  %tile11_5 = AIE.tile(11, 5)
  %tile11_6 = AIE.tile(11, 6)
  %tile11_7 = AIE.tile(11, 7)
  %tile12_7 = AIE.tile(12, 7)
  %tile12_6 = AIE.tile(12, 6)
  %tile12_5 = AIE.tile(12, 5)
  %tile12_4 = AIE.tile(12, 4)
  %tile12_3 = AIE.tile(12, 3)
  %tile12_2 = AIE.tile(12, 2)
  %tile12_1 = AIE.tile(12, 1)
  %tile13_1 = AIE.tile(13, 1)
  %tile13_2 = AIE.tile(13, 2)
  %tile13_3 = AIE.tile(13, 3)
  %tile13_4 = AIE.tile(13, 4)
  %tile13_5 = AIE.tile(13, 5)
  %tile13_6 = AIE.tile(13, 6)
  %tile13_7 = AIE.tile(13, 7)
  %tile14_7 = AIE.tile(14, 7)
  %tile14_6 = AIE.tile(14, 6)
  %tile14_5 = AIE.tile(14, 5)
  %tile14_4 = AIE.tile(14, 4)
  %tile14_3 = AIE.tile(14, 3)
  %tile14_2 = AIE.tile(14, 2)
  %tile14_1 = AIE.tile(14, 1)
  %tile15_1 = AIE.tile(15, 1)
  %tile15_2 = AIE.tile(15, 2)
  %tile15_3 = AIE.tile(15, 3)
  %tile15_4 = AIE.tile(15, 4)
  %tile15_5 = AIE.tile(15, 5)
  %tile15_6 = AIE.tile(15, 6)
  %tile15_7 = AIE.tile(15, 7)
  %tile16_7 = AIE.tile(16, 7)
  %tile16_6 = AIE.tile(16, 6)
  %tile16_5 = AIE.tile(16, 5)
  %tile16_4 = AIE.tile(16, 4)
  %tile16_3 = AIE.tile(16, 3)
  %tile16_2 = AIE.tile(16, 2)
  %tile16_1 = AIE.tile(16, 1)
  %tile17_1 = AIE.tile(17, 1)
  %tile17_2 = AIE.tile(17, 2)
  %tile17_3 = AIE.tile(17, 3)
  %tile17_4 = AIE.tile(17, 4)
  %tile17_5 = AIE.tile(17, 5)
  %tile17_6 = AIE.tile(17, 6)
  %tile17_7 = AIE.tile(17, 7)
  %tile18_7 = AIE.tile(18, 7)
  %tile18_6 = AIE.tile(18, 6)
  %tile18_5 = AIE.tile(18, 5)
  %tile18_4 = AIE.tile(18, 4)
  %tile18_3 = AIE.tile(18, 3)
  %tile18_2 = AIE.tile(18, 2)
  %tile18_1 = AIE.tile(18, 1)
  %tile19_1 = AIE.tile(19, 1)
  %tile19_2 = AIE.tile(19, 2)
  %tile19_3 = AIE.tile(19, 3)
  %tile19_4 = AIE.tile(19, 4)
  %tile19_5 = AIE.tile(19, 5)
  %tile19_6 = AIE.tile(19, 6)
  %tile19_7 = AIE.tile(19, 7)
  %tile20_7 = AIE.tile(20, 7)
  %tile20_6 = AIE.tile(20, 6)
  %tile20_5 = AIE.tile(20, 5)
  %tile20_4 = AIE.tile(20, 4)
  %tile20_3 = AIE.tile(20, 3)
  %tile20_2 = AIE.tile(20, 2)
  %tile20_1 = AIE.tile(20, 1)
  %tile21_1 = AIE.tile(21, 1)
  %tile21_2 = AIE.tile(21, 2)
  %tile21_3 = AIE.tile(21, 3)
  %tile21_4 = AIE.tile(21, 4)
  %tile21_5 = AIE.tile(21, 5)
  %tile21_6 = AIE.tile(21, 6)
  %tile21_7 = AIE.tile(21, 7)
  %tile22_7 = AIE.tile(22, 7)
  %tile22_6 = AIE.tile(22, 6)
  %tile22_5 = AIE.tile(22, 5)
  %tile22_4 = AIE.tile(22, 4)
  %tile22_3 = AIE.tile(22, 3)
  %tile22_2 = AIE.tile(22, 2)
  %tile22_1 = AIE.tile(22, 1)
  %tile23_1 = AIE.tile(23, 1)
  %tile23_2 = AIE.tile(23, 2)
  %tile23_3 = AIE.tile(23, 3)
  %tile23_4 = AIE.tile(23, 4)
  %tile23_5 = AIE.tile(23, 5)
  %tile23_6 = AIE.tile(23, 6)
  %tile23_7 = AIE.tile(23, 7)
  %tile24_7 = AIE.tile(24, 7)
  %tile24_6 = AIE.tile(24, 6)
  %tile24_5 = AIE.tile(24, 5)
  %tile24_4 = AIE.tile(24, 4)
  %tile24_3 = AIE.tile(24, 3)
  %tile24_2 = AIE.tile(24, 2)
  %tile24_1 = AIE.tile(24, 1)
  %tile25_1 = AIE.tile(25, 1)
  %tile25_2 = AIE.tile(25, 2)
  %tile25_3 = AIE.tile(25, 3)
  %tile25_4 = AIE.tile(25, 4)
  %tile25_5 = AIE.tile(25, 5)
  %tile25_6 = AIE.tile(25, 6)
  %tile25_7 = AIE.tile(25, 7)
  %tile26_7 = AIE.tile(26, 7)
  %tile26_6 = AIE.tile(26, 6)
  %tile26_5 = AIE.tile(26, 5)
  %tile26_4 = AIE.tile(26, 4)
  %tile26_3 = AIE.tile(26, 3)
  %tile26_2 = AIE.tile(26, 2)
  %tile26_1 = AIE.tile(26, 1)
  %tile27_1 = AIE.tile(27, 1)
  %tile27_2 = AIE.tile(27, 2)
  %tile27_3 = AIE.tile(27, 3)
  %tile27_4 = AIE.tile(27, 4)
  %tile27_5 = AIE.tile(27, 5)
  %tile27_6 = AIE.tile(27, 6)
  %tile27_7 = AIE.tile(27, 7)
  %tile28_7 = AIE.tile(28, 7)
  %tile28_6 = AIE.tile(28, 6)
  %tile28_5 = AIE.tile(28, 5)
  %tile28_4 = AIE.tile(28, 4)
  %tile28_3 = AIE.tile(28, 3)
  %tile28_2 = AIE.tile(28, 2)
  %tile28_1 = AIE.tile(28, 1)
  %tile29_1 = AIE.tile(29, 1)
  %tile29_2 = AIE.tile(29, 2)
  %tile29_3 = AIE.tile(29, 3)
  %tile29_4 = AIE.tile(29, 4)
  %tile29_5 = AIE.tile(29, 5)
  %tile29_6 = AIE.tile(29, 6)
  %tile29_7 = AIE.tile(29, 7)
  %tile30_7 = AIE.tile(30, 7)
  %tile30_6 = AIE.tile(30, 6)
  %tile30_5 = AIE.tile(30, 5)
  %tile30_4 = AIE.tile(30, 4)
  %tile30_3 = AIE.tile(30, 3)
  %tile30_2 = AIE.tile(30, 2)
  %tile30_1 = AIE.tile(30, 1)
  %tile31_1 = AIE.tile(31, 1)
  %tile31_2 = AIE.tile(31, 2)
  %tile31_3 = AIE.tile(31, 3)
  %tile31_4 = AIE.tile(31, 4)
  %tile31_5 = AIE.tile(31, 5)
  %tile31_6 = AIE.tile(31, 6)
  %tile31_7 = AIE.tile(31, 7)
  %tile32_7 = AIE.tile(32, 7)
  %tile32_6 = AIE.tile(32, 6)
  %tile32_5 = AIE.tile(32, 5)
  %tile32_4 = AIE.tile(32, 4)
  %tile32_3 = AIE.tile(32, 3)
  %tile32_2 = AIE.tile(32, 2)
  %tile32_1 = AIE.tile(32, 1)
  %tile33_1 = AIE.tile(33, 1)
  %tile33_2 = AIE.tile(33, 2)
  %tile33_3 = AIE.tile(33, 3)
  %tile33_4 = AIE.tile(33, 4)
  %tile33_5 = AIE.tile(33, 5)
  %tile33_6 = AIE.tile(33, 6)
  %tile33_7 = AIE.tile(33, 7)
  %tile34_7 = AIE.tile(34, 7)
  %tile34_6 = AIE.tile(34, 6)
  %tile34_5 = AIE.tile(34, 5)
  %tile34_4 = AIE.tile(34, 4)
  %tile34_3 = AIE.tile(34, 3)
  %tile34_2 = AIE.tile(34, 2)
  %tile34_1 = AIE.tile(34, 1)
  %tile35_1 = AIE.tile(35, 1)
  %tile35_2 = AIE.tile(35, 2)
  %tile35_3 = AIE.tile(35, 3)
  %tile35_4 = AIE.tile(35, 4)
  %tile35_5 = AIE.tile(35, 5)
  %tile35_6 = AIE.tile(35, 6)
  %tile35_7 = AIE.tile(35, 7)
  %tile36_7 = AIE.tile(36, 7)
  %tile36_6 = AIE.tile(36, 6)
  %tile36_5 = AIE.tile(36, 5)
  %tile36_4 = AIE.tile(36, 4)
  %tile36_3 = AIE.tile(36, 3)
  %tile36_2 = AIE.tile(36, 2)
  %tile36_1 = AIE.tile(36, 1)
  %tile37_1 = AIE.tile(37, 1)
  %tile37_2 = AIE.tile(37, 2)
  %tile37_3 = AIE.tile(37, 3)
  %tile37_4 = AIE.tile(37, 4)
  %tile37_5 = AIE.tile(37, 5)
  %tile37_6 = AIE.tile(37, 6)
  %tile37_7 = AIE.tile(37, 7)
  %tile38_7 = AIE.tile(38, 7)
  %tile38_6 = AIE.tile(38, 6)
  %tile38_5 = AIE.tile(38, 5)
  %tile38_4 = AIE.tile(38, 4)
  %tile38_3 = AIE.tile(38, 3)
  %tile38_2 = AIE.tile(38, 2)
  %tile38_1 = AIE.tile(38, 1)
  %tile39_1 = AIE.tile(39, 1)
  %tile39_2 = AIE.tile(39, 2)
  %tile39_3 = AIE.tile(39, 3)
  %tile39_4 = AIE.tile(39, 4)
  %tile39_5 = AIE.tile(39, 5)
  %tile39_6 = AIE.tile(39, 6)
  %tile39_7 = AIE.tile(39, 7)
  %tile40_7 = AIE.tile(40, 7)
  %tile40_6 = AIE.tile(40, 6)
  %tile40_5 = AIE.tile(40, 5)
  %tile40_4 = AIE.tile(40, 4)
  %tile40_3 = AIE.tile(40, 3)
  %tile40_2 = AIE.tile(40, 2)
  %tile40_1 = AIE.tile(40, 1)
  %tile41_1 = AIE.tile(41, 1)
  %tile41_2 = AIE.tile(41, 2)
  %tile41_3 = AIE.tile(41, 3)
  %tile41_4 = AIE.tile(41, 4)
  %tile41_5 = AIE.tile(41, 5)
  %tile41_6 = AIE.tile(41, 6)
  %tile41_7 = AIE.tile(41, 7)
  %tile42_7 = AIE.tile(42, 7)
  %tile42_6 = AIE.tile(42, 6)
  %tile42_5 = AIE.tile(42, 5)
  %tile42_4 = AIE.tile(42, 4)
  %tile42_3 = AIE.tile(42, 3)
  %tile42_2 = AIE.tile(42, 2)
  %tile42_1 = AIE.tile(42, 1)
  %tile43_1 = AIE.tile(43, 1)
  %tile43_2 = AIE.tile(43, 2)
  %tile43_3 = AIE.tile(43, 3)
  %tile43_4 = AIE.tile(43, 4)
  %tile43_5 = AIE.tile(43, 5)
  %tile43_6 = AIE.tile(43, 6)
  %tile43_7 = AIE.tile(43, 7)
  %tile44_7 = AIE.tile(44, 7)
  %tile44_6 = AIE.tile(44, 6)
  %tile44_5 = AIE.tile(44, 5)
  %tile44_4 = AIE.tile(44, 4)
  %tile44_3 = AIE.tile(44, 3)
  %tile44_2 = AIE.tile(44, 2)
  %tile44_1 = AIE.tile(44, 1)
  %tile45_1 = AIE.tile(45, 1)
  %tile45_2 = AIE.tile(45, 2)
  %tile45_3 = AIE.tile(45, 3)
  %tile45_4 = AIE.tile(45, 4)
  %tile45_5 = AIE.tile(45, 5)
  %tile45_6 = AIE.tile(45, 6)
  %tile45_7 = AIE.tile(45, 7)
  %tile46_7 = AIE.tile(46, 7)
  %tile46_6 = AIE.tile(46, 6)
  %tile46_5 = AIE.tile(46, 5)
  %tile46_4 = AIE.tile(46, 4)
  %tile46_3 = AIE.tile(46, 3)
  %tile46_2 = AIE.tile(46, 2)
  %tile46_1 = AIE.tile(46, 1)
  %tile47_1 = AIE.tile(47, 1)
  %tile47_2 = AIE.tile(47, 2)
  %tile47_3 = AIE.tile(47, 3)
  %tile47_4 = AIE.tile(47, 4)
  %tile47_5 = AIE.tile(47, 5)
  %tile47_6 = AIE.tile(47, 6)
  %tile47_7 = AIE.tile(47, 7)
  %tile48_7 = AIE.tile(48, 7)
  %tile48_6 = AIE.tile(48, 6)
  %tile48_5 = AIE.tile(48, 5)
  %tile48_4 = AIE.tile(48, 4)
  %tile48_3 = AIE.tile(48, 3)
  %tile48_2 = AIE.tile(48, 2)
  %tile48_1 = AIE.tile(48, 1)

  %lock1_1 = AIE.lock(%tile1_1, 0)
  %lock1_2 = AIE.lock(%tile1_2, 0)
  %lock1_3 = AIE.lock(%tile1_3, 0)
  %lock1_4 = AIE.lock(%tile1_4, 0)
  %lock1_5 = AIE.lock(%tile1_5, 0)
  %lock1_6 = AIE.lock(%tile1_6, 0)
  %lock1_7 = AIE.lock(%tile1_7, 0)
  %lock2_7 = AIE.lock(%tile2_7, 0)
  %lock2_6 = AIE.lock(%tile2_6, 0)
  %lock2_5 = AIE.lock(%tile2_5, 0)
  %lock2_4 = AIE.lock(%tile2_4, 0)
  %lock2_3 = AIE.lock(%tile2_3, 0)
  %lock2_2 = AIE.lock(%tile2_2, 0)
  %lock2_1 = AIE.lock(%tile2_1, 0)
  %lock3_1 = AIE.lock(%tile3_1, 0)
  %lock3_2 = AIE.lock(%tile3_2, 0)
  %lock3_3 = AIE.lock(%tile3_3, 0)
  %lock3_4 = AIE.lock(%tile3_4, 0)
  %lock3_5 = AIE.lock(%tile3_5, 0)
  %lock3_6 = AIE.lock(%tile3_6, 0)
  %lock3_7 = AIE.lock(%tile3_7, 0)
  %lock4_7 = AIE.lock(%tile4_7, 0)
  %lock4_6 = AIE.lock(%tile4_6, 0)
  %lock4_5 = AIE.lock(%tile4_5, 0)
  %lock4_4 = AIE.lock(%tile4_4, 0)
  %lock4_3 = AIE.lock(%tile4_3, 0)
  %lock4_2 = AIE.lock(%tile4_2, 0)
  %lock4_1 = AIE.lock(%tile4_1, 0)
  %lock5_1 = AIE.lock(%tile5_1, 0)
  %lock5_2 = AIE.lock(%tile5_2, 0)
  %lock5_3 = AIE.lock(%tile5_3, 0)
  %lock5_4 = AIE.lock(%tile5_4, 0)
  %lock5_5 = AIE.lock(%tile5_5, 0)
  %lock5_6 = AIE.lock(%tile5_6, 0)
  %lock5_7 = AIE.lock(%tile5_7, 0)
  %lock6_7 = AIE.lock(%tile6_7, 0)
  %lock6_6 = AIE.lock(%tile6_6, 0)
  %lock6_5 = AIE.lock(%tile6_5, 0)
  %lock6_4 = AIE.lock(%tile6_4, 0)
  %lock6_3 = AIE.lock(%tile6_3, 0)
  %lock6_2 = AIE.lock(%tile6_2, 0)
  %lock6_1 = AIE.lock(%tile6_1, 0)
  %lock7_1 = AIE.lock(%tile7_1, 0)
  %lock7_2 = AIE.lock(%tile7_2, 0)
  %lock7_3 = AIE.lock(%tile7_3, 0)
  %lock7_4 = AIE.lock(%tile7_4, 0)
  %lock7_5 = AIE.lock(%tile7_5, 0)
  %lock7_6 = AIE.lock(%tile7_6, 0)
  %lock7_7 = AIE.lock(%tile7_7, 0)
  %lock8_7 = AIE.lock(%tile8_7, 0)
  %lock8_6 = AIE.lock(%tile8_6, 0)
  %lock8_5 = AIE.lock(%tile8_5, 0)
  %lock8_4 = AIE.lock(%tile8_4, 0)
  %lock8_3 = AIE.lock(%tile8_3, 0)
  %lock8_2 = AIE.lock(%tile8_2, 0)
  %lock8_1 = AIE.lock(%tile8_1, 0)
  %lock9_1 = AIE.lock(%tile9_1, 0)
  %lock9_2 = AIE.lock(%tile9_2, 0)
  %lock9_3 = AIE.lock(%tile9_3, 0)
  %lock9_4 = AIE.lock(%tile9_4, 0)
  %lock9_5 = AIE.lock(%tile9_5, 0)
  %lock9_6 = AIE.lock(%tile9_6, 0)
  %lock9_7 = AIE.lock(%tile9_7, 0)
  %lock10_7 = AIE.lock(%tile10_7, 0)
  %lock10_6 = AIE.lock(%tile10_6, 0)
  %lock10_5 = AIE.lock(%tile10_5, 0)
  %lock10_4 = AIE.lock(%tile10_4, 0)
  %lock10_3 = AIE.lock(%tile10_3, 0)
  %lock10_2 = AIE.lock(%tile10_2, 0)
  %lock10_1 = AIE.lock(%tile10_1, 0)
  %lock11_1 = AIE.lock(%tile11_1, 0)
  %lock11_2 = AIE.lock(%tile11_2, 0)
  %lock11_3 = AIE.lock(%tile11_3, 0)
  %lock11_4 = AIE.lock(%tile11_4, 0)
  %lock11_5 = AIE.lock(%tile11_5, 0)
  %lock11_6 = AIE.lock(%tile11_6, 0)
  %lock11_7 = AIE.lock(%tile11_7, 0)
  %lock12_7 = AIE.lock(%tile12_7, 0)
  %lock12_6 = AIE.lock(%tile12_6, 0)
  %lock12_5 = AIE.lock(%tile12_5, 0)
  %lock12_4 = AIE.lock(%tile12_4, 0)
  %lock12_3 = AIE.lock(%tile12_3, 0)
  %lock12_2 = AIE.lock(%tile12_2, 0)
  %lock12_1 = AIE.lock(%tile12_1, 0)
  %lock13_1 = AIE.lock(%tile13_1, 0)
  %lock13_2 = AIE.lock(%tile13_2, 0)
  %lock13_3 = AIE.lock(%tile13_3, 0)
  %lock13_4 = AIE.lock(%tile13_4, 0)
  %lock13_5 = AIE.lock(%tile13_5, 0)
  %lock13_6 = AIE.lock(%tile13_6, 0)
  %lock13_7 = AIE.lock(%tile13_7, 0)
  %lock14_7 = AIE.lock(%tile14_7, 0)
  %lock14_6 = AIE.lock(%tile14_6, 0)
  %lock14_5 = AIE.lock(%tile14_5, 0)
  %lock14_4 = AIE.lock(%tile14_4, 0)
  %lock14_3 = AIE.lock(%tile14_3, 0)
  %lock14_2 = AIE.lock(%tile14_2, 0)
  %lock14_1 = AIE.lock(%tile14_1, 0)
  %lock15_1 = AIE.lock(%tile15_1, 0)
  %lock15_2 = AIE.lock(%tile15_2, 0)
  %lock15_3 = AIE.lock(%tile15_3, 0)
  %lock15_4 = AIE.lock(%tile15_4, 0)
  %lock15_5 = AIE.lock(%tile15_5, 0)
  %lock15_6 = AIE.lock(%tile15_6, 0)
  %lock15_7 = AIE.lock(%tile15_7, 0)
  %lock16_7 = AIE.lock(%tile16_7, 0)
  %lock16_6 = AIE.lock(%tile16_6, 0)
  %lock16_5 = AIE.lock(%tile16_5, 0)
  %lock16_4 = AIE.lock(%tile16_4, 0)
  %lock16_3 = AIE.lock(%tile16_3, 0)
  %lock16_2 = AIE.lock(%tile16_2, 0)
  %lock16_1 = AIE.lock(%tile16_1, 0)
  %lock17_1 = AIE.lock(%tile17_1, 0)
  %lock17_2 = AIE.lock(%tile17_2, 0)
  %lock17_3 = AIE.lock(%tile17_3, 0)
  %lock17_4 = AIE.lock(%tile17_4, 0)
  %lock17_5 = AIE.lock(%tile17_5, 0)
  %lock17_6 = AIE.lock(%tile17_6, 0)
  %lock17_7 = AIE.lock(%tile17_7, 0)
  %lock18_7 = AIE.lock(%tile18_7, 0)
  %lock18_6 = AIE.lock(%tile18_6, 0)
  %lock18_5 = AIE.lock(%tile18_5, 0)
  %lock18_4 = AIE.lock(%tile18_4, 0)
  %lock18_3 = AIE.lock(%tile18_3, 0)
  %lock18_2 = AIE.lock(%tile18_2, 0)
  %lock18_1 = AIE.lock(%tile18_1, 0)
  %lock19_1 = AIE.lock(%tile19_1, 0)
  %lock19_2 = AIE.lock(%tile19_2, 0)
  %lock19_3 = AIE.lock(%tile19_3, 0)
  %lock19_4 = AIE.lock(%tile19_4, 0)
  %lock19_5 = AIE.lock(%tile19_5, 0)
  %lock19_6 = AIE.lock(%tile19_6, 0)
  %lock19_7 = AIE.lock(%tile19_7, 0)
  %lock20_7 = AIE.lock(%tile20_7, 0)
  %lock20_6 = AIE.lock(%tile20_6, 0)
  %lock20_5 = AIE.lock(%tile20_5, 0)
  %lock20_4 = AIE.lock(%tile20_4, 0)
  %lock20_3 = AIE.lock(%tile20_3, 0)
  %lock20_2 = AIE.lock(%tile20_2, 0)
  %lock20_1 = AIE.lock(%tile20_1, 0)
  %lock21_1 = AIE.lock(%tile21_1, 0)
  %lock21_2 = AIE.lock(%tile21_2, 0)
  %lock21_3 = AIE.lock(%tile21_3, 0)
  %lock21_4 = AIE.lock(%tile21_4, 0)
  %lock21_5 = AIE.lock(%tile21_5, 0)
  %lock21_6 = AIE.lock(%tile21_6, 0)
  %lock21_7 = AIE.lock(%tile21_7, 0)
  %lock22_7 = AIE.lock(%tile22_7, 0)
  %lock22_6 = AIE.lock(%tile22_6, 0)
  %lock22_5 = AIE.lock(%tile22_5, 0)
  %lock22_4 = AIE.lock(%tile22_4, 0)
  %lock22_3 = AIE.lock(%tile22_3, 0)
  %lock22_2 = AIE.lock(%tile22_2, 0)
  %lock22_1 = AIE.lock(%tile22_1, 0)
  %lock23_1 = AIE.lock(%tile23_1, 0)
  %lock23_2 = AIE.lock(%tile23_2, 0)
  %lock23_3 = AIE.lock(%tile23_3, 0)
  %lock23_4 = AIE.lock(%tile23_4, 0)
  %lock23_5 = AIE.lock(%tile23_5, 0)
  %lock23_6 = AIE.lock(%tile23_6, 0)
  %lock23_7 = AIE.lock(%tile23_7, 0)
  %lock24_7 = AIE.lock(%tile24_7, 0)
  %lock24_6 = AIE.lock(%tile24_6, 0)
  %lock24_5 = AIE.lock(%tile24_5, 0)
  %lock24_4 = AIE.lock(%tile24_4, 0)
  %lock24_3 = AIE.lock(%tile24_3, 0)
  %lock24_2 = AIE.lock(%tile24_2, 0)
  %lock24_1 = AIE.lock(%tile24_1, 0)
  %lock25_1 = AIE.lock(%tile25_1, 0)
  %lock25_2 = AIE.lock(%tile25_2, 0)
  %lock25_3 = AIE.lock(%tile25_3, 0)
  %lock25_4 = AIE.lock(%tile25_4, 0)
  %lock25_5 = AIE.lock(%tile25_5, 0)
  %lock25_6 = AIE.lock(%tile25_6, 0)
  %lock25_7 = AIE.lock(%tile25_7, 0)
  %lock26_7 = AIE.lock(%tile26_7, 0)
  %lock26_6 = AIE.lock(%tile26_6, 0)
  %lock26_5 = AIE.lock(%tile26_5, 0)
  %lock26_4 = AIE.lock(%tile26_4, 0)
  %lock26_3 = AIE.lock(%tile26_3, 0)
  %lock26_2 = AIE.lock(%tile26_2, 0)
  %lock26_1 = AIE.lock(%tile26_1, 0)
  %lock27_1 = AIE.lock(%tile27_1, 0)
  %lock27_2 = AIE.lock(%tile27_2, 0)
  %lock27_3 = AIE.lock(%tile27_3, 0)
  %lock27_4 = AIE.lock(%tile27_4, 0)
  %lock27_5 = AIE.lock(%tile27_5, 0)
  %lock27_6 = AIE.lock(%tile27_6, 0)
  %lock27_7 = AIE.lock(%tile27_7, 0)
  %lock28_7 = AIE.lock(%tile28_7, 0)
  %lock28_6 = AIE.lock(%tile28_6, 0)
  %lock28_5 = AIE.lock(%tile28_5, 0)
  %lock28_4 = AIE.lock(%tile28_4, 0)
  %lock28_3 = AIE.lock(%tile28_3, 0)
  %lock28_2 = AIE.lock(%tile28_2, 0)
  %lock28_1 = AIE.lock(%tile28_1, 0)
  %lock29_1 = AIE.lock(%tile29_1, 0)
  %lock29_2 = AIE.lock(%tile29_2, 0)
  %lock29_3 = AIE.lock(%tile29_3, 0)
  %lock29_4 = AIE.lock(%tile29_4, 0)
  %lock29_5 = AIE.lock(%tile29_5, 0)
  %lock29_6 = AIE.lock(%tile29_6, 0)
  %lock29_7 = AIE.lock(%tile29_7, 0)
  %lock30_7 = AIE.lock(%tile30_7, 0)
  %lock30_6 = AIE.lock(%tile30_6, 0)
  %lock30_5 = AIE.lock(%tile30_5, 0)
  %lock30_4 = AIE.lock(%tile30_4, 0)
  %lock30_3 = AIE.lock(%tile30_3, 0)
  %lock30_2 = AIE.lock(%tile30_2, 0)
  %lock30_1 = AIE.lock(%tile30_1, 0)
  %lock31_1 = AIE.lock(%tile31_1, 0)
  %lock31_2 = AIE.lock(%tile31_2, 0)
  %lock31_3 = AIE.lock(%tile31_3, 0)
  %lock31_4 = AIE.lock(%tile31_4, 0)
  %lock31_5 = AIE.lock(%tile31_5, 0)
  %lock31_6 = AIE.lock(%tile31_6, 0)
  %lock31_7 = AIE.lock(%tile31_7, 0)
  %lock32_7 = AIE.lock(%tile32_7, 0)
  %lock32_6 = AIE.lock(%tile32_6, 0)
  %lock32_5 = AIE.lock(%tile32_5, 0)
  %lock32_4 = AIE.lock(%tile32_4, 0)
  %lock32_3 = AIE.lock(%tile32_3, 0)
  %lock32_2 = AIE.lock(%tile32_2, 0)
  %lock32_1 = AIE.lock(%tile32_1, 0)
  %lock33_1 = AIE.lock(%tile33_1, 0)
  %lock33_2 = AIE.lock(%tile33_2, 0)
  %lock33_3 = AIE.lock(%tile33_3, 0)
  %lock33_4 = AIE.lock(%tile33_4, 0)
  %lock33_5 = AIE.lock(%tile33_5, 0)
  %lock33_6 = AIE.lock(%tile33_6, 0)
  %lock33_7 = AIE.lock(%tile33_7, 0)
  %lock34_7 = AIE.lock(%tile34_7, 0)
  %lock34_6 = AIE.lock(%tile34_6, 0)
  %lock34_5 = AIE.lock(%tile34_5, 0)
  %lock34_4 = AIE.lock(%tile34_4, 0)
  %lock34_3 = AIE.lock(%tile34_3, 0)
  %lock34_2 = AIE.lock(%tile34_2, 0)
  %lock34_1 = AIE.lock(%tile34_1, 0)
  %lock35_1 = AIE.lock(%tile35_1, 0)
  %lock35_2 = AIE.lock(%tile35_2, 0)
  %lock35_3 = AIE.lock(%tile35_3, 0)
  %lock35_4 = AIE.lock(%tile35_4, 0)
  %lock35_5 = AIE.lock(%tile35_5, 0)
  %lock35_6 = AIE.lock(%tile35_6, 0)
  %lock35_7 = AIE.lock(%tile35_7, 0)
  %lock36_7 = AIE.lock(%tile36_7, 0)
  %lock36_6 = AIE.lock(%tile36_6, 0)
  %lock36_5 = AIE.lock(%tile36_5, 0)
  %lock36_4 = AIE.lock(%tile36_4, 0)
  %lock36_3 = AIE.lock(%tile36_3, 0)
  %lock36_2 = AIE.lock(%tile36_2, 0)
  %lock36_1 = AIE.lock(%tile36_1, 0)
  %lock37_1 = AIE.lock(%tile37_1, 0)
  %lock37_2 = AIE.lock(%tile37_2, 0)
  %lock37_3 = AIE.lock(%tile37_3, 0)
  %lock37_4 = AIE.lock(%tile37_4, 0)
  %lock37_5 = AIE.lock(%tile37_5, 0)
  %lock37_6 = AIE.lock(%tile37_6, 0)
  %lock37_7 = AIE.lock(%tile37_7, 0)
  %lock38_7 = AIE.lock(%tile38_7, 0)
  %lock38_6 = AIE.lock(%tile38_6, 0)
  %lock38_5 = AIE.lock(%tile38_5, 0)
  %lock38_4 = AIE.lock(%tile38_4, 0)
  %lock38_3 = AIE.lock(%tile38_3, 0)
  %lock38_2 = AIE.lock(%tile38_2, 0)
  %lock38_1 = AIE.lock(%tile38_1, 0)
  %lock39_1 = AIE.lock(%tile39_1, 0)
  %lock39_2 = AIE.lock(%tile39_2, 0)
  %lock39_3 = AIE.lock(%tile39_3, 0)
  %lock39_4 = AIE.lock(%tile39_4, 0)
  %lock39_5 = AIE.lock(%tile39_5, 0)
  %lock39_6 = AIE.lock(%tile39_6, 0)
  %lock39_7 = AIE.lock(%tile39_7, 0)
  %lock40_7 = AIE.lock(%tile40_7, 0)
  %lock40_6 = AIE.lock(%tile40_6, 0)
  %lock40_5 = AIE.lock(%tile40_5, 0)
  %lock40_4 = AIE.lock(%tile40_4, 0)
  %lock40_3 = AIE.lock(%tile40_3, 0)
  %lock40_2 = AIE.lock(%tile40_2, 0)
  %lock40_1 = AIE.lock(%tile40_1, 0)
  %lock41_1 = AIE.lock(%tile41_1, 0)
  %lock41_2 = AIE.lock(%tile41_2, 0)
  %lock41_3 = AIE.lock(%tile41_3, 0)
  %lock41_4 = AIE.lock(%tile41_4, 0)
  %lock41_5 = AIE.lock(%tile41_5, 0)
  %lock41_6 = AIE.lock(%tile41_6, 0)
  %lock41_7 = AIE.lock(%tile41_7, 0)
  %lock42_7 = AIE.lock(%tile42_7, 0)
  %lock42_6 = AIE.lock(%tile42_6, 0)
  %lock42_5 = AIE.lock(%tile42_5, 0)
  %lock42_4 = AIE.lock(%tile42_4, 0)
  %lock42_3 = AIE.lock(%tile42_3, 0)
  %lock42_2 = AIE.lock(%tile42_2, 0)
  %lock42_1 = AIE.lock(%tile42_1, 0)
  %lock43_1 = AIE.lock(%tile43_1, 0)
  %lock43_2 = AIE.lock(%tile43_2, 0)
  %lock43_3 = AIE.lock(%tile43_3, 0)
  %lock43_4 = AIE.lock(%tile43_4, 0)
  %lock43_5 = AIE.lock(%tile43_5, 0)
  %lock43_6 = AIE.lock(%tile43_6, 0)
  %lock43_7 = AIE.lock(%tile43_7, 0)
  %lock44_7 = AIE.lock(%tile44_7, 0)
  %lock44_6 = AIE.lock(%tile44_6, 0)
  %lock44_5 = AIE.lock(%tile44_5, 0)
  %lock44_4 = AIE.lock(%tile44_4, 0)
  %lock44_3 = AIE.lock(%tile44_3, 0)
  %lock44_2 = AIE.lock(%tile44_2, 0)
  %lock44_1 = AIE.lock(%tile44_1, 0)
  %lock45_1 = AIE.lock(%tile45_1, 0)
  %lock45_2 = AIE.lock(%tile45_2, 0)
  %lock45_3 = AIE.lock(%tile45_3, 0)
  %lock45_4 = AIE.lock(%tile45_4, 0)
  %lock45_5 = AIE.lock(%tile45_5, 0)
  %lock45_6 = AIE.lock(%tile45_6, 0)
  %lock45_7 = AIE.lock(%tile45_7, 0)
  %lock46_7 = AIE.lock(%tile46_7, 0)
  %lock46_6 = AIE.lock(%tile46_6, 0)
  %lock46_5 = AIE.lock(%tile46_5, 0)
  %lock46_4 = AIE.lock(%tile46_4, 0)
  %lock46_3 = AIE.lock(%tile46_3, 0)
  %lock46_2 = AIE.lock(%tile46_2, 0)
  %lock46_1 = AIE.lock(%tile46_1, 0)
  %lock47_1 = AIE.lock(%tile47_1, 0)
  %lock47_2 = AIE.lock(%tile47_2, 0)
  %lock47_3 = AIE.lock(%tile47_3, 0)
  %lock47_4 = AIE.lock(%tile47_4, 0)
  %lock47_5 = AIE.lock(%tile47_5, 0)
  %lock47_6 = AIE.lock(%tile47_6, 0)
  %lock47_7 = AIE.lock(%tile47_7, 0)
  %lock48_7 = AIE.lock(%tile48_7, 0)
  %lock48_6 = AIE.lock(%tile48_6, 0)
  %lock48_5 = AIE.lock(%tile48_5, 0)
  %lock48_4 = AIE.lock(%tile48_4, 0)
  %lock48_3 = AIE.lock(%tile48_3, 0)
  %lock48_2 = AIE.lock(%tile48_2, 0)
  %lock48_1 = AIE.lock(%tile48_1, 0) { sym_name = "prime_output_lock" }

  %buf1_1 = AIE.buffer(%tile1_1) { sym_name = "a" } : memref<4096xi32>
  %buf1_2 = AIE.buffer(%tile1_2) { sym_name = "prime2" } : memref<4096xi32>
  %buf1_3 = AIE.buffer(%tile1_3) { sym_name = "prime3" } : memref<4096xi32>
  %buf1_4 = AIE.buffer(%tile1_4) { sym_name = "prime5" } : memref<4096xi32>
  %buf1_5 = AIE.buffer(%tile1_5) { sym_name = "prime7" } : memref<4096xi32>
  %buf1_6 = AIE.buffer(%tile1_6) { sym_name = "prime11" } : memref<4096xi32>
  %buf1_7 = AIE.buffer(%tile1_7) { sym_name = "prime13" } : memref<4096xi32>
  %buf2_7 = AIE.buffer(%tile2_7) { sym_name = "prime17" } : memref<4096xi32>
  %buf2_6 = AIE.buffer(%tile2_6) { sym_name = "prime19" } : memref<4096xi32>
  %buf2_5 = AIE.buffer(%tile2_5) { sym_name = "prime23" } : memref<4096xi32>
  %buf2_4 = AIE.buffer(%tile2_4) { sym_name = "prime29" } : memref<4096xi32>
  %buf2_3 = AIE.buffer(%tile2_3) { sym_name = "prime31" } : memref<4096xi32>
  %buf2_2 = AIE.buffer(%tile2_2) { sym_name = "prime37" } : memref<4096xi32>
  %buf2_1 = AIE.buffer(%tile2_1) { sym_name = "prime41" } : memref<4096xi32>
  %buf3_1 = AIE.buffer(%tile3_1) { sym_name = "prime43" } : memref<4096xi32>
  %buf3_2 = AIE.buffer(%tile3_2) { sym_name = "prime47" } : memref<4096xi32>
  %buf3_3 = AIE.buffer(%tile3_3) { sym_name = "prime53" } : memref<4096xi32>
  %buf3_4 = AIE.buffer(%tile3_4) { sym_name = "prime59" } : memref<4096xi32>
  %buf3_5 = AIE.buffer(%tile3_5) { sym_name = "prime61" } : memref<4096xi32>
  %buf3_6 = AIE.buffer(%tile3_6) { sym_name = "prime67" } : memref<4096xi32>
  %buf3_7 = AIE.buffer(%tile3_7) { sym_name = "prime71" } : memref<4096xi32>
  %buf4_7 = AIE.buffer(%tile4_7) { sym_name = "prime73" } : memref<4096xi32>
  %buf4_6 = AIE.buffer(%tile4_6) { sym_name = "prime79" } : memref<4096xi32>
  %buf4_5 = AIE.buffer(%tile4_5) { sym_name = "prime83" } : memref<4096xi32>
  %buf4_4 = AIE.buffer(%tile4_4) { sym_name = "prime89" } : memref<4096xi32>
  %buf4_3 = AIE.buffer(%tile4_3) { sym_name = "prime97" } : memref<4096xi32>
  %buf4_2 = AIE.buffer(%tile4_2) { sym_name = "prime101" } : memref<4096xi32>
  %buf4_1 = AIE.buffer(%tile4_1) { sym_name = "prime103" } : memref<4096xi32>
  %buf5_1 = AIE.buffer(%tile5_1) { sym_name = "prime107" } : memref<4096xi32>
  %buf5_2 = AIE.buffer(%tile5_2) { sym_name = "prime109" } : memref<4096xi32>
  %buf5_3 = AIE.buffer(%tile5_3) { sym_name = "prime113" } : memref<4096xi32>
  %buf5_4 = AIE.buffer(%tile5_4) { sym_name = "prime127" } : memref<4096xi32>
  %buf5_5 = AIE.buffer(%tile5_5) { sym_name = "prime131" } : memref<4096xi32>
  %buf5_6 = AIE.buffer(%tile5_6) { sym_name = "prime137" } : memref<4096xi32>
  %buf5_7 = AIE.buffer(%tile5_7) { sym_name = "prime139" } : memref<4096xi32>
  %buf6_7 = AIE.buffer(%tile6_7) { sym_name = "prime149" } : memref<4096xi32>
  %buf6_6 = AIE.buffer(%tile6_6) { sym_name = "prime151" } : memref<4096xi32>
  %buf6_5 = AIE.buffer(%tile6_5) { sym_name = "prime157" } : memref<4096xi32>
  %buf6_4 = AIE.buffer(%tile6_4) { sym_name = "prime163" } : memref<4096xi32>
  %buf6_3 = AIE.buffer(%tile6_3) { sym_name = "prime167" } : memref<4096xi32>
  %buf6_2 = AIE.buffer(%tile6_2) { sym_name = "prime173" } : memref<4096xi32>
  %buf6_1 = AIE.buffer(%tile6_1) { sym_name = "prime179" } : memref<4096xi32>
  %buf7_1 = AIE.buffer(%tile7_1) { sym_name = "prime181" } : memref<4096xi32>
  %buf7_2 = AIE.buffer(%tile7_2) { sym_name = "prime191" } : memref<4096xi32>
  %buf7_3 = AIE.buffer(%tile7_3) { sym_name = "prime193" } : memref<4096xi32>
  %buf7_4 = AIE.buffer(%tile7_4) { sym_name = "prime197" } : memref<4096xi32>
  %buf7_5 = AIE.buffer(%tile7_5) { sym_name = "prime199" } : memref<4096xi32>
  %buf7_6 = AIE.buffer(%tile7_6) { sym_name = "prime211" } : memref<4096xi32>
  %buf7_7 = AIE.buffer(%tile7_7) { sym_name = "prime223" } : memref<4096xi32>
  %buf8_7 = AIE.buffer(%tile8_7) { sym_name = "prime227" } : memref<4096xi32>
  %buf8_6 = AIE.buffer(%tile8_6) { sym_name = "prime229" } : memref<4096xi32>
  %buf8_5 = AIE.buffer(%tile8_5) { sym_name = "prime233" } : memref<4096xi32>
  %buf8_4 = AIE.buffer(%tile8_4) { sym_name = "prime239" } : memref<4096xi32>
  %buf8_3 = AIE.buffer(%tile8_3) { sym_name = "prime241" } : memref<4096xi32>
  %buf8_2 = AIE.buffer(%tile8_2) { sym_name = "prime251" } : memref<4096xi32>
  %buf8_1 = AIE.buffer(%tile8_1) { sym_name = "prime257" } : memref<4096xi32>
  %buf9_1 = AIE.buffer(%tile9_1) { sym_name = "prime263" } : memref<4096xi32>
  %buf9_2 = AIE.buffer(%tile9_2) { sym_name = "prime269" } : memref<4096xi32>
  %buf9_3 = AIE.buffer(%tile9_3) { sym_name = "prime271" } : memref<4096xi32>
  %buf9_4 = AIE.buffer(%tile9_4) { sym_name = "prime277" } : memref<4096xi32>
  %buf9_5 = AIE.buffer(%tile9_5) { sym_name = "prime281" } : memref<4096xi32>
  %buf9_6 = AIE.buffer(%tile9_6) { sym_name = "prime283" } : memref<4096xi32>
  %buf9_7 = AIE.buffer(%tile9_7) { sym_name = "prime293" } : memref<4096xi32>
  %buf10_7 = AIE.buffer(%tile10_7) { sym_name = "prime307" } : memref<4096xi32>
  %buf10_6 = AIE.buffer(%tile10_6) { sym_name = "prime311" } : memref<4096xi32>
  %buf10_5 = AIE.buffer(%tile10_5) { sym_name = "prime313" } : memref<4096xi32>
  %buf10_4 = AIE.buffer(%tile10_4) { sym_name = "prime317" } : memref<4096xi32>
  %buf10_3 = AIE.buffer(%tile10_3) { sym_name = "prime331" } : memref<4096xi32>
  %buf10_2 = AIE.buffer(%tile10_2) { sym_name = "prime337" } : memref<4096xi32>
  %buf10_1 = AIE.buffer(%tile10_1) { sym_name = "prime347" } : memref<4096xi32>
  %buf11_1 = AIE.buffer(%tile11_1) { sym_name = "prime349" } : memref<4096xi32>
  %buf11_2 = AIE.buffer(%tile11_2) { sym_name = "prime353" } : memref<4096xi32>
  %buf11_3 = AIE.buffer(%tile11_3) { sym_name = "prime359" } : memref<4096xi32>
  %buf11_4 = AIE.buffer(%tile11_4) { sym_name = "prime367" } : memref<4096xi32>
  %buf11_5 = AIE.buffer(%tile11_5) { sym_name = "prime373" } : memref<4096xi32>
  %buf11_6 = AIE.buffer(%tile11_6) { sym_name = "prime379" } : memref<4096xi32>
  %buf11_7 = AIE.buffer(%tile11_7) { sym_name = "prime383" } : memref<4096xi32>
  %buf12_7 = AIE.buffer(%tile12_7) { sym_name = "prime389" } : memref<4096xi32>
  %buf12_6 = AIE.buffer(%tile12_6) { sym_name = "prime397" } : memref<4096xi32>
  %buf12_5 = AIE.buffer(%tile12_5) { sym_name = "prime401" } : memref<4096xi32>
  %buf12_4 = AIE.buffer(%tile12_4) { sym_name = "prime409" } : memref<4096xi32>
  %buf12_3 = AIE.buffer(%tile12_3) { sym_name = "prime419" } : memref<4096xi32>
  %buf12_2 = AIE.buffer(%tile12_2) { sym_name = "prime421" } : memref<4096xi32>
  %buf12_1 = AIE.buffer(%tile12_1) { sym_name = "prime431" } : memref<4096xi32>
  %buf13_1 = AIE.buffer(%tile13_1) { sym_name = "prime433" } : memref<4096xi32>
  %buf13_2 = AIE.buffer(%tile13_2) { sym_name = "prime439" } : memref<4096xi32>
  %buf13_3 = AIE.buffer(%tile13_3) { sym_name = "prime443" } : memref<4096xi32>
  %buf13_4 = AIE.buffer(%tile13_4) { sym_name = "prime449" } : memref<4096xi32>
  %buf13_5 = AIE.buffer(%tile13_5) { sym_name = "prime457" } : memref<4096xi32>
  %buf13_6 = AIE.buffer(%tile13_6) { sym_name = "prime461" } : memref<4096xi32>
  %buf13_7 = AIE.buffer(%tile13_7) { sym_name = "prime463" } : memref<4096xi32>
  %buf14_7 = AIE.buffer(%tile14_7) { sym_name = "prime467" } : memref<4096xi32>
  %buf14_6 = AIE.buffer(%tile14_6) { sym_name = "prime479" } : memref<4096xi32>
  %buf14_5 = AIE.buffer(%tile14_5) { sym_name = "prime487" } : memref<4096xi32>
  %buf14_4 = AIE.buffer(%tile14_4) { sym_name = "prime491" } : memref<4096xi32>
  %buf14_3 = AIE.buffer(%tile14_3) { sym_name = "prime499" } : memref<4096xi32>
  %buf14_2 = AIE.buffer(%tile14_2) { sym_name = "prime503" } : memref<4096xi32>
  %buf14_1 = AIE.buffer(%tile14_1) { sym_name = "prime509" } : memref<4096xi32>
  %buf15_1 = AIE.buffer(%tile15_1) { sym_name = "prime521" } : memref<4096xi32>
  %buf15_2 = AIE.buffer(%tile15_2) { sym_name = "prime523" } : memref<4096xi32>
  %buf15_3 = AIE.buffer(%tile15_3) { sym_name = "prime541" } : memref<4096xi32>
  %buf15_4 = AIE.buffer(%tile15_4) { sym_name = "prime547" } : memref<4096xi32>
  %buf15_5 = AIE.buffer(%tile15_5) { sym_name = "prime557" } : memref<4096xi32>
  %buf15_6 = AIE.buffer(%tile15_6) { sym_name = "prime563" } : memref<4096xi32>
  %buf15_7 = AIE.buffer(%tile15_7) { sym_name = "prime569" } : memref<4096xi32>
  %buf16_7 = AIE.buffer(%tile16_7) { sym_name = "prime571" } : memref<4096xi32>
  %buf16_6 = AIE.buffer(%tile16_6) { sym_name = "prime577" } : memref<4096xi32>
  %buf16_5 = AIE.buffer(%tile16_5) { sym_name = "prime587" } : memref<4096xi32>
  %buf16_4 = AIE.buffer(%tile16_4) { sym_name = "prime593" } : memref<4096xi32>
  %buf16_3 = AIE.buffer(%tile16_3) { sym_name = "prime599" } : memref<4096xi32>
  %buf16_2 = AIE.buffer(%tile16_2) { sym_name = "prime601" } : memref<4096xi32>
  %buf16_1 = AIE.buffer(%tile16_1) { sym_name = "prime607" } : memref<4096xi32>
  %buf17_1 = AIE.buffer(%tile17_1) { sym_name = "prime613" } : memref<4096xi32>
  %buf17_2 = AIE.buffer(%tile17_2) { sym_name = "prime617" } : memref<4096xi32>
  %buf17_3 = AIE.buffer(%tile17_3) { sym_name = "prime619" } : memref<4096xi32>
  %buf17_4 = AIE.buffer(%tile17_4) { sym_name = "prime631" } : memref<4096xi32>
  %buf17_5 = AIE.buffer(%tile17_5) { sym_name = "prime641" } : memref<4096xi32>
  %buf17_6 = AIE.buffer(%tile17_6) { sym_name = "prime643" } : memref<4096xi32>
  %buf17_7 = AIE.buffer(%tile17_7) { sym_name = "prime647" } : memref<4096xi32>
  %buf18_7 = AIE.buffer(%tile18_7) { sym_name = "prime653" } : memref<4096xi32>
  %buf18_6 = AIE.buffer(%tile18_6) { sym_name = "prime659" } : memref<4096xi32>
  %buf18_5 = AIE.buffer(%tile18_5) { sym_name = "prime661" } : memref<4096xi32>
  %buf18_4 = AIE.buffer(%tile18_4) { sym_name = "prime673" } : memref<4096xi32>
  %buf18_3 = AIE.buffer(%tile18_3) { sym_name = "prime677" } : memref<4096xi32>
  %buf18_2 = AIE.buffer(%tile18_2) { sym_name = "prime683" } : memref<4096xi32>
  %buf18_1 = AIE.buffer(%tile18_1) { sym_name = "prime691" } : memref<4096xi32>
  %buf19_1 = AIE.buffer(%tile19_1) { sym_name = "prime701" } : memref<4096xi32>
  %buf19_2 = AIE.buffer(%tile19_2) { sym_name = "prime709" } : memref<4096xi32>
  %buf19_3 = AIE.buffer(%tile19_3) { sym_name = "prime719" } : memref<4096xi32>
  %buf19_4 = AIE.buffer(%tile19_4) { sym_name = "prime727" } : memref<4096xi32>
  %buf19_5 = AIE.buffer(%tile19_5) { sym_name = "prime733" } : memref<4096xi32>
  %buf19_6 = AIE.buffer(%tile19_6) { sym_name = "prime739" } : memref<4096xi32>
  %buf19_7 = AIE.buffer(%tile19_7) { sym_name = "prime743" } : memref<4096xi32>
  %buf20_7 = AIE.buffer(%tile20_7) { sym_name = "prime751" } : memref<4096xi32>
  %buf20_6 = AIE.buffer(%tile20_6) { sym_name = "prime757" } : memref<4096xi32>
  %buf20_5 = AIE.buffer(%tile20_5) { sym_name = "prime761" } : memref<4096xi32>
  %buf20_4 = AIE.buffer(%tile20_4) { sym_name = "prime769" } : memref<4096xi32>
  %buf20_3 = AIE.buffer(%tile20_3) { sym_name = "prime773" } : memref<4096xi32>
  %buf20_2 = AIE.buffer(%tile20_2) { sym_name = "prime787" } : memref<4096xi32>
  %buf20_1 = AIE.buffer(%tile20_1) { sym_name = "prime797" } : memref<4096xi32>
  %buf21_1 = AIE.buffer(%tile21_1) { sym_name = "prime809" } : memref<4096xi32>
  %buf21_2 = AIE.buffer(%tile21_2) { sym_name = "prime811" } : memref<4096xi32>
  %buf21_3 = AIE.buffer(%tile21_3) { sym_name = "prime821" } : memref<4096xi32>
  %buf21_4 = AIE.buffer(%tile21_4) { sym_name = "prime823" } : memref<4096xi32>
  %buf21_5 = AIE.buffer(%tile21_5) { sym_name = "prime827" } : memref<4096xi32>
  %buf21_6 = AIE.buffer(%tile21_6) { sym_name = "prime829" } : memref<4096xi32>
  %buf21_7 = AIE.buffer(%tile21_7) { sym_name = "prime839" } : memref<4096xi32>
  %buf22_7 = AIE.buffer(%tile22_7) { sym_name = "prime853" } : memref<4096xi32>
  %buf22_6 = AIE.buffer(%tile22_6) { sym_name = "prime857" } : memref<4096xi32>
  %buf22_5 = AIE.buffer(%tile22_5) { sym_name = "prime859" } : memref<4096xi32>
  %buf22_4 = AIE.buffer(%tile22_4) { sym_name = "prime863" } : memref<4096xi32>
  %buf22_3 = AIE.buffer(%tile22_3) { sym_name = "prime877" } : memref<4096xi32>
  %buf22_2 = AIE.buffer(%tile22_2) { sym_name = "prime881" } : memref<4096xi32>
  %buf22_1 = AIE.buffer(%tile22_1) { sym_name = "prime883" } : memref<4096xi32>
  %buf23_1 = AIE.buffer(%tile23_1) { sym_name = "prime887" } : memref<4096xi32>
  %buf23_2 = AIE.buffer(%tile23_2) { sym_name = "prime907" } : memref<4096xi32>
  %buf23_3 = AIE.buffer(%tile23_3) { sym_name = "prime911" } : memref<4096xi32>
  %buf23_4 = AIE.buffer(%tile23_4) { sym_name = "prime919" } : memref<4096xi32>
  %buf23_5 = AIE.buffer(%tile23_5) { sym_name = "prime929" } : memref<4096xi32>
  %buf23_6 = AIE.buffer(%tile23_6) { sym_name = "prime937" } : memref<4096xi32>
  %buf23_7 = AIE.buffer(%tile23_7) { sym_name = "prime941" } : memref<4096xi32>
  %buf24_7 = AIE.buffer(%tile24_7) { sym_name = "prime947" } : memref<4096xi32>
  %buf24_6 = AIE.buffer(%tile24_6) { sym_name = "prime953" } : memref<4096xi32>
  %buf24_5 = AIE.buffer(%tile24_5) { sym_name = "prime967" } : memref<4096xi32>
  %buf24_4 = AIE.buffer(%tile24_4) { sym_name = "prime971" } : memref<4096xi32>
  %buf24_3 = AIE.buffer(%tile24_3) { sym_name = "prime977" } : memref<4096xi32>
  %buf24_2 = AIE.buffer(%tile24_2) { sym_name = "prime983" } : memref<4096xi32>
  %buf24_1 = AIE.buffer(%tile24_1) { sym_name = "prime991" } : memref<4096xi32>
  %buf25_1 = AIE.buffer(%tile25_1) { sym_name = "prime997" } : memref<4096xi32>
  %buf25_2 = AIE.buffer(%tile25_2) { sym_name = "prime1009" } : memref<4096xi32>
  %buf25_3 = AIE.buffer(%tile25_3) { sym_name = "prime1013" } : memref<4096xi32>
  %buf25_4 = AIE.buffer(%tile25_4) { sym_name = "prime1019" } : memref<4096xi32>
  %buf25_5 = AIE.buffer(%tile25_5) { sym_name = "prime1021" } : memref<4096xi32>
  %buf25_6 = AIE.buffer(%tile25_6) { sym_name = "prime1031" } : memref<4096xi32>
  %buf25_7 = AIE.buffer(%tile25_7) { sym_name = "prime1033" } : memref<4096xi32>
  %buf26_7 = AIE.buffer(%tile26_7) { sym_name = "prime1039" } : memref<4096xi32>
  %buf26_6 = AIE.buffer(%tile26_6) { sym_name = "prime1049" } : memref<4096xi32>
  %buf26_5 = AIE.buffer(%tile26_5) { sym_name = "prime1051" } : memref<4096xi32>
  %buf26_4 = AIE.buffer(%tile26_4) { sym_name = "prime1061" } : memref<4096xi32>
  %buf26_3 = AIE.buffer(%tile26_3) { sym_name = "prime1063" } : memref<4096xi32>
  %buf26_2 = AIE.buffer(%tile26_2) { sym_name = "prime1069" } : memref<4096xi32>
  %buf26_1 = AIE.buffer(%tile26_1) { sym_name = "prime1087" } : memref<4096xi32>
  %buf27_1 = AIE.buffer(%tile27_1) { sym_name = "prime1091" } : memref<4096xi32>
  %buf27_2 = AIE.buffer(%tile27_2) { sym_name = "prime1093" } : memref<4096xi32>
  %buf27_3 = AIE.buffer(%tile27_3) { sym_name = "prime1097" } : memref<4096xi32>
  %buf27_4 = AIE.buffer(%tile27_4) { sym_name = "prime1103" } : memref<4096xi32>
  %buf27_5 = AIE.buffer(%tile27_5) { sym_name = "prime1109" } : memref<4096xi32>
  %buf27_6 = AIE.buffer(%tile27_6) { sym_name = "prime1117" } : memref<4096xi32>
  %buf27_7 = AIE.buffer(%tile27_7) { sym_name = "prime1123" } : memref<4096xi32>
  %buf28_7 = AIE.buffer(%tile28_7) { sym_name = "prime1129" } : memref<4096xi32>
  %buf28_6 = AIE.buffer(%tile28_6) { sym_name = "prime1151" } : memref<4096xi32>
  %buf28_5 = AIE.buffer(%tile28_5) { sym_name = "prime1153" } : memref<4096xi32>
  %buf28_4 = AIE.buffer(%tile28_4) { sym_name = "prime1163" } : memref<4096xi32>
  %buf28_3 = AIE.buffer(%tile28_3) { sym_name = "prime1171" } : memref<4096xi32>
  %buf28_2 = AIE.buffer(%tile28_2) { sym_name = "prime1181" } : memref<4096xi32>
  %buf28_1 = AIE.buffer(%tile28_1) { sym_name = "prime1187" } : memref<4096xi32>
  %buf29_1 = AIE.buffer(%tile29_1) { sym_name = "prime1193" } : memref<4096xi32>
  %buf29_2 = AIE.buffer(%tile29_2) { sym_name = "prime1201" } : memref<4096xi32>
  %buf29_3 = AIE.buffer(%tile29_3) { sym_name = "prime1213" } : memref<4096xi32>
  %buf29_4 = AIE.buffer(%tile29_4) { sym_name = "prime1217" } : memref<4096xi32>
  %buf29_5 = AIE.buffer(%tile29_5) { sym_name = "prime1223" } : memref<4096xi32>
  %buf29_6 = AIE.buffer(%tile29_6) { sym_name = "prime1229" } : memref<4096xi32>
  %buf29_7 = AIE.buffer(%tile29_7) { sym_name = "prime1231" } : memref<4096xi32>
  %buf30_7 = AIE.buffer(%tile30_7) { sym_name = "prime1237" } : memref<4096xi32>
  %buf30_6 = AIE.buffer(%tile30_6) { sym_name = "prime1249" } : memref<4096xi32>
  %buf30_5 = AIE.buffer(%tile30_5) { sym_name = "prime1259" } : memref<4096xi32>
  %buf30_4 = AIE.buffer(%tile30_4) { sym_name = "prime1277" } : memref<4096xi32>
  %buf30_3 = AIE.buffer(%tile30_3) { sym_name = "prime1279" } : memref<4096xi32>
  %buf30_2 = AIE.buffer(%tile30_2) { sym_name = "prime1283" } : memref<4096xi32>
  %buf30_1 = AIE.buffer(%tile30_1) { sym_name = "prime1289" } : memref<4096xi32>
  %buf31_1 = AIE.buffer(%tile31_1) { sym_name = "prime1291" } : memref<4096xi32>
  %buf31_2 = AIE.buffer(%tile31_2) { sym_name = "prime1297" } : memref<4096xi32>
  %buf31_3 = AIE.buffer(%tile31_3) { sym_name = "prime1301" } : memref<4096xi32>
  %buf31_4 = AIE.buffer(%tile31_4) { sym_name = "prime1303" } : memref<4096xi32>
  %buf31_5 = AIE.buffer(%tile31_5) { sym_name = "prime1307" } : memref<4096xi32>
  %buf31_6 = AIE.buffer(%tile31_6) { sym_name = "prime1319" } : memref<4096xi32>
  %buf31_7 = AIE.buffer(%tile31_7) { sym_name = "prime1321" } : memref<4096xi32>
  %buf32_7 = AIE.buffer(%tile32_7) { sym_name = "prime1327" } : memref<4096xi32>
  %buf32_6 = AIE.buffer(%tile32_6) { sym_name = "prime1361" } : memref<4096xi32>
  %buf32_5 = AIE.buffer(%tile32_5) { sym_name = "prime1367" } : memref<4096xi32>
  %buf32_4 = AIE.buffer(%tile32_4) { sym_name = "prime1373" } : memref<4096xi32>
  %buf32_3 = AIE.buffer(%tile32_3) { sym_name = "prime1381" } : memref<4096xi32>
  %buf32_2 = AIE.buffer(%tile32_2) { sym_name = "prime1399" } : memref<4096xi32>
  %buf32_1 = AIE.buffer(%tile32_1) { sym_name = "prime1409" } : memref<4096xi32>
  %buf33_1 = AIE.buffer(%tile33_1) { sym_name = "prime1423" } : memref<4096xi32>
  %buf33_2 = AIE.buffer(%tile33_2) { sym_name = "prime1427" } : memref<4096xi32>
  %buf33_3 = AIE.buffer(%tile33_3) { sym_name = "prime1429" } : memref<4096xi32>
  %buf33_4 = AIE.buffer(%tile33_4) { sym_name = "prime1433" } : memref<4096xi32>
  %buf33_5 = AIE.buffer(%tile33_5) { sym_name = "prime1439" } : memref<4096xi32>
  %buf33_6 = AIE.buffer(%tile33_6) { sym_name = "prime1447" } : memref<4096xi32>
  %buf33_7 = AIE.buffer(%tile33_7) { sym_name = "prime1451" } : memref<4096xi32>
  %buf34_7 = AIE.buffer(%tile34_7) { sym_name = "prime1453" } : memref<4096xi32>
  %buf34_6 = AIE.buffer(%tile34_6) { sym_name = "prime1459" } : memref<4096xi32>
  %buf34_5 = AIE.buffer(%tile34_5) { sym_name = "prime1471" } : memref<4096xi32>
  %buf34_4 = AIE.buffer(%tile34_4) { sym_name = "prime1481" } : memref<4096xi32>
  %buf34_3 = AIE.buffer(%tile34_3) { sym_name = "prime1483" } : memref<4096xi32>
  %buf34_2 = AIE.buffer(%tile34_2) { sym_name = "prime1487" } : memref<4096xi32>
  %buf34_1 = AIE.buffer(%tile34_1) { sym_name = "prime1489" } : memref<4096xi32>
  %buf35_1 = AIE.buffer(%tile35_1) { sym_name = "prime1493" } : memref<4096xi32>
  %buf35_2 = AIE.buffer(%tile35_2) { sym_name = "prime1499" } : memref<4096xi32>
  %buf35_3 = AIE.buffer(%tile35_3) { sym_name = "prime1511" } : memref<4096xi32>
  %buf35_4 = AIE.buffer(%tile35_4) { sym_name = "prime1523" } : memref<4096xi32>
  %buf35_5 = AIE.buffer(%tile35_5) { sym_name = "prime1531" } : memref<4096xi32>
  %buf35_6 = AIE.buffer(%tile35_6) { sym_name = "prime1543" } : memref<4096xi32>
  %buf35_7 = AIE.buffer(%tile35_7) { sym_name = "prime1549" } : memref<4096xi32>
  %buf36_7 = AIE.buffer(%tile36_7) { sym_name = "prime1553" } : memref<4096xi32>
  %buf36_6 = AIE.buffer(%tile36_6) { sym_name = "prime1559" } : memref<4096xi32>
  %buf36_5 = AIE.buffer(%tile36_5) { sym_name = "prime1567" } : memref<4096xi32>
  %buf36_4 = AIE.buffer(%tile36_4) { sym_name = "prime1571" } : memref<4096xi32>
  %buf36_3 = AIE.buffer(%tile36_3) { sym_name = "prime1579" } : memref<4096xi32>
  %buf36_2 = AIE.buffer(%tile36_2) { sym_name = "prime1583" } : memref<4096xi32>
  %buf36_1 = AIE.buffer(%tile36_1) { sym_name = "prime1597" } : memref<4096xi32>
  %buf37_1 = AIE.buffer(%tile37_1) { sym_name = "prime1601" } : memref<4096xi32>
  %buf37_2 = AIE.buffer(%tile37_2) { sym_name = "prime1607" } : memref<4096xi32>
  %buf37_3 = AIE.buffer(%tile37_3) { sym_name = "prime1609" } : memref<4096xi32>
  %buf37_4 = AIE.buffer(%tile37_4) { sym_name = "prime1613" } : memref<4096xi32>
  %buf37_5 = AIE.buffer(%tile37_5) { sym_name = "prime1619" } : memref<4096xi32>
  %buf37_6 = AIE.buffer(%tile37_6) { sym_name = "prime1621" } : memref<4096xi32>
  %buf37_7 = AIE.buffer(%tile37_7) { sym_name = "prime1627" } : memref<4096xi32>
  %buf38_7 = AIE.buffer(%tile38_7) { sym_name = "prime1637" } : memref<4096xi32>
  %buf38_6 = AIE.buffer(%tile38_6) { sym_name = "prime1657" } : memref<4096xi32>
  %buf38_5 = AIE.buffer(%tile38_5) { sym_name = "prime1663" } : memref<4096xi32>
  %buf38_4 = AIE.buffer(%tile38_4) { sym_name = "prime1667" } : memref<4096xi32>
  %buf38_3 = AIE.buffer(%tile38_3) { sym_name = "prime1669" } : memref<4096xi32>
  %buf38_2 = AIE.buffer(%tile38_2) { sym_name = "prime1693" } : memref<4096xi32>
  %buf38_1 = AIE.buffer(%tile38_1) { sym_name = "prime1697" } : memref<4096xi32>
  %buf39_1 = AIE.buffer(%tile39_1) { sym_name = "prime1699" } : memref<4096xi32>
  %buf39_2 = AIE.buffer(%tile39_2) { sym_name = "prime1709" } : memref<4096xi32>
  %buf39_3 = AIE.buffer(%tile39_3) { sym_name = "prime1721" } : memref<4096xi32>
  %buf39_4 = AIE.buffer(%tile39_4) { sym_name = "prime1723" } : memref<4096xi32>
  %buf39_5 = AIE.buffer(%tile39_5) { sym_name = "prime1733" } : memref<4096xi32>
  %buf39_6 = AIE.buffer(%tile39_6) { sym_name = "prime1741" } : memref<4096xi32>
  %buf39_7 = AIE.buffer(%tile39_7) { sym_name = "prime1747" } : memref<4096xi32>
  %buf40_7 = AIE.buffer(%tile40_7) { sym_name = "prime1753" } : memref<4096xi32>
  %buf40_6 = AIE.buffer(%tile40_6) { sym_name = "prime1759" } : memref<4096xi32>
  %buf40_5 = AIE.buffer(%tile40_5) { sym_name = "prime1777" } : memref<4096xi32>
  %buf40_4 = AIE.buffer(%tile40_4) { sym_name = "prime1783" } : memref<4096xi32>
  %buf40_3 = AIE.buffer(%tile40_3) { sym_name = "prime1787" } : memref<4096xi32>
  %buf40_2 = AIE.buffer(%tile40_2) { sym_name = "prime1789" } : memref<4096xi32>
  %buf40_1 = AIE.buffer(%tile40_1) { sym_name = "prime1801" } : memref<4096xi32>
  %buf41_1 = AIE.buffer(%tile41_1) { sym_name = "prime1811" } : memref<4096xi32>
  %buf41_2 = AIE.buffer(%tile41_2) { sym_name = "prime1823" } : memref<4096xi32>
  %buf41_3 = AIE.buffer(%tile41_3) { sym_name = "prime1831" } : memref<4096xi32>
  %buf41_4 = AIE.buffer(%tile41_4) { sym_name = "prime1847" } : memref<4096xi32>
  %buf41_5 = AIE.buffer(%tile41_5) { sym_name = "prime1861" } : memref<4096xi32>
  %buf41_6 = AIE.buffer(%tile41_6) { sym_name = "prime1867" } : memref<4096xi32>
  %buf41_7 = AIE.buffer(%tile41_7) { sym_name = "prime1871" } : memref<4096xi32>
  %buf42_7 = AIE.buffer(%tile42_7) { sym_name = "prime1873" } : memref<4096xi32>
  %buf42_6 = AIE.buffer(%tile42_6) { sym_name = "prime1877" } : memref<4096xi32>
  %buf42_5 = AIE.buffer(%tile42_5) { sym_name = "prime1879" } : memref<4096xi32>
  %buf42_4 = AIE.buffer(%tile42_4) { sym_name = "prime1889" } : memref<4096xi32>
  %buf42_3 = AIE.buffer(%tile42_3) { sym_name = "prime1901" } : memref<4096xi32>
  %buf42_2 = AIE.buffer(%tile42_2) { sym_name = "prime1907" } : memref<4096xi32>
  %buf42_1 = AIE.buffer(%tile42_1) { sym_name = "prime1913" } : memref<4096xi32>
  %buf43_1 = AIE.buffer(%tile43_1) { sym_name = "prime1931" } : memref<4096xi32>
  %buf43_2 = AIE.buffer(%tile43_2) { sym_name = "prime1933" } : memref<4096xi32>
  %buf43_3 = AIE.buffer(%tile43_3) { sym_name = "prime1949" } : memref<4096xi32>
  %buf43_4 = AIE.buffer(%tile43_4) { sym_name = "prime1951" } : memref<4096xi32>
  %buf43_5 = AIE.buffer(%tile43_5) { sym_name = "prime1973" } : memref<4096xi32>
  %buf43_6 = AIE.buffer(%tile43_6) { sym_name = "prime1979" } : memref<4096xi32>
  %buf43_7 = AIE.buffer(%tile43_7) { sym_name = "prime1987" } : memref<4096xi32>
  %buf44_7 = AIE.buffer(%tile44_7) { sym_name = "prime1993" } : memref<4096xi32>
  %buf44_6 = AIE.buffer(%tile44_6) { sym_name = "prime1997" } : memref<4096xi32>
  %buf44_5 = AIE.buffer(%tile44_5) { sym_name = "prime1999" } : memref<4096xi32>
  %buf44_4 = AIE.buffer(%tile44_4) { sym_name = "prime2003" } : memref<4096xi32>
  %buf44_3 = AIE.buffer(%tile44_3) { sym_name = "prime2011" } : memref<4096xi32>
  %buf44_2 = AIE.buffer(%tile44_2) { sym_name = "prime2017" } : memref<4096xi32>
  %buf44_1 = AIE.buffer(%tile44_1) { sym_name = "prime2027" } : memref<4096xi32>
  %buf45_1 = AIE.buffer(%tile45_1) { sym_name = "prime2029" } : memref<4096xi32>
  %buf45_2 = AIE.buffer(%tile45_2) { sym_name = "prime2039" } : memref<4096xi32>
  %buf45_3 = AIE.buffer(%tile45_3) { sym_name = "prime2053" } : memref<4096xi32>
  %buf45_4 = AIE.buffer(%tile45_4) { sym_name = "prime2063" } : memref<4096xi32>
  %buf45_5 = AIE.buffer(%tile45_5) { sym_name = "prime2069" } : memref<4096xi32>
  %buf45_6 = AIE.buffer(%tile45_6) { sym_name = "prime2081" } : memref<4096xi32>
  %buf45_7 = AIE.buffer(%tile45_7) { sym_name = "prime2083" } : memref<4096xi32>
  %buf46_7 = AIE.buffer(%tile46_7) { sym_name = "prime2087" } : memref<4096xi32>
  %buf46_6 = AIE.buffer(%tile46_6) { sym_name = "prime2089" } : memref<4096xi32>
  %buf46_5 = AIE.buffer(%tile46_5) { sym_name = "prime2099" } : memref<4096xi32>
  %buf46_4 = AIE.buffer(%tile46_4) { sym_name = "prime2111" } : memref<4096xi32>
  %buf46_3 = AIE.buffer(%tile46_3) { sym_name = "prime2113" } : memref<4096xi32>
  %buf46_2 = AIE.buffer(%tile46_2) { sym_name = "prime2129" } : memref<4096xi32>
  %buf46_1 = AIE.buffer(%tile46_1) { sym_name = "prime2131" } : memref<4096xi32>
  %buf47_1 = AIE.buffer(%tile47_1) { sym_name = "prime2137" } : memref<4096xi32>
  %buf47_2 = AIE.buffer(%tile47_2) { sym_name = "prime2141" } : memref<4096xi32>
  %buf47_3 = AIE.buffer(%tile47_3) { sym_name = "prime2143" } : memref<4096xi32>
  %buf47_4 = AIE.buffer(%tile47_4) { sym_name = "prime2153" } : memref<4096xi32>
  %buf47_5 = AIE.buffer(%tile47_5) { sym_name = "prime2161" } : memref<4096xi32>
  %buf47_6 = AIE.buffer(%tile47_6) { sym_name = "prime2179" } : memref<4096xi32>
  %buf47_7 = AIE.buffer(%tile47_7) { sym_name = "prime2203" } : memref<4096xi32>
  %buf48_7 = AIE.buffer(%tile48_7) { sym_name = "prime2207" } : memref<4096xi32>
  %buf48_6 = AIE.buffer(%tile48_6) { sym_name = "prime2213" } : memref<4096xi32>
  %buf48_5 = AIE.buffer(%tile48_5) { sym_name = "prime2221" } : memref<4096xi32>
  %buf48_4 = AIE.buffer(%tile48_4) { sym_name = "prime2237" } : memref<4096xi32>
  %buf48_3 = AIE.buffer(%tile48_3) { sym_name = "prime2239" } : memref<4096xi32>
  %buf48_2 = AIE.buffer(%tile48_2) { sym_name = "prime2243" } : memref<4096xi32>
  %buf48_1 = AIE.buffer(%tile48_1) { sym_name = "prime_output" } : memref<4096xi32>
  
  %core1_1 = AIE.core(%tile1_1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant 4096: index
    %sum_0 = arith.constant 2 : i32
    %t = arith.constant 1 : i32
  
    // output integers starting with 2...
    scf.for %arg0 = %c0 to %cend step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = arith.addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf1_1[%arg0] : memref<4096xi32>
      scf.yield %sum_next : i32
    }
    AIE.useLock(%lock1_1, "Release", 1)
    AIE.end
  }
  func.func @do_sieve(%bufin: memref<4096xi32>, %bufout:memref<4096xi32>) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant 4096 : index
    %count_0 = arith.constant 0 : i32
  
    // The first number we receive is prime
    %prime = memref.load %bufin[%c0] : memref<4096xi32>
  
    // Step through the remaining inputs and sieve out multiples of %prime
    scf.for %arg0 = %c1 to %cend step %c1
      iter_args(%count_iter = %prime, %in_iter = %c1, %out_iter = %c0) -> (i32, index, index) {
      // Get the next input value
      %in_val = memref.load %bufin[%in_iter] : memref<4096xi32>
  
      // Potential next counters
      %count_inc = arith.addi %count_iter, %prime: i32
      %in_inc = arith.addi %in_iter, %c1 : index
      %out_inc = arith.addi %out_iter, %c1 : index
  
      // Compare the input value with the counter
      %b = arith.cmpi "slt", %in_val, %count_iter : i32
      %count_next, %in_next, %out_next = scf.if %b -> (i32, index, index) {
        // Input is less than counter.
        // Pass along the input and continue to the next one.
        memref.store %in_val, %bufout[%out_iter] : memref<4096xi32>
        scf.yield %count_iter, %in_inc, %out_inc : i32, index, index
      } else {
        %b2 = arith.cmpi "eq", %in_val, %count_iter : i32
        %in_next = scf.if %b2 -> (index) {
          // Input is equal to the counter.
          // Increment the counter and continue to the next input.
          scf.yield %in_inc : index
        } else {
          // Input is greater than the counter.
          // Increment the counter and check again.
          scf.yield %in_iter : index
        }
        scf.yield %count_inc, %in_next, %out_iter : i32, index, index
      }
      scf.yield %count_next, %in_next, %out_next : i32, index, index
    }
    return
  }
  
  %core1_2 = AIE.core(%tile1_2) {
    AIE.useLock(%lock1_1, "Acquire", 1)
    AIE.useLock(%lock1_2, "Acquire", 0)
    func.call @do_sieve(%buf1_1, %buf1_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_1, "Release", 0)
    AIE.useLock(%lock1_2, "Release", 1)
    AIE.end
  }

  %core1_3 = AIE.core(%tile1_3) {
    AIE.useLock(%lock1_2, "Acquire", 1)
    AIE.useLock(%lock1_3, "Acquire", 0)
    func.call @do_sieve(%buf1_2, %buf1_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_2, "Release", 0)
    AIE.useLock(%lock1_3, "Release", 1)
    AIE.end
  }

  %core1_4 = AIE.core(%tile1_4) {
    AIE.useLock(%lock1_3, "Acquire", 1)
    AIE.useLock(%lock1_4, "Acquire", 0)
    func.call @do_sieve(%buf1_3, %buf1_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_3, "Release", 0)
    AIE.useLock(%lock1_4, "Release", 1)
    AIE.end
  }

  %core1_5 = AIE.core(%tile1_5) {
    AIE.useLock(%lock1_4, "Acquire", 1)
    AIE.useLock(%lock1_5, "Acquire", 0)
    func.call @do_sieve(%buf1_4, %buf1_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_4, "Release", 0)
    AIE.useLock(%lock1_5, "Release", 1)
    AIE.end
  }

  %core1_6 = AIE.core(%tile1_6) {
    AIE.useLock(%lock1_5, "Acquire", 1)
    AIE.useLock(%lock1_6, "Acquire", 0)
    func.call @do_sieve(%buf1_5, %buf1_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_5, "Release", 0)
    AIE.useLock(%lock1_6, "Release", 1)
    AIE.end
  }

  %core1_7 = AIE.core(%tile1_7) {
    AIE.useLock(%lock1_6, "Acquire", 1)
    AIE.useLock(%lock1_7, "Acquire", 0)
    func.call @do_sieve(%buf1_6, %buf1_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_6, "Release", 0)
    AIE.useLock(%lock1_7, "Release", 1)
    AIE.end
  }

  %core2_7 = AIE.core(%tile2_7) {
    AIE.useLock(%lock1_7, "Acquire", 1)
    AIE.useLock(%lock2_7, "Acquire", 0)
    func.call @do_sieve(%buf1_7, %buf2_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock1_7, "Release", 0)
    AIE.useLock(%lock2_7, "Release", 1)
    AIE.end
  }

  %core2_6 = AIE.core(%tile2_6) {
    AIE.useLock(%lock2_7, "Acquire", 1)
    AIE.useLock(%lock2_6, "Acquire", 0)
    func.call @do_sieve(%buf2_7, %buf2_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_7, "Release", 0)
    AIE.useLock(%lock2_6, "Release", 1)
    AIE.end
  }

  %core2_5 = AIE.core(%tile2_5) {
    AIE.useLock(%lock2_6, "Acquire", 1)
    AIE.useLock(%lock2_5, "Acquire", 0)
    func.call @do_sieve(%buf2_6, %buf2_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_6, "Release", 0)
    AIE.useLock(%lock2_5, "Release", 1)
    AIE.end
  }

  %core2_4 = AIE.core(%tile2_4) {
    AIE.useLock(%lock2_5, "Acquire", 1)
    AIE.useLock(%lock2_4, "Acquire", 0)
    func.call @do_sieve(%buf2_5, %buf2_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_5, "Release", 0)
    AIE.useLock(%lock2_4, "Release", 1)
    AIE.end
  }

  %core2_3 = AIE.core(%tile2_3) {
    AIE.useLock(%lock2_4, "Acquire", 1)
    AIE.useLock(%lock2_3, "Acquire", 0)
    func.call @do_sieve(%buf2_4, %buf2_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_4, "Release", 0)
    AIE.useLock(%lock2_3, "Release", 1)
    AIE.end
  }

  %core2_2 = AIE.core(%tile2_2) {
    AIE.useLock(%lock2_3, "Acquire", 1)
    AIE.useLock(%lock2_2, "Acquire", 0)
    func.call @do_sieve(%buf2_3, %buf2_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_3, "Release", 0)
    AIE.useLock(%lock2_2, "Release", 1)
    AIE.end
  }

  %core2_1 = AIE.core(%tile2_1) {
    AIE.useLock(%lock2_2, "Acquire", 1)
    AIE.useLock(%lock2_1, "Acquire", 0)
    func.call @do_sieve(%buf2_2, %buf2_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_2, "Release", 0)
    AIE.useLock(%lock2_1, "Release", 1)
    AIE.end
  }

  %core3_1 = AIE.core(%tile3_1) {
    AIE.useLock(%lock2_1, "Acquire", 1)
    AIE.useLock(%lock3_1, "Acquire", 0)
    func.call @do_sieve(%buf2_1, %buf3_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock2_1, "Release", 0)
    AIE.useLock(%lock3_1, "Release", 1)
    AIE.end
  }

  %core3_2 = AIE.core(%tile3_2) {
    AIE.useLock(%lock3_1, "Acquire", 1)
    AIE.useLock(%lock3_2, "Acquire", 0)
    func.call @do_sieve(%buf3_1, %buf3_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_1, "Release", 0)
    AIE.useLock(%lock3_2, "Release", 1)
    AIE.end
  }

  %core3_3 = AIE.core(%tile3_3) {
    AIE.useLock(%lock3_2, "Acquire", 1)
    AIE.useLock(%lock3_3, "Acquire", 0)
    func.call @do_sieve(%buf3_2, %buf3_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_2, "Release", 0)
    AIE.useLock(%lock3_3, "Release", 1)
    AIE.end
  }

  %core3_4 = AIE.core(%tile3_4) {
    AIE.useLock(%lock3_3, "Acquire", 1)
    AIE.useLock(%lock3_4, "Acquire", 0)
    func.call @do_sieve(%buf3_3, %buf3_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_3, "Release", 0)
    AIE.useLock(%lock3_4, "Release", 1)
    AIE.end
  }

  %core3_5 = AIE.core(%tile3_5) {
    AIE.useLock(%lock3_4, "Acquire", 1)
    AIE.useLock(%lock3_5, "Acquire", 0)
    func.call @do_sieve(%buf3_4, %buf3_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_4, "Release", 0)
    AIE.useLock(%lock3_5, "Release", 1)
    AIE.end
  }

  %core3_6 = AIE.core(%tile3_6) {
    AIE.useLock(%lock3_5, "Acquire", 1)
    AIE.useLock(%lock3_6, "Acquire", 0)
    func.call @do_sieve(%buf3_5, %buf3_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_5, "Release", 0)
    AIE.useLock(%lock3_6, "Release", 1)
    AIE.end
  }

  %core3_7 = AIE.core(%tile3_7) {
    AIE.useLock(%lock3_6, "Acquire", 1)
    AIE.useLock(%lock3_7, "Acquire", 0)
    func.call @do_sieve(%buf3_6, %buf3_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_6, "Release", 0)
    AIE.useLock(%lock3_7, "Release", 1)
    AIE.end
  }

  %core4_7 = AIE.core(%tile4_7) {
    AIE.useLock(%lock3_7, "Acquire", 1)
    AIE.useLock(%lock4_7, "Acquire", 0)
    func.call @do_sieve(%buf3_7, %buf4_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock3_7, "Release", 0)
    AIE.useLock(%lock4_7, "Release", 1)
    AIE.end
  }

  %core4_6 = AIE.core(%tile4_6) {
    AIE.useLock(%lock4_7, "Acquire", 1)
    AIE.useLock(%lock4_6, "Acquire", 0)
    func.call @do_sieve(%buf4_7, %buf4_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_7, "Release", 0)
    AIE.useLock(%lock4_6, "Release", 1)
    AIE.end
  }

  %core4_5 = AIE.core(%tile4_5) {
    AIE.useLock(%lock4_6, "Acquire", 1)
    AIE.useLock(%lock4_5, "Acquire", 0)
    func.call @do_sieve(%buf4_6, %buf4_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_6, "Release", 0)
    AIE.useLock(%lock4_5, "Release", 1)
    AIE.end
  }

  %core4_4 = AIE.core(%tile4_4) {
    AIE.useLock(%lock4_5, "Acquire", 1)
    AIE.useLock(%lock4_4, "Acquire", 0)
    func.call @do_sieve(%buf4_5, %buf4_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_5, "Release", 0)
    AIE.useLock(%lock4_4, "Release", 1)
    AIE.end
  }

  %core4_3 = AIE.core(%tile4_3) {
    AIE.useLock(%lock4_4, "Acquire", 1)
    AIE.useLock(%lock4_3, "Acquire", 0)
    func.call @do_sieve(%buf4_4, %buf4_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_4, "Release", 0)
    AIE.useLock(%lock4_3, "Release", 1)
    AIE.end
  }

  %core4_2 = AIE.core(%tile4_2) {
    AIE.useLock(%lock4_3, "Acquire", 1)
    AIE.useLock(%lock4_2, "Acquire", 0)
    func.call @do_sieve(%buf4_3, %buf4_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_3, "Release", 0)
    AIE.useLock(%lock4_2, "Release", 1)
    AIE.end
  }

  %core4_1 = AIE.core(%tile4_1) {
    AIE.useLock(%lock4_2, "Acquire", 1)
    AIE.useLock(%lock4_1, "Acquire", 0)
    func.call @do_sieve(%buf4_2, %buf4_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_2, "Release", 0)
    AIE.useLock(%lock4_1, "Release", 1)
    AIE.end
  }

  %core5_1 = AIE.core(%tile5_1) {
    AIE.useLock(%lock4_1, "Acquire", 1)
    AIE.useLock(%lock5_1, "Acquire", 0)
    func.call @do_sieve(%buf4_1, %buf5_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock4_1, "Release", 0)
    AIE.useLock(%lock5_1, "Release", 1)
    AIE.end
  }

  %core5_2 = AIE.core(%tile5_2) {
    AIE.useLock(%lock5_1, "Acquire", 1)
    AIE.useLock(%lock5_2, "Acquire", 0)
    func.call @do_sieve(%buf5_1, %buf5_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_1, "Release", 0)
    AIE.useLock(%lock5_2, "Release", 1)
    AIE.end
  }

  %core5_3 = AIE.core(%tile5_3) {
    AIE.useLock(%lock5_2, "Acquire", 1)
    AIE.useLock(%lock5_3, "Acquire", 0)
    func.call @do_sieve(%buf5_2, %buf5_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_2, "Release", 0)
    AIE.useLock(%lock5_3, "Release", 1)
    AIE.end
  }

  %core5_4 = AIE.core(%tile5_4) {
    AIE.useLock(%lock5_3, "Acquire", 1)
    AIE.useLock(%lock5_4, "Acquire", 0)
    func.call @do_sieve(%buf5_3, %buf5_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_3, "Release", 0)
    AIE.useLock(%lock5_4, "Release", 1)
    AIE.end
  }

  %core5_5 = AIE.core(%tile5_5) {
    AIE.useLock(%lock5_4, "Acquire", 1)
    AIE.useLock(%lock5_5, "Acquire", 0)
    func.call @do_sieve(%buf5_4, %buf5_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_4, "Release", 0)
    AIE.useLock(%lock5_5, "Release", 1)
    AIE.end
  }

  %core5_6 = AIE.core(%tile5_6) {
    AIE.useLock(%lock5_5, "Acquire", 1)
    AIE.useLock(%lock5_6, "Acquire", 0)
    func.call @do_sieve(%buf5_5, %buf5_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_5, "Release", 0)
    AIE.useLock(%lock5_6, "Release", 1)
    AIE.end
  }

  %core5_7 = AIE.core(%tile5_7) {
    AIE.useLock(%lock5_6, "Acquire", 1)
    AIE.useLock(%lock5_7, "Acquire", 0)
    func.call @do_sieve(%buf5_6, %buf5_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_6, "Release", 0)
    AIE.useLock(%lock5_7, "Release", 1)
    AIE.end
  }

  %core6_7 = AIE.core(%tile6_7) {
    AIE.useLock(%lock5_7, "Acquire", 1)
    AIE.useLock(%lock6_7, "Acquire", 0)
    func.call @do_sieve(%buf5_7, %buf6_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock5_7, "Release", 0)
    AIE.useLock(%lock6_7, "Release", 1)
    AIE.end
  }

  %core6_6 = AIE.core(%tile6_6) {
    AIE.useLock(%lock6_7, "Acquire", 1)
    AIE.useLock(%lock6_6, "Acquire", 0)
    func.call @do_sieve(%buf6_7, %buf6_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_7, "Release", 0)
    AIE.useLock(%lock6_6, "Release", 1)
    AIE.end
  }

  %core6_5 = AIE.core(%tile6_5) {
    AIE.useLock(%lock6_6, "Acquire", 1)
    AIE.useLock(%lock6_5, "Acquire", 0)
    func.call @do_sieve(%buf6_6, %buf6_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_6, "Release", 0)
    AIE.useLock(%lock6_5, "Release", 1)
    AIE.end
  }

  %core6_4 = AIE.core(%tile6_4) {
    AIE.useLock(%lock6_5, "Acquire", 1)
    AIE.useLock(%lock6_4, "Acquire", 0)
    func.call @do_sieve(%buf6_5, %buf6_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_5, "Release", 0)
    AIE.useLock(%lock6_4, "Release", 1)
    AIE.end
  }

  %core6_3 = AIE.core(%tile6_3) {
    AIE.useLock(%lock6_4, "Acquire", 1)
    AIE.useLock(%lock6_3, "Acquire", 0)
    func.call @do_sieve(%buf6_4, %buf6_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_4, "Release", 0)
    AIE.useLock(%lock6_3, "Release", 1)
    AIE.end
  }

  %core6_2 = AIE.core(%tile6_2) {
    AIE.useLock(%lock6_3, "Acquire", 1)
    AIE.useLock(%lock6_2, "Acquire", 0)
    func.call @do_sieve(%buf6_3, %buf6_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_3, "Release", 0)
    AIE.useLock(%lock6_2, "Release", 1)
    AIE.end
  }

  %core6_1 = AIE.core(%tile6_1) {
    AIE.useLock(%lock6_2, "Acquire", 1)
    AIE.useLock(%lock6_1, "Acquire", 0)
    func.call @do_sieve(%buf6_2, %buf6_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_2, "Release", 0)
    AIE.useLock(%lock6_1, "Release", 1)
    AIE.end
  }

  %core7_1 = AIE.core(%tile7_1) {
    AIE.useLock(%lock6_1, "Acquire", 1)
    AIE.useLock(%lock7_1, "Acquire", 0)
    func.call @do_sieve(%buf6_1, %buf7_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock6_1, "Release", 0)
    AIE.useLock(%lock7_1, "Release", 1)
    AIE.end
  }

  %core7_2 = AIE.core(%tile7_2) {
    AIE.useLock(%lock7_1, "Acquire", 1)
    AIE.useLock(%lock7_2, "Acquire", 0)
    func.call @do_sieve(%buf7_1, %buf7_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_1, "Release", 0)
    AIE.useLock(%lock7_2, "Release", 1)
    AIE.end
  }

  %core7_3 = AIE.core(%tile7_3) {
    AIE.useLock(%lock7_2, "Acquire", 1)
    AIE.useLock(%lock7_3, "Acquire", 0)
    func.call @do_sieve(%buf7_2, %buf7_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_2, "Release", 0)
    AIE.useLock(%lock7_3, "Release", 1)
    AIE.end
  }

  %core7_4 = AIE.core(%tile7_4) {
    AIE.useLock(%lock7_3, "Acquire", 1)
    AIE.useLock(%lock7_4, "Acquire", 0)
    func.call @do_sieve(%buf7_3, %buf7_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_3, "Release", 0)
    AIE.useLock(%lock7_4, "Release", 1)
    AIE.end
  }

  %core7_5 = AIE.core(%tile7_5) {
    AIE.useLock(%lock7_4, "Acquire", 1)
    AIE.useLock(%lock7_5, "Acquire", 0)
    func.call @do_sieve(%buf7_4, %buf7_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_4, "Release", 0)
    AIE.useLock(%lock7_5, "Release", 1)
    AIE.end
  }

  %core7_6 = AIE.core(%tile7_6) {
    AIE.useLock(%lock7_5, "Acquire", 1)
    AIE.useLock(%lock7_6, "Acquire", 0)
    func.call @do_sieve(%buf7_5, %buf7_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_5, "Release", 0)
    AIE.useLock(%lock7_6, "Release", 1)
    AIE.end
  }

  %core7_7 = AIE.core(%tile7_7) {
    AIE.useLock(%lock7_6, "Acquire", 1)
    AIE.useLock(%lock7_7, "Acquire", 0)
    func.call @do_sieve(%buf7_6, %buf7_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_6, "Release", 0)
    AIE.useLock(%lock7_7, "Release", 1)
    AIE.end
  }

  %core8_7 = AIE.core(%tile8_7) {
    AIE.useLock(%lock7_7, "Acquire", 1)
    AIE.useLock(%lock8_7, "Acquire", 0)
    func.call @do_sieve(%buf7_7, %buf8_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock7_7, "Release", 0)
    AIE.useLock(%lock8_7, "Release", 1)
    AIE.end
  }

  %core8_6 = AIE.core(%tile8_6) {
    AIE.useLock(%lock8_7, "Acquire", 1)
    AIE.useLock(%lock8_6, "Acquire", 0)
    func.call @do_sieve(%buf8_7, %buf8_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_7, "Release", 0)
    AIE.useLock(%lock8_6, "Release", 1)
    AIE.end
  }

  %core8_5 = AIE.core(%tile8_5) {
    AIE.useLock(%lock8_6, "Acquire", 1)
    AIE.useLock(%lock8_5, "Acquire", 0)
    func.call @do_sieve(%buf8_6, %buf8_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_6, "Release", 0)
    AIE.useLock(%lock8_5, "Release", 1)
    AIE.end
  }

  %core8_4 = AIE.core(%tile8_4) {
    AIE.useLock(%lock8_5, "Acquire", 1)
    AIE.useLock(%lock8_4, "Acquire", 0)
    func.call @do_sieve(%buf8_5, %buf8_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_5, "Release", 0)
    AIE.useLock(%lock8_4, "Release", 1)
    AIE.end
  }

  %core8_3 = AIE.core(%tile8_3) {
    AIE.useLock(%lock8_4, "Acquire", 1)
    AIE.useLock(%lock8_3, "Acquire", 0)
    func.call @do_sieve(%buf8_4, %buf8_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_4, "Release", 0)
    AIE.useLock(%lock8_3, "Release", 1)
    AIE.end
  }

  %core8_2 = AIE.core(%tile8_2) {
    AIE.useLock(%lock8_3, "Acquire", 1)
    AIE.useLock(%lock8_2, "Acquire", 0)
    func.call @do_sieve(%buf8_3, %buf8_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_3, "Release", 0)
    AIE.useLock(%lock8_2, "Release", 1)
    AIE.end
  }

  %core8_1 = AIE.core(%tile8_1) {
    AIE.useLock(%lock8_2, "Acquire", 1)
    AIE.useLock(%lock8_1, "Acquire", 0)
    func.call @do_sieve(%buf8_2, %buf8_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_2, "Release", 0)
    AIE.useLock(%lock8_1, "Release", 1)
    AIE.end
  }

  %core9_1 = AIE.core(%tile9_1) {
    AIE.useLock(%lock8_1, "Acquire", 1)
    AIE.useLock(%lock9_1, "Acquire", 0)
    func.call @do_sieve(%buf8_1, %buf9_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock8_1, "Release", 0)
    AIE.useLock(%lock9_1, "Release", 1)
    AIE.end
  }

  %core9_2 = AIE.core(%tile9_2) {
    AIE.useLock(%lock9_1, "Acquire", 1)
    AIE.useLock(%lock9_2, "Acquire", 0)
    func.call @do_sieve(%buf9_1, %buf9_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_1, "Release", 0)
    AIE.useLock(%lock9_2, "Release", 1)
    AIE.end
  }

  %core9_3 = AIE.core(%tile9_3) {
    AIE.useLock(%lock9_2, "Acquire", 1)
    AIE.useLock(%lock9_3, "Acquire", 0)
    func.call @do_sieve(%buf9_2, %buf9_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_2, "Release", 0)
    AIE.useLock(%lock9_3, "Release", 1)
    AIE.end
  }

  %core9_4 = AIE.core(%tile9_4) {
    AIE.useLock(%lock9_3, "Acquire", 1)
    AIE.useLock(%lock9_4, "Acquire", 0)
    func.call @do_sieve(%buf9_3, %buf9_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_3, "Release", 0)
    AIE.useLock(%lock9_4, "Release", 1)
    AIE.end
  }

  %core9_5 = AIE.core(%tile9_5) {
    AIE.useLock(%lock9_4, "Acquire", 1)
    AIE.useLock(%lock9_5, "Acquire", 0)
    func.call @do_sieve(%buf9_4, %buf9_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_4, "Release", 0)
    AIE.useLock(%lock9_5, "Release", 1)
    AIE.end
  }

  %core9_6 = AIE.core(%tile9_6) {
    AIE.useLock(%lock9_5, "Acquire", 1)
    AIE.useLock(%lock9_6, "Acquire", 0)
    func.call @do_sieve(%buf9_5, %buf9_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_5, "Release", 0)
    AIE.useLock(%lock9_6, "Release", 1)
    AIE.end
  }

  %core9_7 = AIE.core(%tile9_7) {
    AIE.useLock(%lock9_6, "Acquire", 1)
    AIE.useLock(%lock9_7, "Acquire", 0)
    func.call @do_sieve(%buf9_6, %buf9_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_6, "Release", 0)
    AIE.useLock(%lock9_7, "Release", 1)
    AIE.end
  }

  %core10_7 = AIE.core(%tile10_7) {
    AIE.useLock(%lock9_7, "Acquire", 1)
    AIE.useLock(%lock10_7, "Acquire", 0)
    func.call @do_sieve(%buf9_7, %buf10_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock9_7, "Release", 0)
    AIE.useLock(%lock10_7, "Release", 1)
    AIE.end
  }

  %core10_6 = AIE.core(%tile10_6) {
    AIE.useLock(%lock10_7, "Acquire", 1)
    AIE.useLock(%lock10_6, "Acquire", 0)
    func.call @do_sieve(%buf10_7, %buf10_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_7, "Release", 0)
    AIE.useLock(%lock10_6, "Release", 1)
    AIE.end
  }

  %core10_5 = AIE.core(%tile10_5) {
    AIE.useLock(%lock10_6, "Acquire", 1)
    AIE.useLock(%lock10_5, "Acquire", 0)
    func.call @do_sieve(%buf10_6, %buf10_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_6, "Release", 0)
    AIE.useLock(%lock10_5, "Release", 1)
    AIE.end
  }

  %core10_4 = AIE.core(%tile10_4) {
    AIE.useLock(%lock10_5, "Acquire", 1)
    AIE.useLock(%lock10_4, "Acquire", 0)
    func.call @do_sieve(%buf10_5, %buf10_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_5, "Release", 0)
    AIE.useLock(%lock10_4, "Release", 1)
    AIE.end
  }

  %core10_3 = AIE.core(%tile10_3) {
    AIE.useLock(%lock10_4, "Acquire", 1)
    AIE.useLock(%lock10_3, "Acquire", 0)
    func.call @do_sieve(%buf10_4, %buf10_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_4, "Release", 0)
    AIE.useLock(%lock10_3, "Release", 1)
    AIE.end
  }

  %core10_2 = AIE.core(%tile10_2) {
    AIE.useLock(%lock10_3, "Acquire", 1)
    AIE.useLock(%lock10_2, "Acquire", 0)
    func.call @do_sieve(%buf10_3, %buf10_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_3, "Release", 0)
    AIE.useLock(%lock10_2, "Release", 1)
    AIE.end
  }

  %core10_1 = AIE.core(%tile10_1) {
    AIE.useLock(%lock10_2, "Acquire", 1)
    AIE.useLock(%lock10_1, "Acquire", 0)
    func.call @do_sieve(%buf10_2, %buf10_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_2, "Release", 0)
    AIE.useLock(%lock10_1, "Release", 1)
    AIE.end
  }

  %core11_1 = AIE.core(%tile11_1) {
    AIE.useLock(%lock10_1, "Acquire", 1)
    AIE.useLock(%lock11_1, "Acquire", 0)
    func.call @do_sieve(%buf10_1, %buf11_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock10_1, "Release", 0)
    AIE.useLock(%lock11_1, "Release", 1)
    AIE.end
  }

  %core11_2 = AIE.core(%tile11_2) {
    AIE.useLock(%lock11_1, "Acquire", 1)
    AIE.useLock(%lock11_2, "Acquire", 0)
    func.call @do_sieve(%buf11_1, %buf11_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_1, "Release", 0)
    AIE.useLock(%lock11_2, "Release", 1)
    AIE.end
  }

  %core11_3 = AIE.core(%tile11_3) {
    AIE.useLock(%lock11_2, "Acquire", 1)
    AIE.useLock(%lock11_3, "Acquire", 0)
    func.call @do_sieve(%buf11_2, %buf11_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_2, "Release", 0)
    AIE.useLock(%lock11_3, "Release", 1)
    AIE.end
  }

  %core11_4 = AIE.core(%tile11_4) {
    AIE.useLock(%lock11_3, "Acquire", 1)
    AIE.useLock(%lock11_4, "Acquire", 0)
    func.call @do_sieve(%buf11_3, %buf11_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_3, "Release", 0)
    AIE.useLock(%lock11_4, "Release", 1)
    AIE.end
  }

  %core11_5 = AIE.core(%tile11_5) {
    AIE.useLock(%lock11_4, "Acquire", 1)
    AIE.useLock(%lock11_5, "Acquire", 0)
    func.call @do_sieve(%buf11_4, %buf11_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_4, "Release", 0)
    AIE.useLock(%lock11_5, "Release", 1)
    AIE.end
  }

  %core11_6 = AIE.core(%tile11_6) {
    AIE.useLock(%lock11_5, "Acquire", 1)
    AIE.useLock(%lock11_6, "Acquire", 0)
    func.call @do_sieve(%buf11_5, %buf11_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_5, "Release", 0)
    AIE.useLock(%lock11_6, "Release", 1)
    AIE.end
  }

  %core11_7 = AIE.core(%tile11_7) {
    AIE.useLock(%lock11_6, "Acquire", 1)
    AIE.useLock(%lock11_7, "Acquire", 0)
    func.call @do_sieve(%buf11_6, %buf11_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_6, "Release", 0)
    AIE.useLock(%lock11_7, "Release", 1)
    AIE.end
  }

  %core12_7 = AIE.core(%tile12_7) {
    AIE.useLock(%lock11_7, "Acquire", 1)
    AIE.useLock(%lock12_7, "Acquire", 0)
    func.call @do_sieve(%buf11_7, %buf12_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock11_7, "Release", 0)
    AIE.useLock(%lock12_7, "Release", 1)
    AIE.end
  }

  %core12_6 = AIE.core(%tile12_6) {
    AIE.useLock(%lock12_7, "Acquire", 1)
    AIE.useLock(%lock12_6, "Acquire", 0)
    func.call @do_sieve(%buf12_7, %buf12_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_7, "Release", 0)
    AIE.useLock(%lock12_6, "Release", 1)
    AIE.end
  }

  %core12_5 = AIE.core(%tile12_5) {
    AIE.useLock(%lock12_6, "Acquire", 1)
    AIE.useLock(%lock12_5, "Acquire", 0)
    func.call @do_sieve(%buf12_6, %buf12_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_6, "Release", 0)
    AIE.useLock(%lock12_5, "Release", 1)
    AIE.end
  }

  %core12_4 = AIE.core(%tile12_4) {
    AIE.useLock(%lock12_5, "Acquire", 1)
    AIE.useLock(%lock12_4, "Acquire", 0)
    func.call @do_sieve(%buf12_5, %buf12_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_5, "Release", 0)
    AIE.useLock(%lock12_4, "Release", 1)
    AIE.end
  }

  %core12_3 = AIE.core(%tile12_3) {
    AIE.useLock(%lock12_4, "Acquire", 1)
    AIE.useLock(%lock12_3, "Acquire", 0)
    func.call @do_sieve(%buf12_4, %buf12_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_4, "Release", 0)
    AIE.useLock(%lock12_3, "Release", 1)
    AIE.end
  }

  %core12_2 = AIE.core(%tile12_2) {
    AIE.useLock(%lock12_3, "Acquire", 1)
    AIE.useLock(%lock12_2, "Acquire", 0)
    func.call @do_sieve(%buf12_3, %buf12_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_3, "Release", 0)
    AIE.useLock(%lock12_2, "Release", 1)
    AIE.end
  }

  %core12_1 = AIE.core(%tile12_1) {
    AIE.useLock(%lock12_2, "Acquire", 1)
    AIE.useLock(%lock12_1, "Acquire", 0)
    func.call @do_sieve(%buf12_2, %buf12_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_2, "Release", 0)
    AIE.useLock(%lock12_1, "Release", 1)
    AIE.end
  }

  %core13_1 = AIE.core(%tile13_1) {
    AIE.useLock(%lock12_1, "Acquire", 1)
    AIE.useLock(%lock13_1, "Acquire", 0)
    func.call @do_sieve(%buf12_1, %buf13_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock12_1, "Release", 0)
    AIE.useLock(%lock13_1, "Release", 1)
    AIE.end
  }

  %core13_2 = AIE.core(%tile13_2) {
    AIE.useLock(%lock13_1, "Acquire", 1)
    AIE.useLock(%lock13_2, "Acquire", 0)
    func.call @do_sieve(%buf13_1, %buf13_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_1, "Release", 0)
    AIE.useLock(%lock13_2, "Release", 1)
    AIE.end
  }

  %core13_3 = AIE.core(%tile13_3) {
    AIE.useLock(%lock13_2, "Acquire", 1)
    AIE.useLock(%lock13_3, "Acquire", 0)
    func.call @do_sieve(%buf13_2, %buf13_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_2, "Release", 0)
    AIE.useLock(%lock13_3, "Release", 1)
    AIE.end
  }

  %core13_4 = AIE.core(%tile13_4) {
    AIE.useLock(%lock13_3, "Acquire", 1)
    AIE.useLock(%lock13_4, "Acquire", 0)
    func.call @do_sieve(%buf13_3, %buf13_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_3, "Release", 0)
    AIE.useLock(%lock13_4, "Release", 1)
    AIE.end
  }

  %core13_5 = AIE.core(%tile13_5) {
    AIE.useLock(%lock13_4, "Acquire", 1)
    AIE.useLock(%lock13_5, "Acquire", 0)
    func.call @do_sieve(%buf13_4, %buf13_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_4, "Release", 0)
    AIE.useLock(%lock13_5, "Release", 1)
    AIE.end
  }

  %core13_6 = AIE.core(%tile13_6) {
    AIE.useLock(%lock13_5, "Acquire", 1)
    AIE.useLock(%lock13_6, "Acquire", 0)
    func.call @do_sieve(%buf13_5, %buf13_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_5, "Release", 0)
    AIE.useLock(%lock13_6, "Release", 1)
    AIE.end
  }

  %core13_7 = AIE.core(%tile13_7) {
    AIE.useLock(%lock13_6, "Acquire", 1)
    AIE.useLock(%lock13_7, "Acquire", 0)
    func.call @do_sieve(%buf13_6, %buf13_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_6, "Release", 0)
    AIE.useLock(%lock13_7, "Release", 1)
    AIE.end
  }

  %core14_7 = AIE.core(%tile14_7) {
    AIE.useLock(%lock13_7, "Acquire", 1)
    AIE.useLock(%lock14_7, "Acquire", 0)
    func.call @do_sieve(%buf13_7, %buf14_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock13_7, "Release", 0)
    AIE.useLock(%lock14_7, "Release", 1)
    AIE.end
  }

  %core14_6 = AIE.core(%tile14_6) {
    AIE.useLock(%lock14_7, "Acquire", 1)
    AIE.useLock(%lock14_6, "Acquire", 0)
    func.call @do_sieve(%buf14_7, %buf14_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_7, "Release", 0)
    AIE.useLock(%lock14_6, "Release", 1)
    AIE.end
  }

  %core14_5 = AIE.core(%tile14_5) {
    AIE.useLock(%lock14_6, "Acquire", 1)
    AIE.useLock(%lock14_5, "Acquire", 0)
    func.call @do_sieve(%buf14_6, %buf14_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_6, "Release", 0)
    AIE.useLock(%lock14_5, "Release", 1)
    AIE.end
  }

  %core14_4 = AIE.core(%tile14_4) {
    AIE.useLock(%lock14_5, "Acquire", 1)
    AIE.useLock(%lock14_4, "Acquire", 0)
    func.call @do_sieve(%buf14_5, %buf14_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_5, "Release", 0)
    AIE.useLock(%lock14_4, "Release", 1)
    AIE.end
  }

  %core14_3 = AIE.core(%tile14_3) {
    AIE.useLock(%lock14_4, "Acquire", 1)
    AIE.useLock(%lock14_3, "Acquire", 0)
    func.call @do_sieve(%buf14_4, %buf14_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_4, "Release", 0)
    AIE.useLock(%lock14_3, "Release", 1)
    AIE.end
  }

  %core14_2 = AIE.core(%tile14_2) {
    AIE.useLock(%lock14_3, "Acquire", 1)
    AIE.useLock(%lock14_2, "Acquire", 0)
    func.call @do_sieve(%buf14_3, %buf14_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_3, "Release", 0)
    AIE.useLock(%lock14_2, "Release", 1)
    AIE.end
  }

  %core14_1 = AIE.core(%tile14_1) {
    AIE.useLock(%lock14_2, "Acquire", 1)
    AIE.useLock(%lock14_1, "Acquire", 0)
    func.call @do_sieve(%buf14_2, %buf14_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_2, "Release", 0)
    AIE.useLock(%lock14_1, "Release", 1)
    AIE.end
  }

  %core15_1 = AIE.core(%tile15_1) {
    AIE.useLock(%lock14_1, "Acquire", 1)
    AIE.useLock(%lock15_1, "Acquire", 0)
    func.call @do_sieve(%buf14_1, %buf15_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock14_1, "Release", 0)
    AIE.useLock(%lock15_1, "Release", 1)
    AIE.end
  }

  %core15_2 = AIE.core(%tile15_2) {
    AIE.useLock(%lock15_1, "Acquire", 1)
    AIE.useLock(%lock15_2, "Acquire", 0)
    func.call @do_sieve(%buf15_1, %buf15_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_1, "Release", 0)
    AIE.useLock(%lock15_2, "Release", 1)
    AIE.end
  }

  %core15_3 = AIE.core(%tile15_3) {
    AIE.useLock(%lock15_2, "Acquire", 1)
    AIE.useLock(%lock15_3, "Acquire", 0)
    func.call @do_sieve(%buf15_2, %buf15_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_2, "Release", 0)
    AIE.useLock(%lock15_3, "Release", 1)
    AIE.end
  }

  %core15_4 = AIE.core(%tile15_4) {
    AIE.useLock(%lock15_3, "Acquire", 1)
    AIE.useLock(%lock15_4, "Acquire", 0)
    func.call @do_sieve(%buf15_3, %buf15_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_3, "Release", 0)
    AIE.useLock(%lock15_4, "Release", 1)
    AIE.end
  }

  %core15_5 = AIE.core(%tile15_5) {
    AIE.useLock(%lock15_4, "Acquire", 1)
    AIE.useLock(%lock15_5, "Acquire", 0)
    func.call @do_sieve(%buf15_4, %buf15_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_4, "Release", 0)
    AIE.useLock(%lock15_5, "Release", 1)
    AIE.end
  }

  %core15_6 = AIE.core(%tile15_6) {
    AIE.useLock(%lock15_5, "Acquire", 1)
    AIE.useLock(%lock15_6, "Acquire", 0)
    func.call @do_sieve(%buf15_5, %buf15_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_5, "Release", 0)
    AIE.useLock(%lock15_6, "Release", 1)
    AIE.end
  }

  %core15_7 = AIE.core(%tile15_7) {
    AIE.useLock(%lock15_6, "Acquire", 1)
    AIE.useLock(%lock15_7, "Acquire", 0)
    func.call @do_sieve(%buf15_6, %buf15_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_6, "Release", 0)
    AIE.useLock(%lock15_7, "Release", 1)
    AIE.end
  }

  %core16_7 = AIE.core(%tile16_7) {
    AIE.useLock(%lock15_7, "Acquire", 1)
    AIE.useLock(%lock16_7, "Acquire", 0)
    func.call @do_sieve(%buf15_7, %buf16_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock15_7, "Release", 0)
    AIE.useLock(%lock16_7, "Release", 1)
    AIE.end
  }

  %core16_6 = AIE.core(%tile16_6) {
    AIE.useLock(%lock16_7, "Acquire", 1)
    AIE.useLock(%lock16_6, "Acquire", 0)
    func.call @do_sieve(%buf16_7, %buf16_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_7, "Release", 0)
    AIE.useLock(%lock16_6, "Release", 1)
    AIE.end
  }

  %core16_5 = AIE.core(%tile16_5) {
    AIE.useLock(%lock16_6, "Acquire", 1)
    AIE.useLock(%lock16_5, "Acquire", 0)
    func.call @do_sieve(%buf16_6, %buf16_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_6, "Release", 0)
    AIE.useLock(%lock16_5, "Release", 1)
    AIE.end
  }

  %core16_4 = AIE.core(%tile16_4) {
    AIE.useLock(%lock16_5, "Acquire", 1)
    AIE.useLock(%lock16_4, "Acquire", 0)
    func.call @do_sieve(%buf16_5, %buf16_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_5, "Release", 0)
    AIE.useLock(%lock16_4, "Release", 1)
    AIE.end
  }

  %core16_3 = AIE.core(%tile16_3) {
    AIE.useLock(%lock16_4, "Acquire", 1)
    AIE.useLock(%lock16_3, "Acquire", 0)
    func.call @do_sieve(%buf16_4, %buf16_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_4, "Release", 0)
    AIE.useLock(%lock16_3, "Release", 1)
    AIE.end
  }

  %core16_2 = AIE.core(%tile16_2) {
    AIE.useLock(%lock16_3, "Acquire", 1)
    AIE.useLock(%lock16_2, "Acquire", 0)
    func.call @do_sieve(%buf16_3, %buf16_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_3, "Release", 0)
    AIE.useLock(%lock16_2, "Release", 1)
    AIE.end
  }

  %core16_1 = AIE.core(%tile16_1) {
    AIE.useLock(%lock16_2, "Acquire", 1)
    AIE.useLock(%lock16_1, "Acquire", 0)
    func.call @do_sieve(%buf16_2, %buf16_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_2, "Release", 0)
    AIE.useLock(%lock16_1, "Release", 1)
    AIE.end
  }

  %core17_1 = AIE.core(%tile17_1) {
    AIE.useLock(%lock16_1, "Acquire", 1)
    AIE.useLock(%lock17_1, "Acquire", 0)
    func.call @do_sieve(%buf16_1, %buf17_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock16_1, "Release", 0)
    AIE.useLock(%lock17_1, "Release", 1)
    AIE.end
  }

  %core17_2 = AIE.core(%tile17_2) {
    AIE.useLock(%lock17_1, "Acquire", 1)
    AIE.useLock(%lock17_2, "Acquire", 0)
    func.call @do_sieve(%buf17_1, %buf17_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_1, "Release", 0)
    AIE.useLock(%lock17_2, "Release", 1)
    AIE.end
  }

  %core17_3 = AIE.core(%tile17_3) {
    AIE.useLock(%lock17_2, "Acquire", 1)
    AIE.useLock(%lock17_3, "Acquire", 0)
    func.call @do_sieve(%buf17_2, %buf17_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_2, "Release", 0)
    AIE.useLock(%lock17_3, "Release", 1)
    AIE.end
  }

  %core17_4 = AIE.core(%tile17_4) {
    AIE.useLock(%lock17_3, "Acquire", 1)
    AIE.useLock(%lock17_4, "Acquire", 0)
    func.call @do_sieve(%buf17_3, %buf17_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_3, "Release", 0)
    AIE.useLock(%lock17_4, "Release", 1)
    AIE.end
  }

  %core17_5 = AIE.core(%tile17_5) {
    AIE.useLock(%lock17_4, "Acquire", 1)
    AIE.useLock(%lock17_5, "Acquire", 0)
    func.call @do_sieve(%buf17_4, %buf17_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_4, "Release", 0)
    AIE.useLock(%lock17_5, "Release", 1)
    AIE.end
  }

  %core17_6 = AIE.core(%tile17_6) {
    AIE.useLock(%lock17_5, "Acquire", 1)
    AIE.useLock(%lock17_6, "Acquire", 0)
    func.call @do_sieve(%buf17_5, %buf17_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_5, "Release", 0)
    AIE.useLock(%lock17_6, "Release", 1)
    AIE.end
  }

  %core17_7 = AIE.core(%tile17_7) {
    AIE.useLock(%lock17_6, "Acquire", 1)
    AIE.useLock(%lock17_7, "Acquire", 0)
    func.call @do_sieve(%buf17_6, %buf17_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_6, "Release", 0)
    AIE.useLock(%lock17_7, "Release", 1)
    AIE.end
  }

  %core18_7 = AIE.core(%tile18_7) {
    AIE.useLock(%lock17_7, "Acquire", 1)
    AIE.useLock(%lock18_7, "Acquire", 0)
    func.call @do_sieve(%buf17_7, %buf18_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock17_7, "Release", 0)
    AIE.useLock(%lock18_7, "Release", 1)
    AIE.end
  }

  %core18_6 = AIE.core(%tile18_6) {
    AIE.useLock(%lock18_7, "Acquire", 1)
    AIE.useLock(%lock18_6, "Acquire", 0)
    func.call @do_sieve(%buf18_7, %buf18_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_7, "Release", 0)
    AIE.useLock(%lock18_6, "Release", 1)
    AIE.end
  }

  %core18_5 = AIE.core(%tile18_5) {
    AIE.useLock(%lock18_6, "Acquire", 1)
    AIE.useLock(%lock18_5, "Acquire", 0)
    func.call @do_sieve(%buf18_6, %buf18_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_6, "Release", 0)
    AIE.useLock(%lock18_5, "Release", 1)
    AIE.end
  }

  %core18_4 = AIE.core(%tile18_4) {
    AIE.useLock(%lock18_5, "Acquire", 1)
    AIE.useLock(%lock18_4, "Acquire", 0)
    func.call @do_sieve(%buf18_5, %buf18_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_5, "Release", 0)
    AIE.useLock(%lock18_4, "Release", 1)
    AIE.end
  }

  %core18_3 = AIE.core(%tile18_3) {
    AIE.useLock(%lock18_4, "Acquire", 1)
    AIE.useLock(%lock18_3, "Acquire", 0)
    func.call @do_sieve(%buf18_4, %buf18_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_4, "Release", 0)
    AIE.useLock(%lock18_3, "Release", 1)
    AIE.end
  }

  %core18_2 = AIE.core(%tile18_2) {
    AIE.useLock(%lock18_3, "Acquire", 1)
    AIE.useLock(%lock18_2, "Acquire", 0)
    func.call @do_sieve(%buf18_3, %buf18_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_3, "Release", 0)
    AIE.useLock(%lock18_2, "Release", 1)
    AIE.end
  }

  %core18_1 = AIE.core(%tile18_1) {
    AIE.useLock(%lock18_2, "Acquire", 1)
    AIE.useLock(%lock18_1, "Acquire", 0)
    func.call @do_sieve(%buf18_2, %buf18_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_2, "Release", 0)
    AIE.useLock(%lock18_1, "Release", 1)
    AIE.end
  }

  %core19_1 = AIE.core(%tile19_1) {
    AIE.useLock(%lock18_1, "Acquire", 1)
    AIE.useLock(%lock19_1, "Acquire", 0)
    func.call @do_sieve(%buf18_1, %buf19_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock18_1, "Release", 0)
    AIE.useLock(%lock19_1, "Release", 1)
    AIE.end
  }

  %core19_2 = AIE.core(%tile19_2) {
    AIE.useLock(%lock19_1, "Acquire", 1)
    AIE.useLock(%lock19_2, "Acquire", 0)
    func.call @do_sieve(%buf19_1, %buf19_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_1, "Release", 0)
    AIE.useLock(%lock19_2, "Release", 1)
    AIE.end
  }

  %core19_3 = AIE.core(%tile19_3) {
    AIE.useLock(%lock19_2, "Acquire", 1)
    AIE.useLock(%lock19_3, "Acquire", 0)
    func.call @do_sieve(%buf19_2, %buf19_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_2, "Release", 0)
    AIE.useLock(%lock19_3, "Release", 1)
    AIE.end
  }

  %core19_4 = AIE.core(%tile19_4) {
    AIE.useLock(%lock19_3, "Acquire", 1)
    AIE.useLock(%lock19_4, "Acquire", 0)
    func.call @do_sieve(%buf19_3, %buf19_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_3, "Release", 0)
    AIE.useLock(%lock19_4, "Release", 1)
    AIE.end
  }

  %core19_5 = AIE.core(%tile19_5) {
    AIE.useLock(%lock19_4, "Acquire", 1)
    AIE.useLock(%lock19_5, "Acquire", 0)
    func.call @do_sieve(%buf19_4, %buf19_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_4, "Release", 0)
    AIE.useLock(%lock19_5, "Release", 1)
    AIE.end
  }

  %core19_6 = AIE.core(%tile19_6) {
    AIE.useLock(%lock19_5, "Acquire", 1)
    AIE.useLock(%lock19_6, "Acquire", 0)
    func.call @do_sieve(%buf19_5, %buf19_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_5, "Release", 0)
    AIE.useLock(%lock19_6, "Release", 1)
    AIE.end
  }

  %core19_7 = AIE.core(%tile19_7) {
    AIE.useLock(%lock19_6, "Acquire", 1)
    AIE.useLock(%lock19_7, "Acquire", 0)
    func.call @do_sieve(%buf19_6, %buf19_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_6, "Release", 0)
    AIE.useLock(%lock19_7, "Release", 1)
    AIE.end
  }

  %core20_7 = AIE.core(%tile20_7) {
    AIE.useLock(%lock19_7, "Acquire", 1)
    AIE.useLock(%lock20_7, "Acquire", 0)
    func.call @do_sieve(%buf19_7, %buf20_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock19_7, "Release", 0)
    AIE.useLock(%lock20_7, "Release", 1)
    AIE.end
  }

  %core20_6 = AIE.core(%tile20_6) {
    AIE.useLock(%lock20_7, "Acquire", 1)
    AIE.useLock(%lock20_6, "Acquire", 0)
    func.call @do_sieve(%buf20_7, %buf20_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_7, "Release", 0)
    AIE.useLock(%lock20_6, "Release", 1)
    AIE.end
  }

  %core20_5 = AIE.core(%tile20_5) {
    AIE.useLock(%lock20_6, "Acquire", 1)
    AIE.useLock(%lock20_5, "Acquire", 0)
    func.call @do_sieve(%buf20_6, %buf20_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_6, "Release", 0)
    AIE.useLock(%lock20_5, "Release", 1)
    AIE.end
  }

  %core20_4 = AIE.core(%tile20_4) {
    AIE.useLock(%lock20_5, "Acquire", 1)
    AIE.useLock(%lock20_4, "Acquire", 0)
    func.call @do_sieve(%buf20_5, %buf20_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_5, "Release", 0)
    AIE.useLock(%lock20_4, "Release", 1)
    AIE.end
  }

  %core20_3 = AIE.core(%tile20_3) {
    AIE.useLock(%lock20_4, "Acquire", 1)
    AIE.useLock(%lock20_3, "Acquire", 0)
    func.call @do_sieve(%buf20_4, %buf20_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_4, "Release", 0)
    AIE.useLock(%lock20_3, "Release", 1)
    AIE.end
  }

  %core20_2 = AIE.core(%tile20_2) {
    AIE.useLock(%lock20_3, "Acquire", 1)
    AIE.useLock(%lock20_2, "Acquire", 0)
    func.call @do_sieve(%buf20_3, %buf20_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_3, "Release", 0)
    AIE.useLock(%lock20_2, "Release", 1)
    AIE.end
  }

  %core20_1 = AIE.core(%tile20_1) {
    AIE.useLock(%lock20_2, "Acquire", 1)
    AIE.useLock(%lock20_1, "Acquire", 0)
    func.call @do_sieve(%buf20_2, %buf20_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_2, "Release", 0)
    AIE.useLock(%lock20_1, "Release", 1)
    AIE.end
  }

  %core21_1 = AIE.core(%tile21_1) {
    AIE.useLock(%lock20_1, "Acquire", 1)
    AIE.useLock(%lock21_1, "Acquire", 0)
    func.call @do_sieve(%buf20_1, %buf21_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock20_1, "Release", 0)
    AIE.useLock(%lock21_1, "Release", 1)
    AIE.end
  }

  %core21_2 = AIE.core(%tile21_2) {
    AIE.useLock(%lock21_1, "Acquire", 1)
    AIE.useLock(%lock21_2, "Acquire", 0)
    func.call @do_sieve(%buf21_1, %buf21_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_1, "Release", 0)
    AIE.useLock(%lock21_2, "Release", 1)
    AIE.end
  }

  %core21_3 = AIE.core(%tile21_3) {
    AIE.useLock(%lock21_2, "Acquire", 1)
    AIE.useLock(%lock21_3, "Acquire", 0)
    func.call @do_sieve(%buf21_2, %buf21_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_2, "Release", 0)
    AIE.useLock(%lock21_3, "Release", 1)
    AIE.end
  }

  %core21_4 = AIE.core(%tile21_4) {
    AIE.useLock(%lock21_3, "Acquire", 1)
    AIE.useLock(%lock21_4, "Acquire", 0)
    func.call @do_sieve(%buf21_3, %buf21_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_3, "Release", 0)
    AIE.useLock(%lock21_4, "Release", 1)
    AIE.end
  }

  %core21_5 = AIE.core(%tile21_5) {
    AIE.useLock(%lock21_4, "Acquire", 1)
    AIE.useLock(%lock21_5, "Acquire", 0)
    func.call @do_sieve(%buf21_4, %buf21_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_4, "Release", 0)
    AIE.useLock(%lock21_5, "Release", 1)
    AIE.end
  }

  %core21_6 = AIE.core(%tile21_6) {
    AIE.useLock(%lock21_5, "Acquire", 1)
    AIE.useLock(%lock21_6, "Acquire", 0)
    func.call @do_sieve(%buf21_5, %buf21_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_5, "Release", 0)
    AIE.useLock(%lock21_6, "Release", 1)
    AIE.end
  }

  %core21_7 = AIE.core(%tile21_7) {
    AIE.useLock(%lock21_6, "Acquire", 1)
    AIE.useLock(%lock21_7, "Acquire", 0)
    func.call @do_sieve(%buf21_6, %buf21_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_6, "Release", 0)
    AIE.useLock(%lock21_7, "Release", 1)
    AIE.end
  }

  %core22_7 = AIE.core(%tile22_7) {
    AIE.useLock(%lock21_7, "Acquire", 1)
    AIE.useLock(%lock22_7, "Acquire", 0)
    func.call @do_sieve(%buf21_7, %buf22_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock21_7, "Release", 0)
    AIE.useLock(%lock22_7, "Release", 1)
    AIE.end
  }

  %core22_6 = AIE.core(%tile22_6) {
    AIE.useLock(%lock22_7, "Acquire", 1)
    AIE.useLock(%lock22_6, "Acquire", 0)
    func.call @do_sieve(%buf22_7, %buf22_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_7, "Release", 0)
    AIE.useLock(%lock22_6, "Release", 1)
    AIE.end
  }

  %core22_5 = AIE.core(%tile22_5) {
    AIE.useLock(%lock22_6, "Acquire", 1)
    AIE.useLock(%lock22_5, "Acquire", 0)
    func.call @do_sieve(%buf22_6, %buf22_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_6, "Release", 0)
    AIE.useLock(%lock22_5, "Release", 1)
    AIE.end
  }

  %core22_4 = AIE.core(%tile22_4) {
    AIE.useLock(%lock22_5, "Acquire", 1)
    AIE.useLock(%lock22_4, "Acquire", 0)
    func.call @do_sieve(%buf22_5, %buf22_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_5, "Release", 0)
    AIE.useLock(%lock22_4, "Release", 1)
    AIE.end
  }

  %core22_3 = AIE.core(%tile22_3) {
    AIE.useLock(%lock22_4, "Acquire", 1)
    AIE.useLock(%lock22_3, "Acquire", 0)
    func.call @do_sieve(%buf22_4, %buf22_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_4, "Release", 0)
    AIE.useLock(%lock22_3, "Release", 1)
    AIE.end
  }

  %core22_2 = AIE.core(%tile22_2) {
    AIE.useLock(%lock22_3, "Acquire", 1)
    AIE.useLock(%lock22_2, "Acquire", 0)
    func.call @do_sieve(%buf22_3, %buf22_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_3, "Release", 0)
    AIE.useLock(%lock22_2, "Release", 1)
    AIE.end
  }

  %core22_1 = AIE.core(%tile22_1) {
    AIE.useLock(%lock22_2, "Acquire", 1)
    AIE.useLock(%lock22_1, "Acquire", 0)
    func.call @do_sieve(%buf22_2, %buf22_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_2, "Release", 0)
    AIE.useLock(%lock22_1, "Release", 1)
    AIE.end
  }

  %core23_1 = AIE.core(%tile23_1) {
    AIE.useLock(%lock22_1, "Acquire", 1)
    AIE.useLock(%lock23_1, "Acquire", 0)
    func.call @do_sieve(%buf22_1, %buf23_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock22_1, "Release", 0)
    AIE.useLock(%lock23_1, "Release", 1)
    AIE.end
  }

  %core23_2 = AIE.core(%tile23_2) {
    AIE.useLock(%lock23_1, "Acquire", 1)
    AIE.useLock(%lock23_2, "Acquire", 0)
    func.call @do_sieve(%buf23_1, %buf23_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_1, "Release", 0)
    AIE.useLock(%lock23_2, "Release", 1)
    AIE.end
  }

  %core23_3 = AIE.core(%tile23_3) {
    AIE.useLock(%lock23_2, "Acquire", 1)
    AIE.useLock(%lock23_3, "Acquire", 0)
    func.call @do_sieve(%buf23_2, %buf23_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_2, "Release", 0)
    AIE.useLock(%lock23_3, "Release", 1)
    AIE.end
  }

  %core23_4 = AIE.core(%tile23_4) {
    AIE.useLock(%lock23_3, "Acquire", 1)
    AIE.useLock(%lock23_4, "Acquire", 0)
    func.call @do_sieve(%buf23_3, %buf23_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_3, "Release", 0)
    AIE.useLock(%lock23_4, "Release", 1)
    AIE.end
  }

  %core23_5 = AIE.core(%tile23_5) {
    AIE.useLock(%lock23_4, "Acquire", 1)
    AIE.useLock(%lock23_5, "Acquire", 0)
    func.call @do_sieve(%buf23_4, %buf23_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_4, "Release", 0)
    AIE.useLock(%lock23_5, "Release", 1)
    AIE.end
  }

  %core23_6 = AIE.core(%tile23_6) {
    AIE.useLock(%lock23_5, "Acquire", 1)
    AIE.useLock(%lock23_6, "Acquire", 0)
    func.call @do_sieve(%buf23_5, %buf23_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_5, "Release", 0)
    AIE.useLock(%lock23_6, "Release", 1)
    AIE.end
  }

  %core23_7 = AIE.core(%tile23_7) {
    AIE.useLock(%lock23_6, "Acquire", 1)
    AIE.useLock(%lock23_7, "Acquire", 0)
    func.call @do_sieve(%buf23_6, %buf23_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_6, "Release", 0)
    AIE.useLock(%lock23_7, "Release", 1)
    AIE.end
  }

  %core24_7 = AIE.core(%tile24_7) {
    AIE.useLock(%lock23_7, "Acquire", 1)
    AIE.useLock(%lock24_7, "Acquire", 0)
    func.call @do_sieve(%buf23_7, %buf24_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock23_7, "Release", 0)
    AIE.useLock(%lock24_7, "Release", 1)
    AIE.end
  }

  %core24_6 = AIE.core(%tile24_6) {
    AIE.useLock(%lock24_7, "Acquire", 1)
    AIE.useLock(%lock24_6, "Acquire", 0)
    func.call @do_sieve(%buf24_7, %buf24_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_7, "Release", 0)
    AIE.useLock(%lock24_6, "Release", 1)
    AIE.end
  }

  %core24_5 = AIE.core(%tile24_5) {
    AIE.useLock(%lock24_6, "Acquire", 1)
    AIE.useLock(%lock24_5, "Acquire", 0)
    func.call @do_sieve(%buf24_6, %buf24_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_6, "Release", 0)
    AIE.useLock(%lock24_5, "Release", 1)
    AIE.end
  }

  %core24_4 = AIE.core(%tile24_4) {
    AIE.useLock(%lock24_5, "Acquire", 1)
    AIE.useLock(%lock24_4, "Acquire", 0)
    func.call @do_sieve(%buf24_5, %buf24_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_5, "Release", 0)
    AIE.useLock(%lock24_4, "Release", 1)
    AIE.end
  }

  %core24_3 = AIE.core(%tile24_3) {
    AIE.useLock(%lock24_4, "Acquire", 1)
    AIE.useLock(%lock24_3, "Acquire", 0)
    func.call @do_sieve(%buf24_4, %buf24_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_4, "Release", 0)
    AIE.useLock(%lock24_3, "Release", 1)
    AIE.end
  }

  %core24_2 = AIE.core(%tile24_2) {
    AIE.useLock(%lock24_3, "Acquire", 1)
    AIE.useLock(%lock24_2, "Acquire", 0)
    func.call @do_sieve(%buf24_3, %buf24_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_3, "Release", 0)
    AIE.useLock(%lock24_2, "Release", 1)
    AIE.end
  }

  %core24_1 = AIE.core(%tile24_1) {
    AIE.useLock(%lock24_2, "Acquire", 1)
    AIE.useLock(%lock24_1, "Acquire", 0)
    func.call @do_sieve(%buf24_2, %buf24_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_2, "Release", 0)
    AIE.useLock(%lock24_1, "Release", 1)
    AIE.end
  }

  %core25_1 = AIE.core(%tile25_1) {
    AIE.useLock(%lock24_1, "Acquire", 1)
    AIE.useLock(%lock25_1, "Acquire", 0)
    func.call @do_sieve(%buf24_1, %buf25_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock24_1, "Release", 0)
    AIE.useLock(%lock25_1, "Release", 1)
    AIE.end
  }

  %core25_2 = AIE.core(%tile25_2) {
    AIE.useLock(%lock25_1, "Acquire", 1)
    AIE.useLock(%lock25_2, "Acquire", 0)
    func.call @do_sieve(%buf25_1, %buf25_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_1, "Release", 0)
    AIE.useLock(%lock25_2, "Release", 1)
    AIE.end
  }

  %core25_3 = AIE.core(%tile25_3) {
    AIE.useLock(%lock25_2, "Acquire", 1)
    AIE.useLock(%lock25_3, "Acquire", 0)
    func.call @do_sieve(%buf25_2, %buf25_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_2, "Release", 0)
    AIE.useLock(%lock25_3, "Release", 1)
    AIE.end
  }

  %core25_4 = AIE.core(%tile25_4) {
    AIE.useLock(%lock25_3, "Acquire", 1)
    AIE.useLock(%lock25_4, "Acquire", 0)
    func.call @do_sieve(%buf25_3, %buf25_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_3, "Release", 0)
    AIE.useLock(%lock25_4, "Release", 1)
    AIE.end
  }

  %core25_5 = AIE.core(%tile25_5) {
    AIE.useLock(%lock25_4, "Acquire", 1)
    AIE.useLock(%lock25_5, "Acquire", 0)
    func.call @do_sieve(%buf25_4, %buf25_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_4, "Release", 0)
    AIE.useLock(%lock25_5, "Release", 1)
    AIE.end
  }

  %core25_6 = AIE.core(%tile25_6) {
    AIE.useLock(%lock25_5, "Acquire", 1)
    AIE.useLock(%lock25_6, "Acquire", 0)
    func.call @do_sieve(%buf25_5, %buf25_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_5, "Release", 0)
    AIE.useLock(%lock25_6, "Release", 1)
    AIE.end
  }

  %core25_7 = AIE.core(%tile25_7) {
    AIE.useLock(%lock25_6, "Acquire", 1)
    AIE.useLock(%lock25_7, "Acquire", 0)
    func.call @do_sieve(%buf25_6, %buf25_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_6, "Release", 0)
    AIE.useLock(%lock25_7, "Release", 1)
    AIE.end
  }

  %core26_7 = AIE.core(%tile26_7) {
    AIE.useLock(%lock25_7, "Acquire", 1)
    AIE.useLock(%lock26_7, "Acquire", 0)
    func.call @do_sieve(%buf25_7, %buf26_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock25_7, "Release", 0)
    AIE.useLock(%lock26_7, "Release", 1)
    AIE.end
  }

  %core26_6 = AIE.core(%tile26_6) {
    AIE.useLock(%lock26_7, "Acquire", 1)
    AIE.useLock(%lock26_6, "Acquire", 0)
    func.call @do_sieve(%buf26_7, %buf26_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_7, "Release", 0)
    AIE.useLock(%lock26_6, "Release", 1)
    AIE.end
  }

  %core26_5 = AIE.core(%tile26_5) {
    AIE.useLock(%lock26_6, "Acquire", 1)
    AIE.useLock(%lock26_5, "Acquire", 0)
    func.call @do_sieve(%buf26_6, %buf26_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_6, "Release", 0)
    AIE.useLock(%lock26_5, "Release", 1)
    AIE.end
  }

  %core26_4 = AIE.core(%tile26_4) {
    AIE.useLock(%lock26_5, "Acquire", 1)
    AIE.useLock(%lock26_4, "Acquire", 0)
    func.call @do_sieve(%buf26_5, %buf26_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_5, "Release", 0)
    AIE.useLock(%lock26_4, "Release", 1)
    AIE.end
  }

  %core26_3 = AIE.core(%tile26_3) {
    AIE.useLock(%lock26_4, "Acquire", 1)
    AIE.useLock(%lock26_3, "Acquire", 0)
    func.call @do_sieve(%buf26_4, %buf26_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_4, "Release", 0)
    AIE.useLock(%lock26_3, "Release", 1)
    AIE.end
  }

  %core26_2 = AIE.core(%tile26_2) {
    AIE.useLock(%lock26_3, "Acquire", 1)
    AIE.useLock(%lock26_2, "Acquire", 0)
    func.call @do_sieve(%buf26_3, %buf26_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_3, "Release", 0)
    AIE.useLock(%lock26_2, "Release", 1)
    AIE.end
  }

  %core26_1 = AIE.core(%tile26_1) {
    AIE.useLock(%lock26_2, "Acquire", 1)
    AIE.useLock(%lock26_1, "Acquire", 0)
    func.call @do_sieve(%buf26_2, %buf26_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_2, "Release", 0)
    AIE.useLock(%lock26_1, "Release", 1)
    AIE.end
  }

  %core27_1 = AIE.core(%tile27_1) {
    AIE.useLock(%lock26_1, "Acquire", 1)
    AIE.useLock(%lock27_1, "Acquire", 0)
    func.call @do_sieve(%buf26_1, %buf27_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock26_1, "Release", 0)
    AIE.useLock(%lock27_1, "Release", 1)
    AIE.end
  }

  %core27_2 = AIE.core(%tile27_2) {
    AIE.useLock(%lock27_1, "Acquire", 1)
    AIE.useLock(%lock27_2, "Acquire", 0)
    func.call @do_sieve(%buf27_1, %buf27_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_1, "Release", 0)
    AIE.useLock(%lock27_2, "Release", 1)
    AIE.end
  }

  %core27_3 = AIE.core(%tile27_3) {
    AIE.useLock(%lock27_2, "Acquire", 1)
    AIE.useLock(%lock27_3, "Acquire", 0)
    func.call @do_sieve(%buf27_2, %buf27_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_2, "Release", 0)
    AIE.useLock(%lock27_3, "Release", 1)
    AIE.end
  }

  %core27_4 = AIE.core(%tile27_4) {
    AIE.useLock(%lock27_3, "Acquire", 1)
    AIE.useLock(%lock27_4, "Acquire", 0)
    func.call @do_sieve(%buf27_3, %buf27_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_3, "Release", 0)
    AIE.useLock(%lock27_4, "Release", 1)
    AIE.end
  }

  %core27_5 = AIE.core(%tile27_5) {
    AIE.useLock(%lock27_4, "Acquire", 1)
    AIE.useLock(%lock27_5, "Acquire", 0)
    func.call @do_sieve(%buf27_4, %buf27_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_4, "Release", 0)
    AIE.useLock(%lock27_5, "Release", 1)
    AIE.end
  }

  %core27_6 = AIE.core(%tile27_6) {
    AIE.useLock(%lock27_5, "Acquire", 1)
    AIE.useLock(%lock27_6, "Acquire", 0)
    func.call @do_sieve(%buf27_5, %buf27_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_5, "Release", 0)
    AIE.useLock(%lock27_6, "Release", 1)
    AIE.end
  }

  %core27_7 = AIE.core(%tile27_7) {
    AIE.useLock(%lock27_6, "Acquire", 1)
    AIE.useLock(%lock27_7, "Acquire", 0)
    func.call @do_sieve(%buf27_6, %buf27_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_6, "Release", 0)
    AIE.useLock(%lock27_7, "Release", 1)
    AIE.end
  }

  %core28_7 = AIE.core(%tile28_7) {
    AIE.useLock(%lock27_7, "Acquire", 1)
    AIE.useLock(%lock28_7, "Acquire", 0)
    func.call @do_sieve(%buf27_7, %buf28_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock27_7, "Release", 0)
    AIE.useLock(%lock28_7, "Release", 1)
    AIE.end
  }

  %core28_6 = AIE.core(%tile28_6) {
    AIE.useLock(%lock28_7, "Acquire", 1)
    AIE.useLock(%lock28_6, "Acquire", 0)
    func.call @do_sieve(%buf28_7, %buf28_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_7, "Release", 0)
    AIE.useLock(%lock28_6, "Release", 1)
    AIE.end
  }

  %core28_5 = AIE.core(%tile28_5) {
    AIE.useLock(%lock28_6, "Acquire", 1)
    AIE.useLock(%lock28_5, "Acquire", 0)
    func.call @do_sieve(%buf28_6, %buf28_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_6, "Release", 0)
    AIE.useLock(%lock28_5, "Release", 1)
    AIE.end
  }

  %core28_4 = AIE.core(%tile28_4) {
    AIE.useLock(%lock28_5, "Acquire", 1)
    AIE.useLock(%lock28_4, "Acquire", 0)
    func.call @do_sieve(%buf28_5, %buf28_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_5, "Release", 0)
    AIE.useLock(%lock28_4, "Release", 1)
    AIE.end
  }

  %core28_3 = AIE.core(%tile28_3) {
    AIE.useLock(%lock28_4, "Acquire", 1)
    AIE.useLock(%lock28_3, "Acquire", 0)
    func.call @do_sieve(%buf28_4, %buf28_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_4, "Release", 0)
    AIE.useLock(%lock28_3, "Release", 1)
    AIE.end
  }

  %core28_2 = AIE.core(%tile28_2) {
    AIE.useLock(%lock28_3, "Acquire", 1)
    AIE.useLock(%lock28_2, "Acquire", 0)
    func.call @do_sieve(%buf28_3, %buf28_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_3, "Release", 0)
    AIE.useLock(%lock28_2, "Release", 1)
    AIE.end
  }

  %core28_1 = AIE.core(%tile28_1) {
    AIE.useLock(%lock28_2, "Acquire", 1)
    AIE.useLock(%lock28_1, "Acquire", 0)
    func.call @do_sieve(%buf28_2, %buf28_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_2, "Release", 0)
    AIE.useLock(%lock28_1, "Release", 1)
    AIE.end
  }

  %core29_1 = AIE.core(%tile29_1) {
    AIE.useLock(%lock28_1, "Acquire", 1)
    AIE.useLock(%lock29_1, "Acquire", 0)
    func.call @do_sieve(%buf28_1, %buf29_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock28_1, "Release", 0)
    AIE.useLock(%lock29_1, "Release", 1)
    AIE.end
  }

  %core29_2 = AIE.core(%tile29_2) {
    AIE.useLock(%lock29_1, "Acquire", 1)
    AIE.useLock(%lock29_2, "Acquire", 0)
    func.call @do_sieve(%buf29_1, %buf29_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_1, "Release", 0)
    AIE.useLock(%lock29_2, "Release", 1)
    AIE.end
  }

  %core29_3 = AIE.core(%tile29_3) {
    AIE.useLock(%lock29_2, "Acquire", 1)
    AIE.useLock(%lock29_3, "Acquire", 0)
    func.call @do_sieve(%buf29_2, %buf29_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_2, "Release", 0)
    AIE.useLock(%lock29_3, "Release", 1)
    AIE.end
  }

  %core29_4 = AIE.core(%tile29_4) {
    AIE.useLock(%lock29_3, "Acquire", 1)
    AIE.useLock(%lock29_4, "Acquire", 0)
    func.call @do_sieve(%buf29_3, %buf29_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_3, "Release", 0)
    AIE.useLock(%lock29_4, "Release", 1)
    AIE.end
  }

  %core29_5 = AIE.core(%tile29_5) {
    AIE.useLock(%lock29_4, "Acquire", 1)
    AIE.useLock(%lock29_5, "Acquire", 0)
    func.call @do_sieve(%buf29_4, %buf29_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_4, "Release", 0)
    AIE.useLock(%lock29_5, "Release", 1)
    AIE.end
  }

  %core29_6 = AIE.core(%tile29_6) {
    AIE.useLock(%lock29_5, "Acquire", 1)
    AIE.useLock(%lock29_6, "Acquire", 0)
    func.call @do_sieve(%buf29_5, %buf29_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_5, "Release", 0)
    AIE.useLock(%lock29_6, "Release", 1)
    AIE.end
  }

  %core29_7 = AIE.core(%tile29_7) {
    AIE.useLock(%lock29_6, "Acquire", 1)
    AIE.useLock(%lock29_7, "Acquire", 0)
    func.call @do_sieve(%buf29_6, %buf29_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_6, "Release", 0)
    AIE.useLock(%lock29_7, "Release", 1)
    AIE.end
  }

  %core30_7 = AIE.core(%tile30_7) {
    AIE.useLock(%lock29_7, "Acquire", 1)
    AIE.useLock(%lock30_7, "Acquire", 0)
    func.call @do_sieve(%buf29_7, %buf30_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock29_7, "Release", 0)
    AIE.useLock(%lock30_7, "Release", 1)
    AIE.end
  }

  %core30_6 = AIE.core(%tile30_6) {
    AIE.useLock(%lock30_7, "Acquire", 1)
    AIE.useLock(%lock30_6, "Acquire", 0)
    func.call @do_sieve(%buf30_7, %buf30_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_7, "Release", 0)
    AIE.useLock(%lock30_6, "Release", 1)
    AIE.end
  }

  %core30_5 = AIE.core(%tile30_5) {
    AIE.useLock(%lock30_6, "Acquire", 1)
    AIE.useLock(%lock30_5, "Acquire", 0)
    func.call @do_sieve(%buf30_6, %buf30_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_6, "Release", 0)
    AIE.useLock(%lock30_5, "Release", 1)
    AIE.end
  }

  %core30_4 = AIE.core(%tile30_4) {
    AIE.useLock(%lock30_5, "Acquire", 1)
    AIE.useLock(%lock30_4, "Acquire", 0)
    func.call @do_sieve(%buf30_5, %buf30_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_5, "Release", 0)
    AIE.useLock(%lock30_4, "Release", 1)
    AIE.end
  }

  %core30_3 = AIE.core(%tile30_3) {
    AIE.useLock(%lock30_4, "Acquire", 1)
    AIE.useLock(%lock30_3, "Acquire", 0)
    func.call @do_sieve(%buf30_4, %buf30_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_4, "Release", 0)
    AIE.useLock(%lock30_3, "Release", 1)
    AIE.end
  }

  %core30_2 = AIE.core(%tile30_2) {
    AIE.useLock(%lock30_3, "Acquire", 1)
    AIE.useLock(%lock30_2, "Acquire", 0)
    func.call @do_sieve(%buf30_3, %buf30_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_3, "Release", 0)
    AIE.useLock(%lock30_2, "Release", 1)
    AIE.end
  }

  %core30_1 = AIE.core(%tile30_1) {
    AIE.useLock(%lock30_2, "Acquire", 1)
    AIE.useLock(%lock30_1, "Acquire", 0)
    func.call @do_sieve(%buf30_2, %buf30_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_2, "Release", 0)
    AIE.useLock(%lock30_1, "Release", 1)
    AIE.end
  }

  %core31_1 = AIE.core(%tile31_1) {
    AIE.useLock(%lock30_1, "Acquire", 1)
    AIE.useLock(%lock31_1, "Acquire", 0)
    func.call @do_sieve(%buf30_1, %buf31_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock30_1, "Release", 0)
    AIE.useLock(%lock31_1, "Release", 1)
    AIE.end
  }

  %core31_2 = AIE.core(%tile31_2) {
    AIE.useLock(%lock31_1, "Acquire", 1)
    AIE.useLock(%lock31_2, "Acquire", 0)
    func.call @do_sieve(%buf31_1, %buf31_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_1, "Release", 0)
    AIE.useLock(%lock31_2, "Release", 1)
    AIE.end
  }

  %core31_3 = AIE.core(%tile31_3) {
    AIE.useLock(%lock31_2, "Acquire", 1)
    AIE.useLock(%lock31_3, "Acquire", 0)
    func.call @do_sieve(%buf31_2, %buf31_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_2, "Release", 0)
    AIE.useLock(%lock31_3, "Release", 1)
    AIE.end
  }

  %core31_4 = AIE.core(%tile31_4) {
    AIE.useLock(%lock31_3, "Acquire", 1)
    AIE.useLock(%lock31_4, "Acquire", 0)
    func.call @do_sieve(%buf31_3, %buf31_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_3, "Release", 0)
    AIE.useLock(%lock31_4, "Release", 1)
    AIE.end
  }

  %core31_5 = AIE.core(%tile31_5) {
    AIE.useLock(%lock31_4, "Acquire", 1)
    AIE.useLock(%lock31_5, "Acquire", 0)
    func.call @do_sieve(%buf31_4, %buf31_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_4, "Release", 0)
    AIE.useLock(%lock31_5, "Release", 1)
    AIE.end
  }

  %core31_6 = AIE.core(%tile31_6) {
    AIE.useLock(%lock31_5, "Acquire", 1)
    AIE.useLock(%lock31_6, "Acquire", 0)
    func.call @do_sieve(%buf31_5, %buf31_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_5, "Release", 0)
    AIE.useLock(%lock31_6, "Release", 1)
    AIE.end
  }

  %core31_7 = AIE.core(%tile31_7) {
    AIE.useLock(%lock31_6, "Acquire", 1)
    AIE.useLock(%lock31_7, "Acquire", 0)
    func.call @do_sieve(%buf31_6, %buf31_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_6, "Release", 0)
    AIE.useLock(%lock31_7, "Release", 1)
    AIE.end
  }

  %core32_7 = AIE.core(%tile32_7) {
    AIE.useLock(%lock31_7, "Acquire", 1)
    AIE.useLock(%lock32_7, "Acquire", 0)
    func.call @do_sieve(%buf31_7, %buf32_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock31_7, "Release", 0)
    AIE.useLock(%lock32_7, "Release", 1)
    AIE.end
  }

  %core32_6 = AIE.core(%tile32_6) {
    AIE.useLock(%lock32_7, "Acquire", 1)
    AIE.useLock(%lock32_6, "Acquire", 0)
    func.call @do_sieve(%buf32_7, %buf32_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_7, "Release", 0)
    AIE.useLock(%lock32_6, "Release", 1)
    AIE.end
  }

  %core32_5 = AIE.core(%tile32_5) {
    AIE.useLock(%lock32_6, "Acquire", 1)
    AIE.useLock(%lock32_5, "Acquire", 0)
    func.call @do_sieve(%buf32_6, %buf32_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_6, "Release", 0)
    AIE.useLock(%lock32_5, "Release", 1)
    AIE.end
  }

  %core32_4 = AIE.core(%tile32_4) {
    AIE.useLock(%lock32_5, "Acquire", 1)
    AIE.useLock(%lock32_4, "Acquire", 0)
    func.call @do_sieve(%buf32_5, %buf32_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_5, "Release", 0)
    AIE.useLock(%lock32_4, "Release", 1)
    AIE.end
  }

  %core32_3 = AIE.core(%tile32_3) {
    AIE.useLock(%lock32_4, "Acquire", 1)
    AIE.useLock(%lock32_3, "Acquire", 0)
    func.call @do_sieve(%buf32_4, %buf32_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_4, "Release", 0)
    AIE.useLock(%lock32_3, "Release", 1)
    AIE.end
  }

  %core32_2 = AIE.core(%tile32_2) {
    AIE.useLock(%lock32_3, "Acquire", 1)
    AIE.useLock(%lock32_2, "Acquire", 0)
    func.call @do_sieve(%buf32_3, %buf32_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_3, "Release", 0)
    AIE.useLock(%lock32_2, "Release", 1)
    AIE.end
  }

  %core32_1 = AIE.core(%tile32_1) {
    AIE.useLock(%lock32_2, "Acquire", 1)
    AIE.useLock(%lock32_1, "Acquire", 0)
    func.call @do_sieve(%buf32_2, %buf32_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_2, "Release", 0)
    AIE.useLock(%lock32_1, "Release", 1)
    AIE.end
  }

  %core33_1 = AIE.core(%tile33_1) {
    AIE.useLock(%lock32_1, "Acquire", 1)
    AIE.useLock(%lock33_1, "Acquire", 0)
    func.call @do_sieve(%buf32_1, %buf33_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock32_1, "Release", 0)
    AIE.useLock(%lock33_1, "Release", 1)
    AIE.end
  }

  %core33_2 = AIE.core(%tile33_2) {
    AIE.useLock(%lock33_1, "Acquire", 1)
    AIE.useLock(%lock33_2, "Acquire", 0)
    func.call @do_sieve(%buf33_1, %buf33_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_1, "Release", 0)
    AIE.useLock(%lock33_2, "Release", 1)
    AIE.end
  }

  %core33_3 = AIE.core(%tile33_3) {
    AIE.useLock(%lock33_2, "Acquire", 1)
    AIE.useLock(%lock33_3, "Acquire", 0)
    func.call @do_sieve(%buf33_2, %buf33_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_2, "Release", 0)
    AIE.useLock(%lock33_3, "Release", 1)
    AIE.end
  }

  %core33_4 = AIE.core(%tile33_4) {
    AIE.useLock(%lock33_3, "Acquire", 1)
    AIE.useLock(%lock33_4, "Acquire", 0)
    func.call @do_sieve(%buf33_3, %buf33_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_3, "Release", 0)
    AIE.useLock(%lock33_4, "Release", 1)
    AIE.end
  }

  %core33_5 = AIE.core(%tile33_5) {
    AIE.useLock(%lock33_4, "Acquire", 1)
    AIE.useLock(%lock33_5, "Acquire", 0)
    func.call @do_sieve(%buf33_4, %buf33_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_4, "Release", 0)
    AIE.useLock(%lock33_5, "Release", 1)
    AIE.end
  }

  %core33_6 = AIE.core(%tile33_6) {
    AIE.useLock(%lock33_5, "Acquire", 1)
    AIE.useLock(%lock33_6, "Acquire", 0)
    func.call @do_sieve(%buf33_5, %buf33_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_5, "Release", 0)
    AIE.useLock(%lock33_6, "Release", 1)
    AIE.end
  }

  %core33_7 = AIE.core(%tile33_7) {
    AIE.useLock(%lock33_6, "Acquire", 1)
    AIE.useLock(%lock33_7, "Acquire", 0)
    func.call @do_sieve(%buf33_6, %buf33_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_6, "Release", 0)
    AIE.useLock(%lock33_7, "Release", 1)
    AIE.end
  }

  %core34_7 = AIE.core(%tile34_7) {
    AIE.useLock(%lock33_7, "Acquire", 1)
    AIE.useLock(%lock34_7, "Acquire", 0)
    func.call @do_sieve(%buf33_7, %buf34_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock33_7, "Release", 0)
    AIE.useLock(%lock34_7, "Release", 1)
    AIE.end
  }

  %core34_6 = AIE.core(%tile34_6) {
    AIE.useLock(%lock34_7, "Acquire", 1)
    AIE.useLock(%lock34_6, "Acquire", 0)
    func.call @do_sieve(%buf34_7, %buf34_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_7, "Release", 0)
    AIE.useLock(%lock34_6, "Release", 1)
    AIE.end
  }

  %core34_5 = AIE.core(%tile34_5) {
    AIE.useLock(%lock34_6, "Acquire", 1)
    AIE.useLock(%lock34_5, "Acquire", 0)
    func.call @do_sieve(%buf34_6, %buf34_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_6, "Release", 0)
    AIE.useLock(%lock34_5, "Release", 1)
    AIE.end
  }

  %core34_4 = AIE.core(%tile34_4) {
    AIE.useLock(%lock34_5, "Acquire", 1)
    AIE.useLock(%lock34_4, "Acquire", 0)
    func.call @do_sieve(%buf34_5, %buf34_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_5, "Release", 0)
    AIE.useLock(%lock34_4, "Release", 1)
    AIE.end
  }

  %core34_3 = AIE.core(%tile34_3) {
    AIE.useLock(%lock34_4, "Acquire", 1)
    AIE.useLock(%lock34_3, "Acquire", 0)
    func.call @do_sieve(%buf34_4, %buf34_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_4, "Release", 0)
    AIE.useLock(%lock34_3, "Release", 1)
    AIE.end
  }

  %core34_2 = AIE.core(%tile34_2) {
    AIE.useLock(%lock34_3, "Acquire", 1)
    AIE.useLock(%lock34_2, "Acquire", 0)
    func.call @do_sieve(%buf34_3, %buf34_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_3, "Release", 0)
    AIE.useLock(%lock34_2, "Release", 1)
    AIE.end
  }

  %core34_1 = AIE.core(%tile34_1) {
    AIE.useLock(%lock34_2, "Acquire", 1)
    AIE.useLock(%lock34_1, "Acquire", 0)
    func.call @do_sieve(%buf34_2, %buf34_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_2, "Release", 0)
    AIE.useLock(%lock34_1, "Release", 1)
    AIE.end
  }

  %core35_1 = AIE.core(%tile35_1) {
    AIE.useLock(%lock34_1, "Acquire", 1)
    AIE.useLock(%lock35_1, "Acquire", 0)
    func.call @do_sieve(%buf34_1, %buf35_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock34_1, "Release", 0)
    AIE.useLock(%lock35_1, "Release", 1)
    AIE.end
  }

  %core35_2 = AIE.core(%tile35_2) {
    AIE.useLock(%lock35_1, "Acquire", 1)
    AIE.useLock(%lock35_2, "Acquire", 0)
    func.call @do_sieve(%buf35_1, %buf35_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_1, "Release", 0)
    AIE.useLock(%lock35_2, "Release", 1)
    AIE.end
  }

  %core35_3 = AIE.core(%tile35_3) {
    AIE.useLock(%lock35_2, "Acquire", 1)
    AIE.useLock(%lock35_3, "Acquire", 0)
    func.call @do_sieve(%buf35_2, %buf35_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_2, "Release", 0)
    AIE.useLock(%lock35_3, "Release", 1)
    AIE.end
  }

  %core35_4 = AIE.core(%tile35_4) {
    AIE.useLock(%lock35_3, "Acquire", 1)
    AIE.useLock(%lock35_4, "Acquire", 0)
    func.call @do_sieve(%buf35_3, %buf35_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_3, "Release", 0)
    AIE.useLock(%lock35_4, "Release", 1)
    AIE.end
  }

  %core35_5 = AIE.core(%tile35_5) {
    AIE.useLock(%lock35_4, "Acquire", 1)
    AIE.useLock(%lock35_5, "Acquire", 0)
    func.call @do_sieve(%buf35_4, %buf35_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_4, "Release", 0)
    AIE.useLock(%lock35_5, "Release", 1)
    AIE.end
  }

  %core35_6 = AIE.core(%tile35_6) {
    AIE.useLock(%lock35_5, "Acquire", 1)
    AIE.useLock(%lock35_6, "Acquire", 0)
    func.call @do_sieve(%buf35_5, %buf35_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_5, "Release", 0)
    AIE.useLock(%lock35_6, "Release", 1)
    AIE.end
  }

  %core35_7 = AIE.core(%tile35_7) {
    AIE.useLock(%lock35_6, "Acquire", 1)
    AIE.useLock(%lock35_7, "Acquire", 0)
    func.call @do_sieve(%buf35_6, %buf35_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_6, "Release", 0)
    AIE.useLock(%lock35_7, "Release", 1)
    AIE.end
  }

  %core36_7 = AIE.core(%tile36_7) {
    AIE.useLock(%lock35_7, "Acquire", 1)
    AIE.useLock(%lock36_7, "Acquire", 0)
    func.call @do_sieve(%buf35_7, %buf36_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock35_7, "Release", 0)
    AIE.useLock(%lock36_7, "Release", 1)
    AIE.end
  }

  %core36_6 = AIE.core(%tile36_6) {
    AIE.useLock(%lock36_7, "Acquire", 1)
    AIE.useLock(%lock36_6, "Acquire", 0)
    func.call @do_sieve(%buf36_7, %buf36_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_7, "Release", 0)
    AIE.useLock(%lock36_6, "Release", 1)
    AIE.end
  }

  %core36_5 = AIE.core(%tile36_5) {
    AIE.useLock(%lock36_6, "Acquire", 1)
    AIE.useLock(%lock36_5, "Acquire", 0)
    func.call @do_sieve(%buf36_6, %buf36_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_6, "Release", 0)
    AIE.useLock(%lock36_5, "Release", 1)
    AIE.end
  }

  %core36_4 = AIE.core(%tile36_4) {
    AIE.useLock(%lock36_5, "Acquire", 1)
    AIE.useLock(%lock36_4, "Acquire", 0)
    func.call @do_sieve(%buf36_5, %buf36_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_5, "Release", 0)
    AIE.useLock(%lock36_4, "Release", 1)
    AIE.end
  }

  %core36_3 = AIE.core(%tile36_3) {
    AIE.useLock(%lock36_4, "Acquire", 1)
    AIE.useLock(%lock36_3, "Acquire", 0)
    func.call @do_sieve(%buf36_4, %buf36_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_4, "Release", 0)
    AIE.useLock(%lock36_3, "Release", 1)
    AIE.end
  }

  %core36_2 = AIE.core(%tile36_2) {
    AIE.useLock(%lock36_3, "Acquire", 1)
    AIE.useLock(%lock36_2, "Acquire", 0)
    func.call @do_sieve(%buf36_3, %buf36_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_3, "Release", 0)
    AIE.useLock(%lock36_2, "Release", 1)
    AIE.end
  }

  %core36_1 = AIE.core(%tile36_1) {
    AIE.useLock(%lock36_2, "Acquire", 1)
    AIE.useLock(%lock36_1, "Acquire", 0)
    func.call @do_sieve(%buf36_2, %buf36_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_2, "Release", 0)
    AIE.useLock(%lock36_1, "Release", 1)
    AIE.end
  }

  %core37_1 = AIE.core(%tile37_1) {
    AIE.useLock(%lock36_1, "Acquire", 1)
    AIE.useLock(%lock37_1, "Acquire", 0)
    func.call @do_sieve(%buf36_1, %buf37_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock36_1, "Release", 0)
    AIE.useLock(%lock37_1, "Release", 1)
    AIE.end
  }

  %core37_2 = AIE.core(%tile37_2) {
    AIE.useLock(%lock37_1, "Acquire", 1)
    AIE.useLock(%lock37_2, "Acquire", 0)
    func.call @do_sieve(%buf37_1, %buf37_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_1, "Release", 0)
    AIE.useLock(%lock37_2, "Release", 1)
    AIE.end
  }

  %core37_3 = AIE.core(%tile37_3) {
    AIE.useLock(%lock37_2, "Acquire", 1)
    AIE.useLock(%lock37_3, "Acquire", 0)
    func.call @do_sieve(%buf37_2, %buf37_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_2, "Release", 0)
    AIE.useLock(%lock37_3, "Release", 1)
    AIE.end
  }

  %core37_4 = AIE.core(%tile37_4) {
    AIE.useLock(%lock37_3, "Acquire", 1)
    AIE.useLock(%lock37_4, "Acquire", 0)
    func.call @do_sieve(%buf37_3, %buf37_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_3, "Release", 0)
    AIE.useLock(%lock37_4, "Release", 1)
    AIE.end
  }

  %core37_5 = AIE.core(%tile37_5) {
    AIE.useLock(%lock37_4, "Acquire", 1)
    AIE.useLock(%lock37_5, "Acquire", 0)
    func.call @do_sieve(%buf37_4, %buf37_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_4, "Release", 0)
    AIE.useLock(%lock37_5, "Release", 1)
    AIE.end
  }

  %core37_6 = AIE.core(%tile37_6) {
    AIE.useLock(%lock37_5, "Acquire", 1)
    AIE.useLock(%lock37_6, "Acquire", 0)
    func.call @do_sieve(%buf37_5, %buf37_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_5, "Release", 0)
    AIE.useLock(%lock37_6, "Release", 1)
    AIE.end
  }

  %core37_7 = AIE.core(%tile37_7) {
    AIE.useLock(%lock37_6, "Acquire", 1)
    AIE.useLock(%lock37_7, "Acquire", 0)
    func.call @do_sieve(%buf37_6, %buf37_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_6, "Release", 0)
    AIE.useLock(%lock37_7, "Release", 1)
    AIE.end
  }

  %core38_7 = AIE.core(%tile38_7) {
    AIE.useLock(%lock37_7, "Acquire", 1)
    AIE.useLock(%lock38_7, "Acquire", 0)
    func.call @do_sieve(%buf37_7, %buf38_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock37_7, "Release", 0)
    AIE.useLock(%lock38_7, "Release", 1)
    AIE.end
  }

  %core38_6 = AIE.core(%tile38_6) {
    AIE.useLock(%lock38_7, "Acquire", 1)
    AIE.useLock(%lock38_6, "Acquire", 0)
    func.call @do_sieve(%buf38_7, %buf38_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_7, "Release", 0)
    AIE.useLock(%lock38_6, "Release", 1)
    AIE.end
  }

  %core38_5 = AIE.core(%tile38_5) {
    AIE.useLock(%lock38_6, "Acquire", 1)
    AIE.useLock(%lock38_5, "Acquire", 0)
    func.call @do_sieve(%buf38_6, %buf38_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_6, "Release", 0)
    AIE.useLock(%lock38_5, "Release", 1)
    AIE.end
  }

  %core38_4 = AIE.core(%tile38_4) {
    AIE.useLock(%lock38_5, "Acquire", 1)
    AIE.useLock(%lock38_4, "Acquire", 0)
    func.call @do_sieve(%buf38_5, %buf38_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_5, "Release", 0)
    AIE.useLock(%lock38_4, "Release", 1)
    AIE.end
  }

  %core38_3 = AIE.core(%tile38_3) {
    AIE.useLock(%lock38_4, "Acquire", 1)
    AIE.useLock(%lock38_3, "Acquire", 0)
    func.call @do_sieve(%buf38_4, %buf38_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_4, "Release", 0)
    AIE.useLock(%lock38_3, "Release", 1)
    AIE.end
  }

  %core38_2 = AIE.core(%tile38_2) {
    AIE.useLock(%lock38_3, "Acquire", 1)
    AIE.useLock(%lock38_2, "Acquire", 0)
    func.call @do_sieve(%buf38_3, %buf38_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_3, "Release", 0)
    AIE.useLock(%lock38_2, "Release", 1)
    AIE.end
  }

  %core38_1 = AIE.core(%tile38_1) {
    AIE.useLock(%lock38_2, "Acquire", 1)
    AIE.useLock(%lock38_1, "Acquire", 0)
    func.call @do_sieve(%buf38_2, %buf38_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_2, "Release", 0)
    AIE.useLock(%lock38_1, "Release", 1)
    AIE.end
  }

  %core39_1 = AIE.core(%tile39_1) {
    AIE.useLock(%lock38_1, "Acquire", 1)
    AIE.useLock(%lock39_1, "Acquire", 0)
    func.call @do_sieve(%buf38_1, %buf39_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock38_1, "Release", 0)
    AIE.useLock(%lock39_1, "Release", 1)
    AIE.end
  }

  %core39_2 = AIE.core(%tile39_2) {
    AIE.useLock(%lock39_1, "Acquire", 1)
    AIE.useLock(%lock39_2, "Acquire", 0)
    func.call @do_sieve(%buf39_1, %buf39_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_1, "Release", 0)
    AIE.useLock(%lock39_2, "Release", 1)
    AIE.end
  }

  %core39_3 = AIE.core(%tile39_3) {
    AIE.useLock(%lock39_2, "Acquire", 1)
    AIE.useLock(%lock39_3, "Acquire", 0)
    func.call @do_sieve(%buf39_2, %buf39_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_2, "Release", 0)
    AIE.useLock(%lock39_3, "Release", 1)
    AIE.end
  }

  %core39_4 = AIE.core(%tile39_4) {
    AIE.useLock(%lock39_3, "Acquire", 1)
    AIE.useLock(%lock39_4, "Acquire", 0)
    func.call @do_sieve(%buf39_3, %buf39_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_3, "Release", 0)
    AIE.useLock(%lock39_4, "Release", 1)
    AIE.end
  }

  %core39_5 = AIE.core(%tile39_5) {
    AIE.useLock(%lock39_4, "Acquire", 1)
    AIE.useLock(%lock39_5, "Acquire", 0)
    func.call @do_sieve(%buf39_4, %buf39_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_4, "Release", 0)
    AIE.useLock(%lock39_5, "Release", 1)
    AIE.end
  }

  %core39_6 = AIE.core(%tile39_6) {
    AIE.useLock(%lock39_5, "Acquire", 1)
    AIE.useLock(%lock39_6, "Acquire", 0)
    func.call @do_sieve(%buf39_5, %buf39_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_5, "Release", 0)
    AIE.useLock(%lock39_6, "Release", 1)
    AIE.end
  }

  %core39_7 = AIE.core(%tile39_7) {
    AIE.useLock(%lock39_6, "Acquire", 1)
    AIE.useLock(%lock39_7, "Acquire", 0)
    func.call @do_sieve(%buf39_6, %buf39_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_6, "Release", 0)
    AIE.useLock(%lock39_7, "Release", 1)
    AIE.end
  }

  %core40_7 = AIE.core(%tile40_7) {
    AIE.useLock(%lock39_7, "Acquire", 1)
    AIE.useLock(%lock40_7, "Acquire", 0)
    func.call @do_sieve(%buf39_7, %buf40_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock39_7, "Release", 0)
    AIE.useLock(%lock40_7, "Release", 1)
    AIE.end
  }

  %core40_6 = AIE.core(%tile40_6) {
    AIE.useLock(%lock40_7, "Acquire", 1)
    AIE.useLock(%lock40_6, "Acquire", 0)
    func.call @do_sieve(%buf40_7, %buf40_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_7, "Release", 0)
    AIE.useLock(%lock40_6, "Release", 1)
    AIE.end
  }

  %core40_5 = AIE.core(%tile40_5) {
    AIE.useLock(%lock40_6, "Acquire", 1)
    AIE.useLock(%lock40_5, "Acquire", 0)
    func.call @do_sieve(%buf40_6, %buf40_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_6, "Release", 0)
    AIE.useLock(%lock40_5, "Release", 1)
    AIE.end
  }

  %core40_4 = AIE.core(%tile40_4) {
    AIE.useLock(%lock40_5, "Acquire", 1)
    AIE.useLock(%lock40_4, "Acquire", 0)
    func.call @do_sieve(%buf40_5, %buf40_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_5, "Release", 0)
    AIE.useLock(%lock40_4, "Release", 1)
    AIE.end
  }

  %core40_3 = AIE.core(%tile40_3) {
    AIE.useLock(%lock40_4, "Acquire", 1)
    AIE.useLock(%lock40_3, "Acquire", 0)
    func.call @do_sieve(%buf40_4, %buf40_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_4, "Release", 0)
    AIE.useLock(%lock40_3, "Release", 1)
    AIE.end
  }

  %core40_2 = AIE.core(%tile40_2) {
    AIE.useLock(%lock40_3, "Acquire", 1)
    AIE.useLock(%lock40_2, "Acquire", 0)
    func.call @do_sieve(%buf40_3, %buf40_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_3, "Release", 0)
    AIE.useLock(%lock40_2, "Release", 1)
    AIE.end
  }

  %core40_1 = AIE.core(%tile40_1) {
    AIE.useLock(%lock40_2, "Acquire", 1)
    AIE.useLock(%lock40_1, "Acquire", 0)
    func.call @do_sieve(%buf40_2, %buf40_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_2, "Release", 0)
    AIE.useLock(%lock40_1, "Release", 1)
    AIE.end
  }

  %core41_1 = AIE.core(%tile41_1) {
    AIE.useLock(%lock40_1, "Acquire", 1)
    AIE.useLock(%lock41_1, "Acquire", 0)
    func.call @do_sieve(%buf40_1, %buf41_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock40_1, "Release", 0)
    AIE.useLock(%lock41_1, "Release", 1)
    AIE.end
  }

  %core41_2 = AIE.core(%tile41_2) {
    AIE.useLock(%lock41_1, "Acquire", 1)
    AIE.useLock(%lock41_2, "Acquire", 0)
    func.call @do_sieve(%buf41_1, %buf41_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_1, "Release", 0)
    AIE.useLock(%lock41_2, "Release", 1)
    AIE.end
  }

  %core41_3 = AIE.core(%tile41_3) {
    AIE.useLock(%lock41_2, "Acquire", 1)
    AIE.useLock(%lock41_3, "Acquire", 0)
    func.call @do_sieve(%buf41_2, %buf41_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_2, "Release", 0)
    AIE.useLock(%lock41_3, "Release", 1)
    AIE.end
  }

  %core41_4 = AIE.core(%tile41_4) {
    AIE.useLock(%lock41_3, "Acquire", 1)
    AIE.useLock(%lock41_4, "Acquire", 0)
    func.call @do_sieve(%buf41_3, %buf41_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_3, "Release", 0)
    AIE.useLock(%lock41_4, "Release", 1)
    AIE.end
  }

  %core41_5 = AIE.core(%tile41_5) {
    AIE.useLock(%lock41_4, "Acquire", 1)
    AIE.useLock(%lock41_5, "Acquire", 0)
    func.call @do_sieve(%buf41_4, %buf41_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_4, "Release", 0)
    AIE.useLock(%lock41_5, "Release", 1)
    AIE.end
  }

  %core41_6 = AIE.core(%tile41_6) {
    AIE.useLock(%lock41_5, "Acquire", 1)
    AIE.useLock(%lock41_6, "Acquire", 0)
    func.call @do_sieve(%buf41_5, %buf41_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_5, "Release", 0)
    AIE.useLock(%lock41_6, "Release", 1)
    AIE.end
  }

  %core41_7 = AIE.core(%tile41_7) {
    AIE.useLock(%lock41_6, "Acquire", 1)
    AIE.useLock(%lock41_7, "Acquire", 0)
    func.call @do_sieve(%buf41_6, %buf41_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_6, "Release", 0)
    AIE.useLock(%lock41_7, "Release", 1)
    AIE.end
  }

  %core42_7 = AIE.core(%tile42_7) {
    AIE.useLock(%lock41_7, "Acquire", 1)
    AIE.useLock(%lock42_7, "Acquire", 0)
    func.call @do_sieve(%buf41_7, %buf42_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock41_7, "Release", 0)
    AIE.useLock(%lock42_7, "Release", 1)
    AIE.end
  }

  %core42_6 = AIE.core(%tile42_6) {
    AIE.useLock(%lock42_7, "Acquire", 1)
    AIE.useLock(%lock42_6, "Acquire", 0)
    func.call @do_sieve(%buf42_7, %buf42_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_7, "Release", 0)
    AIE.useLock(%lock42_6, "Release", 1)
    AIE.end
  }

  %core42_5 = AIE.core(%tile42_5) {
    AIE.useLock(%lock42_6, "Acquire", 1)
    AIE.useLock(%lock42_5, "Acquire", 0)
    func.call @do_sieve(%buf42_6, %buf42_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_6, "Release", 0)
    AIE.useLock(%lock42_5, "Release", 1)
    AIE.end
  }

  %core42_4 = AIE.core(%tile42_4) {
    AIE.useLock(%lock42_5, "Acquire", 1)
    AIE.useLock(%lock42_4, "Acquire", 0)
    func.call @do_sieve(%buf42_5, %buf42_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_5, "Release", 0)
    AIE.useLock(%lock42_4, "Release", 1)
    AIE.end
  }

  %core42_3 = AIE.core(%tile42_3) {
    AIE.useLock(%lock42_4, "Acquire", 1)
    AIE.useLock(%lock42_3, "Acquire", 0)
    func.call @do_sieve(%buf42_4, %buf42_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_4, "Release", 0)
    AIE.useLock(%lock42_3, "Release", 1)
    AIE.end
  }

  %core42_2 = AIE.core(%tile42_2) {
    AIE.useLock(%lock42_3, "Acquire", 1)
    AIE.useLock(%lock42_2, "Acquire", 0)
    func.call @do_sieve(%buf42_3, %buf42_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_3, "Release", 0)
    AIE.useLock(%lock42_2, "Release", 1)
    AIE.end
  }

  %core42_1 = AIE.core(%tile42_1) {
    AIE.useLock(%lock42_2, "Acquire", 1)
    AIE.useLock(%lock42_1, "Acquire", 0)
    func.call @do_sieve(%buf42_2, %buf42_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_2, "Release", 0)
    AIE.useLock(%lock42_1, "Release", 1)
    AIE.end
  }

  %core43_1 = AIE.core(%tile43_1) {
    AIE.useLock(%lock42_1, "Acquire", 1)
    AIE.useLock(%lock43_1, "Acquire", 0)
    func.call @do_sieve(%buf42_1, %buf43_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock42_1, "Release", 0)
    AIE.useLock(%lock43_1, "Release", 1)
    AIE.end
  }

  %core43_2 = AIE.core(%tile43_2) {
    AIE.useLock(%lock43_1, "Acquire", 1)
    AIE.useLock(%lock43_2, "Acquire", 0)
    func.call @do_sieve(%buf43_1, %buf43_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_1, "Release", 0)
    AIE.useLock(%lock43_2, "Release", 1)
    AIE.end
  }

  %core43_3 = AIE.core(%tile43_3) {
    AIE.useLock(%lock43_2, "Acquire", 1)
    AIE.useLock(%lock43_3, "Acquire", 0)
    func.call @do_sieve(%buf43_2, %buf43_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_2, "Release", 0)
    AIE.useLock(%lock43_3, "Release", 1)
    AIE.end
  }

  %core43_4 = AIE.core(%tile43_4) {
    AIE.useLock(%lock43_3, "Acquire", 1)
    AIE.useLock(%lock43_4, "Acquire", 0)
    func.call @do_sieve(%buf43_3, %buf43_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_3, "Release", 0)
    AIE.useLock(%lock43_4, "Release", 1)
    AIE.end
  }

  %core43_5 = AIE.core(%tile43_5) {
    AIE.useLock(%lock43_4, "Acquire", 1)
    AIE.useLock(%lock43_5, "Acquire", 0)
    func.call @do_sieve(%buf43_4, %buf43_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_4, "Release", 0)
    AIE.useLock(%lock43_5, "Release", 1)
    AIE.end
  }

  %core43_6 = AIE.core(%tile43_6) {
    AIE.useLock(%lock43_5, "Acquire", 1)
    AIE.useLock(%lock43_6, "Acquire", 0)
    func.call @do_sieve(%buf43_5, %buf43_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_5, "Release", 0)
    AIE.useLock(%lock43_6, "Release", 1)
    AIE.end
  }

  %core43_7 = AIE.core(%tile43_7) {
    AIE.useLock(%lock43_6, "Acquire", 1)
    AIE.useLock(%lock43_7, "Acquire", 0)
    func.call @do_sieve(%buf43_6, %buf43_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_6, "Release", 0)
    AIE.useLock(%lock43_7, "Release", 1)
    AIE.end
  }

  %core44_7 = AIE.core(%tile44_7) {
    AIE.useLock(%lock43_7, "Acquire", 1)
    AIE.useLock(%lock44_7, "Acquire", 0)
    func.call @do_sieve(%buf43_7, %buf44_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock43_7, "Release", 0)
    AIE.useLock(%lock44_7, "Release", 1)
    AIE.end
  }

  %core44_6 = AIE.core(%tile44_6) {
    AIE.useLock(%lock44_7, "Acquire", 1)
    AIE.useLock(%lock44_6, "Acquire", 0)
    func.call @do_sieve(%buf44_7, %buf44_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_7, "Release", 0)
    AIE.useLock(%lock44_6, "Release", 1)
    AIE.end
  }

  %core44_5 = AIE.core(%tile44_5) {
    AIE.useLock(%lock44_6, "Acquire", 1)
    AIE.useLock(%lock44_5, "Acquire", 0)
    func.call @do_sieve(%buf44_6, %buf44_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_6, "Release", 0)
    AIE.useLock(%lock44_5, "Release", 1)
    AIE.end
  }

  %core44_4 = AIE.core(%tile44_4) {
    AIE.useLock(%lock44_5, "Acquire", 1)
    AIE.useLock(%lock44_4, "Acquire", 0)
    func.call @do_sieve(%buf44_5, %buf44_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_5, "Release", 0)
    AIE.useLock(%lock44_4, "Release", 1)
    AIE.end
  }

  %core44_3 = AIE.core(%tile44_3) {
    AIE.useLock(%lock44_4, "Acquire", 1)
    AIE.useLock(%lock44_3, "Acquire", 0)
    func.call @do_sieve(%buf44_4, %buf44_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_4, "Release", 0)
    AIE.useLock(%lock44_3, "Release", 1)
    AIE.end
  }

  %core44_2 = AIE.core(%tile44_2) {
    AIE.useLock(%lock44_3, "Acquire", 1)
    AIE.useLock(%lock44_2, "Acquire", 0)
    func.call @do_sieve(%buf44_3, %buf44_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_3, "Release", 0)
    AIE.useLock(%lock44_2, "Release", 1)
    AIE.end
  }

  %core44_1 = AIE.core(%tile44_1) {
    AIE.useLock(%lock44_2, "Acquire", 1)
    AIE.useLock(%lock44_1, "Acquire", 0)
    func.call @do_sieve(%buf44_2, %buf44_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_2, "Release", 0)
    AIE.useLock(%lock44_1, "Release", 1)
    AIE.end
  }

  %core45_1 = AIE.core(%tile45_1) {
    AIE.useLock(%lock44_1, "Acquire", 1)
    AIE.useLock(%lock45_1, "Acquire", 0)
    func.call @do_sieve(%buf44_1, %buf45_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock44_1, "Release", 0)
    AIE.useLock(%lock45_1, "Release", 1)
    AIE.end
  }

  %core45_2 = AIE.core(%tile45_2) {
    AIE.useLock(%lock45_1, "Acquire", 1)
    AIE.useLock(%lock45_2, "Acquire", 0)
    func.call @do_sieve(%buf45_1, %buf45_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_1, "Release", 0)
    AIE.useLock(%lock45_2, "Release", 1)
    AIE.end
  }

  %core45_3 = AIE.core(%tile45_3) {
    AIE.useLock(%lock45_2, "Acquire", 1)
    AIE.useLock(%lock45_3, "Acquire", 0)
    func.call @do_sieve(%buf45_2, %buf45_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_2, "Release", 0)
    AIE.useLock(%lock45_3, "Release", 1)
    AIE.end
  }

  %core45_4 = AIE.core(%tile45_4) {
    AIE.useLock(%lock45_3, "Acquire", 1)
    AIE.useLock(%lock45_4, "Acquire", 0)
    func.call @do_sieve(%buf45_3, %buf45_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_3, "Release", 0)
    AIE.useLock(%lock45_4, "Release", 1)
    AIE.end
  }

  %core45_5 = AIE.core(%tile45_5) {
    AIE.useLock(%lock45_4, "Acquire", 1)
    AIE.useLock(%lock45_5, "Acquire", 0)
    func.call @do_sieve(%buf45_4, %buf45_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_4, "Release", 0)
    AIE.useLock(%lock45_5, "Release", 1)
    AIE.end
  }

  %core45_6 = AIE.core(%tile45_6) {
    AIE.useLock(%lock45_5, "Acquire", 1)
    AIE.useLock(%lock45_6, "Acquire", 0)
    func.call @do_sieve(%buf45_5, %buf45_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_5, "Release", 0)
    AIE.useLock(%lock45_6, "Release", 1)
    AIE.end
  }

  %core45_7 = AIE.core(%tile45_7) {
    AIE.useLock(%lock45_6, "Acquire", 1)
    AIE.useLock(%lock45_7, "Acquire", 0)
    func.call @do_sieve(%buf45_6, %buf45_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_6, "Release", 0)
    AIE.useLock(%lock45_7, "Release", 1)
    AIE.end
  }

  %core46_7 = AIE.core(%tile46_7) {
    AIE.useLock(%lock45_7, "Acquire", 1)
    AIE.useLock(%lock46_7, "Acquire", 0)
    func.call @do_sieve(%buf45_7, %buf46_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock45_7, "Release", 0)
    AIE.useLock(%lock46_7, "Release", 1)
    AIE.end
  }

  %core46_6 = AIE.core(%tile46_6) {
    AIE.useLock(%lock46_7, "Acquire", 1)
    AIE.useLock(%lock46_6, "Acquire", 0)
    func.call @do_sieve(%buf46_7, %buf46_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_7, "Release", 0)
    AIE.useLock(%lock46_6, "Release", 1)
    AIE.end
  }

  %core46_5 = AIE.core(%tile46_5) {
    AIE.useLock(%lock46_6, "Acquire", 1)
    AIE.useLock(%lock46_5, "Acquire", 0)
    func.call @do_sieve(%buf46_6, %buf46_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_6, "Release", 0)
    AIE.useLock(%lock46_5, "Release", 1)
    AIE.end
  }

  %core46_4 = AIE.core(%tile46_4) {
    AIE.useLock(%lock46_5, "Acquire", 1)
    AIE.useLock(%lock46_4, "Acquire", 0)
    func.call @do_sieve(%buf46_5, %buf46_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_5, "Release", 0)
    AIE.useLock(%lock46_4, "Release", 1)
    AIE.end
  }

  %core46_3 = AIE.core(%tile46_3) {
    AIE.useLock(%lock46_4, "Acquire", 1)
    AIE.useLock(%lock46_3, "Acquire", 0)
    func.call @do_sieve(%buf46_4, %buf46_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_4, "Release", 0)
    AIE.useLock(%lock46_3, "Release", 1)
    AIE.end
  }

  %core46_2 = AIE.core(%tile46_2) {
    AIE.useLock(%lock46_3, "Acquire", 1)
    AIE.useLock(%lock46_2, "Acquire", 0)
    func.call @do_sieve(%buf46_3, %buf46_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_3, "Release", 0)
    AIE.useLock(%lock46_2, "Release", 1)
    AIE.end
  }

  %core46_1 = AIE.core(%tile46_1) {
    AIE.useLock(%lock46_2, "Acquire", 1)
    AIE.useLock(%lock46_1, "Acquire", 0)
    func.call @do_sieve(%buf46_2, %buf46_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_2, "Release", 0)
    AIE.useLock(%lock46_1, "Release", 1)
    AIE.end
  }

  %core47_1 = AIE.core(%tile47_1) {
    AIE.useLock(%lock46_1, "Acquire", 1)
    AIE.useLock(%lock47_1, "Acquire", 0)
    func.call @do_sieve(%buf46_1, %buf47_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock46_1, "Release", 0)
    AIE.useLock(%lock47_1, "Release", 1)
    AIE.end
  }

  %core47_2 = AIE.core(%tile47_2) {
    AIE.useLock(%lock47_1, "Acquire", 1)
    AIE.useLock(%lock47_2, "Acquire", 0)
    func.call @do_sieve(%buf47_1, %buf47_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_1, "Release", 0)
    AIE.useLock(%lock47_2, "Release", 1)
    AIE.end
  }

  %core47_3 = AIE.core(%tile47_3) {
    AIE.useLock(%lock47_2, "Acquire", 1)
    AIE.useLock(%lock47_3, "Acquire", 0)
    func.call @do_sieve(%buf47_2, %buf47_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_2, "Release", 0)
    AIE.useLock(%lock47_3, "Release", 1)
    AIE.end
  }

  %core47_4 = AIE.core(%tile47_4) {
    AIE.useLock(%lock47_3, "Acquire", 1)
    AIE.useLock(%lock47_4, "Acquire", 0)
    func.call @do_sieve(%buf47_3, %buf47_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_3, "Release", 0)
    AIE.useLock(%lock47_4, "Release", 1)
    AIE.end
  }

  %core47_5 = AIE.core(%tile47_5) {
    AIE.useLock(%lock47_4, "Acquire", 1)
    AIE.useLock(%lock47_5, "Acquire", 0)
    func.call @do_sieve(%buf47_4, %buf47_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_4, "Release", 0)
    AIE.useLock(%lock47_5, "Release", 1)
    AIE.end
  }

  %core47_6 = AIE.core(%tile47_6) {
    AIE.useLock(%lock47_5, "Acquire", 1)
    AIE.useLock(%lock47_6, "Acquire", 0)
    func.call @do_sieve(%buf47_5, %buf47_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_5, "Release", 0)
    AIE.useLock(%lock47_6, "Release", 1)
    AIE.end
  }

  %core47_7 = AIE.core(%tile47_7) {
    AIE.useLock(%lock47_6, "Acquire", 1)
    AIE.useLock(%lock47_7, "Acquire", 0)
    func.call @do_sieve(%buf47_6, %buf47_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_6, "Release", 0)
    AIE.useLock(%lock47_7, "Release", 1)
    AIE.end
  }

  %core48_7 = AIE.core(%tile48_7) {
    AIE.useLock(%lock47_7, "Acquire", 1)
    AIE.useLock(%lock48_7, "Acquire", 0)
    func.call @do_sieve(%buf47_7, %buf48_7) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock47_7, "Release", 0)
    AIE.useLock(%lock48_7, "Release", 1)
    AIE.end
  }

  %core48_6 = AIE.core(%tile48_6) {
    AIE.useLock(%lock48_7, "Acquire", 1)
    AIE.useLock(%lock48_6, "Acquire", 0)
    func.call @do_sieve(%buf48_7, %buf48_6) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock48_7, "Release", 0)
    AIE.useLock(%lock48_6, "Release", 1)
    AIE.end
  }

  %core48_5 = AIE.core(%tile48_5) {
    AIE.useLock(%lock48_6, "Acquire", 1)
    AIE.useLock(%lock48_5, "Acquire", 0)
    func.call @do_sieve(%buf48_6, %buf48_5) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock48_6, "Release", 0)
    AIE.useLock(%lock48_5, "Release", 1)
    AIE.end
  }

  %core48_4 = AIE.core(%tile48_4) {
    AIE.useLock(%lock48_5, "Acquire", 1)
    AIE.useLock(%lock48_4, "Acquire", 0)
    func.call @do_sieve(%buf48_5, %buf48_4) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock48_5, "Release", 0)
    AIE.useLock(%lock48_4, "Release", 1)
    AIE.end
  }

  %core48_3 = AIE.core(%tile48_3) {
    AIE.useLock(%lock48_4, "Acquire", 1)
    AIE.useLock(%lock48_3, "Acquire", 0)
    func.call @do_sieve(%buf48_4, %buf48_3) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock48_4, "Release", 0)
    AIE.useLock(%lock48_3, "Release", 1)
    AIE.end
  }

  %core48_2 = AIE.core(%tile48_2) {
    AIE.useLock(%lock48_3, "Acquire", 1)
    AIE.useLock(%lock48_2, "Acquire", 0)
    func.call @do_sieve(%buf48_3, %buf48_2) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock48_3, "Release", 0)
    AIE.useLock(%lock48_2, "Release", 1)
    AIE.end
  }

  %core48_1 = AIE.core(%tile48_1) {
    AIE.useLock(%lock48_2, "Acquire", 1)
    AIE.useLock(%lock48_1, "Acquire", 0)
    func.call @do_sieve(%buf48_2, %buf48_1) : (memref<4096xi32>, memref<4096xi32>) -> ()
    AIE.useLock(%lock48_2, "Release", 0)
    AIE.useLock(%lock48_1, "Release", 1)
    AIE.end
  }

}
