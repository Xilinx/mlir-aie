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

// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test16_prime_sieve_large {
  %tile0_1 = aie.tile(0, 1)
  %tile0_2 = aie.tile(0, 2)
  %tile0_3 = aie.tile(0, 3)
  %tile0_4 = aie.tile(0, 4)
  %tile0_5 = aie.tile(0, 5)
  %tile0_6 = aie.tile(0, 6)
  %tile0_7 = aie.tile(0, 7)
  %tile0_8 = aie.tile(0, 8)
  %tile1_1 = aie.tile(1, 1)
  %tile1_2 = aie.tile(1, 2)
  %tile1_3 = aie.tile(1, 3)
  %tile1_4 = aie.tile(1, 4)
  %tile1_5 = aie.tile(1, 5)
  %tile1_6 = aie.tile(1, 6)
  %tile1_7 = aie.tile(1, 7)
  %tile1_8 = aie.tile(1, 8)
  %tile2_1 = aie.tile(2, 1)
  %tile2_2 = aie.tile(2, 2)
  %tile2_3 = aie.tile(2, 3)
  %tile2_4 = aie.tile(2, 4)
  %tile2_5 = aie.tile(2, 5)
  %tile2_6 = aie.tile(2, 6)
  %tile2_7 = aie.tile(2, 7)
  %tile2_8 = aie.tile(2, 8)
  %tile3_1 = aie.tile(3, 1)
  %tile3_2 = aie.tile(3, 2)
  %tile3_3 = aie.tile(3, 3)
  %tile3_4 = aie.tile(3, 4)
  %tile3_5 = aie.tile(3, 5)
  %tile3_6 = aie.tile(3, 6)
  %tile3_7 = aie.tile(3, 7)
  %tile3_8 = aie.tile(3, 8)
  %tile4_1 = aie.tile(4, 1)
  %tile4_2 = aie.tile(4, 2)
  %tile4_3 = aie.tile(4, 3)
  %tile4_4 = aie.tile(4, 4)
  %tile4_5 = aie.tile(4, 5)
  %tile4_6 = aie.tile(4, 6)
  %tile4_7 = aie.tile(4, 7)
  %tile4_8 = aie.tile(4, 8)
  %tile5_1 = aie.tile(5, 1)
  %tile5_2 = aie.tile(5, 2)
  %tile5_3 = aie.tile(5, 3)
  %tile5_4 = aie.tile(5, 4)
  %tile5_5 = aie.tile(5, 5)
  %tile5_6 = aie.tile(5, 6)
  %tile5_7 = aie.tile(5, 7)
  %tile5_8 = aie.tile(5, 8)
  %tile6_1 = aie.tile(6, 1)
  %tile6_2 = aie.tile(6, 2)
  %tile6_3 = aie.tile(6, 3)
  %tile6_4 = aie.tile(6, 4)
  %tile6_5 = aie.tile(6, 5)
  %tile6_6 = aie.tile(6, 6)
  %tile6_7 = aie.tile(6, 7)
  %tile6_8 = aie.tile(6, 8)
  %tile7_1 = aie.tile(7, 1)
  %tile7_2 = aie.tile(7, 2)
  %tile7_3 = aie.tile(7, 3)
  %tile7_4 = aie.tile(7, 4)
  %tile7_5 = aie.tile(7, 5)
  %tile7_6 = aie.tile(7, 6)
  %tile7_7 = aie.tile(7, 7)
  %tile7_8 = aie.tile(7, 8)
  %tile8_1 = aie.tile(8, 1)
  %tile8_2 = aie.tile(8, 2)
  %tile8_3 = aie.tile(8, 3)
  %tile8_4 = aie.tile(8, 4)
  %tile8_5 = aie.tile(8, 5)
  %tile8_6 = aie.tile(8, 6)
  %tile8_7 = aie.tile(8, 7)
  %tile8_8 = aie.tile(8, 8)
  %tile9_1 = aie.tile(9, 1)
  %tile9_2 = aie.tile(9, 2)
  %tile9_3 = aie.tile(9, 3)
  %tile9_4 = aie.tile(9, 4)
  %tile9_5 = aie.tile(9, 5)
  %tile9_6 = aie.tile(9, 6)
  %tile9_7 = aie.tile(9, 7)
  %tile9_8 = aie.tile(9, 8)
  %tile10_1 = aie.tile(10, 1)
  %tile10_2 = aie.tile(10, 2)
  %tile10_3 = aie.tile(10, 3)
  %tile10_4 = aie.tile(10, 4)
  %tile10_5 = aie.tile(10, 5)
  %tile10_6 = aie.tile(10, 6)
  %tile10_7 = aie.tile(10, 7)
  %tile10_8 = aie.tile(10, 8)
  %tile11_1 = aie.tile(11, 1)
  %tile11_2 = aie.tile(11, 2)
  %tile11_3 = aie.tile(11, 3)
  %tile11_4 = aie.tile(11, 4)
  %tile11_5 = aie.tile(11, 5)
  %tile11_6 = aie.tile(11, 6)
  %tile11_7 = aie.tile(11, 7)
  %tile11_8 = aie.tile(11, 8)
  %tile12_1 = aie.tile(12, 1)
  %tile12_2 = aie.tile(12, 2)
  %tile12_3 = aie.tile(12, 3)
  %tile12_4 = aie.tile(12, 4)
  %tile12_5 = aie.tile(12, 5)
  %tile12_6 = aie.tile(12, 6)
  %tile12_7 = aie.tile(12, 7)
  %tile12_8 = aie.tile(12, 8)
  %tile13_1 = aie.tile(13, 1)
  %tile13_2 = aie.tile(13, 2)
  %tile13_3 = aie.tile(13, 3)
  %tile13_4 = aie.tile(13, 4)
  %tile13_5 = aie.tile(13, 5)
  %tile13_6 = aie.tile(13, 6)
  %tile13_7 = aie.tile(13, 7)
  %tile13_8 = aie.tile(13, 8)
  %tile14_1 = aie.tile(14, 1)
  %tile14_2 = aie.tile(14, 2)
  %tile14_3 = aie.tile(14, 3)
  %tile14_4 = aie.tile(14, 4)
  %tile14_5 = aie.tile(14, 5)
  %tile14_6 = aie.tile(14, 6)
  %tile14_7 = aie.tile(14, 7)
  %tile14_8 = aie.tile(14, 8)
  %tile15_1 = aie.tile(15, 1)
  %tile15_2 = aie.tile(15, 2)
  %tile15_3 = aie.tile(15, 3)
  %tile15_4 = aie.tile(15, 4)
  %tile15_5 = aie.tile(15, 5)
  %tile15_6 = aie.tile(15, 6)
  %tile15_7 = aie.tile(15, 7)
  %tile15_8 = aie.tile(15, 8)
  %tile16_1 = aie.tile(16, 1)
  %tile16_2 = aie.tile(16, 2)
  %tile16_3 = aie.tile(16, 3)
  %tile16_4 = aie.tile(16, 4)
  %tile16_5 = aie.tile(16, 5)
  %tile16_6 = aie.tile(16, 6)
  %tile16_7 = aie.tile(16, 7)
  %tile16_8 = aie.tile(16, 8)
  %tile17_1 = aie.tile(17, 1)
  %tile17_2 = aie.tile(17, 2)
  %tile17_3 = aie.tile(17, 3)
  %tile17_4 = aie.tile(17, 4)
  %tile17_5 = aie.tile(17, 5)
  %tile17_6 = aie.tile(17, 6)
  %tile17_7 = aie.tile(17, 7)
  %tile17_8 = aie.tile(17, 8)
  %tile18_1 = aie.tile(18, 1)
  %tile18_2 = aie.tile(18, 2)
  %tile18_3 = aie.tile(18, 3)
  %tile18_4 = aie.tile(18, 4)
  %tile18_5 = aie.tile(18, 5)
  %tile18_6 = aie.tile(18, 6)
  %tile18_7 = aie.tile(18, 7)
  %tile18_8 = aie.tile(18, 8)
  %tile19_1 = aie.tile(19, 1)
  %tile19_2 = aie.tile(19, 2)
  %tile19_3 = aie.tile(19, 3)
  %tile19_4 = aie.tile(19, 4)
  %tile19_5 = aie.tile(19, 5)
  %tile19_6 = aie.tile(19, 6)
  %tile19_7 = aie.tile(19, 7)
  %tile19_8 = aie.tile(19, 8)
  %tile20_1 = aie.tile(20, 1)
  %tile20_2 = aie.tile(20, 2)
  %tile20_3 = aie.tile(20, 3)
  %tile20_4 = aie.tile(20, 4)
  %tile20_5 = aie.tile(20, 5)
  %tile20_6 = aie.tile(20, 6)
  %tile20_7 = aie.tile(20, 7)
  %tile20_8 = aie.tile(20, 8)
  %tile21_1 = aie.tile(21, 1)
  %tile21_2 = aie.tile(21, 2)
  %tile21_3 = aie.tile(21, 3)
  %tile21_4 = aie.tile(21, 4)
  %tile21_5 = aie.tile(21, 5)
  %tile21_6 = aie.tile(21, 6)
  %tile21_7 = aie.tile(21, 7)
  %tile21_8 = aie.tile(21, 8)
  %tile22_1 = aie.tile(22, 1)
  %tile22_2 = aie.tile(22, 2)
  %tile22_3 = aie.tile(22, 3)
  %tile22_4 = aie.tile(22, 4)
  %tile22_5 = aie.tile(22, 5)
  %tile22_6 = aie.tile(22, 6)
  %tile22_7 = aie.tile(22, 7)
  %tile22_8 = aie.tile(22, 8)
  %tile23_1 = aie.tile(23, 1)
  %tile23_2 = aie.tile(23, 2)
  %tile23_3 = aie.tile(23, 3)
  %tile23_4 = aie.tile(23, 4)
  %tile23_5 = aie.tile(23, 5)
  %tile23_6 = aie.tile(23, 6)
  %tile23_7 = aie.tile(23, 7)
  %tile23_8 = aie.tile(23, 8)
  %tile24_1 = aie.tile(24, 1)
  %tile24_2 = aie.tile(24, 2)
  %tile24_3 = aie.tile(24, 3)
  %tile24_4 = aie.tile(24, 4)
  %tile24_5 = aie.tile(24, 5)
  %tile24_6 = aie.tile(24, 6)
  %tile24_7 = aie.tile(24, 7)
  %tile24_8 = aie.tile(24, 8)
  %tile25_1 = aie.tile(25, 1)
  %tile25_2 = aie.tile(25, 2)
  %tile25_3 = aie.tile(25, 3)
  %tile25_4 = aie.tile(25, 4)
  %tile25_5 = aie.tile(25, 5)
  %tile25_6 = aie.tile(25, 6)
  %tile25_7 = aie.tile(25, 7)
  %tile25_8 = aie.tile(25, 8)
  %tile26_1 = aie.tile(26, 1)
  %tile26_2 = aie.tile(26, 2)
  %tile26_3 = aie.tile(26, 3)
  %tile26_4 = aie.tile(26, 4)
  %tile26_5 = aie.tile(26, 5)
  %tile26_6 = aie.tile(26, 6)
  %tile26_7 = aie.tile(26, 7)
  %tile26_8 = aie.tile(26, 8)
  %tile27_1 = aie.tile(27, 1)
  %tile27_2 = aie.tile(27, 2)
  %tile27_3 = aie.tile(27, 3)
  %tile27_4 = aie.tile(27, 4)
  %tile27_5 = aie.tile(27, 5)
  %tile27_6 = aie.tile(27, 6)
  %tile27_7 = aie.tile(27, 7)
  %tile27_8 = aie.tile(27, 8)
  %tile28_1 = aie.tile(28, 1)
  %tile28_2 = aie.tile(28, 2)
  %tile28_3 = aie.tile(28, 3)
  %tile28_4 = aie.tile(28, 4)
  %tile28_5 = aie.tile(28, 5)
  %tile28_6 = aie.tile(28, 6)
  %tile28_7 = aie.tile(28, 7)
  %tile28_8 = aie.tile(28, 8)
  %tile29_1 = aie.tile(29, 1)
  %tile29_2 = aie.tile(29, 2)
  %tile29_3 = aie.tile(29, 3)
  %tile29_4 = aie.tile(29, 4)
  %tile29_5 = aie.tile(29, 5)
  %tile29_6 = aie.tile(29, 6)
  %tile29_7 = aie.tile(29, 7)
  %tile29_8 = aie.tile(29, 8)
  %tile30_1 = aie.tile(30, 1)
  %tile30_2 = aie.tile(30, 2)
  %tile30_3 = aie.tile(30, 3)
  %tile30_4 = aie.tile(30, 4)
  %tile30_5 = aie.tile(30, 5)
  %tile30_6 = aie.tile(30, 6)
  %tile30_7 = aie.tile(30, 7)
  %tile30_8 = aie.tile(30, 8)
  %tile31_1 = aie.tile(31, 1)
  %tile31_2 = aie.tile(31, 2)
  %tile31_3 = aie.tile(31, 3)
  %tile31_4 = aie.tile(31, 4)
  %tile31_5 = aie.tile(31, 5)
  %tile31_6 = aie.tile(31, 6)
  %tile31_7 = aie.tile(31, 7)
  %tile31_8 = aie.tile(31, 8)
  %tile32_1 = aie.tile(32, 1)
  %tile32_2 = aie.tile(32, 2)
  %tile32_3 = aie.tile(32, 3)
  %tile32_4 = aie.tile(32, 4)
  %tile32_5 = aie.tile(32, 5)
  %tile32_6 = aie.tile(32, 6)
  %tile32_7 = aie.tile(32, 7)
  %tile32_8 = aie.tile(32, 8)
  %tile33_1 = aie.tile(33, 1)
  %tile33_2 = aie.tile(33, 2)
  %tile33_3 = aie.tile(33, 3)
  %tile33_4 = aie.tile(33, 4)
  %tile33_5 = aie.tile(33, 5)
  %tile33_6 = aie.tile(33, 6)
  %tile33_7 = aie.tile(33, 7)
  %tile33_8 = aie.tile(33, 8)
  %tile34_1 = aie.tile(34, 1)
  %tile34_2 = aie.tile(34, 2)
  %tile34_3 = aie.tile(34, 3)
  %tile34_4 = aie.tile(34, 4)
  %tile34_5 = aie.tile(34, 5)
  %tile34_6 = aie.tile(34, 6)
  %tile34_7 = aie.tile(34, 7)
  %tile34_8 = aie.tile(34, 8)
  %tile35_1 = aie.tile(35, 1)
  %tile35_2 = aie.tile(35, 2)
  %tile35_3 = aie.tile(35, 3)
  %tile35_4 = aie.tile(35, 4)
  %tile35_5 = aie.tile(35, 5)
  %tile35_6 = aie.tile(35, 6)
  %tile35_7 = aie.tile(35, 7)
  %tile35_8 = aie.tile(35, 8)
  %tile36_1 = aie.tile(36, 1)
  %tile36_2 = aie.tile(36, 2)
  %tile36_3 = aie.tile(36, 3)
  %tile36_4 = aie.tile(36, 4)
  %tile36_5 = aie.tile(36, 5)
  %tile36_6 = aie.tile(36, 6)
  %tile36_7 = aie.tile(36, 7)
  %tile36_8 = aie.tile(36, 8)
  %tile37_1 = aie.tile(37, 1)
  %tile37_2 = aie.tile(37, 2)
  %tile37_3 = aie.tile(37, 3)
  %tile37_4 = aie.tile(37, 4)
  %tile37_5 = aie.tile(37, 5)
  %tile37_6 = aie.tile(37, 6)
  %tile37_7 = aie.tile(37, 7)
  %tile37_8 = aie.tile(37, 8)
  %tile38_1 = aie.tile(38, 1)
  %tile38_2 = aie.tile(38, 2)
  %tile38_3 = aie.tile(38, 3)
  %tile38_4 = aie.tile(38, 4)
  %tile38_5 = aie.tile(38, 5)
  %tile38_6 = aie.tile(38, 6)
  %tile38_7 = aie.tile(38, 7)
  %tile38_8 = aie.tile(38, 8)
  %tile39_1 = aie.tile(39, 1)
  %tile39_2 = aie.tile(39, 2)
  %tile39_3 = aie.tile(39, 3)
  %tile39_4 = aie.tile(39, 4)
  %tile39_5 = aie.tile(39, 5)
  %tile39_6 = aie.tile(39, 6)
  %tile39_7 = aie.tile(39, 7)
  %tile39_8 = aie.tile(39, 8)
  %tile40_1 = aie.tile(40, 1)
  %tile40_2 = aie.tile(40, 2)
  %tile40_3 = aie.tile(40, 3)
  %tile40_4 = aie.tile(40, 4)
  %tile40_5 = aie.tile(40, 5)
  %tile40_6 = aie.tile(40, 6)
  %tile40_7 = aie.tile(40, 7)
  %tile40_8 = aie.tile(40, 8)
  %tile41_1 = aie.tile(41, 1)
  %tile41_2 = aie.tile(41, 2)
  %tile41_3 = aie.tile(41, 3)
  %tile41_4 = aie.tile(41, 4)
  %tile41_5 = aie.tile(41, 5)
  %tile41_6 = aie.tile(41, 6)
  %tile41_7 = aie.tile(41, 7)
  %tile41_8 = aie.tile(41, 8)
  %tile42_1 = aie.tile(42, 1)
  %tile42_2 = aie.tile(42, 2)
  %tile42_3 = aie.tile(42, 3)
  %tile42_4 = aie.tile(42, 4)
  %tile42_5 = aie.tile(42, 5)
  %tile42_6 = aie.tile(42, 6)
  %tile42_7 = aie.tile(42, 7)
  %tile42_8 = aie.tile(42, 8)
  %tile43_1 = aie.tile(43, 1)
  %tile43_2 = aie.tile(43, 2)
  %tile43_3 = aie.tile(43, 3)
  %tile43_4 = aie.tile(43, 4)
  %tile43_5 = aie.tile(43, 5)
  %tile43_6 = aie.tile(43, 6)
  %tile43_7 = aie.tile(43, 7)
  %tile43_8 = aie.tile(43, 8)
  %tile44_1 = aie.tile(44, 1)
  %tile44_2 = aie.tile(44, 2)
  %tile44_3 = aie.tile(44, 3)
  %tile44_4 = aie.tile(44, 4)
  %tile44_5 = aie.tile(44, 5)
  %tile44_6 = aie.tile(44, 6)
  %tile44_7 = aie.tile(44, 7)
  %tile44_8 = aie.tile(44, 8)
  %tile45_1 = aie.tile(45, 1)
  %tile45_2 = aie.tile(45, 2)
  %tile45_3 = aie.tile(45, 3)
  %tile45_4 = aie.tile(45, 4)
  %tile45_5 = aie.tile(45, 5)
  %tile45_6 = aie.tile(45, 6)
  %tile45_7 = aie.tile(45, 7)
  %tile45_8 = aie.tile(45, 8)
  %tile46_1 = aie.tile(46, 1)
  %tile46_2 = aie.tile(46, 2)
  %tile46_3 = aie.tile(46, 3)
  %tile46_4 = aie.tile(46, 4)
  %tile46_5 = aie.tile(46, 5)
  %tile46_6 = aie.tile(46, 6)
  %tile46_7 = aie.tile(46, 7)
  %tile46_8 = aie.tile(46, 8)
  %tile47_1 = aie.tile(47, 1)
  %tile47_2 = aie.tile(47, 2)
  %tile47_3 = aie.tile(47, 3)
  %tile47_4 = aie.tile(47, 4)
  %tile47_5 = aie.tile(47, 5)
  %tile47_6 = aie.tile(47, 6)
  %tile47_7 = aie.tile(47, 7)
  %tile47_8 = aie.tile(47, 8)
  %tile48_1 = aie.tile(48, 1)
  %tile48_2 = aie.tile(48, 2)
  %tile48_3 = aie.tile(48, 3)
  %tile48_4 = aie.tile(48, 4)
  %tile48_5 = aie.tile(48, 5)
  %tile48_6 = aie.tile(48, 6)
  %tile48_7 = aie.tile(48, 7)
  %tile48_8 = aie.tile(48, 8)
  %tile49_1 = aie.tile(49, 1)
  %tile49_2 = aie.tile(49, 2)
  %tile49_3 = aie.tile(49, 3)
  %tile49_4 = aie.tile(49, 4)
  %tile49_5 = aie.tile(49, 5)
  %tile49_6 = aie.tile(49, 6)
  %tile49_7 = aie.tile(49, 7)
  %tile49_8 = aie.tile(49, 8)

  %lock0_1 = aie.lock(%tile0_1)
  %lock0_2 = aie.lock(%tile0_2)
  %lock0_3 = aie.lock(%tile0_3)
  %lock0_4 = aie.lock(%tile0_4)
  %lock0_5 = aie.lock(%tile0_5)
  %lock0_6 = aie.lock(%tile0_6)
  %lock0_7 = aie.lock(%tile0_7)
  %lock0_8 = aie.lock(%tile1_8)
  %lock1_8 = aie.lock(%tile1_8)
  %lock1_7 = aie.lock(%tile1_7)
  %lock1_6 = aie.lock(%tile1_6)
  %lock1_5 = aie.lock(%tile1_5)
  %lock1_4 = aie.lock(%tile1_4)
  %lock1_3 = aie.lock(%tile1_3)
  %lock1_2 = aie.lock(%tile1_2)
  %lock1_1 = aie.lock(%tile1_1)
  %lock2_1 = aie.lock(%tile2_1)
  %lock2_2 = aie.lock(%tile2_2)
  %lock2_3 = aie.lock(%tile2_3)
  %lock2_4 = aie.lock(%tile2_4)
  %lock2_5 = aie.lock(%tile2_5)
  %lock2_6 = aie.lock(%tile2_6)
  %lock2_7 = aie.lock(%tile2_7)
  %lock2_8 = aie.lock(%tile3_8)
  %lock3_8 = aie.lock(%tile3_8)
  %lock3_7 = aie.lock(%tile3_7)
  %lock3_6 = aie.lock(%tile3_6)
  %lock3_5 = aie.lock(%tile3_5)
  %lock3_4 = aie.lock(%tile3_4)
  %lock3_3 = aie.lock(%tile3_3)
  %lock3_2 = aie.lock(%tile3_2)
  %lock3_1 = aie.lock(%tile3_1)
  %lock4_1 = aie.lock(%tile4_1)
  %lock4_2 = aie.lock(%tile4_2)
  %lock4_3 = aie.lock(%tile4_3)
  %lock4_4 = aie.lock(%tile4_4)
  %lock4_5 = aie.lock(%tile4_5)
  %lock4_6 = aie.lock(%tile4_6)
  %lock4_7 = aie.lock(%tile4_7)
  %lock4_8 = aie.lock(%tile5_8)
  %lock5_8 = aie.lock(%tile5_8)
  %lock5_7 = aie.lock(%tile5_7)
  %lock5_6 = aie.lock(%tile5_6)
  %lock5_5 = aie.lock(%tile5_5)
  %lock5_4 = aie.lock(%tile5_4)
  %lock5_3 = aie.lock(%tile5_3)
  %lock5_2 = aie.lock(%tile5_2)
  %lock5_1 = aie.lock(%tile5_1)
  %lock6_1 = aie.lock(%tile6_1)
  %lock6_2 = aie.lock(%tile6_2)
  %lock6_3 = aie.lock(%tile6_3)
  %lock6_4 = aie.lock(%tile6_4)
  %lock6_5 = aie.lock(%tile6_5)
  %lock6_6 = aie.lock(%tile6_6)
  %lock6_7 = aie.lock(%tile6_7)
  %lock6_8 = aie.lock(%tile7_8)
  %lock7_8 = aie.lock(%tile7_8)
  %lock7_7 = aie.lock(%tile7_7)
  %lock7_6 = aie.lock(%tile7_6)
  %lock7_5 = aie.lock(%tile7_5)
  %lock7_4 = aie.lock(%tile7_4)
  %lock7_3 = aie.lock(%tile7_3)
  %lock7_2 = aie.lock(%tile7_2)
  %lock7_1 = aie.lock(%tile7_1)
  %lock8_1 = aie.lock(%tile8_1)
  %lock8_2 = aie.lock(%tile8_2)
  %lock8_3 = aie.lock(%tile8_3)
  %lock8_4 = aie.lock(%tile8_4)
  %lock8_5 = aie.lock(%tile8_5)
  %lock8_6 = aie.lock(%tile8_6)
  %lock8_7 = aie.lock(%tile8_7)
  %lock8_8 = aie.lock(%tile9_8)
  %lock9_8 = aie.lock(%tile9_8)
  %lock9_7 = aie.lock(%tile9_7)
  %lock9_6 = aie.lock(%tile9_6)
  %lock9_5 = aie.lock(%tile9_5)
  %lock9_4 = aie.lock(%tile9_4)
  %lock9_3 = aie.lock(%tile9_3)
  %lock9_2 = aie.lock(%tile9_2)
  %lock9_1 = aie.lock(%tile9_1)
  %lock10_1 = aie.lock(%tile10_1)
  %lock10_2 = aie.lock(%tile10_2)
  %lock10_3 = aie.lock(%tile10_3)
  %lock10_4 = aie.lock(%tile10_4)
  %lock10_5 = aie.lock(%tile10_5)
  %lock10_6 = aie.lock(%tile10_6)
  %lock10_7 = aie.lock(%tile10_7)
  %lock10_8 = aie.lock(%tile11_8)
  %lock11_8 = aie.lock(%tile11_8)
  %lock11_7 = aie.lock(%tile11_7)
  %lock11_6 = aie.lock(%tile11_6)
  %lock11_5 = aie.lock(%tile11_5)
  %lock11_4 = aie.lock(%tile11_4)
  %lock11_3 = aie.lock(%tile11_3)
  %lock11_2 = aie.lock(%tile11_2)
  %lock11_1 = aie.lock(%tile11_1)
  %lock12_1 = aie.lock(%tile12_1)
  %lock12_2 = aie.lock(%tile12_2)
  %lock12_3 = aie.lock(%tile12_3)
  %lock12_4 = aie.lock(%tile12_4)
  %lock12_5 = aie.lock(%tile12_5)
  %lock12_6 = aie.lock(%tile12_6)
  %lock12_7 = aie.lock(%tile12_7)
  %lock12_8 = aie.lock(%tile13_8)
  %lock13_8 = aie.lock(%tile13_8)
  %lock13_7 = aie.lock(%tile13_7)
  %lock13_6 = aie.lock(%tile13_6)
  %lock13_5 = aie.lock(%tile13_5)
  %lock13_4 = aie.lock(%tile13_4)
  %lock13_3 = aie.lock(%tile13_3)
  %lock13_2 = aie.lock(%tile13_2)
  %lock13_1 = aie.lock(%tile13_1)
  %lock14_1 = aie.lock(%tile14_1)
  %lock14_2 = aie.lock(%tile14_2)
  %lock14_3 = aie.lock(%tile14_3)
  %lock14_4 = aie.lock(%tile14_4)
  %lock14_5 = aie.lock(%tile14_5)
  %lock14_6 = aie.lock(%tile14_6)
  %lock14_7 = aie.lock(%tile14_7)
  %lock14_8 = aie.lock(%tile15_8)
  %lock15_8 = aie.lock(%tile15_8)
  %lock15_7 = aie.lock(%tile15_7)
  %lock15_6 = aie.lock(%tile15_6)
  %lock15_5 = aie.lock(%tile15_5)
  %lock15_4 = aie.lock(%tile15_4)
  %lock15_3 = aie.lock(%tile15_3)
  %lock15_2 = aie.lock(%tile15_2)
  %lock15_1 = aie.lock(%tile15_1)
  %lock16_1 = aie.lock(%tile16_1)
  %lock16_2 = aie.lock(%tile16_2)
  %lock16_3 = aie.lock(%tile16_3)
  %lock16_4 = aie.lock(%tile16_4)
  %lock16_5 = aie.lock(%tile16_5)
  %lock16_6 = aie.lock(%tile16_6)
  %lock16_7 = aie.lock(%tile16_7)
  %lock16_8 = aie.lock(%tile17_8)
  %lock17_8 = aie.lock(%tile17_8)
  %lock17_7 = aie.lock(%tile17_7)
  %lock17_6 = aie.lock(%tile17_6)
  %lock17_5 = aie.lock(%tile17_5)
  %lock17_4 = aie.lock(%tile17_4)
  %lock17_3 = aie.lock(%tile17_3)
  %lock17_2 = aie.lock(%tile17_2)
  %lock17_1 = aie.lock(%tile17_1)
  %lock18_1 = aie.lock(%tile18_1)
  %lock18_2 = aie.lock(%tile18_2)
  %lock18_3 = aie.lock(%tile18_3)
  %lock18_4 = aie.lock(%tile18_4)
  %lock18_5 = aie.lock(%tile18_5)
  %lock18_6 = aie.lock(%tile18_6)
  %lock18_7 = aie.lock(%tile18_7)
  %lock18_8 = aie.lock(%tile19_8)
  %lock19_8 = aie.lock(%tile19_8)
  %lock19_7 = aie.lock(%tile19_7)
  %lock19_6 = aie.lock(%tile19_6)
  %lock19_5 = aie.lock(%tile19_5)
  %lock19_4 = aie.lock(%tile19_4)
  %lock19_3 = aie.lock(%tile19_3)
  %lock19_2 = aie.lock(%tile19_2)
  %lock19_1 = aie.lock(%tile19_1)
  %lock20_1 = aie.lock(%tile20_1)
  %lock20_2 = aie.lock(%tile20_2)
  %lock20_3 = aie.lock(%tile20_3)
  %lock20_4 = aie.lock(%tile20_4)
  %lock20_5 = aie.lock(%tile20_5)
  %lock20_6 = aie.lock(%tile20_6)
  %lock20_7 = aie.lock(%tile20_7)
  %lock20_8 = aie.lock(%tile21_8)
  %lock21_8 = aie.lock(%tile21_8)
  %lock21_7 = aie.lock(%tile21_7)
  %lock21_6 = aie.lock(%tile21_6)
  %lock21_5 = aie.lock(%tile21_5)
  %lock21_4 = aie.lock(%tile21_4)
  %lock21_3 = aie.lock(%tile21_3)
  %lock21_2 = aie.lock(%tile21_2)
  %lock21_1 = aie.lock(%tile21_1)
  %lock22_1 = aie.lock(%tile22_1)
  %lock22_2 = aie.lock(%tile22_2)
  %lock22_3 = aie.lock(%tile22_3)
  %lock22_4 = aie.lock(%tile22_4)
  %lock22_5 = aie.lock(%tile22_5)
  %lock22_6 = aie.lock(%tile22_6)
  %lock22_7 = aie.lock(%tile22_7)
  %lock22_8 = aie.lock(%tile23_8)
  %lock23_8 = aie.lock(%tile23_8)
  %lock23_7 = aie.lock(%tile23_7)
  %lock23_6 = aie.lock(%tile23_6)
  %lock23_5 = aie.lock(%tile23_5)
  %lock23_4 = aie.lock(%tile23_4)
  %lock23_3 = aie.lock(%tile23_3)
  %lock23_2 = aie.lock(%tile23_2)
  %lock23_1 = aie.lock(%tile23_1)
  %lock24_1 = aie.lock(%tile24_1)
  %lock24_2 = aie.lock(%tile24_2)
  %lock24_3 = aie.lock(%tile24_3)
  %lock24_4 = aie.lock(%tile24_4)
  %lock24_5 = aie.lock(%tile24_5)
  %lock24_6 = aie.lock(%tile24_6)
  %lock24_7 = aie.lock(%tile24_7)
  %lock24_8 = aie.lock(%tile25_8)
  %lock25_8 = aie.lock(%tile25_8)
  %lock25_7 = aie.lock(%tile25_7)
  %lock25_6 = aie.lock(%tile25_6)
  %lock25_5 = aie.lock(%tile25_5)
  %lock25_4 = aie.lock(%tile25_4)
  %lock25_3 = aie.lock(%tile25_3)
  %lock25_2 = aie.lock(%tile25_2)
  %lock25_1 = aie.lock(%tile25_1)
  %lock26_1 = aie.lock(%tile26_1)
  %lock26_2 = aie.lock(%tile26_2)
  %lock26_3 = aie.lock(%tile26_3)
  %lock26_4 = aie.lock(%tile26_4)
  %lock26_5 = aie.lock(%tile26_5)
  %lock26_6 = aie.lock(%tile26_6)
  %lock26_7 = aie.lock(%tile26_7)
  %lock26_8 = aie.lock(%tile27_8)
  %lock27_8 = aie.lock(%tile27_8)
  %lock27_7 = aie.lock(%tile27_7)
  %lock27_6 = aie.lock(%tile27_6)
  %lock27_5 = aie.lock(%tile27_5)
  %lock27_4 = aie.lock(%tile27_4)
  %lock27_3 = aie.lock(%tile27_3)
  %lock27_2 = aie.lock(%tile27_2)
  %lock27_1 = aie.lock(%tile27_1)
  %lock28_1 = aie.lock(%tile28_1)
  %lock28_2 = aie.lock(%tile28_2)
  %lock28_3 = aie.lock(%tile28_3)
  %lock28_4 = aie.lock(%tile28_4)
  %lock28_5 = aie.lock(%tile28_5)
  %lock28_6 = aie.lock(%tile28_6)
  %lock28_7 = aie.lock(%tile28_7)
  %lock28_8 = aie.lock(%tile29_8)
  %lock29_8 = aie.lock(%tile29_8)
  %lock29_7 = aie.lock(%tile29_7)
  %lock29_6 = aie.lock(%tile29_6)
  %lock29_5 = aie.lock(%tile29_5)
  %lock29_4 = aie.lock(%tile29_4)
  %lock29_3 = aie.lock(%tile29_3)
  %lock29_2 = aie.lock(%tile29_2)
  %lock29_1 = aie.lock(%tile29_1)
  %lock30_1 = aie.lock(%tile30_1)
  %lock30_2 = aie.lock(%tile30_2)
  %lock30_3 = aie.lock(%tile30_3)
  %lock30_4 = aie.lock(%tile30_4)
  %lock30_5 = aie.lock(%tile30_5)
  %lock30_6 = aie.lock(%tile30_6)
  %lock30_7 = aie.lock(%tile30_7)
  %lock30_8 = aie.lock(%tile31_8)
  %lock31_8 = aie.lock(%tile31_8)
  %lock31_7 = aie.lock(%tile31_7)
  %lock31_6 = aie.lock(%tile31_6)
  %lock31_5 = aie.lock(%tile31_5)
  %lock31_4 = aie.lock(%tile31_4)
  %lock31_3 = aie.lock(%tile31_3)
  %lock31_2 = aie.lock(%tile31_2)
  %lock31_1 = aie.lock(%tile31_1)
  %lock32_1 = aie.lock(%tile32_1)
  %lock32_2 = aie.lock(%tile32_2)
  %lock32_3 = aie.lock(%tile32_3)
  %lock32_4 = aie.lock(%tile32_4)
  %lock32_5 = aie.lock(%tile32_5)
  %lock32_6 = aie.lock(%tile32_6)
  %lock32_7 = aie.lock(%tile32_7)
  %lock32_8 = aie.lock(%tile33_8)
  %lock33_8 = aie.lock(%tile33_8)
  %lock33_7 = aie.lock(%tile33_7)
  %lock33_6 = aie.lock(%tile33_6)
  %lock33_5 = aie.lock(%tile33_5)
  %lock33_4 = aie.lock(%tile33_4)
  %lock33_3 = aie.lock(%tile33_3)
  %lock33_2 = aie.lock(%tile33_2)
  %lock33_1 = aie.lock(%tile33_1)
  %lock34_1 = aie.lock(%tile34_1)
  %lock34_2 = aie.lock(%tile34_2)
  %lock34_3 = aie.lock(%tile34_3)
  %lock34_4 = aie.lock(%tile34_4)
  %lock34_5 = aie.lock(%tile34_5)
  %lock34_6 = aie.lock(%tile34_6)
  %lock34_7 = aie.lock(%tile34_7)
  %lock34_8 = aie.lock(%tile35_8)
  %lock35_8 = aie.lock(%tile35_8)
  %lock35_7 = aie.lock(%tile35_7)
  %lock35_6 = aie.lock(%tile35_6)
  %lock35_5 = aie.lock(%tile35_5)
  %lock35_4 = aie.lock(%tile35_4)
  %lock35_3 = aie.lock(%tile35_3)
  %lock35_2 = aie.lock(%tile35_2)
  %lock35_1 = aie.lock(%tile35_1)
  %lock36_1 = aie.lock(%tile36_1)
  %lock36_2 = aie.lock(%tile36_2)
  %lock36_3 = aie.lock(%tile36_3)
  %lock36_4 = aie.lock(%tile36_4)
  %lock36_5 = aie.lock(%tile36_5)
  %lock36_6 = aie.lock(%tile36_6)
  %lock36_7 = aie.lock(%tile36_7)
  %lock36_8 = aie.lock(%tile37_8)
  %lock37_8 = aie.lock(%tile37_8)
  %lock37_7 = aie.lock(%tile37_7)
  %lock37_6 = aie.lock(%tile37_6)
  %lock37_5 = aie.lock(%tile37_5)
  %lock37_4 = aie.lock(%tile37_4)
  %lock37_3 = aie.lock(%tile37_3)
  %lock37_2 = aie.lock(%tile37_2)
  %lock37_1 = aie.lock(%tile37_1)
  %lock38_1 = aie.lock(%tile38_1)
  %lock38_2 = aie.lock(%tile38_2)
  %lock38_3 = aie.lock(%tile38_3)
  %lock38_4 = aie.lock(%tile38_4)
  %lock38_5 = aie.lock(%tile38_5)
  %lock38_6 = aie.lock(%tile38_6)
  %lock38_7 = aie.lock(%tile38_7)
  %lock38_8 = aie.lock(%tile39_8)
  %lock39_8 = aie.lock(%tile39_8)
  %lock39_7 = aie.lock(%tile39_7)
  %lock39_6 = aie.lock(%tile39_6)
  %lock39_5 = aie.lock(%tile39_5)
  %lock39_4 = aie.lock(%tile39_4)
  %lock39_3 = aie.lock(%tile39_3)
  %lock39_2 = aie.lock(%tile39_2)
  %lock39_1 = aie.lock(%tile39_1)
  %lock40_1 = aie.lock(%tile40_1)
  %lock40_2 = aie.lock(%tile40_2)
  %lock40_3 = aie.lock(%tile40_3)
  %lock40_4 = aie.lock(%tile40_4)
  %lock40_5 = aie.lock(%tile40_5)
  %lock40_6 = aie.lock(%tile40_6)
  %lock40_7 = aie.lock(%tile40_7)
  %lock40_8 = aie.lock(%tile41_8)
  %lock41_8 = aie.lock(%tile41_8)
  %lock41_7 = aie.lock(%tile41_7)
  %lock41_6 = aie.lock(%tile41_6)
  %lock41_5 = aie.lock(%tile41_5)
  %lock41_4 = aie.lock(%tile41_4)
  %lock41_3 = aie.lock(%tile41_3)
  %lock41_2 = aie.lock(%tile41_2)
  %lock41_1 = aie.lock(%tile41_1)
  %lock42_1 = aie.lock(%tile42_1)
  %lock42_2 = aie.lock(%tile42_2)
  %lock42_3 = aie.lock(%tile42_3)
  %lock42_4 = aie.lock(%tile42_4)
  %lock42_5 = aie.lock(%tile42_5)
  %lock42_6 = aie.lock(%tile42_6)
  %lock42_7 = aie.lock(%tile42_7)
  %lock42_8 = aie.lock(%tile43_8)
  %lock43_8 = aie.lock(%tile43_8)
  %lock43_7 = aie.lock(%tile43_7)
  %lock43_6 = aie.lock(%tile43_6)
  %lock43_5 = aie.lock(%tile43_5)
  %lock43_4 = aie.lock(%tile43_4)
  %lock43_3 = aie.lock(%tile43_3)
  %lock43_2 = aie.lock(%tile43_2)
  %lock43_1 = aie.lock(%tile43_1)
  %lock44_1 = aie.lock(%tile44_1)
  %lock44_2 = aie.lock(%tile44_2)
  %lock44_3 = aie.lock(%tile44_3)
  %lock44_4 = aie.lock(%tile44_4)
  %lock44_5 = aie.lock(%tile44_5)
  %lock44_6 = aie.lock(%tile44_6)
  %lock44_7 = aie.lock(%tile44_7)
  %lock44_8 = aie.lock(%tile45_8)
  %lock45_8 = aie.lock(%tile45_8)
  %lock45_7 = aie.lock(%tile45_7)
  %lock45_6 = aie.lock(%tile45_6)
  %lock45_5 = aie.lock(%tile45_5)
  %lock45_4 = aie.lock(%tile45_4)
  %lock45_3 = aie.lock(%tile45_3)
  %lock45_2 = aie.lock(%tile45_2)
  %lock45_1 = aie.lock(%tile45_1)
  %lock46_1 = aie.lock(%tile46_1)
  %lock46_2 = aie.lock(%tile46_2)
  %lock46_3 = aie.lock(%tile46_3)
  %lock46_4 = aie.lock(%tile46_4)
  %lock46_5 = aie.lock(%tile46_5)
  %lock46_6 = aie.lock(%tile46_6)
  %lock46_7 = aie.lock(%tile46_7)
  %lock46_8 = aie.lock(%tile47_8)
  %lock47_8 = aie.lock(%tile47_8)
  %lock47_7 = aie.lock(%tile47_7)
  %lock47_6 = aie.lock(%tile47_6)
  %lock47_5 = aie.lock(%tile47_5)
  %lock47_4 = aie.lock(%tile47_4)
  %lock47_3 = aie.lock(%tile47_3)
  %lock47_2 = aie.lock(%tile47_2)
  %lock47_1 = aie.lock(%tile47_1)
  %lock48_1 = aie.lock(%tile48_1)
  %lock48_2 = aie.lock(%tile48_2)
  %lock48_3 = aie.lock(%tile48_3)
  %lock48_4 = aie.lock(%tile48_4)
  %lock48_5 = aie.lock(%tile48_5)
  %lock48_6 = aie.lock(%tile48_6)
  %lock48_7 = aie.lock(%tile48_7)
  %lock48_8 = aie.lock(%tile49_8)
  %lock49_8 = aie.lock(%tile49_8)
  %lock49_7 = aie.lock(%tile49_7)
  %lock49_6 = aie.lock(%tile49_6)
  %lock49_5 = aie.lock(%tile49_5)
  %lock49_4 = aie.lock(%tile49_4)
  %lock49_3 = aie.lock(%tile49_3)
  %lock49_2 = aie.lock(%tile49_2)
  %lock49_1 = aie.lock(%tile49_1) { sym_name = "prime_output_lock" }

  %buf0_1 = aie.buffer(%tile0_1) { sym_name = "a" } : memref<3072xi32>
  %buf0_2 = aie.buffer(%tile0_2) { sym_name = "prime2" } : memref<3072xi32>
  %buf0_3 = aie.buffer(%tile0_3) { sym_name = "prime3" } : memref<3072xi32>
  %buf0_4 = aie.buffer(%tile0_4) { sym_name = "prime5" } : memref<3072xi32>
  %buf0_5 = aie.buffer(%tile0_5) { sym_name = "prime7" } : memref<3072xi32>
  %buf0_6 = aie.buffer(%tile0_6) { sym_name = "prime11" } : memref<3072xi32>
  %buf0_7 = aie.buffer(%tile0_7) { sym_name = "prime13" } : memref<3072xi32>
  %buf0_8 = aie.buffer(%tile1_8) { sym_name = "prime17" } : memref<3072xi32>
  %buf1_8 = aie.buffer(%tile1_8) { sym_name = "prime19" } : memref<3072xi32>
  %buf1_7 = aie.buffer(%tile1_7) { sym_name = "prime23" } : memref<3072xi32>
  %buf1_6 = aie.buffer(%tile1_6) { sym_name = "prime29" } : memref<3072xi32>
  %buf1_5 = aie.buffer(%tile1_5) { sym_name = "prime31" } : memref<3072xi32>
  %buf1_4 = aie.buffer(%tile1_4) { sym_name = "prime37" } : memref<3072xi32>
  %buf1_3 = aie.buffer(%tile1_3) { sym_name = "prime41" } : memref<3072xi32>
  %buf1_2 = aie.buffer(%tile1_2) { sym_name = "prime43" } : memref<3072xi32>
  %buf1_1 = aie.buffer(%tile1_1) { sym_name = "prime47" } : memref<3072xi32>
  %buf2_1 = aie.buffer(%tile2_1) { sym_name = "prime53" } : memref<3072xi32>
  %buf2_2 = aie.buffer(%tile2_2) { sym_name = "prime59" } : memref<3072xi32>
  %buf2_3 = aie.buffer(%tile2_3) { sym_name = "prime61" } : memref<3072xi32>
  %buf2_4 = aie.buffer(%tile2_4) { sym_name = "prime67" } : memref<3072xi32>
  %buf2_5 = aie.buffer(%tile2_5) { sym_name = "prime71" } : memref<3072xi32>
  %buf2_6 = aie.buffer(%tile2_6) { sym_name = "prime73" } : memref<3072xi32>
  %buf2_7 = aie.buffer(%tile2_7) { sym_name = "prime79" } : memref<3072xi32>
  %buf2_8 = aie.buffer(%tile3_8) { sym_name = "prime83" } : memref<3072xi32>
  %buf3_8 = aie.buffer(%tile3_8) { sym_name = "prime89" } : memref<3072xi32>
  %buf3_7 = aie.buffer(%tile3_7) { sym_name = "prime97" } : memref<3072xi32>
  %buf3_6 = aie.buffer(%tile3_6) { sym_name = "prime101" } : memref<3072xi32>
  %buf3_5 = aie.buffer(%tile3_5) { sym_name = "prime103" } : memref<3072xi32>
  %buf3_4 = aie.buffer(%tile3_4) { sym_name = "prime107" } : memref<3072xi32>
  %buf3_3 = aie.buffer(%tile3_3) { sym_name = "prime109" } : memref<3072xi32>
  %buf3_2 = aie.buffer(%tile3_2) { sym_name = "prime113" } : memref<3072xi32>
  %buf3_1 = aie.buffer(%tile3_1) { sym_name = "prime127" } : memref<3072xi32>
  %buf4_1 = aie.buffer(%tile4_1) { sym_name = "prime131" } : memref<3072xi32>
  %buf4_2 = aie.buffer(%tile4_2) { sym_name = "prime137" } : memref<3072xi32>
  %buf4_3 = aie.buffer(%tile4_3) { sym_name = "prime139" } : memref<3072xi32>
  %buf4_4 = aie.buffer(%tile4_4) { sym_name = "prime149" } : memref<3072xi32>
  %buf4_5 = aie.buffer(%tile4_5) { sym_name = "prime151" } : memref<3072xi32>
  %buf4_6 = aie.buffer(%tile4_6) { sym_name = "prime157" } : memref<3072xi32>
  %buf4_7 = aie.buffer(%tile4_7) { sym_name = "prime163" } : memref<3072xi32>
  %buf4_8 = aie.buffer(%tile5_8) { sym_name = "prime167" } : memref<3072xi32>
  %buf5_8 = aie.buffer(%tile5_8) { sym_name = "prime173" } : memref<3072xi32>
  %buf5_7 = aie.buffer(%tile5_7) { sym_name = "prime179" } : memref<3072xi32>
  %buf5_6 = aie.buffer(%tile5_6) { sym_name = "prime181" } : memref<3072xi32>
  %buf5_5 = aie.buffer(%tile5_5) { sym_name = "prime191" } : memref<3072xi32>
  %buf5_4 = aie.buffer(%tile5_4) { sym_name = "prime193" } : memref<3072xi32>
  %buf5_3 = aie.buffer(%tile5_3) { sym_name = "prime197" } : memref<3072xi32>
  %buf5_2 = aie.buffer(%tile5_2) { sym_name = "prime199" } : memref<3072xi32>
  %buf5_1 = aie.buffer(%tile5_1) { sym_name = "prime211" } : memref<3072xi32>
  %buf6_1 = aie.buffer(%tile6_1) { sym_name = "prime223" } : memref<3072xi32>
  %buf6_2 = aie.buffer(%tile6_2) { sym_name = "prime227" } : memref<3072xi32>
  %buf6_3 = aie.buffer(%tile6_3) { sym_name = "prime229" } : memref<3072xi32>
  %buf6_4 = aie.buffer(%tile6_4) { sym_name = "prime233" } : memref<3072xi32>
  %buf6_5 = aie.buffer(%tile6_5) { sym_name = "prime239" } : memref<3072xi32>
  %buf6_6 = aie.buffer(%tile6_6) { sym_name = "prime241" } : memref<3072xi32>
  %buf6_7 = aie.buffer(%tile6_7) { sym_name = "prime251" } : memref<3072xi32>
  %buf6_8 = aie.buffer(%tile7_8) { sym_name = "prime257" } : memref<3072xi32>
  %buf7_8 = aie.buffer(%tile7_8) { sym_name = "prime263" } : memref<3072xi32>
  %buf7_7 = aie.buffer(%tile7_7) { sym_name = "prime269" } : memref<3072xi32>
  %buf7_6 = aie.buffer(%tile7_6) { sym_name = "prime271" } : memref<3072xi32>
  %buf7_5 = aie.buffer(%tile7_5) { sym_name = "prime277" } : memref<3072xi32>
  %buf7_4 = aie.buffer(%tile7_4) { sym_name = "prime281" } : memref<3072xi32>
  %buf7_3 = aie.buffer(%tile7_3) { sym_name = "prime283" } : memref<3072xi32>
  %buf7_2 = aie.buffer(%tile7_2) { sym_name = "prime293" } : memref<3072xi32>
  %buf7_1 = aie.buffer(%tile7_1) { sym_name = "prime307" } : memref<3072xi32>
  %buf8_1 = aie.buffer(%tile8_1) { sym_name = "prime311" } : memref<3072xi32>
  %buf8_2 = aie.buffer(%tile8_2) { sym_name = "prime313" } : memref<3072xi32>
  %buf8_3 = aie.buffer(%tile8_3) { sym_name = "prime317" } : memref<3072xi32>
  %buf8_4 = aie.buffer(%tile8_4) { sym_name = "prime331" } : memref<3072xi32>
  %buf8_5 = aie.buffer(%tile8_5) { sym_name = "prime337" } : memref<3072xi32>
  %buf8_6 = aie.buffer(%tile8_6) { sym_name = "prime347" } : memref<3072xi32>
  %buf8_7 = aie.buffer(%tile8_7) { sym_name = "prime349" } : memref<3072xi32>
  %buf8_8 = aie.buffer(%tile9_8) { sym_name = "prime353" } : memref<3072xi32>
  %buf9_8 = aie.buffer(%tile9_8) { sym_name = "prime359" } : memref<3072xi32>
  %buf9_7 = aie.buffer(%tile9_7) { sym_name = "prime367" } : memref<3072xi32>
  %buf9_6 = aie.buffer(%tile9_6) { sym_name = "prime373" } : memref<3072xi32>
  %buf9_5 = aie.buffer(%tile9_5) { sym_name = "prime379" } : memref<3072xi32>
  %buf9_4 = aie.buffer(%tile9_4) { sym_name = "prime383" } : memref<3072xi32>
  %buf9_3 = aie.buffer(%tile9_3) { sym_name = "prime389" } : memref<3072xi32>
  %buf9_2 = aie.buffer(%tile9_2) { sym_name = "prime397" } : memref<3072xi32>
  %buf9_1 = aie.buffer(%tile9_1) { sym_name = "prime401" } : memref<3072xi32>
  %buf10_1 = aie.buffer(%tile10_1) { sym_name = "prime409" } : memref<3072xi32>
  %buf10_2 = aie.buffer(%tile10_2) { sym_name = "prime419" } : memref<3072xi32>
  %buf10_3 = aie.buffer(%tile10_3) { sym_name = "prime421" } : memref<3072xi32>
  %buf10_4 = aie.buffer(%tile10_4) { sym_name = "prime431" } : memref<3072xi32>
  %buf10_5 = aie.buffer(%tile10_5) { sym_name = "prime433" } : memref<3072xi32>
  %buf10_6 = aie.buffer(%tile10_6) { sym_name = "prime439" } : memref<3072xi32>
  %buf10_7 = aie.buffer(%tile10_7) { sym_name = "prime443" } : memref<3072xi32>
  %buf10_8 = aie.buffer(%tile11_8) { sym_name = "prime449" } : memref<3072xi32>
  %buf11_8 = aie.buffer(%tile11_8) { sym_name = "prime457" } : memref<3072xi32>
  %buf11_7 = aie.buffer(%tile11_7) { sym_name = "prime461" } : memref<3072xi32>
  %buf11_6 = aie.buffer(%tile11_6) { sym_name = "prime463" } : memref<3072xi32>
  %buf11_5 = aie.buffer(%tile11_5) { sym_name = "prime467" } : memref<3072xi32>
  %buf11_4 = aie.buffer(%tile11_4) { sym_name = "prime479" } : memref<3072xi32>
  %buf11_3 = aie.buffer(%tile11_3) { sym_name = "prime487" } : memref<3072xi32>
  %buf11_2 = aie.buffer(%tile11_2) { sym_name = "prime491" } : memref<3072xi32>
  %buf11_1 = aie.buffer(%tile11_1) { sym_name = "prime499" } : memref<3072xi32>
  %buf12_1 = aie.buffer(%tile12_1) { sym_name = "prime503" } : memref<3072xi32>
  %buf12_2 = aie.buffer(%tile12_2) { sym_name = "prime509" } : memref<3072xi32>
  %buf12_3 = aie.buffer(%tile12_3) { sym_name = "prime521" } : memref<3072xi32>
  %buf12_4 = aie.buffer(%tile12_4) { sym_name = "prime523" } : memref<3072xi32>
  %buf12_5 = aie.buffer(%tile12_5) { sym_name = "prime541" } : memref<3072xi32>
  %buf12_6 = aie.buffer(%tile12_6) { sym_name = "prime547" } : memref<3072xi32>
  %buf12_7 = aie.buffer(%tile12_7) { sym_name = "prime557" } : memref<3072xi32>
  %buf12_8 = aie.buffer(%tile13_8) { sym_name = "prime563" } : memref<3072xi32>
  %buf13_8 = aie.buffer(%tile13_8) { sym_name = "prime569" } : memref<3072xi32>
  %buf13_7 = aie.buffer(%tile13_7) { sym_name = "prime571" } : memref<3072xi32>
  %buf13_6 = aie.buffer(%tile13_6) { sym_name = "prime577" } : memref<3072xi32>
  %buf13_5 = aie.buffer(%tile13_5) { sym_name = "prime587" } : memref<3072xi32>
  %buf13_4 = aie.buffer(%tile13_4) { sym_name = "prime593" } : memref<3072xi32>
  %buf13_3 = aie.buffer(%tile13_3) { sym_name = "prime599" } : memref<3072xi32>
  %buf13_2 = aie.buffer(%tile13_2) { sym_name = "prime601" } : memref<3072xi32>
  %buf13_1 = aie.buffer(%tile13_1) { sym_name = "prime607" } : memref<3072xi32>
  %buf14_1 = aie.buffer(%tile14_1) { sym_name = "prime613" } : memref<3072xi32>
  %buf14_2 = aie.buffer(%tile14_2) { sym_name = "prime617" } : memref<3072xi32>
  %buf14_3 = aie.buffer(%tile14_3) { sym_name = "prime619" } : memref<3072xi32>
  %buf14_4 = aie.buffer(%tile14_4) { sym_name = "prime631" } : memref<3072xi32>
  %buf14_5 = aie.buffer(%tile14_5) { sym_name = "prime641" } : memref<3072xi32>
  %buf14_6 = aie.buffer(%tile14_6) { sym_name = "prime643" } : memref<3072xi32>
  %buf14_7 = aie.buffer(%tile14_7) { sym_name = "prime647" } : memref<3072xi32>
  %buf14_8 = aie.buffer(%tile15_8) { sym_name = "prime653" } : memref<3072xi32>
  %buf15_8 = aie.buffer(%tile15_8) { sym_name = "prime659" } : memref<3072xi32>
  %buf15_7 = aie.buffer(%tile15_7) { sym_name = "prime661" } : memref<3072xi32>
  %buf15_6 = aie.buffer(%tile15_6) { sym_name = "prime673" } : memref<3072xi32>
  %buf15_5 = aie.buffer(%tile15_5) { sym_name = "prime677" } : memref<3072xi32>
  %buf15_4 = aie.buffer(%tile15_4) { sym_name = "prime683" } : memref<3072xi32>
  %buf15_3 = aie.buffer(%tile15_3) { sym_name = "prime691" } : memref<3072xi32>
  %buf15_2 = aie.buffer(%tile15_2) { sym_name = "prime701" } : memref<3072xi32>
  %buf15_1 = aie.buffer(%tile15_1) { sym_name = "prime709" } : memref<3072xi32>
  %buf16_1 = aie.buffer(%tile16_1) { sym_name = "prime719" } : memref<3072xi32>
  %buf16_2 = aie.buffer(%tile16_2) { sym_name = "prime727" } : memref<3072xi32>
  %buf16_3 = aie.buffer(%tile16_3) { sym_name = "prime733" } : memref<3072xi32>
  %buf16_4 = aie.buffer(%tile16_4) { sym_name = "prime739" } : memref<3072xi32>
  %buf16_5 = aie.buffer(%tile16_5) { sym_name = "prime743" } : memref<3072xi32>
  %buf16_6 = aie.buffer(%tile16_6) { sym_name = "prime751" } : memref<3072xi32>
  %buf16_7 = aie.buffer(%tile16_7) { sym_name = "prime757" } : memref<3072xi32>
  %buf16_8 = aie.buffer(%tile17_8) { sym_name = "prime761" } : memref<3072xi32>
  %buf17_8 = aie.buffer(%tile17_8) { sym_name = "prime769" } : memref<3072xi32>
  %buf17_7 = aie.buffer(%tile17_7) { sym_name = "prime773" } : memref<3072xi32>
  %buf17_6 = aie.buffer(%tile17_6) { sym_name = "prime787" } : memref<3072xi32>
  %buf17_5 = aie.buffer(%tile17_5) { sym_name = "prime797" } : memref<3072xi32>
  %buf17_4 = aie.buffer(%tile17_4) { sym_name = "prime809" } : memref<3072xi32>
  %buf17_3 = aie.buffer(%tile17_3) { sym_name = "prime811" } : memref<3072xi32>
  %buf17_2 = aie.buffer(%tile17_2) { sym_name = "prime821" } : memref<3072xi32>
  %buf17_1 = aie.buffer(%tile17_1) { sym_name = "prime823" } : memref<3072xi32>
  %buf18_1 = aie.buffer(%tile18_1) { sym_name = "prime827" } : memref<3072xi32>
  %buf18_2 = aie.buffer(%tile18_2) { sym_name = "prime829" } : memref<3072xi32>
  %buf18_3 = aie.buffer(%tile18_3) { sym_name = "prime839" } : memref<3072xi32>
  %buf18_4 = aie.buffer(%tile18_4) { sym_name = "prime853" } : memref<3072xi32>
  %buf18_5 = aie.buffer(%tile18_5) { sym_name = "prime857" } : memref<3072xi32>
  %buf18_6 = aie.buffer(%tile18_6) { sym_name = "prime859" } : memref<3072xi32>
  %buf18_7 = aie.buffer(%tile18_7) { sym_name = "prime863" } : memref<3072xi32>
  %buf18_8 = aie.buffer(%tile19_8) { sym_name = "prime877" } : memref<3072xi32>
  %buf19_8 = aie.buffer(%tile19_8) { sym_name = "prime881" } : memref<3072xi32>
  %buf19_7 = aie.buffer(%tile19_7) { sym_name = "prime883" } : memref<3072xi32>
  %buf19_6 = aie.buffer(%tile19_6) { sym_name = "prime887" } : memref<3072xi32>
  %buf19_5 = aie.buffer(%tile19_5) { sym_name = "prime907" } : memref<3072xi32>
  %buf19_4 = aie.buffer(%tile19_4) { sym_name = "prime911" } : memref<3072xi32>
  %buf19_3 = aie.buffer(%tile19_3) { sym_name = "prime919" } : memref<3072xi32>
  %buf19_2 = aie.buffer(%tile19_2) { sym_name = "prime929" } : memref<3072xi32>
  %buf19_1 = aie.buffer(%tile19_1) { sym_name = "prime937" } : memref<3072xi32>
  %buf20_1 = aie.buffer(%tile20_1) { sym_name = "prime941" } : memref<3072xi32>
  %buf20_2 = aie.buffer(%tile20_2) { sym_name = "prime947" } : memref<3072xi32>
  %buf20_3 = aie.buffer(%tile20_3) { sym_name = "prime953" } : memref<3072xi32>
  %buf20_4 = aie.buffer(%tile20_4) { sym_name = "prime967" } : memref<3072xi32>
  %buf20_5 = aie.buffer(%tile20_5) { sym_name = "prime971" } : memref<3072xi32>
  %buf20_6 = aie.buffer(%tile20_6) { sym_name = "prime977" } : memref<3072xi32>
  %buf20_7 = aie.buffer(%tile20_7) { sym_name = "prime983" } : memref<3072xi32>
  %buf20_8 = aie.buffer(%tile21_8) { sym_name = "prime991" } : memref<3072xi32>
  %buf21_8 = aie.buffer(%tile21_8) { sym_name = "prime997" } : memref<3072xi32>
  %buf21_7 = aie.buffer(%tile21_7) { sym_name = "prime1009" } : memref<3072xi32>
  %buf21_6 = aie.buffer(%tile21_6) { sym_name = "prime1013" } : memref<3072xi32>
  %buf21_5 = aie.buffer(%tile21_5) { sym_name = "prime1019" } : memref<3072xi32>
  %buf21_4 = aie.buffer(%tile21_4) { sym_name = "prime1021" } : memref<3072xi32>
  %buf21_3 = aie.buffer(%tile21_3) { sym_name = "prime1031" } : memref<3072xi32>
  %buf21_2 = aie.buffer(%tile21_2) { sym_name = "prime1033" } : memref<3072xi32>
  %buf21_1 = aie.buffer(%tile21_1) { sym_name = "prime1039" } : memref<3072xi32>
  %buf22_1 = aie.buffer(%tile22_1) { sym_name = "prime1049" } : memref<3072xi32>
  %buf22_2 = aie.buffer(%tile22_2) { sym_name = "prime1051" } : memref<3072xi32>
  %buf22_3 = aie.buffer(%tile22_3) { sym_name = "prime1061" } : memref<3072xi32>
  %buf22_4 = aie.buffer(%tile22_4) { sym_name = "prime1063" } : memref<3072xi32>
  %buf22_5 = aie.buffer(%tile22_5) { sym_name = "prime1069" } : memref<3072xi32>
  %buf22_6 = aie.buffer(%tile22_6) { sym_name = "prime1087" } : memref<3072xi32>
  %buf22_7 = aie.buffer(%tile22_7) { sym_name = "prime1091" } : memref<3072xi32>
  %buf22_8 = aie.buffer(%tile23_8) { sym_name = "prime1093" } : memref<3072xi32>
  %buf23_8 = aie.buffer(%tile23_8) { sym_name = "prime1097" } : memref<3072xi32>
  %buf23_7 = aie.buffer(%tile23_7) { sym_name = "prime1103" } : memref<3072xi32>
  %buf23_6 = aie.buffer(%tile23_6) { sym_name = "prime1109" } : memref<3072xi32>
  %buf23_5 = aie.buffer(%tile23_5) { sym_name = "prime1117" } : memref<3072xi32>
  %buf23_4 = aie.buffer(%tile23_4) { sym_name = "prime1123" } : memref<3072xi32>
  %buf23_3 = aie.buffer(%tile23_3) { sym_name = "prime1129" } : memref<3072xi32>
  %buf23_2 = aie.buffer(%tile23_2) { sym_name = "prime1151" } : memref<3072xi32>
  %buf23_1 = aie.buffer(%tile23_1) { sym_name = "prime1153" } : memref<3072xi32>
  %buf24_1 = aie.buffer(%tile24_1) { sym_name = "prime1163" } : memref<3072xi32>
  %buf24_2 = aie.buffer(%tile24_2) { sym_name = "prime1171" } : memref<3072xi32>
  %buf24_3 = aie.buffer(%tile24_3) { sym_name = "prime1181" } : memref<3072xi32>
  %buf24_4 = aie.buffer(%tile24_4) { sym_name = "prime1187" } : memref<3072xi32>
  %buf24_5 = aie.buffer(%tile24_5) { sym_name = "prime1193" } : memref<3072xi32>
  %buf24_6 = aie.buffer(%tile24_6) { sym_name = "prime1201" } : memref<3072xi32>
  %buf24_7 = aie.buffer(%tile24_7) { sym_name = "prime1213" } : memref<3072xi32>
  %buf24_8 = aie.buffer(%tile25_8) { sym_name = "prime1217" } : memref<3072xi32>
  %buf25_8 = aie.buffer(%tile25_8) { sym_name = "prime1223" } : memref<3072xi32>
  %buf25_7 = aie.buffer(%tile25_7) { sym_name = "prime1229" } : memref<3072xi32>
  %buf25_6 = aie.buffer(%tile25_6) { sym_name = "prime1231" } : memref<3072xi32>
  %buf25_5 = aie.buffer(%tile25_5) { sym_name = "prime1237" } : memref<3072xi32>
  %buf25_4 = aie.buffer(%tile25_4) { sym_name = "prime1249" } : memref<3072xi32>
  %buf25_3 = aie.buffer(%tile25_3) { sym_name = "prime1259" } : memref<3072xi32>
  %buf25_2 = aie.buffer(%tile25_2) { sym_name = "prime1277" } : memref<3072xi32>
  %buf25_1 = aie.buffer(%tile25_1) { sym_name = "prime1279" } : memref<3072xi32>
  %buf26_1 = aie.buffer(%tile26_1) { sym_name = "prime1283" } : memref<3072xi32>
  %buf26_2 = aie.buffer(%tile26_2) { sym_name = "prime1289" } : memref<3072xi32>
  %buf26_3 = aie.buffer(%tile26_3) { sym_name = "prime1291" } : memref<3072xi32>
  %buf26_4 = aie.buffer(%tile26_4) { sym_name = "prime1297" } : memref<3072xi32>
  %buf26_5 = aie.buffer(%tile26_5) { sym_name = "prime1301" } : memref<3072xi32>
  %buf26_6 = aie.buffer(%tile26_6) { sym_name = "prime1303" } : memref<3072xi32>
  %buf26_7 = aie.buffer(%tile26_7) { sym_name = "prime1307" } : memref<3072xi32>
  %buf26_8 = aie.buffer(%tile27_8) { sym_name = "prime1319" } : memref<3072xi32>
  %buf27_8 = aie.buffer(%tile27_8) { sym_name = "prime1321" } : memref<3072xi32>
  %buf27_7 = aie.buffer(%tile27_7) { sym_name = "prime1327" } : memref<3072xi32>
  %buf27_6 = aie.buffer(%tile27_6) { sym_name = "prime1361" } : memref<3072xi32>
  %buf27_5 = aie.buffer(%tile27_5) { sym_name = "prime1367" } : memref<3072xi32>
  %buf27_4 = aie.buffer(%tile27_4) { sym_name = "prime1373" } : memref<3072xi32>
  %buf27_3 = aie.buffer(%tile27_3) { sym_name = "prime1381" } : memref<3072xi32>
  %buf27_2 = aie.buffer(%tile27_2) { sym_name = "prime1399" } : memref<3072xi32>
  %buf27_1 = aie.buffer(%tile27_1) { sym_name = "prime1409" } : memref<3072xi32>
  %buf28_1 = aie.buffer(%tile28_1) { sym_name = "prime1423" } : memref<3072xi32>
  %buf28_2 = aie.buffer(%tile28_2) { sym_name = "prime1427" } : memref<3072xi32>
  %buf28_3 = aie.buffer(%tile28_3) { sym_name = "prime1429" } : memref<3072xi32>
  %buf28_4 = aie.buffer(%tile28_4) { sym_name = "prime1433" } : memref<3072xi32>
  %buf28_5 = aie.buffer(%tile28_5) { sym_name = "prime1439" } : memref<3072xi32>
  %buf28_6 = aie.buffer(%tile28_6) { sym_name = "prime1447" } : memref<3072xi32>
  %buf28_7 = aie.buffer(%tile28_7) { sym_name = "prime1451" } : memref<3072xi32>
  %buf28_8 = aie.buffer(%tile29_8) { sym_name = "prime1453" } : memref<3072xi32>
  %buf29_8 = aie.buffer(%tile29_8) { sym_name = "prime1459" } : memref<3072xi32>
  %buf29_7 = aie.buffer(%tile29_7) { sym_name = "prime1471" } : memref<3072xi32>
  %buf29_6 = aie.buffer(%tile29_6) { sym_name = "prime1481" } : memref<3072xi32>
  %buf29_5 = aie.buffer(%tile29_5) { sym_name = "prime1483" } : memref<3072xi32>
  %buf29_4 = aie.buffer(%tile29_4) { sym_name = "prime1487" } : memref<3072xi32>
  %buf29_3 = aie.buffer(%tile29_3) { sym_name = "prime1489" } : memref<3072xi32>
  %buf29_2 = aie.buffer(%tile29_2) { sym_name = "prime1493" } : memref<3072xi32>
  %buf29_1 = aie.buffer(%tile29_1) { sym_name = "prime1499" } : memref<3072xi32>
  %buf30_1 = aie.buffer(%tile30_1) { sym_name = "prime1511" } : memref<3072xi32>
  %buf30_2 = aie.buffer(%tile30_2) { sym_name = "prime1523" } : memref<3072xi32>
  %buf30_3 = aie.buffer(%tile30_3) { sym_name = "prime1531" } : memref<3072xi32>
  %buf30_4 = aie.buffer(%tile30_4) { sym_name = "prime1543" } : memref<3072xi32>
  %buf30_5 = aie.buffer(%tile30_5) { sym_name = "prime1549" } : memref<3072xi32>
  %buf30_6 = aie.buffer(%tile30_6) { sym_name = "prime1553" } : memref<3072xi32>
  %buf30_7 = aie.buffer(%tile30_7) { sym_name = "prime1559" } : memref<3072xi32>
  %buf30_8 = aie.buffer(%tile31_8) { sym_name = "prime1567" } : memref<3072xi32>
  %buf31_8 = aie.buffer(%tile31_8) { sym_name = "prime1571" } : memref<3072xi32>
  %buf31_7 = aie.buffer(%tile31_7) { sym_name = "prime1579" } : memref<3072xi32>
  %buf31_6 = aie.buffer(%tile31_6) { sym_name = "prime1583" } : memref<3072xi32>
  %buf31_5 = aie.buffer(%tile31_5) { sym_name = "prime1597" } : memref<3072xi32>
  %buf31_4 = aie.buffer(%tile31_4) { sym_name = "prime1601" } : memref<3072xi32>
  %buf31_3 = aie.buffer(%tile31_3) { sym_name = "prime1607" } : memref<3072xi32>
  %buf31_2 = aie.buffer(%tile31_2) { sym_name = "prime1609" } : memref<3072xi32>
  %buf31_1 = aie.buffer(%tile31_1) { sym_name = "prime1613" } : memref<3072xi32>
  %buf32_1 = aie.buffer(%tile32_1) { sym_name = "prime1619" } : memref<3072xi32>
  %buf32_2 = aie.buffer(%tile32_2) { sym_name = "prime1621" } : memref<3072xi32>
  %buf32_3 = aie.buffer(%tile32_3) { sym_name = "prime1627" } : memref<3072xi32>
  %buf32_4 = aie.buffer(%tile32_4) { sym_name = "prime1637" } : memref<3072xi32>
  %buf32_5 = aie.buffer(%tile32_5) { sym_name = "prime1657" } : memref<3072xi32>
  %buf32_6 = aie.buffer(%tile32_6) { sym_name = "prime1663" } : memref<3072xi32>
  %buf32_7 = aie.buffer(%tile32_7) { sym_name = "prime1667" } : memref<3072xi32>
  %buf32_8 = aie.buffer(%tile33_8) { sym_name = "prime1669" } : memref<3072xi32>
  %buf33_8 = aie.buffer(%tile33_8) { sym_name = "prime1693" } : memref<3072xi32>
  %buf33_7 = aie.buffer(%tile33_7) { sym_name = "prime1697" } : memref<3072xi32>
  %buf33_6 = aie.buffer(%tile33_6) { sym_name = "prime1699" } : memref<3072xi32>
  %buf33_5 = aie.buffer(%tile33_5) { sym_name = "prime1709" } : memref<3072xi32>
  %buf33_4 = aie.buffer(%tile33_4) { sym_name = "prime1721" } : memref<3072xi32>
  %buf33_3 = aie.buffer(%tile33_3) { sym_name = "prime1723" } : memref<3072xi32>
  %buf33_2 = aie.buffer(%tile33_2) { sym_name = "prime1733" } : memref<3072xi32>
  %buf33_1 = aie.buffer(%tile33_1) { sym_name = "prime1741" } : memref<3072xi32>
  %buf34_1 = aie.buffer(%tile34_1) { sym_name = "prime1747" } : memref<3072xi32>
  %buf34_2 = aie.buffer(%tile34_2) { sym_name = "prime1753" } : memref<3072xi32>
  %buf34_3 = aie.buffer(%tile34_3) { sym_name = "prime1759" } : memref<3072xi32>
  %buf34_4 = aie.buffer(%tile34_4) { sym_name = "prime1777" } : memref<3072xi32>
  %buf34_5 = aie.buffer(%tile34_5) { sym_name = "prime1783" } : memref<3072xi32>
  %buf34_6 = aie.buffer(%tile34_6) { sym_name = "prime1787" } : memref<3072xi32>
  %buf34_7 = aie.buffer(%tile34_7) { sym_name = "prime1789" } : memref<3072xi32>
  %buf34_8 = aie.buffer(%tile35_8) { sym_name = "prime1801" } : memref<3072xi32>
  %buf35_8 = aie.buffer(%tile35_8) { sym_name = "prime1811" } : memref<3072xi32>
  %buf35_7 = aie.buffer(%tile35_7) { sym_name = "prime1823" } : memref<3072xi32>
  %buf35_6 = aie.buffer(%tile35_6) { sym_name = "prime1831" } : memref<3072xi32>
  %buf35_5 = aie.buffer(%tile35_5) { sym_name = "prime1847" } : memref<3072xi32>
  %buf35_4 = aie.buffer(%tile35_4) { sym_name = "prime1861" } : memref<3072xi32>
  %buf35_3 = aie.buffer(%tile35_3) { sym_name = "prime1867" } : memref<3072xi32>
  %buf35_2 = aie.buffer(%tile35_2) { sym_name = "prime1871" } : memref<3072xi32>
  %buf35_1 = aie.buffer(%tile35_1) { sym_name = "prime1873" } : memref<3072xi32>
  %buf36_1 = aie.buffer(%tile36_1) { sym_name = "prime1877" } : memref<3072xi32>
  %buf36_2 = aie.buffer(%tile36_2) { sym_name = "prime1879" } : memref<3072xi32>
  %buf36_3 = aie.buffer(%tile36_3) { sym_name = "prime1889" } : memref<3072xi32>
  %buf36_4 = aie.buffer(%tile36_4) { sym_name = "prime1901" } : memref<3072xi32>
  %buf36_5 = aie.buffer(%tile36_5) { sym_name = "prime1907" } : memref<3072xi32>
  %buf36_6 = aie.buffer(%tile36_6) { sym_name = "prime1913" } : memref<3072xi32>
  %buf36_7 = aie.buffer(%tile36_7) { sym_name = "prime1931" } : memref<3072xi32>
  %buf36_8 = aie.buffer(%tile37_8) { sym_name = "prime1933" } : memref<3072xi32>
  %buf37_8 = aie.buffer(%tile37_8) { sym_name = "prime1949" } : memref<3072xi32>
  %buf37_7 = aie.buffer(%tile37_7) { sym_name = "prime1951" } : memref<3072xi32>
  %buf37_6 = aie.buffer(%tile37_6) { sym_name = "prime1973" } : memref<3072xi32>
  %buf37_5 = aie.buffer(%tile37_5) { sym_name = "prime1979" } : memref<3072xi32>
  %buf37_4 = aie.buffer(%tile37_4) { sym_name = "prime1987" } : memref<3072xi32>
  %buf37_3 = aie.buffer(%tile37_3) { sym_name = "prime1993" } : memref<3072xi32>
  %buf37_2 = aie.buffer(%tile37_2) { sym_name = "prime1997" } : memref<3072xi32>
  %buf37_1 = aie.buffer(%tile37_1) { sym_name = "prime1999" } : memref<3072xi32>
  %buf38_1 = aie.buffer(%tile38_1) { sym_name = "prime2003" } : memref<3072xi32>
  %buf38_2 = aie.buffer(%tile38_2) { sym_name = "prime2011" } : memref<3072xi32>
  %buf38_3 = aie.buffer(%tile38_3) { sym_name = "prime2017" } : memref<3072xi32>
  %buf38_4 = aie.buffer(%tile38_4) { sym_name = "prime2027" } : memref<3072xi32>
  %buf38_5 = aie.buffer(%tile38_5) { sym_name = "prime2029" } : memref<3072xi32>
  %buf38_6 = aie.buffer(%tile38_6) { sym_name = "prime2039" } : memref<3072xi32>
  %buf38_7 = aie.buffer(%tile38_7) { sym_name = "prime2053" } : memref<3072xi32>
  %buf38_8 = aie.buffer(%tile39_8) { sym_name = "prime2063" } : memref<3072xi32>
  %buf39_8 = aie.buffer(%tile39_8) { sym_name = "prime2069" } : memref<3072xi32>
  %buf39_7 = aie.buffer(%tile39_7) { sym_name = "prime2081" } : memref<3072xi32>
  %buf39_6 = aie.buffer(%tile39_6) { sym_name = "prime2083" } : memref<3072xi32>
  %buf39_5 = aie.buffer(%tile39_5) { sym_name = "prime2087" } : memref<3072xi32>
  %buf39_4 = aie.buffer(%tile39_4) { sym_name = "prime2089" } : memref<3072xi32>
  %buf39_3 = aie.buffer(%tile39_3) { sym_name = "prime2099" } : memref<3072xi32>
  %buf39_2 = aie.buffer(%tile39_2) { sym_name = "prime2111" } : memref<3072xi32>
  %buf39_1 = aie.buffer(%tile39_1) { sym_name = "prime2113" } : memref<3072xi32>
  %buf40_1 = aie.buffer(%tile40_1) { sym_name = "prime2129" } : memref<3072xi32>
  %buf40_2 = aie.buffer(%tile40_2) { sym_name = "prime2131" } : memref<3072xi32>
  %buf40_3 = aie.buffer(%tile40_3) { sym_name = "prime2137" } : memref<3072xi32>
  %buf40_4 = aie.buffer(%tile40_4) { sym_name = "prime2141" } : memref<3072xi32>
  %buf40_5 = aie.buffer(%tile40_5) { sym_name = "prime2143" } : memref<3072xi32>
  %buf40_6 = aie.buffer(%tile40_6) { sym_name = "prime2153" } : memref<3072xi32>
  %buf40_7 = aie.buffer(%tile40_7) { sym_name = "prime2161" } : memref<3072xi32>
  %buf40_8 = aie.buffer(%tile41_8) { sym_name = "prime2179" } : memref<3072xi32>
  %buf41_8 = aie.buffer(%tile41_8) { sym_name = "prime2203" } : memref<3072xi32>
  %buf41_7 = aie.buffer(%tile41_7) { sym_name = "prime2207" } : memref<3072xi32>
  %buf41_6 = aie.buffer(%tile41_6) { sym_name = "prime2213" } : memref<3072xi32>
  %buf41_5 = aie.buffer(%tile41_5) { sym_name = "prime2221" } : memref<3072xi32>
  %buf41_4 = aie.buffer(%tile41_4) { sym_name = "prime2237" } : memref<3072xi32>
  %buf41_3 = aie.buffer(%tile41_3) { sym_name = "prime2239" } : memref<3072xi32>
  %buf41_2 = aie.buffer(%tile41_2) { sym_name = "prime2243" } : memref<3072xi32>
  %buf41_1 = aie.buffer(%tile41_1) { sym_name = "prime2251" } : memref<3072xi32>
  %buf42_1 = aie.buffer(%tile42_1) { sym_name = "prime2267" } : memref<3072xi32>
  %buf42_2 = aie.buffer(%tile42_2) { sym_name = "prime2269" } : memref<3072xi32>
  %buf42_3 = aie.buffer(%tile42_3) { sym_name = "prime2273" } : memref<3072xi32>
  %buf42_4 = aie.buffer(%tile42_4) { sym_name = "prime2281" } : memref<3072xi32>
  %buf42_5 = aie.buffer(%tile42_5) { sym_name = "prime2287" } : memref<3072xi32>
  %buf42_6 = aie.buffer(%tile42_6) { sym_name = "prime2293" } : memref<3072xi32>
  %buf42_7 = aie.buffer(%tile42_7) { sym_name = "prime2297" } : memref<3072xi32>
  %buf42_8 = aie.buffer(%tile43_8) { sym_name = "prime2309" } : memref<3072xi32>
  %buf43_8 = aie.buffer(%tile43_8) { sym_name = "prime2311" } : memref<3072xi32>
  %buf43_7 = aie.buffer(%tile43_7) { sym_name = "prime2333" } : memref<3072xi32>
  %buf43_6 = aie.buffer(%tile43_6) { sym_name = "prime2339" } : memref<3072xi32>
  %buf43_5 = aie.buffer(%tile43_5) { sym_name = "prime2341" } : memref<3072xi32>
  %buf43_4 = aie.buffer(%tile43_4) { sym_name = "prime2347" } : memref<3072xi32>
  %buf43_3 = aie.buffer(%tile43_3) { sym_name = "prime2351" } : memref<3072xi32>
  %buf43_2 = aie.buffer(%tile43_2) { sym_name = "prime2357" } : memref<3072xi32>
  %buf43_1 = aie.buffer(%tile43_1) { sym_name = "prime2371" } : memref<3072xi32>
  %buf44_1 = aie.buffer(%tile44_1) { sym_name = "prime2377" } : memref<3072xi32>
  %buf44_2 = aie.buffer(%tile44_2) { sym_name = "prime2381" } : memref<3072xi32>
  %buf44_3 = aie.buffer(%tile44_3) { sym_name = "prime2383" } : memref<3072xi32>
  %buf44_4 = aie.buffer(%tile44_4) { sym_name = "prime2389" } : memref<3072xi32>
  %buf44_5 = aie.buffer(%tile44_5) { sym_name = "prime2393" } : memref<3072xi32>
  %buf44_6 = aie.buffer(%tile44_6) { sym_name = "prime2399" } : memref<3072xi32>
  %buf44_7 = aie.buffer(%tile44_7) { sym_name = "prime2411" } : memref<3072xi32>
  %buf44_8 = aie.buffer(%tile45_8) { sym_name = "prime2417" } : memref<3072xi32>
  %buf45_8 = aie.buffer(%tile45_8) { sym_name = "prime2423" } : memref<3072xi32>
  %buf45_7 = aie.buffer(%tile45_7) { sym_name = "prime2437" } : memref<3072xi32>
  %buf45_6 = aie.buffer(%tile45_6) { sym_name = "prime2441" } : memref<3072xi32>
  %buf45_5 = aie.buffer(%tile45_5) { sym_name = "prime2447" } : memref<3072xi32>
  %buf45_4 = aie.buffer(%tile45_4) { sym_name = "prime2459" } : memref<3072xi32>
  %buf45_3 = aie.buffer(%tile45_3) { sym_name = "prime2467" } : memref<3072xi32>
  %buf45_2 = aie.buffer(%tile45_2) { sym_name = "prime2473" } : memref<3072xi32>
  %buf45_1 = aie.buffer(%tile45_1) { sym_name = "prime2477" } : memref<3072xi32>
  %buf46_1 = aie.buffer(%tile46_1) { sym_name = "prime2503" } : memref<3072xi32>
  %buf46_2 = aie.buffer(%tile46_2) { sym_name = "prime2521" } : memref<3072xi32>
  %buf46_3 = aie.buffer(%tile46_3) { sym_name = "prime2531" } : memref<3072xi32>
  %buf46_4 = aie.buffer(%tile46_4) { sym_name = "prime2539" } : memref<3072xi32>
  %buf46_5 = aie.buffer(%tile46_5) { sym_name = "prime2543" } : memref<3072xi32>
  %buf46_6 = aie.buffer(%tile46_6) { sym_name = "prime2549" } : memref<3072xi32>
  %buf46_7 = aie.buffer(%tile46_7) { sym_name = "prime2551" } : memref<3072xi32>
  %buf46_8 = aie.buffer(%tile47_8) { sym_name = "prime2557" } : memref<3072xi32>
  %buf47_8 = aie.buffer(%tile47_8) { sym_name = "prime2579" } : memref<3072xi32>
  %buf47_7 = aie.buffer(%tile47_7) { sym_name = "prime2591" } : memref<3072xi32>
  %buf47_6 = aie.buffer(%tile47_6) { sym_name = "prime2593" } : memref<3072xi32>
  %buf47_5 = aie.buffer(%tile47_5) { sym_name = "prime2609" } : memref<3072xi32>
  %buf47_4 = aie.buffer(%tile47_4) { sym_name = "prime2617" } : memref<3072xi32>
  %buf47_3 = aie.buffer(%tile47_3) { sym_name = "prime2621" } : memref<3072xi32>
  %buf47_2 = aie.buffer(%tile47_2) { sym_name = "prime2633" } : memref<3072xi32>
  %buf47_1 = aie.buffer(%tile47_1) { sym_name = "prime2647" } : memref<3072xi32>
  %buf48_1 = aie.buffer(%tile48_1) { sym_name = "prime2657" } : memref<3072xi32>
  %buf48_2 = aie.buffer(%tile48_2) { sym_name = "prime2659" } : memref<3072xi32>
  %buf48_3 = aie.buffer(%tile48_3) { sym_name = "prime2663" } : memref<3072xi32>
  %buf48_4 = aie.buffer(%tile48_4) { sym_name = "prime2671" } : memref<3072xi32>
  %buf48_5 = aie.buffer(%tile48_5) { sym_name = "prime2677" } : memref<3072xi32>
  %buf48_6 = aie.buffer(%tile48_6) { sym_name = "prime2683" } : memref<3072xi32>
  %buf48_7 = aie.buffer(%tile48_7) { sym_name = "prime2687" } : memref<3072xi32>
  %buf48_8 = aie.buffer(%tile49_8) { sym_name = "prime2689" } : memref<3072xi32>
  %buf49_8 = aie.buffer(%tile49_8) { sym_name = "prime2693" } : memref<3072xi32>
  %buf49_7 = aie.buffer(%tile49_7) { sym_name = "prime2699" } : memref<3072xi32>
  %buf49_6 = aie.buffer(%tile49_6) { sym_name = "prime2707" } : memref<3072xi32>
  %buf49_5 = aie.buffer(%tile49_5) { sym_name = "prime2711" } : memref<3072xi32>
  %buf49_4 = aie.buffer(%tile49_4) { sym_name = "prime2713" } : memref<3072xi32>
  %buf49_3 = aie.buffer(%tile49_3) { sym_name = "prime2719" } : memref<3072xi32>
  %buf49_2 = aie.buffer(%tile49_2) { sym_name = "prime2729" } : memref<3072xi32>
  %buf49_1 = aie.buffer(%tile49_1) { sym_name = "prime_output" } : memref<3072xi32>
  
  %core0_1 = aie.core(%tile0_1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant 3072: index
    %sum_0 = arith.constant 2 : i32
    %t = arith.constant 1 : i32
  
    // store the index of the next prime number
    memref.store %t, %buf0_1[%c0] : memref<3072xi32>

    // output integers starting with 2...
    scf.for %arg0 = %c1 to %cend step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = arith.addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf0_1[%arg0] : memref<3072xi32>
      scf.yield %sum_next : i32
    }
    aie.use_lock(%lock0_1, "Release", 1)
    aie.end
  }
  func.func @do_sieve(%bufin: memref<3072xi32>, %bufout:memref<3072xi32>) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant 3072 : index
    %count_0 = arith.constant 0 : i32
    %one = arith.constant 1 : i32
  
    // The first number we receive is the index of the next prime
    %id = memref.load %bufin[%c0] : memref<3072xi32>

    // Compute the next id and store it in the output buffer
    %nextid = arith.addi %id, %one : i32
    memref.store %nextid, %bufout[%c0] : memref<3072xi32>

    // Copy the prior inputs
    %id_index = arith.index_cast %id : i32 to index
    %nextid_index = arith.index_cast %nextid : i32 to index
    scf.for %arg0 = %c1 to %nextid_index step %c1 {
      %in_val = memref.load %bufin[%arg0] : memref<3072xi32>
      memref.store %in_val, %bufout[%arg0] : memref<3072xi32>
    }
    %prime = memref.load %bufin[%id_index] : memref<3072xi32>

    // Step through the remaining inputs and sieve out multiples of %prime
    scf.for %arg0 = %nextid_index to %cend step %c1
      iter_args(%count_iter = %prime, %in_iter = %nextid_index, %out_iter = %nextid_index) -> (i32, index, index) {
      // Get the next input value
      %in_val = memref.load %bufin[%in_iter] : memref<3072xi32>

      // Potential next counters
      %count_inc = arith.addi %count_iter, %prime: i32
      %in_inc = arith.addi %in_iter, %c1 : index
      %out_inc = arith.addi %out_iter, %c1 : index

      // Compare the input value with the counter
      %b = arith.cmpi "slt", %in_val, %count_iter : i32
      %count_next, %in_next, %out_next = scf.if %b -> (i32, index, index) {
        // Input is less than counter.
        // Pass along the input and continue to the next one.
        memref.store %in_val, %bufout[%out_iter] : memref<3072xi32>
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
  
  %core0_2 = aie.core(%tile0_2) {
    aie.use_lock(%lock0_1, "Acquire", 1)
    aie.use_lock(%lock0_2, "Acquire", 0)
    func.call @do_sieve(%buf0_1, %buf0_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_1, "Release", 0)
    aie.use_lock(%lock0_2, "Release", 1)
    aie.end
  }

  %core0_3 = aie.core(%tile0_3) {
    aie.use_lock(%lock0_2, "Acquire", 1)
    aie.use_lock(%lock0_3, "Acquire", 0)
    func.call @do_sieve(%buf0_2, %buf0_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_2, "Release", 0)
    aie.use_lock(%lock0_3, "Release", 1)
    aie.end
  }

  %core0_4 = aie.core(%tile0_4) {
    aie.use_lock(%lock0_3, "Acquire", 1)
    aie.use_lock(%lock0_4, "Acquire", 0)
    func.call @do_sieve(%buf0_3, %buf0_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_3, "Release", 0)
    aie.use_lock(%lock0_4, "Release", 1)
    aie.end
  }

  %core0_5 = aie.core(%tile0_5) {
    aie.use_lock(%lock0_4, "Acquire", 1)
    aie.use_lock(%lock0_5, "Acquire", 0)
    func.call @do_sieve(%buf0_4, %buf0_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_4, "Release", 0)
    aie.use_lock(%lock0_5, "Release", 1)
    aie.end
  }

  %core0_6 = aie.core(%tile0_6) {
    aie.use_lock(%lock0_5, "Acquire", 1)
    aie.use_lock(%lock0_6, "Acquire", 0)
    func.call @do_sieve(%buf0_5, %buf0_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_5, "Release", 0)
    aie.use_lock(%lock0_6, "Release", 1)
    aie.end
  }

  %core0_7 = aie.core(%tile0_7) {
    aie.use_lock(%lock0_6, "Acquire", 1)
    aie.use_lock(%lock0_7, "Acquire", 0)
    func.call @do_sieve(%buf0_6, %buf0_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_6, "Release", 0)
    aie.use_lock(%lock0_7, "Release", 1)
    aie.end
  }

  %core0_8 = aie.core(%tile0_8) {
    aie.use_lock(%lock0_7, "Acquire", 1)
    aie.use_lock(%lock0_8, "Acquire", 0)
    func.call @do_sieve(%buf0_7, %buf0_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_7, "Release", 0)
    aie.use_lock(%lock0_8, "Release", 1)
    aie.end
  }

  %core1_8 = aie.core(%tile1_8) {
    aie.use_lock(%lock0_8, "Acquire", 1)
    aie.use_lock(%lock1_8, "Acquire", 0)
    func.call @do_sieve(%buf0_8, %buf1_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock0_8, "Release", 0)
    aie.use_lock(%lock1_8, "Release", 1)
    aie.end
  }

  %core1_7 = aie.core(%tile1_7) {
    aie.use_lock(%lock1_8, "Acquire", 1)
    aie.use_lock(%lock1_7, "Acquire", 0)
    func.call @do_sieve(%buf1_8, %buf1_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_8, "Release", 0)
    aie.use_lock(%lock1_7, "Release", 1)
    aie.end
  }

  %core1_6 = aie.core(%tile1_6) {
    aie.use_lock(%lock1_7, "Acquire", 1)
    aie.use_lock(%lock1_6, "Acquire", 0)
    func.call @do_sieve(%buf1_7, %buf1_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_7, "Release", 0)
    aie.use_lock(%lock1_6, "Release", 1)
    aie.end
  }

  %core1_5 = aie.core(%tile1_5) {
    aie.use_lock(%lock1_6, "Acquire", 1)
    aie.use_lock(%lock1_5, "Acquire", 0)
    func.call @do_sieve(%buf1_6, %buf1_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_6, "Release", 0)
    aie.use_lock(%lock1_5, "Release", 1)
    aie.end
  }

  %core1_4 = aie.core(%tile1_4) {
    aie.use_lock(%lock1_5, "Acquire", 1)
    aie.use_lock(%lock1_4, "Acquire", 0)
    func.call @do_sieve(%buf1_5, %buf1_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_5, "Release", 0)
    aie.use_lock(%lock1_4, "Release", 1)
    aie.end
  }

  %core1_3 = aie.core(%tile1_3) {
    aie.use_lock(%lock1_4, "Acquire", 1)
    aie.use_lock(%lock1_3, "Acquire", 0)
    func.call @do_sieve(%buf1_4, %buf1_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_4, "Release", 0)
    aie.use_lock(%lock1_3, "Release", 1)
    aie.end
  }

  %core1_2 = aie.core(%tile1_2) {
    aie.use_lock(%lock1_3, "Acquire", 1)
    aie.use_lock(%lock1_2, "Acquire", 0)
    func.call @do_sieve(%buf1_3, %buf1_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_3, "Release", 0)
    aie.use_lock(%lock1_2, "Release", 1)
    aie.end
  }

  %core1_1 = aie.core(%tile1_1) {
    aie.use_lock(%lock1_2, "Acquire", 1)
    aie.use_lock(%lock1_1, "Acquire", 0)
    func.call @do_sieve(%buf1_2, %buf1_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_2, "Release", 0)
    aie.use_lock(%lock1_1, "Release", 1)
    aie.end
  }

  %core2_1 = aie.core(%tile2_1) {
    aie.use_lock(%lock1_1, "Acquire", 1)
    aie.use_lock(%lock2_1, "Acquire", 0)
    func.call @do_sieve(%buf1_1, %buf2_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock1_1, "Release", 0)
    aie.use_lock(%lock2_1, "Release", 1)
    aie.end
  }

  %core2_2 = aie.core(%tile2_2) {
    aie.use_lock(%lock2_1, "Acquire", 1)
    aie.use_lock(%lock2_2, "Acquire", 0)
    func.call @do_sieve(%buf2_1, %buf2_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_1, "Release", 0)
    aie.use_lock(%lock2_2, "Release", 1)
    aie.end
  }

  %core2_3 = aie.core(%tile2_3) {
    aie.use_lock(%lock2_2, "Acquire", 1)
    aie.use_lock(%lock2_3, "Acquire", 0)
    func.call @do_sieve(%buf2_2, %buf2_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_2, "Release", 0)
    aie.use_lock(%lock2_3, "Release", 1)
    aie.end
  }

  %core2_4 = aie.core(%tile2_4) {
    aie.use_lock(%lock2_3, "Acquire", 1)
    aie.use_lock(%lock2_4, "Acquire", 0)
    func.call @do_sieve(%buf2_3, %buf2_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_3, "Release", 0)
    aie.use_lock(%lock2_4, "Release", 1)
    aie.end
  }

  %core2_5 = aie.core(%tile2_5) {
    aie.use_lock(%lock2_4, "Acquire", 1)
    aie.use_lock(%lock2_5, "Acquire", 0)
    func.call @do_sieve(%buf2_4, %buf2_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_4, "Release", 0)
    aie.use_lock(%lock2_5, "Release", 1)
    aie.end
  }

  %core2_6 = aie.core(%tile2_6) {
    aie.use_lock(%lock2_5, "Acquire", 1)
    aie.use_lock(%lock2_6, "Acquire", 0)
    func.call @do_sieve(%buf2_5, %buf2_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_5, "Release", 0)
    aie.use_lock(%lock2_6, "Release", 1)
    aie.end
  }

  %core2_7 = aie.core(%tile2_7) {
    aie.use_lock(%lock2_6, "Acquire", 1)
    aie.use_lock(%lock2_7, "Acquire", 0)
    func.call @do_sieve(%buf2_6, %buf2_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_6, "Release", 0)
    aie.use_lock(%lock2_7, "Release", 1)
    aie.end
  }

  %core2_8 = aie.core(%tile2_8) {
    aie.use_lock(%lock2_7, "Acquire", 1)
    aie.use_lock(%lock2_8, "Acquire", 0)
    func.call @do_sieve(%buf2_7, %buf2_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_7, "Release", 0)
    aie.use_lock(%lock2_8, "Release", 1)
    aie.end
  }

  %core3_8 = aie.core(%tile3_8) {
    aie.use_lock(%lock2_8, "Acquire", 1)
    aie.use_lock(%lock3_8, "Acquire", 0)
    func.call @do_sieve(%buf2_8, %buf3_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock2_8, "Release", 0)
    aie.use_lock(%lock3_8, "Release", 1)
    aie.end
  }

  %core3_7 = aie.core(%tile3_7) {
    aie.use_lock(%lock3_8, "Acquire", 1)
    aie.use_lock(%lock3_7, "Acquire", 0)
    func.call @do_sieve(%buf3_8, %buf3_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_8, "Release", 0)
    aie.use_lock(%lock3_7, "Release", 1)
    aie.end
  }

  %core3_6 = aie.core(%tile3_6) {
    aie.use_lock(%lock3_7, "Acquire", 1)
    aie.use_lock(%lock3_6, "Acquire", 0)
    func.call @do_sieve(%buf3_7, %buf3_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_7, "Release", 0)
    aie.use_lock(%lock3_6, "Release", 1)
    aie.end
  }

  %core3_5 = aie.core(%tile3_5) {
    aie.use_lock(%lock3_6, "Acquire", 1)
    aie.use_lock(%lock3_5, "Acquire", 0)
    func.call @do_sieve(%buf3_6, %buf3_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_6, "Release", 0)
    aie.use_lock(%lock3_5, "Release", 1)
    aie.end
  }

  %core3_4 = aie.core(%tile3_4) {
    aie.use_lock(%lock3_5, "Acquire", 1)
    aie.use_lock(%lock3_4, "Acquire", 0)
    func.call @do_sieve(%buf3_5, %buf3_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_5, "Release", 0)
    aie.use_lock(%lock3_4, "Release", 1)
    aie.end
  }

  %core3_3 = aie.core(%tile3_3) {
    aie.use_lock(%lock3_4, "Acquire", 1)
    aie.use_lock(%lock3_3, "Acquire", 0)
    func.call @do_sieve(%buf3_4, %buf3_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_4, "Release", 0)
    aie.use_lock(%lock3_3, "Release", 1)
    aie.end
  }

  %core3_2 = aie.core(%tile3_2) {
    aie.use_lock(%lock3_3, "Acquire", 1)
    aie.use_lock(%lock3_2, "Acquire", 0)
    func.call @do_sieve(%buf3_3, %buf3_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_3, "Release", 0)
    aie.use_lock(%lock3_2, "Release", 1)
    aie.end
  }

  %core3_1 = aie.core(%tile3_1) {
    aie.use_lock(%lock3_2, "Acquire", 1)
    aie.use_lock(%lock3_1, "Acquire", 0)
    func.call @do_sieve(%buf3_2, %buf3_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_2, "Release", 0)
    aie.use_lock(%lock3_1, "Release", 1)
    aie.end
  }

  %core4_1 = aie.core(%tile4_1) {
    aie.use_lock(%lock3_1, "Acquire", 1)
    aie.use_lock(%lock4_1, "Acquire", 0)
    func.call @do_sieve(%buf3_1, %buf4_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock3_1, "Release", 0)
    aie.use_lock(%lock4_1, "Release", 1)
    aie.end
  }

  %core4_2 = aie.core(%tile4_2) {
    aie.use_lock(%lock4_1, "Acquire", 1)
    aie.use_lock(%lock4_2, "Acquire", 0)
    func.call @do_sieve(%buf4_1, %buf4_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_1, "Release", 0)
    aie.use_lock(%lock4_2, "Release", 1)
    aie.end
  }

  %core4_3 = aie.core(%tile4_3) {
    aie.use_lock(%lock4_2, "Acquire", 1)
    aie.use_lock(%lock4_3, "Acquire", 0)
    func.call @do_sieve(%buf4_2, %buf4_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_2, "Release", 0)
    aie.use_lock(%lock4_3, "Release", 1)
    aie.end
  }

  %core4_4 = aie.core(%tile4_4) {
    aie.use_lock(%lock4_3, "Acquire", 1)
    aie.use_lock(%lock4_4, "Acquire", 0)
    func.call @do_sieve(%buf4_3, %buf4_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_3, "Release", 0)
    aie.use_lock(%lock4_4, "Release", 1)
    aie.end
  }

  %core4_5 = aie.core(%tile4_5) {
    aie.use_lock(%lock4_4, "Acquire", 1)
    aie.use_lock(%lock4_5, "Acquire", 0)
    func.call @do_sieve(%buf4_4, %buf4_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_4, "Release", 0)
    aie.use_lock(%lock4_5, "Release", 1)
    aie.end
  }

  %core4_6 = aie.core(%tile4_6) {
    aie.use_lock(%lock4_5, "Acquire", 1)
    aie.use_lock(%lock4_6, "Acquire", 0)
    func.call @do_sieve(%buf4_5, %buf4_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_5, "Release", 0)
    aie.use_lock(%lock4_6, "Release", 1)
    aie.end
  }

  %core4_7 = aie.core(%tile4_7) {
    aie.use_lock(%lock4_6, "Acquire", 1)
    aie.use_lock(%lock4_7, "Acquire", 0)
    func.call @do_sieve(%buf4_6, %buf4_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_6, "Release", 0)
    aie.use_lock(%lock4_7, "Release", 1)
    aie.end
  }

  %core4_8 = aie.core(%tile4_8) {
    aie.use_lock(%lock4_7, "Acquire", 1)
    aie.use_lock(%lock4_8, "Acquire", 0)
    func.call @do_sieve(%buf4_7, %buf4_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_7, "Release", 0)
    aie.use_lock(%lock4_8, "Release", 1)
    aie.end
  }

  %core5_8 = aie.core(%tile5_8) {
    aie.use_lock(%lock4_8, "Acquire", 1)
    aie.use_lock(%lock5_8, "Acquire", 0)
    func.call @do_sieve(%buf4_8, %buf5_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock4_8, "Release", 0)
    aie.use_lock(%lock5_8, "Release", 1)
    aie.end
  }

  %core5_7 = aie.core(%tile5_7) {
    aie.use_lock(%lock5_8, "Acquire", 1)
    aie.use_lock(%lock5_7, "Acquire", 0)
    func.call @do_sieve(%buf5_8, %buf5_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_8, "Release", 0)
    aie.use_lock(%lock5_7, "Release", 1)
    aie.end
  }

  %core5_6 = aie.core(%tile5_6) {
    aie.use_lock(%lock5_7, "Acquire", 1)
    aie.use_lock(%lock5_6, "Acquire", 0)
    func.call @do_sieve(%buf5_7, %buf5_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_7, "Release", 0)
    aie.use_lock(%lock5_6, "Release", 1)
    aie.end
  }

  %core5_5 = aie.core(%tile5_5) {
    aie.use_lock(%lock5_6, "Acquire", 1)
    aie.use_lock(%lock5_5, "Acquire", 0)
    func.call @do_sieve(%buf5_6, %buf5_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_6, "Release", 0)
    aie.use_lock(%lock5_5, "Release", 1)
    aie.end
  }

  %core5_4 = aie.core(%tile5_4) {
    aie.use_lock(%lock5_5, "Acquire", 1)
    aie.use_lock(%lock5_4, "Acquire", 0)
    func.call @do_sieve(%buf5_5, %buf5_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_5, "Release", 0)
    aie.use_lock(%lock5_4, "Release", 1)
    aie.end
  }

  %core5_3 = aie.core(%tile5_3) {
    aie.use_lock(%lock5_4, "Acquire", 1)
    aie.use_lock(%lock5_3, "Acquire", 0)
    func.call @do_sieve(%buf5_4, %buf5_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_4, "Release", 0)
    aie.use_lock(%lock5_3, "Release", 1)
    aie.end
  }

  %core5_2 = aie.core(%tile5_2) {
    aie.use_lock(%lock5_3, "Acquire", 1)
    aie.use_lock(%lock5_2, "Acquire", 0)
    func.call @do_sieve(%buf5_3, %buf5_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_3, "Release", 0)
    aie.use_lock(%lock5_2, "Release", 1)
    aie.end
  }

  %core5_1 = aie.core(%tile5_1) {
    aie.use_lock(%lock5_2, "Acquire", 1)
    aie.use_lock(%lock5_1, "Acquire", 0)
    func.call @do_sieve(%buf5_2, %buf5_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_2, "Release", 0)
    aie.use_lock(%lock5_1, "Release", 1)
    aie.end
  }

  %core6_1 = aie.core(%tile6_1) {
    aie.use_lock(%lock5_1, "Acquire", 1)
    aie.use_lock(%lock6_1, "Acquire", 0)
    func.call @do_sieve(%buf5_1, %buf6_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock5_1, "Release", 0)
    aie.use_lock(%lock6_1, "Release", 1)
    aie.end
  }

  %core6_2 = aie.core(%tile6_2) {
    aie.use_lock(%lock6_1, "Acquire", 1)
    aie.use_lock(%lock6_2, "Acquire", 0)
    func.call @do_sieve(%buf6_1, %buf6_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_1, "Release", 0)
    aie.use_lock(%lock6_2, "Release", 1)
    aie.end
  }

  %core6_3 = aie.core(%tile6_3) {
    aie.use_lock(%lock6_2, "Acquire", 1)
    aie.use_lock(%lock6_3, "Acquire", 0)
    func.call @do_sieve(%buf6_2, %buf6_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_2, "Release", 0)
    aie.use_lock(%lock6_3, "Release", 1)
    aie.end
  }

  %core6_4 = aie.core(%tile6_4) {
    aie.use_lock(%lock6_3, "Acquire", 1)
    aie.use_lock(%lock6_4, "Acquire", 0)
    func.call @do_sieve(%buf6_3, %buf6_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_3, "Release", 0)
    aie.use_lock(%lock6_4, "Release", 1)
    aie.end
  }

  %core6_5 = aie.core(%tile6_5) {
    aie.use_lock(%lock6_4, "Acquire", 1)
    aie.use_lock(%lock6_5, "Acquire", 0)
    func.call @do_sieve(%buf6_4, %buf6_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_4, "Release", 0)
    aie.use_lock(%lock6_5, "Release", 1)
    aie.end
  }

  %core6_6 = aie.core(%tile6_6) {
    aie.use_lock(%lock6_5, "Acquire", 1)
    aie.use_lock(%lock6_6, "Acquire", 0)
    func.call @do_sieve(%buf6_5, %buf6_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_5, "Release", 0)
    aie.use_lock(%lock6_6, "Release", 1)
    aie.end
  }

  %core6_7 = aie.core(%tile6_7) {
    aie.use_lock(%lock6_6, "Acquire", 1)
    aie.use_lock(%lock6_7, "Acquire", 0)
    func.call @do_sieve(%buf6_6, %buf6_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_6, "Release", 0)
    aie.use_lock(%lock6_7, "Release", 1)
    aie.end
  }

  %core6_8 = aie.core(%tile6_8) {
    aie.use_lock(%lock6_7, "Acquire", 1)
    aie.use_lock(%lock6_8, "Acquire", 0)
    func.call @do_sieve(%buf6_7, %buf6_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_7, "Release", 0)
    aie.use_lock(%lock6_8, "Release", 1)
    aie.end
  }

  %core7_8 = aie.core(%tile7_8) {
    aie.use_lock(%lock6_8, "Acquire", 1)
    aie.use_lock(%lock7_8, "Acquire", 0)
    func.call @do_sieve(%buf6_8, %buf7_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock6_8, "Release", 0)
    aie.use_lock(%lock7_8, "Release", 1)
    aie.end
  }

  %core7_7 = aie.core(%tile7_7) {
    aie.use_lock(%lock7_8, "Acquire", 1)
    aie.use_lock(%lock7_7, "Acquire", 0)
    func.call @do_sieve(%buf7_8, %buf7_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_8, "Release", 0)
    aie.use_lock(%lock7_7, "Release", 1)
    aie.end
  }

  %core7_6 = aie.core(%tile7_6) {
    aie.use_lock(%lock7_7, "Acquire", 1)
    aie.use_lock(%lock7_6, "Acquire", 0)
    func.call @do_sieve(%buf7_7, %buf7_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_7, "Release", 0)
    aie.use_lock(%lock7_6, "Release", 1)
    aie.end
  }

  %core7_5 = aie.core(%tile7_5) {
    aie.use_lock(%lock7_6, "Acquire", 1)
    aie.use_lock(%lock7_5, "Acquire", 0)
    func.call @do_sieve(%buf7_6, %buf7_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_6, "Release", 0)
    aie.use_lock(%lock7_5, "Release", 1)
    aie.end
  }

  %core7_4 = aie.core(%tile7_4) {
    aie.use_lock(%lock7_5, "Acquire", 1)
    aie.use_lock(%lock7_4, "Acquire", 0)
    func.call @do_sieve(%buf7_5, %buf7_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_5, "Release", 0)
    aie.use_lock(%lock7_4, "Release", 1)
    aie.end
  }

  %core7_3 = aie.core(%tile7_3) {
    aie.use_lock(%lock7_4, "Acquire", 1)
    aie.use_lock(%lock7_3, "Acquire", 0)
    func.call @do_sieve(%buf7_4, %buf7_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_4, "Release", 0)
    aie.use_lock(%lock7_3, "Release", 1)
    aie.end
  }

  %core7_2 = aie.core(%tile7_2) {
    aie.use_lock(%lock7_3, "Acquire", 1)
    aie.use_lock(%lock7_2, "Acquire", 0)
    func.call @do_sieve(%buf7_3, %buf7_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_3, "Release", 0)
    aie.use_lock(%lock7_2, "Release", 1)
    aie.end
  }

  %core7_1 = aie.core(%tile7_1) {
    aie.use_lock(%lock7_2, "Acquire", 1)
    aie.use_lock(%lock7_1, "Acquire", 0)
    func.call @do_sieve(%buf7_2, %buf7_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_2, "Release", 0)
    aie.use_lock(%lock7_1, "Release", 1)
    aie.end
  }

  %core8_1 = aie.core(%tile8_1) {
    aie.use_lock(%lock7_1, "Acquire", 1)
    aie.use_lock(%lock8_1, "Acquire", 0)
    func.call @do_sieve(%buf7_1, %buf8_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock7_1, "Release", 0)
    aie.use_lock(%lock8_1, "Release", 1)
    aie.end
  }

  %core8_2 = aie.core(%tile8_2) {
    aie.use_lock(%lock8_1, "Acquire", 1)
    aie.use_lock(%lock8_2, "Acquire", 0)
    func.call @do_sieve(%buf8_1, %buf8_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_1, "Release", 0)
    aie.use_lock(%lock8_2, "Release", 1)
    aie.end
  }

  %core8_3 = aie.core(%tile8_3) {
    aie.use_lock(%lock8_2, "Acquire", 1)
    aie.use_lock(%lock8_3, "Acquire", 0)
    func.call @do_sieve(%buf8_2, %buf8_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_2, "Release", 0)
    aie.use_lock(%lock8_3, "Release", 1)
    aie.end
  }

  %core8_4 = aie.core(%tile8_4) {
    aie.use_lock(%lock8_3, "Acquire", 1)
    aie.use_lock(%lock8_4, "Acquire", 0)
    func.call @do_sieve(%buf8_3, %buf8_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_3, "Release", 0)
    aie.use_lock(%lock8_4, "Release", 1)
    aie.end
  }

  %core8_5 = aie.core(%tile8_5) {
    aie.use_lock(%lock8_4, "Acquire", 1)
    aie.use_lock(%lock8_5, "Acquire", 0)
    func.call @do_sieve(%buf8_4, %buf8_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_4, "Release", 0)
    aie.use_lock(%lock8_5, "Release", 1)
    aie.end
  }

  %core8_6 = aie.core(%tile8_6) {
    aie.use_lock(%lock8_5, "Acquire", 1)
    aie.use_lock(%lock8_6, "Acquire", 0)
    func.call @do_sieve(%buf8_5, %buf8_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_5, "Release", 0)
    aie.use_lock(%lock8_6, "Release", 1)
    aie.end
  }

  %core8_7 = aie.core(%tile8_7) {
    aie.use_lock(%lock8_6, "Acquire", 1)
    aie.use_lock(%lock8_7, "Acquire", 0)
    func.call @do_sieve(%buf8_6, %buf8_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_6, "Release", 0)
    aie.use_lock(%lock8_7, "Release", 1)
    aie.end
  }

  %core8_8 = aie.core(%tile8_8) {
    aie.use_lock(%lock8_7, "Acquire", 1)
    aie.use_lock(%lock8_8, "Acquire", 0)
    func.call @do_sieve(%buf8_7, %buf8_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_7, "Release", 0)
    aie.use_lock(%lock8_8, "Release", 1)
    aie.end
  }

  %core9_8 = aie.core(%tile9_8) {
    aie.use_lock(%lock8_8, "Acquire", 1)
    aie.use_lock(%lock9_8, "Acquire", 0)
    func.call @do_sieve(%buf8_8, %buf9_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock8_8, "Release", 0)
    aie.use_lock(%lock9_8, "Release", 1)
    aie.end
  }

  %core9_7 = aie.core(%tile9_7) {
    aie.use_lock(%lock9_8, "Acquire", 1)
    aie.use_lock(%lock9_7, "Acquire", 0)
    func.call @do_sieve(%buf9_8, %buf9_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_8, "Release", 0)
    aie.use_lock(%lock9_7, "Release", 1)
    aie.end
  }

  %core9_6 = aie.core(%tile9_6) {
    aie.use_lock(%lock9_7, "Acquire", 1)
    aie.use_lock(%lock9_6, "Acquire", 0)
    func.call @do_sieve(%buf9_7, %buf9_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_7, "Release", 0)
    aie.use_lock(%lock9_6, "Release", 1)
    aie.end
  }

  %core9_5 = aie.core(%tile9_5) {
    aie.use_lock(%lock9_6, "Acquire", 1)
    aie.use_lock(%lock9_5, "Acquire", 0)
    func.call @do_sieve(%buf9_6, %buf9_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_6, "Release", 0)
    aie.use_lock(%lock9_5, "Release", 1)
    aie.end
  }

  %core9_4 = aie.core(%tile9_4) {
    aie.use_lock(%lock9_5, "Acquire", 1)
    aie.use_lock(%lock9_4, "Acquire", 0)
    func.call @do_sieve(%buf9_5, %buf9_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_5, "Release", 0)
    aie.use_lock(%lock9_4, "Release", 1)
    aie.end
  }

  %core9_3 = aie.core(%tile9_3) {
    aie.use_lock(%lock9_4, "Acquire", 1)
    aie.use_lock(%lock9_3, "Acquire", 0)
    func.call @do_sieve(%buf9_4, %buf9_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_4, "Release", 0)
    aie.use_lock(%lock9_3, "Release", 1)
    aie.end
  }

  %core9_2 = aie.core(%tile9_2) {
    aie.use_lock(%lock9_3, "Acquire", 1)
    aie.use_lock(%lock9_2, "Acquire", 0)
    func.call @do_sieve(%buf9_3, %buf9_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_3, "Release", 0)
    aie.use_lock(%lock9_2, "Release", 1)
    aie.end
  }

  %core9_1 = aie.core(%tile9_1) {
    aie.use_lock(%lock9_2, "Acquire", 1)
    aie.use_lock(%lock9_1, "Acquire", 0)
    func.call @do_sieve(%buf9_2, %buf9_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_2, "Release", 0)
    aie.use_lock(%lock9_1, "Release", 1)
    aie.end
  }

  %core10_1 = aie.core(%tile10_1) {
    aie.use_lock(%lock9_1, "Acquire", 1)
    aie.use_lock(%lock10_1, "Acquire", 0)
    func.call @do_sieve(%buf9_1, %buf10_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock9_1, "Release", 0)
    aie.use_lock(%lock10_1, "Release", 1)
    aie.end
  }

  %core10_2 = aie.core(%tile10_2) {
    aie.use_lock(%lock10_1, "Acquire", 1)
    aie.use_lock(%lock10_2, "Acquire", 0)
    func.call @do_sieve(%buf10_1, %buf10_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_1, "Release", 0)
    aie.use_lock(%lock10_2, "Release", 1)
    aie.end
  }

  %core10_3 = aie.core(%tile10_3) {
    aie.use_lock(%lock10_2, "Acquire", 1)
    aie.use_lock(%lock10_3, "Acquire", 0)
    func.call @do_sieve(%buf10_2, %buf10_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_2, "Release", 0)
    aie.use_lock(%lock10_3, "Release", 1)
    aie.end
  }

  %core10_4 = aie.core(%tile10_4) {
    aie.use_lock(%lock10_3, "Acquire", 1)
    aie.use_lock(%lock10_4, "Acquire", 0)
    func.call @do_sieve(%buf10_3, %buf10_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_3, "Release", 0)
    aie.use_lock(%lock10_4, "Release", 1)
    aie.end
  }

  %core10_5 = aie.core(%tile10_5) {
    aie.use_lock(%lock10_4, "Acquire", 1)
    aie.use_lock(%lock10_5, "Acquire", 0)
    func.call @do_sieve(%buf10_4, %buf10_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_4, "Release", 0)
    aie.use_lock(%lock10_5, "Release", 1)
    aie.end
  }

  %core10_6 = aie.core(%tile10_6) {
    aie.use_lock(%lock10_5, "Acquire", 1)
    aie.use_lock(%lock10_6, "Acquire", 0)
    func.call @do_sieve(%buf10_5, %buf10_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_5, "Release", 0)
    aie.use_lock(%lock10_6, "Release", 1)
    aie.end
  }

  %core10_7 = aie.core(%tile10_7) {
    aie.use_lock(%lock10_6, "Acquire", 1)
    aie.use_lock(%lock10_7, "Acquire", 0)
    func.call @do_sieve(%buf10_6, %buf10_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_6, "Release", 0)
    aie.use_lock(%lock10_7, "Release", 1)
    aie.end
  }

  %core10_8 = aie.core(%tile10_8) {
    aie.use_lock(%lock10_7, "Acquire", 1)
    aie.use_lock(%lock10_8, "Acquire", 0)
    func.call @do_sieve(%buf10_7, %buf10_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_7, "Release", 0)
    aie.use_lock(%lock10_8, "Release", 1)
    aie.end
  }

  %core11_8 = aie.core(%tile11_8) {
    aie.use_lock(%lock10_8, "Acquire", 1)
    aie.use_lock(%lock11_8, "Acquire", 0)
    func.call @do_sieve(%buf10_8, %buf11_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock10_8, "Release", 0)
    aie.use_lock(%lock11_8, "Release", 1)
    aie.end
  }

  %core11_7 = aie.core(%tile11_7) {
    aie.use_lock(%lock11_8, "Acquire", 1)
    aie.use_lock(%lock11_7, "Acquire", 0)
    func.call @do_sieve(%buf11_8, %buf11_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_8, "Release", 0)
    aie.use_lock(%lock11_7, "Release", 1)
    aie.end
  }

  %core11_6 = aie.core(%tile11_6) {
    aie.use_lock(%lock11_7, "Acquire", 1)
    aie.use_lock(%lock11_6, "Acquire", 0)
    func.call @do_sieve(%buf11_7, %buf11_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_7, "Release", 0)
    aie.use_lock(%lock11_6, "Release", 1)
    aie.end
  }

  %core11_5 = aie.core(%tile11_5) {
    aie.use_lock(%lock11_6, "Acquire", 1)
    aie.use_lock(%lock11_5, "Acquire", 0)
    func.call @do_sieve(%buf11_6, %buf11_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_6, "Release", 0)
    aie.use_lock(%lock11_5, "Release", 1)
    aie.end
  }

  %core11_4 = aie.core(%tile11_4) {
    aie.use_lock(%lock11_5, "Acquire", 1)
    aie.use_lock(%lock11_4, "Acquire", 0)
    func.call @do_sieve(%buf11_5, %buf11_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_5, "Release", 0)
    aie.use_lock(%lock11_4, "Release", 1)
    aie.end
  }

  %core11_3 = aie.core(%tile11_3) {
    aie.use_lock(%lock11_4, "Acquire", 1)
    aie.use_lock(%lock11_3, "Acquire", 0)
    func.call @do_sieve(%buf11_4, %buf11_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_4, "Release", 0)
    aie.use_lock(%lock11_3, "Release", 1)
    aie.end
  }

  %core11_2 = aie.core(%tile11_2) {
    aie.use_lock(%lock11_3, "Acquire", 1)
    aie.use_lock(%lock11_2, "Acquire", 0)
    func.call @do_sieve(%buf11_3, %buf11_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_3, "Release", 0)
    aie.use_lock(%lock11_2, "Release", 1)
    aie.end
  }

  %core11_1 = aie.core(%tile11_1) {
    aie.use_lock(%lock11_2, "Acquire", 1)
    aie.use_lock(%lock11_1, "Acquire", 0)
    func.call @do_sieve(%buf11_2, %buf11_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_2, "Release", 0)
    aie.use_lock(%lock11_1, "Release", 1)
    aie.end
  }

  %core12_1 = aie.core(%tile12_1) {
    aie.use_lock(%lock11_1, "Acquire", 1)
    aie.use_lock(%lock12_1, "Acquire", 0)
    func.call @do_sieve(%buf11_1, %buf12_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock11_1, "Release", 0)
    aie.use_lock(%lock12_1, "Release", 1)
    aie.end
  }

  %core12_2 = aie.core(%tile12_2) {
    aie.use_lock(%lock12_1, "Acquire", 1)
    aie.use_lock(%lock12_2, "Acquire", 0)
    func.call @do_sieve(%buf12_1, %buf12_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_1, "Release", 0)
    aie.use_lock(%lock12_2, "Release", 1)
    aie.end
  }

  %core12_3 = aie.core(%tile12_3) {
    aie.use_lock(%lock12_2, "Acquire", 1)
    aie.use_lock(%lock12_3, "Acquire", 0)
    func.call @do_sieve(%buf12_2, %buf12_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_2, "Release", 0)
    aie.use_lock(%lock12_3, "Release", 1)
    aie.end
  }

  %core12_4 = aie.core(%tile12_4) {
    aie.use_lock(%lock12_3, "Acquire", 1)
    aie.use_lock(%lock12_4, "Acquire", 0)
    func.call @do_sieve(%buf12_3, %buf12_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_3, "Release", 0)
    aie.use_lock(%lock12_4, "Release", 1)
    aie.end
  }

  %core12_5 = aie.core(%tile12_5) {
    aie.use_lock(%lock12_4, "Acquire", 1)
    aie.use_lock(%lock12_5, "Acquire", 0)
    func.call @do_sieve(%buf12_4, %buf12_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_4, "Release", 0)
    aie.use_lock(%lock12_5, "Release", 1)
    aie.end
  }

  %core12_6 = aie.core(%tile12_6) {
    aie.use_lock(%lock12_5, "Acquire", 1)
    aie.use_lock(%lock12_6, "Acquire", 0)
    func.call @do_sieve(%buf12_5, %buf12_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_5, "Release", 0)
    aie.use_lock(%lock12_6, "Release", 1)
    aie.end
  }

  %core12_7 = aie.core(%tile12_7) {
    aie.use_lock(%lock12_6, "Acquire", 1)
    aie.use_lock(%lock12_7, "Acquire", 0)
    func.call @do_sieve(%buf12_6, %buf12_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_6, "Release", 0)
    aie.use_lock(%lock12_7, "Release", 1)
    aie.end
  }

  %core12_8 = aie.core(%tile12_8) {
    aie.use_lock(%lock12_7, "Acquire", 1)
    aie.use_lock(%lock12_8, "Acquire", 0)
    func.call @do_sieve(%buf12_7, %buf12_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_7, "Release", 0)
    aie.use_lock(%lock12_8, "Release", 1)
    aie.end
  }

  %core13_8 = aie.core(%tile13_8) {
    aie.use_lock(%lock12_8, "Acquire", 1)
    aie.use_lock(%lock13_8, "Acquire", 0)
    func.call @do_sieve(%buf12_8, %buf13_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock12_8, "Release", 0)
    aie.use_lock(%lock13_8, "Release", 1)
    aie.end
  }

  %core13_7 = aie.core(%tile13_7) {
    aie.use_lock(%lock13_8, "Acquire", 1)
    aie.use_lock(%lock13_7, "Acquire", 0)
    func.call @do_sieve(%buf13_8, %buf13_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_8, "Release", 0)
    aie.use_lock(%lock13_7, "Release", 1)
    aie.end
  }

  %core13_6 = aie.core(%tile13_6) {
    aie.use_lock(%lock13_7, "Acquire", 1)
    aie.use_lock(%lock13_6, "Acquire", 0)
    func.call @do_sieve(%buf13_7, %buf13_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_7, "Release", 0)
    aie.use_lock(%lock13_6, "Release", 1)
    aie.end
  }

  %core13_5 = aie.core(%tile13_5) {
    aie.use_lock(%lock13_6, "Acquire", 1)
    aie.use_lock(%lock13_5, "Acquire", 0)
    func.call @do_sieve(%buf13_6, %buf13_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_6, "Release", 0)
    aie.use_lock(%lock13_5, "Release", 1)
    aie.end
  }

  %core13_4 = aie.core(%tile13_4) {
    aie.use_lock(%lock13_5, "Acquire", 1)
    aie.use_lock(%lock13_4, "Acquire", 0)
    func.call @do_sieve(%buf13_5, %buf13_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_5, "Release", 0)
    aie.use_lock(%lock13_4, "Release", 1)
    aie.end
  }

  %core13_3 = aie.core(%tile13_3) {
    aie.use_lock(%lock13_4, "Acquire", 1)
    aie.use_lock(%lock13_3, "Acquire", 0)
    func.call @do_sieve(%buf13_4, %buf13_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_4, "Release", 0)
    aie.use_lock(%lock13_3, "Release", 1)
    aie.end
  }

  %core13_2 = aie.core(%tile13_2) {
    aie.use_lock(%lock13_3, "Acquire", 1)
    aie.use_lock(%lock13_2, "Acquire", 0)
    func.call @do_sieve(%buf13_3, %buf13_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_3, "Release", 0)
    aie.use_lock(%lock13_2, "Release", 1)
    aie.end
  }

  %core13_1 = aie.core(%tile13_1) {
    aie.use_lock(%lock13_2, "Acquire", 1)
    aie.use_lock(%lock13_1, "Acquire", 0)
    func.call @do_sieve(%buf13_2, %buf13_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_2, "Release", 0)
    aie.use_lock(%lock13_1, "Release", 1)
    aie.end
  }

  %core14_1 = aie.core(%tile14_1) {
    aie.use_lock(%lock13_1, "Acquire", 1)
    aie.use_lock(%lock14_1, "Acquire", 0)
    func.call @do_sieve(%buf13_1, %buf14_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock13_1, "Release", 0)
    aie.use_lock(%lock14_1, "Release", 1)
    aie.end
  }

  %core14_2 = aie.core(%tile14_2) {
    aie.use_lock(%lock14_1, "Acquire", 1)
    aie.use_lock(%lock14_2, "Acquire", 0)
    func.call @do_sieve(%buf14_1, %buf14_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_1, "Release", 0)
    aie.use_lock(%lock14_2, "Release", 1)
    aie.end
  }

  %core14_3 = aie.core(%tile14_3) {
    aie.use_lock(%lock14_2, "Acquire", 1)
    aie.use_lock(%lock14_3, "Acquire", 0)
    func.call @do_sieve(%buf14_2, %buf14_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_2, "Release", 0)
    aie.use_lock(%lock14_3, "Release", 1)
    aie.end
  }

  %core14_4 = aie.core(%tile14_4) {
    aie.use_lock(%lock14_3, "Acquire", 1)
    aie.use_lock(%lock14_4, "Acquire", 0)
    func.call @do_sieve(%buf14_3, %buf14_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_3, "Release", 0)
    aie.use_lock(%lock14_4, "Release", 1)
    aie.end
  }

  %core14_5 = aie.core(%tile14_5) {
    aie.use_lock(%lock14_4, "Acquire", 1)
    aie.use_lock(%lock14_5, "Acquire", 0)
    func.call @do_sieve(%buf14_4, %buf14_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_4, "Release", 0)
    aie.use_lock(%lock14_5, "Release", 1)
    aie.end
  }

  %core14_6 = aie.core(%tile14_6) {
    aie.use_lock(%lock14_5, "Acquire", 1)
    aie.use_lock(%lock14_6, "Acquire", 0)
    func.call @do_sieve(%buf14_5, %buf14_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_5, "Release", 0)
    aie.use_lock(%lock14_6, "Release", 1)
    aie.end
  }

  %core14_7 = aie.core(%tile14_7) {
    aie.use_lock(%lock14_6, "Acquire", 1)
    aie.use_lock(%lock14_7, "Acquire", 0)
    func.call @do_sieve(%buf14_6, %buf14_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_6, "Release", 0)
    aie.use_lock(%lock14_7, "Release", 1)
    aie.end
  }

  %core14_8 = aie.core(%tile14_8) {
    aie.use_lock(%lock14_7, "Acquire", 1)
    aie.use_lock(%lock14_8, "Acquire", 0)
    func.call @do_sieve(%buf14_7, %buf14_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_7, "Release", 0)
    aie.use_lock(%lock14_8, "Release", 1)
    aie.end
  }

  %core15_8 = aie.core(%tile15_8) {
    aie.use_lock(%lock14_8, "Acquire", 1)
    aie.use_lock(%lock15_8, "Acquire", 0)
    func.call @do_sieve(%buf14_8, %buf15_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock14_8, "Release", 0)
    aie.use_lock(%lock15_8, "Release", 1)
    aie.end
  }

  %core15_7 = aie.core(%tile15_7) {
    aie.use_lock(%lock15_8, "Acquire", 1)
    aie.use_lock(%lock15_7, "Acquire", 0)
    func.call @do_sieve(%buf15_8, %buf15_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_8, "Release", 0)
    aie.use_lock(%lock15_7, "Release", 1)
    aie.end
  }

  %core15_6 = aie.core(%tile15_6) {
    aie.use_lock(%lock15_7, "Acquire", 1)
    aie.use_lock(%lock15_6, "Acquire", 0)
    func.call @do_sieve(%buf15_7, %buf15_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_7, "Release", 0)
    aie.use_lock(%lock15_6, "Release", 1)
    aie.end
  }

  %core15_5 = aie.core(%tile15_5) {
    aie.use_lock(%lock15_6, "Acquire", 1)
    aie.use_lock(%lock15_5, "Acquire", 0)
    func.call @do_sieve(%buf15_6, %buf15_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_6, "Release", 0)
    aie.use_lock(%lock15_5, "Release", 1)
    aie.end
  }

  %core15_4 = aie.core(%tile15_4) {
    aie.use_lock(%lock15_5, "Acquire", 1)
    aie.use_lock(%lock15_4, "Acquire", 0)
    func.call @do_sieve(%buf15_5, %buf15_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_5, "Release", 0)
    aie.use_lock(%lock15_4, "Release", 1)
    aie.end
  }

  %core15_3 = aie.core(%tile15_3) {
    aie.use_lock(%lock15_4, "Acquire", 1)
    aie.use_lock(%lock15_3, "Acquire", 0)
    func.call @do_sieve(%buf15_4, %buf15_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_4, "Release", 0)
    aie.use_lock(%lock15_3, "Release", 1)
    aie.end
  }

  %core15_2 = aie.core(%tile15_2) {
    aie.use_lock(%lock15_3, "Acquire", 1)
    aie.use_lock(%lock15_2, "Acquire", 0)
    func.call @do_sieve(%buf15_3, %buf15_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_3, "Release", 0)
    aie.use_lock(%lock15_2, "Release", 1)
    aie.end
  }

  %core15_1 = aie.core(%tile15_1) {
    aie.use_lock(%lock15_2, "Acquire", 1)
    aie.use_lock(%lock15_1, "Acquire", 0)
    func.call @do_sieve(%buf15_2, %buf15_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_2, "Release", 0)
    aie.use_lock(%lock15_1, "Release", 1)
    aie.end
  }

  %core16_1 = aie.core(%tile16_1) {
    aie.use_lock(%lock15_1, "Acquire", 1)
    aie.use_lock(%lock16_1, "Acquire", 0)
    func.call @do_sieve(%buf15_1, %buf16_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock15_1, "Release", 0)
    aie.use_lock(%lock16_1, "Release", 1)
    aie.end
  }

  %core16_2 = aie.core(%tile16_2) {
    aie.use_lock(%lock16_1, "Acquire", 1)
    aie.use_lock(%lock16_2, "Acquire", 0)
    func.call @do_sieve(%buf16_1, %buf16_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_1, "Release", 0)
    aie.use_lock(%lock16_2, "Release", 1)
    aie.end
  }

  %core16_3 = aie.core(%tile16_3) {
    aie.use_lock(%lock16_2, "Acquire", 1)
    aie.use_lock(%lock16_3, "Acquire", 0)
    func.call @do_sieve(%buf16_2, %buf16_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_2, "Release", 0)
    aie.use_lock(%lock16_3, "Release", 1)
    aie.end
  }

  %core16_4 = aie.core(%tile16_4) {
    aie.use_lock(%lock16_3, "Acquire", 1)
    aie.use_lock(%lock16_4, "Acquire", 0)
    func.call @do_sieve(%buf16_3, %buf16_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_3, "Release", 0)
    aie.use_lock(%lock16_4, "Release", 1)
    aie.end
  }

  %core16_5 = aie.core(%tile16_5) {
    aie.use_lock(%lock16_4, "Acquire", 1)
    aie.use_lock(%lock16_5, "Acquire", 0)
    func.call @do_sieve(%buf16_4, %buf16_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_4, "Release", 0)
    aie.use_lock(%lock16_5, "Release", 1)
    aie.end
  }

  %core16_6 = aie.core(%tile16_6) {
    aie.use_lock(%lock16_5, "Acquire", 1)
    aie.use_lock(%lock16_6, "Acquire", 0)
    func.call @do_sieve(%buf16_5, %buf16_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_5, "Release", 0)
    aie.use_lock(%lock16_6, "Release", 1)
    aie.end
  }

  %core16_7 = aie.core(%tile16_7) {
    aie.use_lock(%lock16_6, "Acquire", 1)
    aie.use_lock(%lock16_7, "Acquire", 0)
    func.call @do_sieve(%buf16_6, %buf16_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_6, "Release", 0)
    aie.use_lock(%lock16_7, "Release", 1)
    aie.end
  }

  %core16_8 = aie.core(%tile16_8) {
    aie.use_lock(%lock16_7, "Acquire", 1)
    aie.use_lock(%lock16_8, "Acquire", 0)
    func.call @do_sieve(%buf16_7, %buf16_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_7, "Release", 0)
    aie.use_lock(%lock16_8, "Release", 1)
    aie.end
  }

  %core17_8 = aie.core(%tile17_8) {
    aie.use_lock(%lock16_8, "Acquire", 1)
    aie.use_lock(%lock17_8, "Acquire", 0)
    func.call @do_sieve(%buf16_8, %buf17_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock16_8, "Release", 0)
    aie.use_lock(%lock17_8, "Release", 1)
    aie.end
  }

  %core17_7 = aie.core(%tile17_7) {
    aie.use_lock(%lock17_8, "Acquire", 1)
    aie.use_lock(%lock17_7, "Acquire", 0)
    func.call @do_sieve(%buf17_8, %buf17_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_8, "Release", 0)
    aie.use_lock(%lock17_7, "Release", 1)
    aie.end
  }

  %core17_6 = aie.core(%tile17_6) {
    aie.use_lock(%lock17_7, "Acquire", 1)
    aie.use_lock(%lock17_6, "Acquire", 0)
    func.call @do_sieve(%buf17_7, %buf17_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_7, "Release", 0)
    aie.use_lock(%lock17_6, "Release", 1)
    aie.end
  }

  %core17_5 = aie.core(%tile17_5) {
    aie.use_lock(%lock17_6, "Acquire", 1)
    aie.use_lock(%lock17_5, "Acquire", 0)
    func.call @do_sieve(%buf17_6, %buf17_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_6, "Release", 0)
    aie.use_lock(%lock17_5, "Release", 1)
    aie.end
  }

  %core17_4 = aie.core(%tile17_4) {
    aie.use_lock(%lock17_5, "Acquire", 1)
    aie.use_lock(%lock17_4, "Acquire", 0)
    func.call @do_sieve(%buf17_5, %buf17_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_5, "Release", 0)
    aie.use_lock(%lock17_4, "Release", 1)
    aie.end
  }

  %core17_3 = aie.core(%tile17_3) {
    aie.use_lock(%lock17_4, "Acquire", 1)
    aie.use_lock(%lock17_3, "Acquire", 0)
    func.call @do_sieve(%buf17_4, %buf17_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_4, "Release", 0)
    aie.use_lock(%lock17_3, "Release", 1)
    aie.end
  }

  %core17_2 = aie.core(%tile17_2) {
    aie.use_lock(%lock17_3, "Acquire", 1)
    aie.use_lock(%lock17_2, "Acquire", 0)
    func.call @do_sieve(%buf17_3, %buf17_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_3, "Release", 0)
    aie.use_lock(%lock17_2, "Release", 1)
    aie.end
  }

  %core17_1 = aie.core(%tile17_1) {
    aie.use_lock(%lock17_2, "Acquire", 1)
    aie.use_lock(%lock17_1, "Acquire", 0)
    func.call @do_sieve(%buf17_2, %buf17_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_2, "Release", 0)
    aie.use_lock(%lock17_1, "Release", 1)
    aie.end
  }

  %core18_1 = aie.core(%tile18_1) {
    aie.use_lock(%lock17_1, "Acquire", 1)
    aie.use_lock(%lock18_1, "Acquire", 0)
    func.call @do_sieve(%buf17_1, %buf18_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock17_1, "Release", 0)
    aie.use_lock(%lock18_1, "Release", 1)
    aie.end
  }

  %core18_2 = aie.core(%tile18_2) {
    aie.use_lock(%lock18_1, "Acquire", 1)
    aie.use_lock(%lock18_2, "Acquire", 0)
    func.call @do_sieve(%buf18_1, %buf18_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_1, "Release", 0)
    aie.use_lock(%lock18_2, "Release", 1)
    aie.end
  }

  %core18_3 = aie.core(%tile18_3) {
    aie.use_lock(%lock18_2, "Acquire", 1)
    aie.use_lock(%lock18_3, "Acquire", 0)
    func.call @do_sieve(%buf18_2, %buf18_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_2, "Release", 0)
    aie.use_lock(%lock18_3, "Release", 1)
    aie.end
  }

  %core18_4 = aie.core(%tile18_4) {
    aie.use_lock(%lock18_3, "Acquire", 1)
    aie.use_lock(%lock18_4, "Acquire", 0)
    func.call @do_sieve(%buf18_3, %buf18_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_3, "Release", 0)
    aie.use_lock(%lock18_4, "Release", 1)
    aie.end
  }

  %core18_5 = aie.core(%tile18_5) {
    aie.use_lock(%lock18_4, "Acquire", 1)
    aie.use_lock(%lock18_5, "Acquire", 0)
    func.call @do_sieve(%buf18_4, %buf18_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_4, "Release", 0)
    aie.use_lock(%lock18_5, "Release", 1)
    aie.end
  }

  %core18_6 = aie.core(%tile18_6) {
    aie.use_lock(%lock18_5, "Acquire", 1)
    aie.use_lock(%lock18_6, "Acquire", 0)
    func.call @do_sieve(%buf18_5, %buf18_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_5, "Release", 0)
    aie.use_lock(%lock18_6, "Release", 1)
    aie.end
  }

  %core18_7 = aie.core(%tile18_7) {
    aie.use_lock(%lock18_6, "Acquire", 1)
    aie.use_lock(%lock18_7, "Acquire", 0)
    func.call @do_sieve(%buf18_6, %buf18_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_6, "Release", 0)
    aie.use_lock(%lock18_7, "Release", 1)
    aie.end
  }

  %core18_8 = aie.core(%tile18_8) {
    aie.use_lock(%lock18_7, "Acquire", 1)
    aie.use_lock(%lock18_8, "Acquire", 0)
    func.call @do_sieve(%buf18_7, %buf18_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_7, "Release", 0)
    aie.use_lock(%lock18_8, "Release", 1)
    aie.end
  }

  %core19_8 = aie.core(%tile19_8) {
    aie.use_lock(%lock18_8, "Acquire", 1)
    aie.use_lock(%lock19_8, "Acquire", 0)
    func.call @do_sieve(%buf18_8, %buf19_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock18_8, "Release", 0)
    aie.use_lock(%lock19_8, "Release", 1)
    aie.end
  }

  %core19_7 = aie.core(%tile19_7) {
    aie.use_lock(%lock19_8, "Acquire", 1)
    aie.use_lock(%lock19_7, "Acquire", 0)
    func.call @do_sieve(%buf19_8, %buf19_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_8, "Release", 0)
    aie.use_lock(%lock19_7, "Release", 1)
    aie.end
  }

  %core19_6 = aie.core(%tile19_6) {
    aie.use_lock(%lock19_7, "Acquire", 1)
    aie.use_lock(%lock19_6, "Acquire", 0)
    func.call @do_sieve(%buf19_7, %buf19_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_7, "Release", 0)
    aie.use_lock(%lock19_6, "Release", 1)
    aie.end
  }

  %core19_5 = aie.core(%tile19_5) {
    aie.use_lock(%lock19_6, "Acquire", 1)
    aie.use_lock(%lock19_5, "Acquire", 0)
    func.call @do_sieve(%buf19_6, %buf19_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_6, "Release", 0)
    aie.use_lock(%lock19_5, "Release", 1)
    aie.end
  }

  %core19_4 = aie.core(%tile19_4) {
    aie.use_lock(%lock19_5, "Acquire", 1)
    aie.use_lock(%lock19_4, "Acquire", 0)
    func.call @do_sieve(%buf19_5, %buf19_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_5, "Release", 0)
    aie.use_lock(%lock19_4, "Release", 1)
    aie.end
  }

  %core19_3 = aie.core(%tile19_3) {
    aie.use_lock(%lock19_4, "Acquire", 1)
    aie.use_lock(%lock19_3, "Acquire", 0)
    func.call @do_sieve(%buf19_4, %buf19_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_4, "Release", 0)
    aie.use_lock(%lock19_3, "Release", 1)
    aie.end
  }

  %core19_2 = aie.core(%tile19_2) {
    aie.use_lock(%lock19_3, "Acquire", 1)
    aie.use_lock(%lock19_2, "Acquire", 0)
    func.call @do_sieve(%buf19_3, %buf19_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_3, "Release", 0)
    aie.use_lock(%lock19_2, "Release", 1)
    aie.end
  }

  %core19_1 = aie.core(%tile19_1) {
    aie.use_lock(%lock19_2, "Acquire", 1)
    aie.use_lock(%lock19_1, "Acquire", 0)
    func.call @do_sieve(%buf19_2, %buf19_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_2, "Release", 0)
    aie.use_lock(%lock19_1, "Release", 1)
    aie.end
  }

  %core20_1 = aie.core(%tile20_1) {
    aie.use_lock(%lock19_1, "Acquire", 1)
    aie.use_lock(%lock20_1, "Acquire", 0)
    func.call @do_sieve(%buf19_1, %buf20_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock19_1, "Release", 0)
    aie.use_lock(%lock20_1, "Release", 1)
    aie.end
  }

  %core20_2 = aie.core(%tile20_2) {
    aie.use_lock(%lock20_1, "Acquire", 1)
    aie.use_lock(%lock20_2, "Acquire", 0)
    func.call @do_sieve(%buf20_1, %buf20_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_1, "Release", 0)
    aie.use_lock(%lock20_2, "Release", 1)
    aie.end
  }

  %core20_3 = aie.core(%tile20_3) {
    aie.use_lock(%lock20_2, "Acquire", 1)
    aie.use_lock(%lock20_3, "Acquire", 0)
    func.call @do_sieve(%buf20_2, %buf20_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_2, "Release", 0)
    aie.use_lock(%lock20_3, "Release", 1)
    aie.end
  }

  %core20_4 = aie.core(%tile20_4) {
    aie.use_lock(%lock20_3, "Acquire", 1)
    aie.use_lock(%lock20_4, "Acquire", 0)
    func.call @do_sieve(%buf20_3, %buf20_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_3, "Release", 0)
    aie.use_lock(%lock20_4, "Release", 1)
    aie.end
  }

  %core20_5 = aie.core(%tile20_5) {
    aie.use_lock(%lock20_4, "Acquire", 1)
    aie.use_lock(%lock20_5, "Acquire", 0)
    func.call @do_sieve(%buf20_4, %buf20_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_4, "Release", 0)
    aie.use_lock(%lock20_5, "Release", 1)
    aie.end
  }

  %core20_6 = aie.core(%tile20_6) {
    aie.use_lock(%lock20_5, "Acquire", 1)
    aie.use_lock(%lock20_6, "Acquire", 0)
    func.call @do_sieve(%buf20_5, %buf20_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_5, "Release", 0)
    aie.use_lock(%lock20_6, "Release", 1)
    aie.end
  }

  %core20_7 = aie.core(%tile20_7) {
    aie.use_lock(%lock20_6, "Acquire", 1)
    aie.use_lock(%lock20_7, "Acquire", 0)
    func.call @do_sieve(%buf20_6, %buf20_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_6, "Release", 0)
    aie.use_lock(%lock20_7, "Release", 1)
    aie.end
  }

  %core20_8 = aie.core(%tile20_8) {
    aie.use_lock(%lock20_7, "Acquire", 1)
    aie.use_lock(%lock20_8, "Acquire", 0)
    func.call @do_sieve(%buf20_7, %buf20_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_7, "Release", 0)
    aie.use_lock(%lock20_8, "Release", 1)
    aie.end
  }

  %core21_8 = aie.core(%tile21_8) {
    aie.use_lock(%lock20_8, "Acquire", 1)
    aie.use_lock(%lock21_8, "Acquire", 0)
    func.call @do_sieve(%buf20_8, %buf21_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock20_8, "Release", 0)
    aie.use_lock(%lock21_8, "Release", 1)
    aie.end
  }

  %core21_7 = aie.core(%tile21_7) {
    aie.use_lock(%lock21_8, "Acquire", 1)
    aie.use_lock(%lock21_7, "Acquire", 0)
    func.call @do_sieve(%buf21_8, %buf21_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_8, "Release", 0)
    aie.use_lock(%lock21_7, "Release", 1)
    aie.end
  }

  %core21_6 = aie.core(%tile21_6) {
    aie.use_lock(%lock21_7, "Acquire", 1)
    aie.use_lock(%lock21_6, "Acquire", 0)
    func.call @do_sieve(%buf21_7, %buf21_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_7, "Release", 0)
    aie.use_lock(%lock21_6, "Release", 1)
    aie.end
  }

  %core21_5 = aie.core(%tile21_5) {
    aie.use_lock(%lock21_6, "Acquire", 1)
    aie.use_lock(%lock21_5, "Acquire", 0)
    func.call @do_sieve(%buf21_6, %buf21_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_6, "Release", 0)
    aie.use_lock(%lock21_5, "Release", 1)
    aie.end
  }

  %core21_4 = aie.core(%tile21_4) {
    aie.use_lock(%lock21_5, "Acquire", 1)
    aie.use_lock(%lock21_4, "Acquire", 0)
    func.call @do_sieve(%buf21_5, %buf21_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_5, "Release", 0)
    aie.use_lock(%lock21_4, "Release", 1)
    aie.end
  }

  %core21_3 = aie.core(%tile21_3) {
    aie.use_lock(%lock21_4, "Acquire", 1)
    aie.use_lock(%lock21_3, "Acquire", 0)
    func.call @do_sieve(%buf21_4, %buf21_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_4, "Release", 0)
    aie.use_lock(%lock21_3, "Release", 1)
    aie.end
  }

  %core21_2 = aie.core(%tile21_2) {
    aie.use_lock(%lock21_3, "Acquire", 1)
    aie.use_lock(%lock21_2, "Acquire", 0)
    func.call @do_sieve(%buf21_3, %buf21_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_3, "Release", 0)
    aie.use_lock(%lock21_2, "Release", 1)
    aie.end
  }

  %core21_1 = aie.core(%tile21_1) {
    aie.use_lock(%lock21_2, "Acquire", 1)
    aie.use_lock(%lock21_1, "Acquire", 0)
    func.call @do_sieve(%buf21_2, %buf21_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_2, "Release", 0)
    aie.use_lock(%lock21_1, "Release", 1)
    aie.end
  }

  %core22_1 = aie.core(%tile22_1) {
    aie.use_lock(%lock21_1, "Acquire", 1)
    aie.use_lock(%lock22_1, "Acquire", 0)
    func.call @do_sieve(%buf21_1, %buf22_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock21_1, "Release", 0)
    aie.use_lock(%lock22_1, "Release", 1)
    aie.end
  }

  %core22_2 = aie.core(%tile22_2) {
    aie.use_lock(%lock22_1, "Acquire", 1)
    aie.use_lock(%lock22_2, "Acquire", 0)
    func.call @do_sieve(%buf22_1, %buf22_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_1, "Release", 0)
    aie.use_lock(%lock22_2, "Release", 1)
    aie.end
  }

  %core22_3 = aie.core(%tile22_3) {
    aie.use_lock(%lock22_2, "Acquire", 1)
    aie.use_lock(%lock22_3, "Acquire", 0)
    func.call @do_sieve(%buf22_2, %buf22_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_2, "Release", 0)
    aie.use_lock(%lock22_3, "Release", 1)
    aie.end
  }

  %core22_4 = aie.core(%tile22_4) {
    aie.use_lock(%lock22_3, "Acquire", 1)
    aie.use_lock(%lock22_4, "Acquire", 0)
    func.call @do_sieve(%buf22_3, %buf22_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_3, "Release", 0)
    aie.use_lock(%lock22_4, "Release", 1)
    aie.end
  }

  %core22_5 = aie.core(%tile22_5) {
    aie.use_lock(%lock22_4, "Acquire", 1)
    aie.use_lock(%lock22_5, "Acquire", 0)
    func.call @do_sieve(%buf22_4, %buf22_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_4, "Release", 0)
    aie.use_lock(%lock22_5, "Release", 1)
    aie.end
  }

  %core22_6 = aie.core(%tile22_6) {
    aie.use_lock(%lock22_5, "Acquire", 1)
    aie.use_lock(%lock22_6, "Acquire", 0)
    func.call @do_sieve(%buf22_5, %buf22_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_5, "Release", 0)
    aie.use_lock(%lock22_6, "Release", 1)
    aie.end
  }

  %core22_7 = aie.core(%tile22_7) {
    aie.use_lock(%lock22_6, "Acquire", 1)
    aie.use_lock(%lock22_7, "Acquire", 0)
    func.call @do_sieve(%buf22_6, %buf22_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_6, "Release", 0)
    aie.use_lock(%lock22_7, "Release", 1)
    aie.end
  }

  %core22_8 = aie.core(%tile22_8) {
    aie.use_lock(%lock22_7, "Acquire", 1)
    aie.use_lock(%lock22_8, "Acquire", 0)
    func.call @do_sieve(%buf22_7, %buf22_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_7, "Release", 0)
    aie.use_lock(%lock22_8, "Release", 1)
    aie.end
  }

  %core23_8 = aie.core(%tile23_8) {
    aie.use_lock(%lock22_8, "Acquire", 1)
    aie.use_lock(%lock23_8, "Acquire", 0)
    func.call @do_sieve(%buf22_8, %buf23_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock22_8, "Release", 0)
    aie.use_lock(%lock23_8, "Release", 1)
    aie.end
  }

  %core23_7 = aie.core(%tile23_7) {
    aie.use_lock(%lock23_8, "Acquire", 1)
    aie.use_lock(%lock23_7, "Acquire", 0)
    func.call @do_sieve(%buf23_8, %buf23_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_8, "Release", 0)
    aie.use_lock(%lock23_7, "Release", 1)
    aie.end
  }

  %core23_6 = aie.core(%tile23_6) {
    aie.use_lock(%lock23_7, "Acquire", 1)
    aie.use_lock(%lock23_6, "Acquire", 0)
    func.call @do_sieve(%buf23_7, %buf23_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_7, "Release", 0)
    aie.use_lock(%lock23_6, "Release", 1)
    aie.end
  }

  %core23_5 = aie.core(%tile23_5) {
    aie.use_lock(%lock23_6, "Acquire", 1)
    aie.use_lock(%lock23_5, "Acquire", 0)
    func.call @do_sieve(%buf23_6, %buf23_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_6, "Release", 0)
    aie.use_lock(%lock23_5, "Release", 1)
    aie.end
  }

  %core23_4 = aie.core(%tile23_4) {
    aie.use_lock(%lock23_5, "Acquire", 1)
    aie.use_lock(%lock23_4, "Acquire", 0)
    func.call @do_sieve(%buf23_5, %buf23_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_5, "Release", 0)
    aie.use_lock(%lock23_4, "Release", 1)
    aie.end
  }

  %core23_3 = aie.core(%tile23_3) {
    aie.use_lock(%lock23_4, "Acquire", 1)
    aie.use_lock(%lock23_3, "Acquire", 0)
    func.call @do_sieve(%buf23_4, %buf23_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_4, "Release", 0)
    aie.use_lock(%lock23_3, "Release", 1)
    aie.end
  }

  %core23_2 = aie.core(%tile23_2) {
    aie.use_lock(%lock23_3, "Acquire", 1)
    aie.use_lock(%lock23_2, "Acquire", 0)
    func.call @do_sieve(%buf23_3, %buf23_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_3, "Release", 0)
    aie.use_lock(%lock23_2, "Release", 1)
    aie.end
  }

  %core23_1 = aie.core(%tile23_1) {
    aie.use_lock(%lock23_2, "Acquire", 1)
    aie.use_lock(%lock23_1, "Acquire", 0)
    func.call @do_sieve(%buf23_2, %buf23_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_2, "Release", 0)
    aie.use_lock(%lock23_1, "Release", 1)
    aie.end
  }

  %core24_1 = aie.core(%tile24_1) {
    aie.use_lock(%lock23_1, "Acquire", 1)
    aie.use_lock(%lock24_1, "Acquire", 0)
    func.call @do_sieve(%buf23_1, %buf24_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock23_1, "Release", 0)
    aie.use_lock(%lock24_1, "Release", 1)
    aie.end
  }

  %core24_2 = aie.core(%tile24_2) {
    aie.use_lock(%lock24_1, "Acquire", 1)
    aie.use_lock(%lock24_2, "Acquire", 0)
    func.call @do_sieve(%buf24_1, %buf24_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_1, "Release", 0)
    aie.use_lock(%lock24_2, "Release", 1)
    aie.end
  }

  %core24_3 = aie.core(%tile24_3) {
    aie.use_lock(%lock24_2, "Acquire", 1)
    aie.use_lock(%lock24_3, "Acquire", 0)
    func.call @do_sieve(%buf24_2, %buf24_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_2, "Release", 0)
    aie.use_lock(%lock24_3, "Release", 1)
    aie.end
  }

  %core24_4 = aie.core(%tile24_4) {
    aie.use_lock(%lock24_3, "Acquire", 1)
    aie.use_lock(%lock24_4, "Acquire", 0)
    func.call @do_sieve(%buf24_3, %buf24_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_3, "Release", 0)
    aie.use_lock(%lock24_4, "Release", 1)
    aie.end
  }

  %core24_5 = aie.core(%tile24_5) {
    aie.use_lock(%lock24_4, "Acquire", 1)
    aie.use_lock(%lock24_5, "Acquire", 0)
    func.call @do_sieve(%buf24_4, %buf24_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_4, "Release", 0)
    aie.use_lock(%lock24_5, "Release", 1)
    aie.end
  }

  %core24_6 = aie.core(%tile24_6) {
    aie.use_lock(%lock24_5, "Acquire", 1)
    aie.use_lock(%lock24_6, "Acquire", 0)
    func.call @do_sieve(%buf24_5, %buf24_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_5, "Release", 0)
    aie.use_lock(%lock24_6, "Release", 1)
    aie.end
  }

  %core24_7 = aie.core(%tile24_7) {
    aie.use_lock(%lock24_6, "Acquire", 1)
    aie.use_lock(%lock24_7, "Acquire", 0)
    func.call @do_sieve(%buf24_6, %buf24_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_6, "Release", 0)
    aie.use_lock(%lock24_7, "Release", 1)
    aie.end
  }

  %core24_8 = aie.core(%tile24_8) {
    aie.use_lock(%lock24_7, "Acquire", 1)
    aie.use_lock(%lock24_8, "Acquire", 0)
    func.call @do_sieve(%buf24_7, %buf24_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_7, "Release", 0)
    aie.use_lock(%lock24_8, "Release", 1)
    aie.end
  }

  %core25_8 = aie.core(%tile25_8) {
    aie.use_lock(%lock24_8, "Acquire", 1)
    aie.use_lock(%lock25_8, "Acquire", 0)
    func.call @do_sieve(%buf24_8, %buf25_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock24_8, "Release", 0)
    aie.use_lock(%lock25_8, "Release", 1)
    aie.end
  }

  %core25_7 = aie.core(%tile25_7) {
    aie.use_lock(%lock25_8, "Acquire", 1)
    aie.use_lock(%lock25_7, "Acquire", 0)
    func.call @do_sieve(%buf25_8, %buf25_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_8, "Release", 0)
    aie.use_lock(%lock25_7, "Release", 1)
    aie.end
  }

  %core25_6 = aie.core(%tile25_6) {
    aie.use_lock(%lock25_7, "Acquire", 1)
    aie.use_lock(%lock25_6, "Acquire", 0)
    func.call @do_sieve(%buf25_7, %buf25_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_7, "Release", 0)
    aie.use_lock(%lock25_6, "Release", 1)
    aie.end
  }

  %core25_5 = aie.core(%tile25_5) {
    aie.use_lock(%lock25_6, "Acquire", 1)
    aie.use_lock(%lock25_5, "Acquire", 0)
    func.call @do_sieve(%buf25_6, %buf25_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_6, "Release", 0)
    aie.use_lock(%lock25_5, "Release", 1)
    aie.end
  }

  %core25_4 = aie.core(%tile25_4) {
    aie.use_lock(%lock25_5, "Acquire", 1)
    aie.use_lock(%lock25_4, "Acquire", 0)
    func.call @do_sieve(%buf25_5, %buf25_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_5, "Release", 0)
    aie.use_lock(%lock25_4, "Release", 1)
    aie.end
  }

  %core25_3 = aie.core(%tile25_3) {
    aie.use_lock(%lock25_4, "Acquire", 1)
    aie.use_lock(%lock25_3, "Acquire", 0)
    func.call @do_sieve(%buf25_4, %buf25_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_4, "Release", 0)
    aie.use_lock(%lock25_3, "Release", 1)
    aie.end
  }

  %core25_2 = aie.core(%tile25_2) {
    aie.use_lock(%lock25_3, "Acquire", 1)
    aie.use_lock(%lock25_2, "Acquire", 0)
    func.call @do_sieve(%buf25_3, %buf25_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_3, "Release", 0)
    aie.use_lock(%lock25_2, "Release", 1)
    aie.end
  }

  %core25_1 = aie.core(%tile25_1) {
    aie.use_lock(%lock25_2, "Acquire", 1)
    aie.use_lock(%lock25_1, "Acquire", 0)
    func.call @do_sieve(%buf25_2, %buf25_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_2, "Release", 0)
    aie.use_lock(%lock25_1, "Release", 1)
    aie.end
  }

  %core26_1 = aie.core(%tile26_1) {
    aie.use_lock(%lock25_1, "Acquire", 1)
    aie.use_lock(%lock26_1, "Acquire", 0)
    func.call @do_sieve(%buf25_1, %buf26_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock25_1, "Release", 0)
    aie.use_lock(%lock26_1, "Release", 1)
    aie.end
  }

  %core26_2 = aie.core(%tile26_2) {
    aie.use_lock(%lock26_1, "Acquire", 1)
    aie.use_lock(%lock26_2, "Acquire", 0)
    func.call @do_sieve(%buf26_1, %buf26_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_1, "Release", 0)
    aie.use_lock(%lock26_2, "Release", 1)
    aie.end
  }

  %core26_3 = aie.core(%tile26_3) {
    aie.use_lock(%lock26_2, "Acquire", 1)
    aie.use_lock(%lock26_3, "Acquire", 0)
    func.call @do_sieve(%buf26_2, %buf26_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_2, "Release", 0)
    aie.use_lock(%lock26_3, "Release", 1)
    aie.end
  }

  %core26_4 = aie.core(%tile26_4) {
    aie.use_lock(%lock26_3, "Acquire", 1)
    aie.use_lock(%lock26_4, "Acquire", 0)
    func.call @do_sieve(%buf26_3, %buf26_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_3, "Release", 0)
    aie.use_lock(%lock26_4, "Release", 1)
    aie.end
  }

  %core26_5 = aie.core(%tile26_5) {
    aie.use_lock(%lock26_4, "Acquire", 1)
    aie.use_lock(%lock26_5, "Acquire", 0)
    func.call @do_sieve(%buf26_4, %buf26_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_4, "Release", 0)
    aie.use_lock(%lock26_5, "Release", 1)
    aie.end
  }

  %core26_6 = aie.core(%tile26_6) {
    aie.use_lock(%lock26_5, "Acquire", 1)
    aie.use_lock(%lock26_6, "Acquire", 0)
    func.call @do_sieve(%buf26_5, %buf26_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_5, "Release", 0)
    aie.use_lock(%lock26_6, "Release", 1)
    aie.end
  }

  %core26_7 = aie.core(%tile26_7) {
    aie.use_lock(%lock26_6, "Acquire", 1)
    aie.use_lock(%lock26_7, "Acquire", 0)
    func.call @do_sieve(%buf26_6, %buf26_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_6, "Release", 0)
    aie.use_lock(%lock26_7, "Release", 1)
    aie.end
  }

  %core26_8 = aie.core(%tile26_8) {
    aie.use_lock(%lock26_7, "Acquire", 1)
    aie.use_lock(%lock26_8, "Acquire", 0)
    func.call @do_sieve(%buf26_7, %buf26_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_7, "Release", 0)
    aie.use_lock(%lock26_8, "Release", 1)
    aie.end
  }

  %core27_8 = aie.core(%tile27_8) {
    aie.use_lock(%lock26_8, "Acquire", 1)
    aie.use_lock(%lock27_8, "Acquire", 0)
    func.call @do_sieve(%buf26_8, %buf27_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock26_8, "Release", 0)
    aie.use_lock(%lock27_8, "Release", 1)
    aie.end
  }

  %core27_7 = aie.core(%tile27_7) {
    aie.use_lock(%lock27_8, "Acquire", 1)
    aie.use_lock(%lock27_7, "Acquire", 0)
    func.call @do_sieve(%buf27_8, %buf27_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_8, "Release", 0)
    aie.use_lock(%lock27_7, "Release", 1)
    aie.end
  }

  %core27_6 = aie.core(%tile27_6) {
    aie.use_lock(%lock27_7, "Acquire", 1)
    aie.use_lock(%lock27_6, "Acquire", 0)
    func.call @do_sieve(%buf27_7, %buf27_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_7, "Release", 0)
    aie.use_lock(%lock27_6, "Release", 1)
    aie.end
  }

  %core27_5 = aie.core(%tile27_5) {
    aie.use_lock(%lock27_6, "Acquire", 1)
    aie.use_lock(%lock27_5, "Acquire", 0)
    func.call @do_sieve(%buf27_6, %buf27_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_6, "Release", 0)
    aie.use_lock(%lock27_5, "Release", 1)
    aie.end
  }

  %core27_4 = aie.core(%tile27_4) {
    aie.use_lock(%lock27_5, "Acquire", 1)
    aie.use_lock(%lock27_4, "Acquire", 0)
    func.call @do_sieve(%buf27_5, %buf27_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_5, "Release", 0)
    aie.use_lock(%lock27_4, "Release", 1)
    aie.end
  }

  %core27_3 = aie.core(%tile27_3) {
    aie.use_lock(%lock27_4, "Acquire", 1)
    aie.use_lock(%lock27_3, "Acquire", 0)
    func.call @do_sieve(%buf27_4, %buf27_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_4, "Release", 0)
    aie.use_lock(%lock27_3, "Release", 1)
    aie.end
  }

  %core27_2 = aie.core(%tile27_2) {
    aie.use_lock(%lock27_3, "Acquire", 1)
    aie.use_lock(%lock27_2, "Acquire", 0)
    func.call @do_sieve(%buf27_3, %buf27_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_3, "Release", 0)
    aie.use_lock(%lock27_2, "Release", 1)
    aie.end
  }

  %core27_1 = aie.core(%tile27_1) {
    aie.use_lock(%lock27_2, "Acquire", 1)
    aie.use_lock(%lock27_1, "Acquire", 0)
    func.call @do_sieve(%buf27_2, %buf27_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_2, "Release", 0)
    aie.use_lock(%lock27_1, "Release", 1)
    aie.end
  }

  %core28_1 = aie.core(%tile28_1) {
    aie.use_lock(%lock27_1, "Acquire", 1)
    aie.use_lock(%lock28_1, "Acquire", 0)
    func.call @do_sieve(%buf27_1, %buf28_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock27_1, "Release", 0)
    aie.use_lock(%lock28_1, "Release", 1)
    aie.end
  }

  %core28_2 = aie.core(%tile28_2) {
    aie.use_lock(%lock28_1, "Acquire", 1)
    aie.use_lock(%lock28_2, "Acquire", 0)
    func.call @do_sieve(%buf28_1, %buf28_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_1, "Release", 0)
    aie.use_lock(%lock28_2, "Release", 1)
    aie.end
  }

  %core28_3 = aie.core(%tile28_3) {
    aie.use_lock(%lock28_2, "Acquire", 1)
    aie.use_lock(%lock28_3, "Acquire", 0)
    func.call @do_sieve(%buf28_2, %buf28_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_2, "Release", 0)
    aie.use_lock(%lock28_3, "Release", 1)
    aie.end
  }

  %core28_4 = aie.core(%tile28_4) {
    aie.use_lock(%lock28_3, "Acquire", 1)
    aie.use_lock(%lock28_4, "Acquire", 0)
    func.call @do_sieve(%buf28_3, %buf28_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_3, "Release", 0)
    aie.use_lock(%lock28_4, "Release", 1)
    aie.end
  }

  %core28_5 = aie.core(%tile28_5) {
    aie.use_lock(%lock28_4, "Acquire", 1)
    aie.use_lock(%lock28_5, "Acquire", 0)
    func.call @do_sieve(%buf28_4, %buf28_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_4, "Release", 0)
    aie.use_lock(%lock28_5, "Release", 1)
    aie.end
  }

  %core28_6 = aie.core(%tile28_6) {
    aie.use_lock(%lock28_5, "Acquire", 1)
    aie.use_lock(%lock28_6, "Acquire", 0)
    func.call @do_sieve(%buf28_5, %buf28_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_5, "Release", 0)
    aie.use_lock(%lock28_6, "Release", 1)
    aie.end
  }

  %core28_7 = aie.core(%tile28_7) {
    aie.use_lock(%lock28_6, "Acquire", 1)
    aie.use_lock(%lock28_7, "Acquire", 0)
    func.call @do_sieve(%buf28_6, %buf28_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_6, "Release", 0)
    aie.use_lock(%lock28_7, "Release", 1)
    aie.end
  }

  %core28_8 = aie.core(%tile28_8) {
    aie.use_lock(%lock28_7, "Acquire", 1)
    aie.use_lock(%lock28_8, "Acquire", 0)
    func.call @do_sieve(%buf28_7, %buf28_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_7, "Release", 0)
    aie.use_lock(%lock28_8, "Release", 1)
    aie.end
  }

  %core29_8 = aie.core(%tile29_8) {
    aie.use_lock(%lock28_8, "Acquire", 1)
    aie.use_lock(%lock29_8, "Acquire", 0)
    func.call @do_sieve(%buf28_8, %buf29_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock28_8, "Release", 0)
    aie.use_lock(%lock29_8, "Release", 1)
    aie.end
  }

  %core29_7 = aie.core(%tile29_7) {
    aie.use_lock(%lock29_8, "Acquire", 1)
    aie.use_lock(%lock29_7, "Acquire", 0)
    func.call @do_sieve(%buf29_8, %buf29_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_8, "Release", 0)
    aie.use_lock(%lock29_7, "Release", 1)
    aie.end
  }

  %core29_6 = aie.core(%tile29_6) {
    aie.use_lock(%lock29_7, "Acquire", 1)
    aie.use_lock(%lock29_6, "Acquire", 0)
    func.call @do_sieve(%buf29_7, %buf29_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_7, "Release", 0)
    aie.use_lock(%lock29_6, "Release", 1)
    aie.end
  }

  %core29_5 = aie.core(%tile29_5) {
    aie.use_lock(%lock29_6, "Acquire", 1)
    aie.use_lock(%lock29_5, "Acquire", 0)
    func.call @do_sieve(%buf29_6, %buf29_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_6, "Release", 0)
    aie.use_lock(%lock29_5, "Release", 1)
    aie.end
  }

  %core29_4 = aie.core(%tile29_4) {
    aie.use_lock(%lock29_5, "Acquire", 1)
    aie.use_lock(%lock29_4, "Acquire", 0)
    func.call @do_sieve(%buf29_5, %buf29_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_5, "Release", 0)
    aie.use_lock(%lock29_4, "Release", 1)
    aie.end
  }

  %core29_3 = aie.core(%tile29_3) {
    aie.use_lock(%lock29_4, "Acquire", 1)
    aie.use_lock(%lock29_3, "Acquire", 0)
    func.call @do_sieve(%buf29_4, %buf29_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_4, "Release", 0)
    aie.use_lock(%lock29_3, "Release", 1)
    aie.end
  }

  %core29_2 = aie.core(%tile29_2) {
    aie.use_lock(%lock29_3, "Acquire", 1)
    aie.use_lock(%lock29_2, "Acquire", 0)
    func.call @do_sieve(%buf29_3, %buf29_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_3, "Release", 0)
    aie.use_lock(%lock29_2, "Release", 1)
    aie.end
  }

  %core29_1 = aie.core(%tile29_1) {
    aie.use_lock(%lock29_2, "Acquire", 1)
    aie.use_lock(%lock29_1, "Acquire", 0)
    func.call @do_sieve(%buf29_2, %buf29_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_2, "Release", 0)
    aie.use_lock(%lock29_1, "Release", 1)
    aie.end
  }

  %core30_1 = aie.core(%tile30_1) {
    aie.use_lock(%lock29_1, "Acquire", 1)
    aie.use_lock(%lock30_1, "Acquire", 0)
    func.call @do_sieve(%buf29_1, %buf30_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock29_1, "Release", 0)
    aie.use_lock(%lock30_1, "Release", 1)
    aie.end
  }

  %core30_2 = aie.core(%tile30_2) {
    aie.use_lock(%lock30_1, "Acquire", 1)
    aie.use_lock(%lock30_2, "Acquire", 0)
    func.call @do_sieve(%buf30_1, %buf30_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_1, "Release", 0)
    aie.use_lock(%lock30_2, "Release", 1)
    aie.end
  }

  %core30_3 = aie.core(%tile30_3) {
    aie.use_lock(%lock30_2, "Acquire", 1)
    aie.use_lock(%lock30_3, "Acquire", 0)
    func.call @do_sieve(%buf30_2, %buf30_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_2, "Release", 0)
    aie.use_lock(%lock30_3, "Release", 1)
    aie.end
  }

  %core30_4 = aie.core(%tile30_4) {
    aie.use_lock(%lock30_3, "Acquire", 1)
    aie.use_lock(%lock30_4, "Acquire", 0)
    func.call @do_sieve(%buf30_3, %buf30_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_3, "Release", 0)
    aie.use_lock(%lock30_4, "Release", 1)
    aie.end
  }

  %core30_5 = aie.core(%tile30_5) {
    aie.use_lock(%lock30_4, "Acquire", 1)
    aie.use_lock(%lock30_5, "Acquire", 0)
    func.call @do_sieve(%buf30_4, %buf30_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_4, "Release", 0)
    aie.use_lock(%lock30_5, "Release", 1)
    aie.end
  }

  %core30_6 = aie.core(%tile30_6) {
    aie.use_lock(%lock30_5, "Acquire", 1)
    aie.use_lock(%lock30_6, "Acquire", 0)
    func.call @do_sieve(%buf30_5, %buf30_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_5, "Release", 0)
    aie.use_lock(%lock30_6, "Release", 1)
    aie.end
  }

  %core30_7 = aie.core(%tile30_7) {
    aie.use_lock(%lock30_6, "Acquire", 1)
    aie.use_lock(%lock30_7, "Acquire", 0)
    func.call @do_sieve(%buf30_6, %buf30_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_6, "Release", 0)
    aie.use_lock(%lock30_7, "Release", 1)
    aie.end
  }

  %core30_8 = aie.core(%tile30_8) {
    aie.use_lock(%lock30_7, "Acquire", 1)
    aie.use_lock(%lock30_8, "Acquire", 0)
    func.call @do_sieve(%buf30_7, %buf30_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_7, "Release", 0)
    aie.use_lock(%lock30_8, "Release", 1)
    aie.end
  }

  %core31_8 = aie.core(%tile31_8) {
    aie.use_lock(%lock30_8, "Acquire", 1)
    aie.use_lock(%lock31_8, "Acquire", 0)
    func.call @do_sieve(%buf30_8, %buf31_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock30_8, "Release", 0)
    aie.use_lock(%lock31_8, "Release", 1)
    aie.end
  }

  %core31_7 = aie.core(%tile31_7) {
    aie.use_lock(%lock31_8, "Acquire", 1)
    aie.use_lock(%lock31_7, "Acquire", 0)
    func.call @do_sieve(%buf31_8, %buf31_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_8, "Release", 0)
    aie.use_lock(%lock31_7, "Release", 1)
    aie.end
  }

  %core31_6 = aie.core(%tile31_6) {
    aie.use_lock(%lock31_7, "Acquire", 1)
    aie.use_lock(%lock31_6, "Acquire", 0)
    func.call @do_sieve(%buf31_7, %buf31_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_7, "Release", 0)
    aie.use_lock(%lock31_6, "Release", 1)
    aie.end
  }

  %core31_5 = aie.core(%tile31_5) {
    aie.use_lock(%lock31_6, "Acquire", 1)
    aie.use_lock(%lock31_5, "Acquire", 0)
    func.call @do_sieve(%buf31_6, %buf31_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_6, "Release", 0)
    aie.use_lock(%lock31_5, "Release", 1)
    aie.end
  }

  %core31_4 = aie.core(%tile31_4) {
    aie.use_lock(%lock31_5, "Acquire", 1)
    aie.use_lock(%lock31_4, "Acquire", 0)
    func.call @do_sieve(%buf31_5, %buf31_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_5, "Release", 0)
    aie.use_lock(%lock31_4, "Release", 1)
    aie.end
  }

  %core31_3 = aie.core(%tile31_3) {
    aie.use_lock(%lock31_4, "Acquire", 1)
    aie.use_lock(%lock31_3, "Acquire", 0)
    func.call @do_sieve(%buf31_4, %buf31_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_4, "Release", 0)
    aie.use_lock(%lock31_3, "Release", 1)
    aie.end
  }

  %core31_2 = aie.core(%tile31_2) {
    aie.use_lock(%lock31_3, "Acquire", 1)
    aie.use_lock(%lock31_2, "Acquire", 0)
    func.call @do_sieve(%buf31_3, %buf31_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_3, "Release", 0)
    aie.use_lock(%lock31_2, "Release", 1)
    aie.end
  }

  %core31_1 = aie.core(%tile31_1) {
    aie.use_lock(%lock31_2, "Acquire", 1)
    aie.use_lock(%lock31_1, "Acquire", 0)
    func.call @do_sieve(%buf31_2, %buf31_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_2, "Release", 0)
    aie.use_lock(%lock31_1, "Release", 1)
    aie.end
  }

  %core32_1 = aie.core(%tile32_1) {
    aie.use_lock(%lock31_1, "Acquire", 1)
    aie.use_lock(%lock32_1, "Acquire", 0)
    func.call @do_sieve(%buf31_1, %buf32_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock31_1, "Release", 0)
    aie.use_lock(%lock32_1, "Release", 1)
    aie.end
  }

  %core32_2 = aie.core(%tile32_2) {
    aie.use_lock(%lock32_1, "Acquire", 1)
    aie.use_lock(%lock32_2, "Acquire", 0)
    func.call @do_sieve(%buf32_1, %buf32_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_1, "Release", 0)
    aie.use_lock(%lock32_2, "Release", 1)
    aie.end
  }

  %core32_3 = aie.core(%tile32_3) {
    aie.use_lock(%lock32_2, "Acquire", 1)
    aie.use_lock(%lock32_3, "Acquire", 0)
    func.call @do_sieve(%buf32_2, %buf32_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_2, "Release", 0)
    aie.use_lock(%lock32_3, "Release", 1)
    aie.end
  }

  %core32_4 = aie.core(%tile32_4) {
    aie.use_lock(%lock32_3, "Acquire", 1)
    aie.use_lock(%lock32_4, "Acquire", 0)
    func.call @do_sieve(%buf32_3, %buf32_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_3, "Release", 0)
    aie.use_lock(%lock32_4, "Release", 1)
    aie.end
  }

  %core32_5 = aie.core(%tile32_5) {
    aie.use_lock(%lock32_4, "Acquire", 1)
    aie.use_lock(%lock32_5, "Acquire", 0)
    func.call @do_sieve(%buf32_4, %buf32_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_4, "Release", 0)
    aie.use_lock(%lock32_5, "Release", 1)
    aie.end
  }

  %core32_6 = aie.core(%tile32_6) {
    aie.use_lock(%lock32_5, "Acquire", 1)
    aie.use_lock(%lock32_6, "Acquire", 0)
    func.call @do_sieve(%buf32_5, %buf32_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_5, "Release", 0)
    aie.use_lock(%lock32_6, "Release", 1)
    aie.end
  }

  %core32_7 = aie.core(%tile32_7) {
    aie.use_lock(%lock32_6, "Acquire", 1)
    aie.use_lock(%lock32_7, "Acquire", 0)
    func.call @do_sieve(%buf32_6, %buf32_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_6, "Release", 0)
    aie.use_lock(%lock32_7, "Release", 1)
    aie.end
  }

  %core32_8 = aie.core(%tile32_8) {
    aie.use_lock(%lock32_7, "Acquire", 1)
    aie.use_lock(%lock32_8, "Acquire", 0)
    func.call @do_sieve(%buf32_7, %buf32_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_7, "Release", 0)
    aie.use_lock(%lock32_8, "Release", 1)
    aie.end
  }

  %core33_8 = aie.core(%tile33_8) {
    aie.use_lock(%lock32_8, "Acquire", 1)
    aie.use_lock(%lock33_8, "Acquire", 0)
    func.call @do_sieve(%buf32_8, %buf33_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock32_8, "Release", 0)
    aie.use_lock(%lock33_8, "Release", 1)
    aie.end
  }

  %core33_7 = aie.core(%tile33_7) {
    aie.use_lock(%lock33_8, "Acquire", 1)
    aie.use_lock(%lock33_7, "Acquire", 0)
    func.call @do_sieve(%buf33_8, %buf33_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_8, "Release", 0)
    aie.use_lock(%lock33_7, "Release", 1)
    aie.end
  }

  %core33_6 = aie.core(%tile33_6) {
    aie.use_lock(%lock33_7, "Acquire", 1)
    aie.use_lock(%lock33_6, "Acquire", 0)
    func.call @do_sieve(%buf33_7, %buf33_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_7, "Release", 0)
    aie.use_lock(%lock33_6, "Release", 1)
    aie.end
  }

  %core33_5 = aie.core(%tile33_5) {
    aie.use_lock(%lock33_6, "Acquire", 1)
    aie.use_lock(%lock33_5, "Acquire", 0)
    func.call @do_sieve(%buf33_6, %buf33_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_6, "Release", 0)
    aie.use_lock(%lock33_5, "Release", 1)
    aie.end
  }

  %core33_4 = aie.core(%tile33_4) {
    aie.use_lock(%lock33_5, "Acquire", 1)
    aie.use_lock(%lock33_4, "Acquire", 0)
    func.call @do_sieve(%buf33_5, %buf33_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_5, "Release", 0)
    aie.use_lock(%lock33_4, "Release", 1)
    aie.end
  }

  %core33_3 = aie.core(%tile33_3) {
    aie.use_lock(%lock33_4, "Acquire", 1)
    aie.use_lock(%lock33_3, "Acquire", 0)
    func.call @do_sieve(%buf33_4, %buf33_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_4, "Release", 0)
    aie.use_lock(%lock33_3, "Release", 1)
    aie.end
  }

  %core33_2 = aie.core(%tile33_2) {
    aie.use_lock(%lock33_3, "Acquire", 1)
    aie.use_lock(%lock33_2, "Acquire", 0)
    func.call @do_sieve(%buf33_3, %buf33_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_3, "Release", 0)
    aie.use_lock(%lock33_2, "Release", 1)
    aie.end
  }

  %core33_1 = aie.core(%tile33_1) {
    aie.use_lock(%lock33_2, "Acquire", 1)
    aie.use_lock(%lock33_1, "Acquire", 0)
    func.call @do_sieve(%buf33_2, %buf33_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_2, "Release", 0)
    aie.use_lock(%lock33_1, "Release", 1)
    aie.end
  }

  %core34_1 = aie.core(%tile34_1) {
    aie.use_lock(%lock33_1, "Acquire", 1)
    aie.use_lock(%lock34_1, "Acquire", 0)
    func.call @do_sieve(%buf33_1, %buf34_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock33_1, "Release", 0)
    aie.use_lock(%lock34_1, "Release", 1)
    aie.end
  }

  %core34_2 = aie.core(%tile34_2) {
    aie.use_lock(%lock34_1, "Acquire", 1)
    aie.use_lock(%lock34_2, "Acquire", 0)
    func.call @do_sieve(%buf34_1, %buf34_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_1, "Release", 0)
    aie.use_lock(%lock34_2, "Release", 1)
    aie.end
  }

  %core34_3 = aie.core(%tile34_3) {
    aie.use_lock(%lock34_2, "Acquire", 1)
    aie.use_lock(%lock34_3, "Acquire", 0)
    func.call @do_sieve(%buf34_2, %buf34_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_2, "Release", 0)
    aie.use_lock(%lock34_3, "Release", 1)
    aie.end
  }

  %core34_4 = aie.core(%tile34_4) {
    aie.use_lock(%lock34_3, "Acquire", 1)
    aie.use_lock(%lock34_4, "Acquire", 0)
    func.call @do_sieve(%buf34_3, %buf34_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_3, "Release", 0)
    aie.use_lock(%lock34_4, "Release", 1)
    aie.end
  }

  %core34_5 = aie.core(%tile34_5) {
    aie.use_lock(%lock34_4, "Acquire", 1)
    aie.use_lock(%lock34_5, "Acquire", 0)
    func.call @do_sieve(%buf34_4, %buf34_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_4, "Release", 0)
    aie.use_lock(%lock34_5, "Release", 1)
    aie.end
  }

  %core34_6 = aie.core(%tile34_6) {
    aie.use_lock(%lock34_5, "Acquire", 1)
    aie.use_lock(%lock34_6, "Acquire", 0)
    func.call @do_sieve(%buf34_5, %buf34_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_5, "Release", 0)
    aie.use_lock(%lock34_6, "Release", 1)
    aie.end
  }

  %core34_7 = aie.core(%tile34_7) {
    aie.use_lock(%lock34_6, "Acquire", 1)
    aie.use_lock(%lock34_7, "Acquire", 0)
    func.call @do_sieve(%buf34_6, %buf34_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_6, "Release", 0)
    aie.use_lock(%lock34_7, "Release", 1)
    aie.end
  }

  %core34_8 = aie.core(%tile34_8) {
    aie.use_lock(%lock34_7, "Acquire", 1)
    aie.use_lock(%lock34_8, "Acquire", 0)
    func.call @do_sieve(%buf34_7, %buf34_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_7, "Release", 0)
    aie.use_lock(%lock34_8, "Release", 1)
    aie.end
  }

  %core35_8 = aie.core(%tile35_8) {
    aie.use_lock(%lock34_8, "Acquire", 1)
    aie.use_lock(%lock35_8, "Acquire", 0)
    func.call @do_sieve(%buf34_8, %buf35_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock34_8, "Release", 0)
    aie.use_lock(%lock35_8, "Release", 1)
    aie.end
  }

  %core35_7 = aie.core(%tile35_7) {
    aie.use_lock(%lock35_8, "Acquire", 1)
    aie.use_lock(%lock35_7, "Acquire", 0)
    func.call @do_sieve(%buf35_8, %buf35_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_8, "Release", 0)
    aie.use_lock(%lock35_7, "Release", 1)
    aie.end
  }

  %core35_6 = aie.core(%tile35_6) {
    aie.use_lock(%lock35_7, "Acquire", 1)
    aie.use_lock(%lock35_6, "Acquire", 0)
    func.call @do_sieve(%buf35_7, %buf35_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_7, "Release", 0)
    aie.use_lock(%lock35_6, "Release", 1)
    aie.end
  }

  %core35_5 = aie.core(%tile35_5) {
    aie.use_lock(%lock35_6, "Acquire", 1)
    aie.use_lock(%lock35_5, "Acquire", 0)
    func.call @do_sieve(%buf35_6, %buf35_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_6, "Release", 0)
    aie.use_lock(%lock35_5, "Release", 1)
    aie.end
  }

  %core35_4 = aie.core(%tile35_4) {
    aie.use_lock(%lock35_5, "Acquire", 1)
    aie.use_lock(%lock35_4, "Acquire", 0)
    func.call @do_sieve(%buf35_5, %buf35_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_5, "Release", 0)
    aie.use_lock(%lock35_4, "Release", 1)
    aie.end
  }

  %core35_3 = aie.core(%tile35_3) {
    aie.use_lock(%lock35_4, "Acquire", 1)
    aie.use_lock(%lock35_3, "Acquire", 0)
    func.call @do_sieve(%buf35_4, %buf35_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_4, "Release", 0)
    aie.use_lock(%lock35_3, "Release", 1)
    aie.end
  }

  %core35_2 = aie.core(%tile35_2) {
    aie.use_lock(%lock35_3, "Acquire", 1)
    aie.use_lock(%lock35_2, "Acquire", 0)
    func.call @do_sieve(%buf35_3, %buf35_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_3, "Release", 0)
    aie.use_lock(%lock35_2, "Release", 1)
    aie.end
  }

  %core35_1 = aie.core(%tile35_1) {
    aie.use_lock(%lock35_2, "Acquire", 1)
    aie.use_lock(%lock35_1, "Acquire", 0)
    func.call @do_sieve(%buf35_2, %buf35_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_2, "Release", 0)
    aie.use_lock(%lock35_1, "Release", 1)
    aie.end
  }

  %core36_1 = aie.core(%tile36_1) {
    aie.use_lock(%lock35_1, "Acquire", 1)
    aie.use_lock(%lock36_1, "Acquire", 0)
    func.call @do_sieve(%buf35_1, %buf36_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock35_1, "Release", 0)
    aie.use_lock(%lock36_1, "Release", 1)
    aie.end
  }

  %core36_2 = aie.core(%tile36_2) {
    aie.use_lock(%lock36_1, "Acquire", 1)
    aie.use_lock(%lock36_2, "Acquire", 0)
    func.call @do_sieve(%buf36_1, %buf36_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_1, "Release", 0)
    aie.use_lock(%lock36_2, "Release", 1)
    aie.end
  }

  %core36_3 = aie.core(%tile36_3) {
    aie.use_lock(%lock36_2, "Acquire", 1)
    aie.use_lock(%lock36_3, "Acquire", 0)
    func.call @do_sieve(%buf36_2, %buf36_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_2, "Release", 0)
    aie.use_lock(%lock36_3, "Release", 1)
    aie.end
  }

  %core36_4 = aie.core(%tile36_4) {
    aie.use_lock(%lock36_3, "Acquire", 1)
    aie.use_lock(%lock36_4, "Acquire", 0)
    func.call @do_sieve(%buf36_3, %buf36_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_3, "Release", 0)
    aie.use_lock(%lock36_4, "Release", 1)
    aie.end
  }

  %core36_5 = aie.core(%tile36_5) {
    aie.use_lock(%lock36_4, "Acquire", 1)
    aie.use_lock(%lock36_5, "Acquire", 0)
    func.call @do_sieve(%buf36_4, %buf36_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_4, "Release", 0)
    aie.use_lock(%lock36_5, "Release", 1)
    aie.end
  }

  %core36_6 = aie.core(%tile36_6) {
    aie.use_lock(%lock36_5, "Acquire", 1)
    aie.use_lock(%lock36_6, "Acquire", 0)
    func.call @do_sieve(%buf36_5, %buf36_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_5, "Release", 0)
    aie.use_lock(%lock36_6, "Release", 1)
    aie.end
  }

  %core36_7 = aie.core(%tile36_7) {
    aie.use_lock(%lock36_6, "Acquire", 1)
    aie.use_lock(%lock36_7, "Acquire", 0)
    func.call @do_sieve(%buf36_6, %buf36_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_6, "Release", 0)
    aie.use_lock(%lock36_7, "Release", 1)
    aie.end
  }

  %core36_8 = aie.core(%tile36_8) {
    aie.use_lock(%lock36_7, "Acquire", 1)
    aie.use_lock(%lock36_8, "Acquire", 0)
    func.call @do_sieve(%buf36_7, %buf36_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_7, "Release", 0)
    aie.use_lock(%lock36_8, "Release", 1)
    aie.end
  }

  %core37_8 = aie.core(%tile37_8) {
    aie.use_lock(%lock36_8, "Acquire", 1)
    aie.use_lock(%lock37_8, "Acquire", 0)
    func.call @do_sieve(%buf36_8, %buf37_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock36_8, "Release", 0)
    aie.use_lock(%lock37_8, "Release", 1)
    aie.end
  }

  %core37_7 = aie.core(%tile37_7) {
    aie.use_lock(%lock37_8, "Acquire", 1)
    aie.use_lock(%lock37_7, "Acquire", 0)
    func.call @do_sieve(%buf37_8, %buf37_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_8, "Release", 0)
    aie.use_lock(%lock37_7, "Release", 1)
    aie.end
  }

  %core37_6 = aie.core(%tile37_6) {
    aie.use_lock(%lock37_7, "Acquire", 1)
    aie.use_lock(%lock37_6, "Acquire", 0)
    func.call @do_sieve(%buf37_7, %buf37_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_7, "Release", 0)
    aie.use_lock(%lock37_6, "Release", 1)
    aie.end
  }

  %core37_5 = aie.core(%tile37_5) {
    aie.use_lock(%lock37_6, "Acquire", 1)
    aie.use_lock(%lock37_5, "Acquire", 0)
    func.call @do_sieve(%buf37_6, %buf37_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_6, "Release", 0)
    aie.use_lock(%lock37_5, "Release", 1)
    aie.end
  }

  %core37_4 = aie.core(%tile37_4) {
    aie.use_lock(%lock37_5, "Acquire", 1)
    aie.use_lock(%lock37_4, "Acquire", 0)
    func.call @do_sieve(%buf37_5, %buf37_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_5, "Release", 0)
    aie.use_lock(%lock37_4, "Release", 1)
    aie.end
  }

  %core37_3 = aie.core(%tile37_3) {
    aie.use_lock(%lock37_4, "Acquire", 1)
    aie.use_lock(%lock37_3, "Acquire", 0)
    func.call @do_sieve(%buf37_4, %buf37_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_4, "Release", 0)
    aie.use_lock(%lock37_3, "Release", 1)
    aie.end
  }

  %core37_2 = aie.core(%tile37_2) {
    aie.use_lock(%lock37_3, "Acquire", 1)
    aie.use_lock(%lock37_2, "Acquire", 0)
    func.call @do_sieve(%buf37_3, %buf37_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_3, "Release", 0)
    aie.use_lock(%lock37_2, "Release", 1)
    aie.end
  }

  %core37_1 = aie.core(%tile37_1) {
    aie.use_lock(%lock37_2, "Acquire", 1)
    aie.use_lock(%lock37_1, "Acquire", 0)
    func.call @do_sieve(%buf37_2, %buf37_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_2, "Release", 0)
    aie.use_lock(%lock37_1, "Release", 1)
    aie.end
  }

  %core38_1 = aie.core(%tile38_1) {
    aie.use_lock(%lock37_1, "Acquire", 1)
    aie.use_lock(%lock38_1, "Acquire", 0)
    func.call @do_sieve(%buf37_1, %buf38_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock37_1, "Release", 0)
    aie.use_lock(%lock38_1, "Release", 1)
    aie.end
  }

  %core38_2 = aie.core(%tile38_2) {
    aie.use_lock(%lock38_1, "Acquire", 1)
    aie.use_lock(%lock38_2, "Acquire", 0)
    func.call @do_sieve(%buf38_1, %buf38_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_1, "Release", 0)
    aie.use_lock(%lock38_2, "Release", 1)
    aie.end
  }

  %core38_3 = aie.core(%tile38_3) {
    aie.use_lock(%lock38_2, "Acquire", 1)
    aie.use_lock(%lock38_3, "Acquire", 0)
    func.call @do_sieve(%buf38_2, %buf38_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_2, "Release", 0)
    aie.use_lock(%lock38_3, "Release", 1)
    aie.end
  }

  %core38_4 = aie.core(%tile38_4) {
    aie.use_lock(%lock38_3, "Acquire", 1)
    aie.use_lock(%lock38_4, "Acquire", 0)
    func.call @do_sieve(%buf38_3, %buf38_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_3, "Release", 0)
    aie.use_lock(%lock38_4, "Release", 1)
    aie.end
  }

  %core38_5 = aie.core(%tile38_5) {
    aie.use_lock(%lock38_4, "Acquire", 1)
    aie.use_lock(%lock38_5, "Acquire", 0)
    func.call @do_sieve(%buf38_4, %buf38_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_4, "Release", 0)
    aie.use_lock(%lock38_5, "Release", 1)
    aie.end
  }

  %core38_6 = aie.core(%tile38_6) {
    aie.use_lock(%lock38_5, "Acquire", 1)
    aie.use_lock(%lock38_6, "Acquire", 0)
    func.call @do_sieve(%buf38_5, %buf38_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_5, "Release", 0)
    aie.use_lock(%lock38_6, "Release", 1)
    aie.end
  }

  %core38_7 = aie.core(%tile38_7) {
    aie.use_lock(%lock38_6, "Acquire", 1)
    aie.use_lock(%lock38_7, "Acquire", 0)
    func.call @do_sieve(%buf38_6, %buf38_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_6, "Release", 0)
    aie.use_lock(%lock38_7, "Release", 1)
    aie.end
  }

  %core38_8 = aie.core(%tile38_8) {
    aie.use_lock(%lock38_7, "Acquire", 1)
    aie.use_lock(%lock38_8, "Acquire", 0)
    func.call @do_sieve(%buf38_7, %buf38_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_7, "Release", 0)
    aie.use_lock(%lock38_8, "Release", 1)
    aie.end
  }

  %core39_8 = aie.core(%tile39_8) {
    aie.use_lock(%lock38_8, "Acquire", 1)
    aie.use_lock(%lock39_8, "Acquire", 0)
    func.call @do_sieve(%buf38_8, %buf39_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock38_8, "Release", 0)
    aie.use_lock(%lock39_8, "Release", 1)
    aie.end
  }

  %core39_7 = aie.core(%tile39_7) {
    aie.use_lock(%lock39_8, "Acquire", 1)
    aie.use_lock(%lock39_7, "Acquire", 0)
    func.call @do_sieve(%buf39_8, %buf39_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_8, "Release", 0)
    aie.use_lock(%lock39_7, "Release", 1)
    aie.end
  }

  %core39_6 = aie.core(%tile39_6) {
    aie.use_lock(%lock39_7, "Acquire", 1)
    aie.use_lock(%lock39_6, "Acquire", 0)
    func.call @do_sieve(%buf39_7, %buf39_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_7, "Release", 0)
    aie.use_lock(%lock39_6, "Release", 1)
    aie.end
  }

  %core39_5 = aie.core(%tile39_5) {
    aie.use_lock(%lock39_6, "Acquire", 1)
    aie.use_lock(%lock39_5, "Acquire", 0)
    func.call @do_sieve(%buf39_6, %buf39_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_6, "Release", 0)
    aie.use_lock(%lock39_5, "Release", 1)
    aie.end
  }

  %core39_4 = aie.core(%tile39_4) {
    aie.use_lock(%lock39_5, "Acquire", 1)
    aie.use_lock(%lock39_4, "Acquire", 0)
    func.call @do_sieve(%buf39_5, %buf39_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_5, "Release", 0)
    aie.use_lock(%lock39_4, "Release", 1)
    aie.end
  }

  %core39_3 = aie.core(%tile39_3) {
    aie.use_lock(%lock39_4, "Acquire", 1)
    aie.use_lock(%lock39_3, "Acquire", 0)
    func.call @do_sieve(%buf39_4, %buf39_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_4, "Release", 0)
    aie.use_lock(%lock39_3, "Release", 1)
    aie.end
  }

  %core39_2 = aie.core(%tile39_2) {
    aie.use_lock(%lock39_3, "Acquire", 1)
    aie.use_lock(%lock39_2, "Acquire", 0)
    func.call @do_sieve(%buf39_3, %buf39_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_3, "Release", 0)
    aie.use_lock(%lock39_2, "Release", 1)
    aie.end
  }

  %core39_1 = aie.core(%tile39_1) {
    aie.use_lock(%lock39_2, "Acquire", 1)
    aie.use_lock(%lock39_1, "Acquire", 0)
    func.call @do_sieve(%buf39_2, %buf39_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_2, "Release", 0)
    aie.use_lock(%lock39_1, "Release", 1)
    aie.end
  }

  %core40_1 = aie.core(%tile40_1) {
    aie.use_lock(%lock39_1, "Acquire", 1)
    aie.use_lock(%lock40_1, "Acquire", 0)
    func.call @do_sieve(%buf39_1, %buf40_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock39_1, "Release", 0)
    aie.use_lock(%lock40_1, "Release", 1)
    aie.end
  }

  %core40_2 = aie.core(%tile40_2) {
    aie.use_lock(%lock40_1, "Acquire", 1)
    aie.use_lock(%lock40_2, "Acquire", 0)
    func.call @do_sieve(%buf40_1, %buf40_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_1, "Release", 0)
    aie.use_lock(%lock40_2, "Release", 1)
    aie.end
  }

  %core40_3 = aie.core(%tile40_3) {
    aie.use_lock(%lock40_2, "Acquire", 1)
    aie.use_lock(%lock40_3, "Acquire", 0)
    func.call @do_sieve(%buf40_2, %buf40_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_2, "Release", 0)
    aie.use_lock(%lock40_3, "Release", 1)
    aie.end
  }

  %core40_4 = aie.core(%tile40_4) {
    aie.use_lock(%lock40_3, "Acquire", 1)
    aie.use_lock(%lock40_4, "Acquire", 0)
    func.call @do_sieve(%buf40_3, %buf40_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_3, "Release", 0)
    aie.use_lock(%lock40_4, "Release", 1)
    aie.end
  }

  %core40_5 = aie.core(%tile40_5) {
    aie.use_lock(%lock40_4, "Acquire", 1)
    aie.use_lock(%lock40_5, "Acquire", 0)
    func.call @do_sieve(%buf40_4, %buf40_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_4, "Release", 0)
    aie.use_lock(%lock40_5, "Release", 1)
    aie.end
  }

  %core40_6 = aie.core(%tile40_6) {
    aie.use_lock(%lock40_5, "Acquire", 1)
    aie.use_lock(%lock40_6, "Acquire", 0)
    func.call @do_sieve(%buf40_5, %buf40_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_5, "Release", 0)
    aie.use_lock(%lock40_6, "Release", 1)
    aie.end
  }

  %core40_7 = aie.core(%tile40_7) {
    aie.use_lock(%lock40_6, "Acquire", 1)
    aie.use_lock(%lock40_7, "Acquire", 0)
    func.call @do_sieve(%buf40_6, %buf40_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_6, "Release", 0)
    aie.use_lock(%lock40_7, "Release", 1)
    aie.end
  }

  %core40_8 = aie.core(%tile40_8) {
    aie.use_lock(%lock40_7, "Acquire", 1)
    aie.use_lock(%lock40_8, "Acquire", 0)
    func.call @do_sieve(%buf40_7, %buf40_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_7, "Release", 0)
    aie.use_lock(%lock40_8, "Release", 1)
    aie.end
  }

  %core41_8 = aie.core(%tile41_8) {
    aie.use_lock(%lock40_8, "Acquire", 1)
    aie.use_lock(%lock41_8, "Acquire", 0)
    func.call @do_sieve(%buf40_8, %buf41_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock40_8, "Release", 0)
    aie.use_lock(%lock41_8, "Release", 1)
    aie.end
  }

  %core41_7 = aie.core(%tile41_7) {
    aie.use_lock(%lock41_8, "Acquire", 1)
    aie.use_lock(%lock41_7, "Acquire", 0)
    func.call @do_sieve(%buf41_8, %buf41_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_8, "Release", 0)
    aie.use_lock(%lock41_7, "Release", 1)
    aie.end
  }

  %core41_6 = aie.core(%tile41_6) {
    aie.use_lock(%lock41_7, "Acquire", 1)
    aie.use_lock(%lock41_6, "Acquire", 0)
    func.call @do_sieve(%buf41_7, %buf41_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_7, "Release", 0)
    aie.use_lock(%lock41_6, "Release", 1)
    aie.end
  }

  %core41_5 = aie.core(%tile41_5) {
    aie.use_lock(%lock41_6, "Acquire", 1)
    aie.use_lock(%lock41_5, "Acquire", 0)
    func.call @do_sieve(%buf41_6, %buf41_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_6, "Release", 0)
    aie.use_lock(%lock41_5, "Release", 1)
    aie.end
  }

  %core41_4 = aie.core(%tile41_4) {
    aie.use_lock(%lock41_5, "Acquire", 1)
    aie.use_lock(%lock41_4, "Acquire", 0)
    func.call @do_sieve(%buf41_5, %buf41_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_5, "Release", 0)
    aie.use_lock(%lock41_4, "Release", 1)
    aie.end
  }

  %core41_3 = aie.core(%tile41_3) {
    aie.use_lock(%lock41_4, "Acquire", 1)
    aie.use_lock(%lock41_3, "Acquire", 0)
    func.call @do_sieve(%buf41_4, %buf41_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_4, "Release", 0)
    aie.use_lock(%lock41_3, "Release", 1)
    aie.end
  }

  %core41_2 = aie.core(%tile41_2) {
    aie.use_lock(%lock41_3, "Acquire", 1)
    aie.use_lock(%lock41_2, "Acquire", 0)
    func.call @do_sieve(%buf41_3, %buf41_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_3, "Release", 0)
    aie.use_lock(%lock41_2, "Release", 1)
    aie.end
  }

  %core41_1 = aie.core(%tile41_1) {
    aie.use_lock(%lock41_2, "Acquire", 1)
    aie.use_lock(%lock41_1, "Acquire", 0)
    func.call @do_sieve(%buf41_2, %buf41_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_2, "Release", 0)
    aie.use_lock(%lock41_1, "Release", 1)
    aie.end
  }

  %core42_1 = aie.core(%tile42_1) {
    aie.use_lock(%lock41_1, "Acquire", 1)
    aie.use_lock(%lock42_1, "Acquire", 0)
    func.call @do_sieve(%buf41_1, %buf42_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock41_1, "Release", 0)
    aie.use_lock(%lock42_1, "Release", 1)
    aie.end
  }

  %core42_2 = aie.core(%tile42_2) {
    aie.use_lock(%lock42_1, "Acquire", 1)
    aie.use_lock(%lock42_2, "Acquire", 0)
    func.call @do_sieve(%buf42_1, %buf42_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_1, "Release", 0)
    aie.use_lock(%lock42_2, "Release", 1)
    aie.end
  }

  %core42_3 = aie.core(%tile42_3) {
    aie.use_lock(%lock42_2, "Acquire", 1)
    aie.use_lock(%lock42_3, "Acquire", 0)
    func.call @do_sieve(%buf42_2, %buf42_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_2, "Release", 0)
    aie.use_lock(%lock42_3, "Release", 1)
    aie.end
  }

  %core42_4 = aie.core(%tile42_4) {
    aie.use_lock(%lock42_3, "Acquire", 1)
    aie.use_lock(%lock42_4, "Acquire", 0)
    func.call @do_sieve(%buf42_3, %buf42_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_3, "Release", 0)
    aie.use_lock(%lock42_4, "Release", 1)
    aie.end
  }

  %core42_5 = aie.core(%tile42_5) {
    aie.use_lock(%lock42_4, "Acquire", 1)
    aie.use_lock(%lock42_5, "Acquire", 0)
    func.call @do_sieve(%buf42_4, %buf42_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_4, "Release", 0)
    aie.use_lock(%lock42_5, "Release", 1)
    aie.end
  }

  %core42_6 = aie.core(%tile42_6) {
    aie.use_lock(%lock42_5, "Acquire", 1)
    aie.use_lock(%lock42_6, "Acquire", 0)
    func.call @do_sieve(%buf42_5, %buf42_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_5, "Release", 0)
    aie.use_lock(%lock42_6, "Release", 1)
    aie.end
  }

  %core42_7 = aie.core(%tile42_7) {
    aie.use_lock(%lock42_6, "Acquire", 1)
    aie.use_lock(%lock42_7, "Acquire", 0)
    func.call @do_sieve(%buf42_6, %buf42_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_6, "Release", 0)
    aie.use_lock(%lock42_7, "Release", 1)
    aie.end
  }

  %core42_8 = aie.core(%tile42_8) {
    aie.use_lock(%lock42_7, "Acquire", 1)
    aie.use_lock(%lock42_8, "Acquire", 0)
    func.call @do_sieve(%buf42_7, %buf42_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_7, "Release", 0)
    aie.use_lock(%lock42_8, "Release", 1)
    aie.end
  }

  %core43_8 = aie.core(%tile43_8) {
    aie.use_lock(%lock42_8, "Acquire", 1)
    aie.use_lock(%lock43_8, "Acquire", 0)
    func.call @do_sieve(%buf42_8, %buf43_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock42_8, "Release", 0)
    aie.use_lock(%lock43_8, "Release", 1)
    aie.end
  }

  %core43_7 = aie.core(%tile43_7) {
    aie.use_lock(%lock43_8, "Acquire", 1)
    aie.use_lock(%lock43_7, "Acquire", 0)
    func.call @do_sieve(%buf43_8, %buf43_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_8, "Release", 0)
    aie.use_lock(%lock43_7, "Release", 1)
    aie.end
  }

  %core43_6 = aie.core(%tile43_6) {
    aie.use_lock(%lock43_7, "Acquire", 1)
    aie.use_lock(%lock43_6, "Acquire", 0)
    func.call @do_sieve(%buf43_7, %buf43_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_7, "Release", 0)
    aie.use_lock(%lock43_6, "Release", 1)
    aie.end
  }

  %core43_5 = aie.core(%tile43_5) {
    aie.use_lock(%lock43_6, "Acquire", 1)
    aie.use_lock(%lock43_5, "Acquire", 0)
    func.call @do_sieve(%buf43_6, %buf43_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_6, "Release", 0)
    aie.use_lock(%lock43_5, "Release", 1)
    aie.end
  }

  %core43_4 = aie.core(%tile43_4) {
    aie.use_lock(%lock43_5, "Acquire", 1)
    aie.use_lock(%lock43_4, "Acquire", 0)
    func.call @do_sieve(%buf43_5, %buf43_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_5, "Release", 0)
    aie.use_lock(%lock43_4, "Release", 1)
    aie.end
  }

  %core43_3 = aie.core(%tile43_3) {
    aie.use_lock(%lock43_4, "Acquire", 1)
    aie.use_lock(%lock43_3, "Acquire", 0)
    func.call @do_sieve(%buf43_4, %buf43_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_4, "Release", 0)
    aie.use_lock(%lock43_3, "Release", 1)
    aie.end
  }

  %core43_2 = aie.core(%tile43_2) {
    aie.use_lock(%lock43_3, "Acquire", 1)
    aie.use_lock(%lock43_2, "Acquire", 0)
    func.call @do_sieve(%buf43_3, %buf43_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_3, "Release", 0)
    aie.use_lock(%lock43_2, "Release", 1)
    aie.end
  }

  %core43_1 = aie.core(%tile43_1) {
    aie.use_lock(%lock43_2, "Acquire", 1)
    aie.use_lock(%lock43_1, "Acquire", 0)
    func.call @do_sieve(%buf43_2, %buf43_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_2, "Release", 0)
    aie.use_lock(%lock43_1, "Release", 1)
    aie.end
  }

  %core44_1 = aie.core(%tile44_1) {
    aie.use_lock(%lock43_1, "Acquire", 1)
    aie.use_lock(%lock44_1, "Acquire", 0)
    func.call @do_sieve(%buf43_1, %buf44_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock43_1, "Release", 0)
    aie.use_lock(%lock44_1, "Release", 1)
    aie.end
  }

  %core44_2 = aie.core(%tile44_2) {
    aie.use_lock(%lock44_1, "Acquire", 1)
    aie.use_lock(%lock44_2, "Acquire", 0)
    func.call @do_sieve(%buf44_1, %buf44_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_1, "Release", 0)
    aie.use_lock(%lock44_2, "Release", 1)
    aie.end
  }

  %core44_3 = aie.core(%tile44_3) {
    aie.use_lock(%lock44_2, "Acquire", 1)
    aie.use_lock(%lock44_3, "Acquire", 0)
    func.call @do_sieve(%buf44_2, %buf44_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_2, "Release", 0)
    aie.use_lock(%lock44_3, "Release", 1)
    aie.end
  }

  %core44_4 = aie.core(%tile44_4) {
    aie.use_lock(%lock44_3, "Acquire", 1)
    aie.use_lock(%lock44_4, "Acquire", 0)
    func.call @do_sieve(%buf44_3, %buf44_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_3, "Release", 0)
    aie.use_lock(%lock44_4, "Release", 1)
    aie.end
  }

  %core44_5 = aie.core(%tile44_5) {
    aie.use_lock(%lock44_4, "Acquire", 1)
    aie.use_lock(%lock44_5, "Acquire", 0)
    func.call @do_sieve(%buf44_4, %buf44_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_4, "Release", 0)
    aie.use_lock(%lock44_5, "Release", 1)
    aie.end
  }

  %core44_6 = aie.core(%tile44_6) {
    aie.use_lock(%lock44_5, "Acquire", 1)
    aie.use_lock(%lock44_6, "Acquire", 0)
    func.call @do_sieve(%buf44_5, %buf44_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_5, "Release", 0)
    aie.use_lock(%lock44_6, "Release", 1)
    aie.end
  }

  %core44_7 = aie.core(%tile44_7) {
    aie.use_lock(%lock44_6, "Acquire", 1)
    aie.use_lock(%lock44_7, "Acquire", 0)
    func.call @do_sieve(%buf44_6, %buf44_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_6, "Release", 0)
    aie.use_lock(%lock44_7, "Release", 1)
    aie.end
  }

  %core44_8 = aie.core(%tile44_8) {
    aie.use_lock(%lock44_7, "Acquire", 1)
    aie.use_lock(%lock44_8, "Acquire", 0)
    func.call @do_sieve(%buf44_7, %buf44_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_7, "Release", 0)
    aie.use_lock(%lock44_8, "Release", 1)
    aie.end
  }

  %core45_8 = aie.core(%tile45_8) {
    aie.use_lock(%lock44_8, "Acquire", 1)
    aie.use_lock(%lock45_8, "Acquire", 0)
    func.call @do_sieve(%buf44_8, %buf45_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock44_8, "Release", 0)
    aie.use_lock(%lock45_8, "Release", 1)
    aie.end
  }

  %core45_7 = aie.core(%tile45_7) {
    aie.use_lock(%lock45_8, "Acquire", 1)
    aie.use_lock(%lock45_7, "Acquire", 0)
    func.call @do_sieve(%buf45_8, %buf45_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_8, "Release", 0)
    aie.use_lock(%lock45_7, "Release", 1)
    aie.end
  }

  %core45_6 = aie.core(%tile45_6) {
    aie.use_lock(%lock45_7, "Acquire", 1)
    aie.use_lock(%lock45_6, "Acquire", 0)
    func.call @do_sieve(%buf45_7, %buf45_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_7, "Release", 0)
    aie.use_lock(%lock45_6, "Release", 1)
    aie.end
  }

  %core45_5 = aie.core(%tile45_5) {
    aie.use_lock(%lock45_6, "Acquire", 1)
    aie.use_lock(%lock45_5, "Acquire", 0)
    func.call @do_sieve(%buf45_6, %buf45_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_6, "Release", 0)
    aie.use_lock(%lock45_5, "Release", 1)
    aie.end
  }

  %core45_4 = aie.core(%tile45_4) {
    aie.use_lock(%lock45_5, "Acquire", 1)
    aie.use_lock(%lock45_4, "Acquire", 0)
    func.call @do_sieve(%buf45_5, %buf45_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_5, "Release", 0)
    aie.use_lock(%lock45_4, "Release", 1)
    aie.end
  }

  %core45_3 = aie.core(%tile45_3) {
    aie.use_lock(%lock45_4, "Acquire", 1)
    aie.use_lock(%lock45_3, "Acquire", 0)
    func.call @do_sieve(%buf45_4, %buf45_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_4, "Release", 0)
    aie.use_lock(%lock45_3, "Release", 1)
    aie.end
  }

  %core45_2 = aie.core(%tile45_2) {
    aie.use_lock(%lock45_3, "Acquire", 1)
    aie.use_lock(%lock45_2, "Acquire", 0)
    func.call @do_sieve(%buf45_3, %buf45_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_3, "Release", 0)
    aie.use_lock(%lock45_2, "Release", 1)
    aie.end
  }

  %core45_1 = aie.core(%tile45_1) {
    aie.use_lock(%lock45_2, "Acquire", 1)
    aie.use_lock(%lock45_1, "Acquire", 0)
    func.call @do_sieve(%buf45_2, %buf45_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_2, "Release", 0)
    aie.use_lock(%lock45_1, "Release", 1)
    aie.end
  }

  %core46_1 = aie.core(%tile46_1) {
    aie.use_lock(%lock45_1, "Acquire", 1)
    aie.use_lock(%lock46_1, "Acquire", 0)
    func.call @do_sieve(%buf45_1, %buf46_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock45_1, "Release", 0)
    aie.use_lock(%lock46_1, "Release", 1)
    aie.end
  }

  %core46_2 = aie.core(%tile46_2) {
    aie.use_lock(%lock46_1, "Acquire", 1)
    aie.use_lock(%lock46_2, "Acquire", 0)
    func.call @do_sieve(%buf46_1, %buf46_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_1, "Release", 0)
    aie.use_lock(%lock46_2, "Release", 1)
    aie.end
  }

  %core46_3 = aie.core(%tile46_3) {
    aie.use_lock(%lock46_2, "Acquire", 1)
    aie.use_lock(%lock46_3, "Acquire", 0)
    func.call @do_sieve(%buf46_2, %buf46_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_2, "Release", 0)
    aie.use_lock(%lock46_3, "Release", 1)
    aie.end
  }

  %core46_4 = aie.core(%tile46_4) {
    aie.use_lock(%lock46_3, "Acquire", 1)
    aie.use_lock(%lock46_4, "Acquire", 0)
    func.call @do_sieve(%buf46_3, %buf46_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_3, "Release", 0)
    aie.use_lock(%lock46_4, "Release", 1)
    aie.end
  }

  %core46_5 = aie.core(%tile46_5) {
    aie.use_lock(%lock46_4, "Acquire", 1)
    aie.use_lock(%lock46_5, "Acquire", 0)
    func.call @do_sieve(%buf46_4, %buf46_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_4, "Release", 0)
    aie.use_lock(%lock46_5, "Release", 1)
    aie.end
  }

  %core46_6 = aie.core(%tile46_6) {
    aie.use_lock(%lock46_5, "Acquire", 1)
    aie.use_lock(%lock46_6, "Acquire", 0)
    func.call @do_sieve(%buf46_5, %buf46_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_5, "Release", 0)
    aie.use_lock(%lock46_6, "Release", 1)
    aie.end
  }

  %core46_7 = aie.core(%tile46_7) {
    aie.use_lock(%lock46_6, "Acquire", 1)
    aie.use_lock(%lock46_7, "Acquire", 0)
    func.call @do_sieve(%buf46_6, %buf46_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_6, "Release", 0)
    aie.use_lock(%lock46_7, "Release", 1)
    aie.end
  }

  %core46_8 = aie.core(%tile46_8) {
    aie.use_lock(%lock46_7, "Acquire", 1)
    aie.use_lock(%lock46_8, "Acquire", 0)
    func.call @do_sieve(%buf46_7, %buf46_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_7, "Release", 0)
    aie.use_lock(%lock46_8, "Release", 1)
    aie.end
  }

  %core47_8 = aie.core(%tile47_8) {
    aie.use_lock(%lock46_8, "Acquire", 1)
    aie.use_lock(%lock47_8, "Acquire", 0)
    func.call @do_sieve(%buf46_8, %buf47_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock46_8, "Release", 0)
    aie.use_lock(%lock47_8, "Release", 1)
    aie.end
  }

  %core47_7 = aie.core(%tile47_7) {
    aie.use_lock(%lock47_8, "Acquire", 1)
    aie.use_lock(%lock47_7, "Acquire", 0)
    func.call @do_sieve(%buf47_8, %buf47_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_8, "Release", 0)
    aie.use_lock(%lock47_7, "Release", 1)
    aie.end
  }

  %core47_6 = aie.core(%tile47_6) {
    aie.use_lock(%lock47_7, "Acquire", 1)
    aie.use_lock(%lock47_6, "Acquire", 0)
    func.call @do_sieve(%buf47_7, %buf47_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_7, "Release", 0)
    aie.use_lock(%lock47_6, "Release", 1)
    aie.end
  }

  %core47_5 = aie.core(%tile47_5) {
    aie.use_lock(%lock47_6, "Acquire", 1)
    aie.use_lock(%lock47_5, "Acquire", 0)
    func.call @do_sieve(%buf47_6, %buf47_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_6, "Release", 0)
    aie.use_lock(%lock47_5, "Release", 1)
    aie.end
  }

  %core47_4 = aie.core(%tile47_4) {
    aie.use_lock(%lock47_5, "Acquire", 1)
    aie.use_lock(%lock47_4, "Acquire", 0)
    func.call @do_sieve(%buf47_5, %buf47_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_5, "Release", 0)
    aie.use_lock(%lock47_4, "Release", 1)
    aie.end
  }

  %core47_3 = aie.core(%tile47_3) {
    aie.use_lock(%lock47_4, "Acquire", 1)
    aie.use_lock(%lock47_3, "Acquire", 0)
    func.call @do_sieve(%buf47_4, %buf47_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_4, "Release", 0)
    aie.use_lock(%lock47_3, "Release", 1)
    aie.end
  }

  %core47_2 = aie.core(%tile47_2) {
    aie.use_lock(%lock47_3, "Acquire", 1)
    aie.use_lock(%lock47_2, "Acquire", 0)
    func.call @do_sieve(%buf47_3, %buf47_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_3, "Release", 0)
    aie.use_lock(%lock47_2, "Release", 1)
    aie.end
  }

  %core47_1 = aie.core(%tile47_1) {
    aie.use_lock(%lock47_2, "Acquire", 1)
    aie.use_lock(%lock47_1, "Acquire", 0)
    func.call @do_sieve(%buf47_2, %buf47_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_2, "Release", 0)
    aie.use_lock(%lock47_1, "Release", 1)
    aie.end
  }

  %core48_1 = aie.core(%tile48_1) {
    aie.use_lock(%lock47_1, "Acquire", 1)
    aie.use_lock(%lock48_1, "Acquire", 0)
    func.call @do_sieve(%buf47_1, %buf48_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock47_1, "Release", 0)
    aie.use_lock(%lock48_1, "Release", 1)
    aie.end
  }

  %core48_2 = aie.core(%tile48_2) {
    aie.use_lock(%lock48_1, "Acquire", 1)
    aie.use_lock(%lock48_2, "Acquire", 0)
    func.call @do_sieve(%buf48_1, %buf48_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_1, "Release", 0)
    aie.use_lock(%lock48_2, "Release", 1)
    aie.end
  }

  %core48_3 = aie.core(%tile48_3) {
    aie.use_lock(%lock48_2, "Acquire", 1)
    aie.use_lock(%lock48_3, "Acquire", 0)
    func.call @do_sieve(%buf48_2, %buf48_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_2, "Release", 0)
    aie.use_lock(%lock48_3, "Release", 1)
    aie.end
  }

  %core48_4 = aie.core(%tile48_4) {
    aie.use_lock(%lock48_3, "Acquire", 1)
    aie.use_lock(%lock48_4, "Acquire", 0)
    func.call @do_sieve(%buf48_3, %buf48_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_3, "Release", 0)
    aie.use_lock(%lock48_4, "Release", 1)
    aie.end
  }

  %core48_5 = aie.core(%tile48_5) {
    aie.use_lock(%lock48_4, "Acquire", 1)
    aie.use_lock(%lock48_5, "Acquire", 0)
    func.call @do_sieve(%buf48_4, %buf48_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_4, "Release", 0)
    aie.use_lock(%lock48_5, "Release", 1)
    aie.end
  }

  %core48_6 = aie.core(%tile48_6) {
    aie.use_lock(%lock48_5, "Acquire", 1)
    aie.use_lock(%lock48_6, "Acquire", 0)
    func.call @do_sieve(%buf48_5, %buf48_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_5, "Release", 0)
    aie.use_lock(%lock48_6, "Release", 1)
    aie.end
  }

  %core48_7 = aie.core(%tile48_7) {
    aie.use_lock(%lock48_6, "Acquire", 1)
    aie.use_lock(%lock48_7, "Acquire", 0)
    func.call @do_sieve(%buf48_6, %buf48_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_6, "Release", 0)
    aie.use_lock(%lock48_7, "Release", 1)
    aie.end
  }

  %core48_8 = aie.core(%tile48_8) {
    aie.use_lock(%lock48_7, "Acquire", 1)
    aie.use_lock(%lock48_8, "Acquire", 0)
    func.call @do_sieve(%buf48_7, %buf48_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_7, "Release", 0)
    aie.use_lock(%lock48_8, "Release", 1)
    aie.end
  }

  %core49_8 = aie.core(%tile49_8) {
    aie.use_lock(%lock48_8, "Acquire", 1)
    aie.use_lock(%lock49_8, "Acquire", 0)
    func.call @do_sieve(%buf48_8, %buf49_8) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock48_8, "Release", 0)
    aie.use_lock(%lock49_8, "Release", 1)
    aie.end
  }

  %core49_7 = aie.core(%tile49_7) {
    aie.use_lock(%lock49_8, "Acquire", 1)
    aie.use_lock(%lock49_7, "Acquire", 0)
    func.call @do_sieve(%buf49_8, %buf49_7) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_8, "Release", 0)
    aie.use_lock(%lock49_7, "Release", 1)
    aie.end
  }

  %core49_6 = aie.core(%tile49_6) {
    aie.use_lock(%lock49_7, "Acquire", 1)
    aie.use_lock(%lock49_6, "Acquire", 0)
    func.call @do_sieve(%buf49_7, %buf49_6) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_7, "Release", 0)
    aie.use_lock(%lock49_6, "Release", 1)
    aie.end
  }

  %core49_5 = aie.core(%tile49_5) {
    aie.use_lock(%lock49_6, "Acquire", 1)
    aie.use_lock(%lock49_5, "Acquire", 0)
    func.call @do_sieve(%buf49_6, %buf49_5) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_6, "Release", 0)
    aie.use_lock(%lock49_5, "Release", 1)
    aie.end
  }

  %core49_4 = aie.core(%tile49_4) {
    aie.use_lock(%lock49_5, "Acquire", 1)
    aie.use_lock(%lock49_4, "Acquire", 0)
    func.call @do_sieve(%buf49_5, %buf49_4) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_5, "Release", 0)
    aie.use_lock(%lock49_4, "Release", 1)
    aie.end
  }

  %core49_3 = aie.core(%tile49_3) {
    aie.use_lock(%lock49_4, "Acquire", 1)
    aie.use_lock(%lock49_3, "Acquire", 0)
    func.call @do_sieve(%buf49_4, %buf49_3) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_4, "Release", 0)
    aie.use_lock(%lock49_3, "Release", 1)
    aie.end
  }

  %core49_2 = aie.core(%tile49_2) {
    aie.use_lock(%lock49_3, "Acquire", 1)
    aie.use_lock(%lock49_2, "Acquire", 0)
    func.call @do_sieve(%buf49_3, %buf49_2) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_3, "Release", 0)
    aie.use_lock(%lock49_2, "Release", 1)
    aie.end
  }

  %core49_1 = aie.core(%tile49_1) {
    aie.use_lock(%lock49_2, "Acquire", 1)
    aie.use_lock(%lock49_1, "Acquire", 0)
    func.call @do_sieve(%buf49_2, %buf49_1) : (memref<3072xi32>, memref<3072xi32>) -> ()
    aie.use_lock(%lock49_2, "Release", 0)
    aie.use_lock(%lock49_1, "Release", 1)
    aie.end
  }

}
