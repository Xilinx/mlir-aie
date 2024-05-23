//===- AIEVecAttributes.td - AIE vector attributes def. ----*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECATTRIBUTES_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECATTRIBUTES_TD

include "aie/Dialect/AIEVec/IR/AIEVecDialect.td"
include "mlir/IR/EnumAttr.td"

// Shuffle modes for shuffle ops.
def SHUFFLE_MODE_T8_64X2_LO   : I32EnumAttrCase<"T8_64X2_LO",    0, "t8_64x2_lo">;
def SHUFFLE_MODE_T8_64X2_HI   : I32EnumAttrCase<"T8_64X2_HI",    1, "t8_64x2_hi">;
def SHUFFLE_MODE_T16_32X2_LO  : I32EnumAttrCase<"T16_32X2_LO",   2, "t16_32x2_lo">;
def SHUFFLE_MODE_T16_32X2_HI  : I32EnumAttrCase<"T16_32X2_HI",   3, "t16_32x2_hi">;
def SHUFFLE_MODE_T32_16X2_LO  : I32EnumAttrCase<"T32_16X2_LO",   4, "t32_16x2_lo">;
def SHUFFLE_MODE_T32_16X2_HI  : I32EnumAttrCase<"T32_16X2_HI",   5, "t32_16x2_hi">;
def SHUFFLE_MODE_T64_8X2_LO   : I32EnumAttrCase<"T64_8X2_LO",    6, "t64_8x2_lo">;
def SHUFFLE_MODE_T64_8X2_HI   : I32EnumAttrCase<"T64_8X2_HI",    7, "t64_8x2_hi">;
def SHUFFLE_MODE_T128_4X2_LO  : I32EnumAttrCase<"T128_4X2_LO",   8, "t128_4x2_lo">;
def SHUFFLE_MODE_T128_4X2_HI  : I32EnumAttrCase<"T128_4X2_HI",   9, "t128_4x2_hi">;
def SHUFFLE_MODE_T256_2X2_LO  : I32EnumAttrCase<"T256_2X2_LO",  10, "t256_2x2_lo">;
def SHUFFLE_MODE_T256_2X2_HI  : I32EnumAttrCase<"T256_2X2_HI",  11, "t256_2x2_hi">;
def SHUFFLE_MODE_T128_2X4_LO  : I32EnumAttrCase<"T128_2X4_LO",  12, "t128_2x4_lo">;
def SHUFFLE_MODE_T128_2X4_HI  : I32EnumAttrCase<"T128_2X4_HI",  13, "t128_2x4_hi">;
def SHUFFLE_MODE_T64_2X8_LO   : I32EnumAttrCase<"T64_2X8_LO",   14, "t64_2x8_lo">;
def SHUFFLE_MODE_T64_2X8_HI   : I32EnumAttrCase<"T64_2X8_HI",   15, "t64_2x8_hi">;
def SHUFFLE_MODE_T32_2X16_LO  : I32EnumAttrCase<"T32_2X16_LO",  16, "t32_2x16_lo">;
def SHUFFLE_MODE_T32_2X16_HI  : I32EnumAttrCase<"T32_2X16_HI",  17, "t32_2x16_hi">;
def SHUFFLE_MODE_T16_2X32_LO  : I32EnumAttrCase<"T16_2X32_LO",  18, "t16_2x32_lo">;
def SHUFFLE_MODE_T16_2X32_HI  : I32EnumAttrCase<"T16_2X32_HI",  19, "t16_2x32_hi">;
def SHUFFLE_MODE_T8_2X64_LO   : I32EnumAttrCase<"T8_2X64_LO",   20, "t8_2x64_lo">;
def SHUFFLE_MODE_T8_2X64_HI   : I32EnumAttrCase<"T8_2X64_HI",   21, "t8_2x64_hi">;
def SHUFFLE_MODE_T512_1X2_LO  : I32EnumAttrCase<"T512_1X2_LO",  22, "t512_1x2_lo">;
def SHUFFLE_MODE_T512_1X2_HI  : I32EnumAttrCase<"T512_1X2_HI",  23, "t512_1x2_hi">;
def SHUFFLE_MODE_T16_16X4_LO  : I32EnumAttrCase<"T16_16X4_LO",  24, "t16_16x4_lo">;
def SHUFFLE_MODE_T16_16X4_HI  : I32EnumAttrCase<"T16_16X4_HI",  25, "t16_16x4_hi">;
def SHUFFLE_MODE_T16_4X16_LO  : I32EnumAttrCase<"T16_4X16_LO",  26, "t16_4x16_lo">;
def SHUFFLE_MODE_T16_4X16_HI  : I32EnumAttrCase<"T16_4X16_HI",  27, "t16_4x16_hi">;
def SHUFFLE_MODE_T16_8X4      : I32EnumAttrCase<"T16_8X4",      28, "t16_8x4">;
def SHUFFLE_MODE_T16_4X8      : I32EnumAttrCase<"T16_4X8",      29, "t16_4x8">;
def SHUFFLE_MODE_T32_8X4_LO   : I32EnumAttrCase<"T32_8X4_LO",   30, "t32_8x4_lo">;
def SHUFFLE_MODE_T32_8X4_HI   : I32EnumAttrCase<"T32_8X4_HI",   31, "t32_8x4_hi">;
def SHUFFLE_MODE_T32_4X8_LO   : I32EnumAttrCase<"T32_4X8_LO",   32, "t32_4x8_lo">;
def SHUFFLE_MODE_T32_4X8_HI   : I32EnumAttrCase<"T32_4X8_HI",   33, "t32_4x8_hi">;
def SHUFFLE_MODE_T32_4X4      : I32EnumAttrCase<"T32_4X4",      34, "t32_4x4">;
def SHUFFLE_MODE_T8_8X8       : I32EnumAttrCase<"T8_8X8",       35, "t8_8x8">;
def SHUFFLE_MODE_T8_16X4      : I32EnumAttrCase<"T8_16X4",      36, "t8_16x4">;
def SHUFFLE_MODE_T8_4X16      : I32EnumAttrCase<"T8_4X16",      37, "t8_4x16">;
def SHUFFLE_MODE_T16_1X2_flip : I32EnumAttrCase<"T16_1X2_flip", 38, "t16_1x2_flip">;
def SHUFFLE_MODE_T16_4X4      : I32EnumAttrCase<"T16_4X4",      39, "t16_4x4">;
def SHUFFLE_MODE_T16_4X2      : I32EnumAttrCase<"T16_4X2",      40, "t16_4x2">;
def SHUFFLE_MODE_T16_2X4      : I32EnumAttrCase<"T16_2X4",      41, "t16_2x4">;
def SHUFFLE_MODE_T16_8X2      : I32EnumAttrCase<"T16_8X2",      42, "t16_8x2">;
def SHUFFLE_MODE_T16_2X8      : I32EnumAttrCase<"T16_2X8",      43, "t16_2x8">;
def SHUFFLE_MODE_T16_16X2     : I32EnumAttrCase<"T16_16X2",     44, "t16_16x2">;
def SHUFFLE_MODE_T16_2X16     : I32EnumAttrCase<"T16_2X16",     45, "t16_2x16">;
def SHUFFLE_MODE_T8_8X4       : I32EnumAttrCase<"T8_8X4",       46, "t8_8x4">;
def SHUFFLE_MODE_T8_4X8       : I32EnumAttrCase<"T8_4X8",       47, "t8_4x8">;

def ShuffleMode : I32EnumAttr<
    "ShuffleMode",
    "Shuffle mode for AIEVec shuffle operations",
    [SHUFFLE_MODE_T8_64X2_LO, SHUFFLE_MODE_T8_64X2_HI, SHUFFLE_MODE_T16_32X2_LO,
     SHUFFLE_MODE_T16_32X2_HI, SHUFFLE_MODE_T32_16X2_LO, SHUFFLE_MODE_T32_16X2_HI,
     SHUFFLE_MODE_T64_8X2_LO, SHUFFLE_MODE_T64_8X2_HI, SHUFFLE_MODE_T128_4X2_LO,
     SHUFFLE_MODE_T128_4X2_HI, SHUFFLE_MODE_T256_2X2_LO, SHUFFLE_MODE_T256_2X2_HI,
     SHUFFLE_MODE_T128_2X4_LO, SHUFFLE_MODE_T128_2X4_HI, SHUFFLE_MODE_T64_2X8_LO,
     SHUFFLE_MODE_T64_2X8_HI, SHUFFLE_MODE_T32_2X16_LO, SHUFFLE_MODE_T32_2X16_HI,
     SHUFFLE_MODE_T16_2X32_LO, SHUFFLE_MODE_T16_2X32_HI, SHUFFLE_MODE_T8_2X64_LO,
     SHUFFLE_MODE_T8_2X64_HI, SHUFFLE_MODE_T512_1X2_LO, SHUFFLE_MODE_T512_1X2_HI,
     SHUFFLE_MODE_T16_16X4_LO, SHUFFLE_MODE_T16_16X4_HI, SHUFFLE_MODE_T16_4X16_LO,
     SHUFFLE_MODE_T16_4X16_HI, SHUFFLE_MODE_T16_8X4, SHUFFLE_MODE_T16_4X8,
     SHUFFLE_MODE_T32_8X4_LO, SHUFFLE_MODE_T32_8X4_HI, SHUFFLE_MODE_T32_4X8_LO,
     SHUFFLE_MODE_T32_4X8_HI, SHUFFLE_MODE_T32_4X4, SHUFFLE_MODE_T8_8X8,
     SHUFFLE_MODE_T8_16X4, SHUFFLE_MODE_T8_4X16, SHUFFLE_MODE_T16_1X2_flip,
     SHUFFLE_MODE_T16_4X4, SHUFFLE_MODE_T16_4X2, SHUFFLE_MODE_T16_2X4,
     SHUFFLE_MODE_T16_8X2, SHUFFLE_MODE_T16_2X8, SHUFFLE_MODE_T16_16X2,
     SHUFFLE_MODE_T16_2X16, SHUFFLE_MODE_T8_8X4, SHUFFLE_MODE_T8_4X8]> {
  let cppNamespace = "::xilinx::aievec";
  let genSpecializedAttr = 0;
}

def AIEVec_ShuffleModeAttr : EnumAttr<AIEVec_Dialect, ShuffleMode, "mode"> {
  let assemblyFormat = "`[` $value `]`";
}

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECATTRIBUTES_TD
