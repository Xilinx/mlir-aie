// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Three adjacent mem tiles, each with six links whose 120 BD ids need four
// of them walked whole: 22 walk sets a tile, 10648 choices for the spill
// component, and no buffer room for any. The search stops at its bound, and
// the error says so rather than that no placement exists.
// CHECK: error: 'aie.objectfifo.pool' op could not place buffers: the search for which pools to walk whole stopped after 1024 placements tried in one spill component
module @search_stops {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %s4 = aie.tile(4, 0)
    %s5 = aie.tile(5, 0)
    %m0 = aie.tile(0, 1)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<520192xi8>
    %m1 = aie.tile(1, 1)
    %fill1 = aie.buffer(%m1) {sym_name = "fill1"} : memref<520192xi8>
    %m2 = aie.tile(2, 1)
    %fill2 = aie.buffer(%m2) {sym_name = "fill2"} : memref<520192xi8>
    aie.objectfifo @p00(%s0, {%m0}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @p00_out(%m0, {%s0}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@p00] -> [@p00_out] ([] [])
    aie.objectfifo @p01(%s0, {%m0}, 10 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @p01_out(%m0, {%s0}, 10 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@p01] -> [@p01_out] ([] [])
    aie.objectfifo @p02(%s0, {%m0}, 10 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo @p02_out(%m0, {%s0}, 10 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo.link [@p02] -> [@p02_out] ([] [])
    aie.objectfifo @p03(%s1, {%m0}, 10 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @p03_out(%m0, {%s1}, 10 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@p03] -> [@p03_out] ([] [])
    aie.objectfifo @p04(%s1, {%m0}, 10 : i32) : !aie.objectfifo<memref<320xi32>>
    aie.objectfifo @p04_out(%m0, {%s1}, 10 : i32) : !aie.objectfifo<memref<320xi32>>
    aie.objectfifo.link [@p04] -> [@p04_out] ([] [])
    aie.objectfifo @p05(%s1, {%m0}, 10 : i32) : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo @p05_out(%m0, {%s1}, 10 : i32) : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo.link [@p05] -> [@p05_out] ([] [])
    aie.objectfifo @p10(%s2, {%m1}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @p10_out(%m1, {%s2}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@p10] -> [@p10_out] ([] [])
    aie.objectfifo @p11(%s2, {%m1}, 10 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @p11_out(%m1, {%s2}, 10 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@p11] -> [@p11_out] ([] [])
    aie.objectfifo @p12(%s2, {%m1}, 10 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo @p12_out(%m1, {%s2}, 10 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo.link [@p12] -> [@p12_out] ([] [])
    aie.objectfifo @p13(%s3, {%m1}, 10 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @p13_out(%m1, {%s3}, 10 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@p13] -> [@p13_out] ([] [])
    aie.objectfifo @p14(%s3, {%m1}, 10 : i32) : !aie.objectfifo<memref<320xi32>>
    aie.objectfifo @p14_out(%m1, {%s3}, 10 : i32) : !aie.objectfifo<memref<320xi32>>
    aie.objectfifo.link [@p14] -> [@p14_out] ([] [])
    aie.objectfifo @p15(%s3, {%m1}, 10 : i32) : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo @p15_out(%m1, {%s3}, 10 : i32) : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo.link [@p15] -> [@p15_out] ([] [])
    aie.objectfifo @p20(%s4, {%m2}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @p20_out(%m2, {%s4}, 10 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@p20] -> [@p20_out] ([] [])
    aie.objectfifo @p21(%s4, {%m2}, 10 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @p21_out(%m2, {%s4}, 10 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@p21] -> [@p21_out] ([] [])
    aie.objectfifo @p22(%s4, {%m2}, 10 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo @p22_out(%m2, {%s4}, 10 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo.link [@p22] -> [@p22_out] ([] [])
    aie.objectfifo @p23(%s5, {%m2}, 10 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @p23_out(%m2, {%s5}, 10 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@p23] -> [@p23_out] ([] [])
    aie.objectfifo @p24(%s5, {%m2}, 10 : i32) : !aie.objectfifo<memref<320xi32>>
    aie.objectfifo @p24_out(%m2, {%s5}, 10 : i32) : !aie.objectfifo<memref<320xi32>>
    aie.objectfifo.link [@p24] -> [@p24_out] ([] [])
    aie.objectfifo @p25(%s5, {%m2}, 10 : i32) : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo @p25_out(%m2, {%s5}, 10 : i32) : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo.link [@p25] -> [@p25_out] ([] [])
  }
}
