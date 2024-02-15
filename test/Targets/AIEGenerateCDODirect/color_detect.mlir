// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module {
  aie.device(ipu) {
    // // func.func private @rgba2hueLine(memref<256xui8>, memref<64xui8>, i32)
    // // func.func private @thresholdLine(memref<64xui8>, memref<64xui8>, i32, i16, i16, i8)
    // // func.func private @bitwiseORLine(memref<64xui8>, memref<64xui8>, memref<64xui8>, i32)
    // // func.func private @gray2rgbaLine(memref<64xui8>, memref<256xui8>, i32)
    // // func.func private @bitwiseANDLine(memref<256xui8>, memref<256xui8>, memref<256xui8>, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @inOF_L3L2(%tile_0_0, {%tile_0_2, %tile_0_1}, [2 : i32, 2 : i32, 6 : i32]) : !aie.objectfifo<memref<256xui8>>
    aie.objectfifo @inOF_L2L1(%tile_0_1, {%tile_0_5}, 6 : i32) : !aie.objectfifo<memref<256xui8>>
    aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1]()
    aie.objectfifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<256xui8>>
    aie.objectfifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<256xui8>>
    aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3]()
    aie.objectfifo @OF_2to34(%tile_0_2, {%tile_0_3, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64xui8>>
    aie.objectfifo @OF_3to3(%tile_0_3, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<64xui8>>
    aie.objectfifo @OF_3to5(%tile_0_3, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64xui8>>
    aie.objectfifo @OF_4to4(%tile_0_4, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<64xui8>>
    aie.objectfifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64xui8>>
    aie.objectfifo @OF_5to5a(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<64xui8>>
    aie.objectfifo @OF_5to5b(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<256xui8>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOF_L3L2(Consume, 1) : !aie.objectfifosubview<memref<256xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
        %2 = aie.objectfifo.acquire @OF_2to34(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %c64_i32 = arith.constant 64 : i32
        // func.call @rgba2hueLine(%1, %3, %c64_i32) : (memref<256xui8>, memref<64xui8>, i32) -> ()
        aie.objectfifo.release @inOF_L3L2(Consume, 1)
        aie.objectfifo.release @OF_2to34(Produce, 1)
      }
      aie.end
    } // {link_with = "rgba2hue.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c40_i16 = arith.constant 40 : i16
      %c30_i16 = arith.constant 30 : i16
      %c255_i16 = arith.constant 255 : i16
      %c4_i8 = arith.constant 4 : i8
      %c0_i8 = arith.constant 0 : i8
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_2to34(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %2 = aie.objectfifo.acquire @OF_3to3(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %c64_i32 = arith.constant 64 : i32
        // func.call @thresholdLine(%1, %3, %c64_i32, %c40_i16, %c255_i16, %c4_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_2to34(Consume, 1)
        aie.objectfifo.release @OF_3to3(Produce, 1)
        %4 = aie.objectfifo.acquire @OF_3to3(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %6 = aie.objectfifo.acquire @OF_3to5(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %c64_i32_0 = arith.constant 64 : i32
        // func.call @thresholdLine(%5, %7, %c64_i32_0, %c30_i16, %c255_i16, %c0_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_3to3(Consume, 1)
        aie.objectfifo.release @OF_3to5(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c160_i16 = arith.constant 160 : i16
      %c90_i16 = arith.constant 90 : i16
      %c255_i16 = arith.constant 255 : i16
      %c4_i8 = arith.constant 4 : i8
      %c0_i8 = arith.constant 0 : i8
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_2to34(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %2 = aie.objectfifo.acquire @OF_4to4(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %c64_i32 = arith.constant 64 : i32
        // func.call @thresholdLine(%1, %3, %c64_i32, %c160_i16, %c255_i16, %c4_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_2to34(Consume, 1)
        aie.objectfifo.release @OF_4to4(Produce, 1)
        %4 = aie.objectfifo.acquire @OF_4to4(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %6 = aie.objectfifo.acquire @OF_4to5(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %c64_i32_0 = arith.constant 64 : i32
        // func.call @thresholdLine(%5, %7, %c64_i32_0, %c90_i16, %c255_i16, %c0_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_4to4(Consume, 1)
        aie.objectfifo.release @OF_4to5(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_3to5(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %2 = aie.objectfifo.acquire @OF_4to5(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %4 = aie.objectfifo.acquire @OF_5to5a(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %c64_i32 = arith.constant 64 : i32
        // func.call @bitwiseORLine(%1, %3, %5, %c64_i32) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, i32) -> ()
        aie.objectfifo.release @OF_3to5(Consume, 1)
        aie.objectfifo.release @OF_4to5(Consume, 1)
        aie.objectfifo.release @OF_5to5a(Produce, 1)
        %6 = aie.objectfifo.acquire @OF_5to5a(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
        %8 = aie.objectfifo.acquire @OF_5to5b(Produce, 1) : !aie.objectfifosubview<memref<256xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
        %c64_i32_0 = arith.constant 64 : i32
        // func.call @gray2rgbaLine(%7, %9, %c64_i32_0) : (memref<64xui8>, memref<256xui8>, i32) -> ()
        aie.objectfifo.release @OF_5to5a(Consume, 1)
        aie.objectfifo.release @OF_5to5b(Produce, 1)
        %10 = aie.objectfifo.acquire @OF_5to5b(Consume, 1) : !aie.objectfifosubview<memref<256xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
        %12 = aie.objectfifo.acquire @inOF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<256xui8>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
        %14 = aie.objectfifo.acquire @outOF_L1L2(Produce, 1) : !aie.objectfifosubview<memref<256xui8>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
        %c256_i32 = arith.constant 256 : i32
        // func.call @bitwiseANDLine(%11, %13, %15, %c256_i32) : (memref<256xui8>, memref<256xui8>, memref<256xui8>, i32) -> ()
        aie.objectfifo.release @OF_5to5b(Consume, 1)
        aie.objectfifo.release @inOF_L2L1(Consume, 1)
        aie.objectfifo.release @outOF_L1L2(Produce, 1)
      }
      aie.end
    } // {link_with = "combined_bitwiseOR_gray2rgba_bitwiseAND.a"}
    func.func @sequence(%arg0: memref<2304xi32>, %arg1: memref<16x16xi32>, %arg2: memref<2304xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 2304][0, 0, 0]) {id = 1 : i64, metadata = @inOF_L3L2} : memref<2304xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 2304][0, 0, 0]) {id = 0 : i64, metadata = @outOF_L2L3} : memref<2304xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}