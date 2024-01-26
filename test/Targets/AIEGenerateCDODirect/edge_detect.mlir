// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module {
  aie.device(ipu) {
    // // func.func private @rgba2grayLine(memref<7680xui8>, memref<1920xui8>, i32)
    // // func.func private @filter2dLine(memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>)
    // // func.func private @thresholdLine(memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8)
    // // func.func private @gray2rgbaLine(memref<1920xui8>, memref<7680xui8>, i32)
    // // func.func private @addWeightedLine(memref<7680xui8>, memref<7680xui8>, memref<7680xui8>, i32, i16, i16, i8)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @inOF_L3L2(%tile_0_0, {%tile_0_2, %tile_0_1}, [2 : i32, 2 : i32, 7 : i32]) : !aie.objectfifo<memref<7680xui8>>
    aie.objectfifo @inOF_L2L1(%tile_0_1, {%tile_0_5}, 7 : i32) : !aie.objectfifo<memref<7680xui8>>
    aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1]()
    aie.objectfifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<7680xui8>>
    aie.objectfifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<7680xui8>>
    aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3]()
    aie.objectfifo @OF_2to3(%tile_0_2, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<1920xui8>>
    aie.objectfifo @OF_3to4(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1920xui8>>
    aie.objectfifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1920xui8>>
    aie.objectfifo @OF_5to5(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<7680xui8>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @inOF_L3L2(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %2 = aie.objectfifo.acquire @OF_2to3(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        // func.call @rgba2grayLine(%1, %3, %c1920_i32) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
        aie.objectfifo.release @inOF_L3L2(Consume, 1)
        aie.objectfifo.release @OF_2to3(Produce, 1)
      }
      aie.end
    } // {link_with = "rgba2gray.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %alloc = memref.alloc() : memref<3x3xi16>
      %c0_i16 = arith.constant 0 : i16
      %c4096_i16 = arith.constant 4096 : i16
      %c-16384_i16 = arith.constant -16384 : i16
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      memref.store %c0_i16, %alloc[%c0, %c0_0] : memref<3x3xi16>
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      memref.store %c4096_i16, %alloc[%c0_1, %c1] : memref<3x3xi16>
      %c0_2 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      memref.store %c0_i16, %alloc[%c0_2, %c2] : memref<3x3xi16>
      %c1_3 = arith.constant 1 : index
      %c0_4 = arith.constant 0 : index
      memref.store %c4096_i16, %alloc[%c1_3, %c0_4] : memref<3x3xi16>
      %c1_5 = arith.constant 1 : index
      %c1_6 = arith.constant 1 : index
      memref.store %c-16384_i16, %alloc[%c1_5, %c1_6] : memref<3x3xi16>
      %c1_7 = arith.constant 1 : index
      %c2_8 = arith.constant 2 : index
      memref.store %c4096_i16, %alloc[%c1_7, %c2_8] : memref<3x3xi16>
      %c2_9 = arith.constant 2 : index
      %c0_10 = arith.constant 0 : index
      memref.store %c0_i16, %alloc[%c2_9, %c0_10] : memref<3x3xi16>
      %c2_11 = arith.constant 2 : index
      %c1_12 = arith.constant 1 : index
      memref.store %c4096_i16, %alloc[%c2_11, %c1_12] : memref<3x3xi16>
      %c2_13 = arith.constant 2 : index
      %c2_14 = arith.constant 2 : index
      memref.store %c0_i16, %alloc[%c2_13, %c2_14] : memref<3x3xi16>
      %c0_15 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1_16 = arith.constant 1 : index
      scf.for %arg0 = %c0_15 to %c4294967295 step %c1_16 {
        %0 = aie.objectfifo.acquire @OF_2to3(Consume, 2) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %3 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        // func.call @filter2dLine(%1, %1, %2, %4, %c1920_i32, %alloc) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>) -> ()
        aie.objectfifo.release @OF_3to4(Produce, 1)
        %c1_17 = arith.constant 1 : index
        %c1079 = arith.constant 1079 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c1_17 to %c1079 step %c1_18 {
          %10 = aie.objectfifo.acquire @OF_2to3(Consume, 3) : !aie.objectfifosubview<memref<1920xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %12 = aie.objectfifo.subview.access %10[1] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %13 = aie.objectfifo.subview.access %10[2] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %14 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %c1920_i32_20 = arith.constant 1920 : i32
          // func.call @filter2dLine(%11, %12, %13, %15, %c1920_i32_20, %alloc) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>) -> ()
          aie.objectfifo.release @OF_2to3(Consume, 1)
          aie.objectfifo.release @OF_3to4(Produce, 1)
        }
        %5 = aie.objectfifo.acquire @OF_2to3(Consume, 2) : !aie.objectfifosubview<memref<1920xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %8 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32_19 = arith.constant 1920 : i32
        // func.call @filter2dLine(%6, %7, %7, %9, %c1920_i32_19, %alloc) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>) -> ()
        aie.objectfifo.release @OF_2to3(Consume, 2)
        aie.objectfifo.release @OF_3to4(Produce, 1)
      }
      aie.end
    } // {link_with = "filter2d.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c10_i16 = arith.constant 10 : i16
      %c255_i16 = arith.constant 255 : i16
      %c0_i8 = arith.constant 0 : i8
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @OF_3to4(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_4to5(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        // func.call @thresholdLine(%1, %3, %c1920_i32, %c10_i16, %c255_i16, %c0_i8) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_3to4(Consume, 1)
        aie.objectfifo.release @OF_4to5(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @OF_4to5(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_5to5(Produce, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %c1920_i32 = arith.constant 1920 : i32
        // func.call @gray2rgbaLine(%1, %3, %c1920_i32) : (memref<1920xui8>, memref<7680xui8>, i32) -> ()
        aie.objectfifo.release @OF_4to5(Consume, 1)
        aie.objectfifo.release @OF_5to5(Produce, 1)
        %4 = aie.objectfifo.acquire @OF_5to5(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %6 = aie.objectfifo.acquire @inOF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %8 = aie.objectfifo.acquire @outOF_L1L2(Produce, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %c16384_i16 = arith.constant 16384 : i16
        %c16384_i16_0 = arith.constant 16384 : i16
        %c0_i8 = arith.constant 0 : i8
        %c7680_i32 = arith.constant 7680 : i32
        // func.call @addWeightedLine(%5, %7, %9, %c7680_i32, %c16384_i16, %c16384_i16_0, %c0_i8) : (memref<7680xui8>, memref<7680xui8>, memref<7680xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_5to5(Consume, 1)
        aie.objectfifo.release @inOF_L2L1(Consume, 1)
        aie.objectfifo.release @outOF_L1L2(Produce, 1)
      }
      aie.end
    } // {link_with = "combined_gray2rgba_addWeighted.a"}
    func.func @sequence(%arg0: memref<2073600xi32>, %arg1: memref<16x16xi32>, %arg2: memref<2073600xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 2073600][0, 0, 0]) {id = 0 : i64, metadata = @outOF_L2L3} : memref<2073600xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 2073600][0, 0, 0]) {id = 1 : i64, metadata = @inOF_L3L2} : memref<2073600xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}
