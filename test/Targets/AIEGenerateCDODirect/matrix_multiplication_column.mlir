// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module {
  aie.device(ipu) {
    // // func.func private @zero_scalar_i16(memref<64x64xi16>)
    // // func.func private @zero_i16(memref<64x64xi16>)
    // // func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    // // func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @inA(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8192xi16>>
    aie.objectfifo @memA0(%tile_0_1 toStream [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xi16>>
    aie.objectfifo @memA1(%tile_0_1 toStream [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>], {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<64x32xi16>>
    aie.objectfifo @memA2(%tile_0_1 toStream [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>], {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64x32xi16>>
    aie.objectfifo @memA3(%tile_0_1 toStream [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi16>>
    aie.objectfifo.link [@inA] -> [@memA0, @memA1, @memA2, @memA3]()
    aie.objectfifo @inB(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xi16>>
    aie.objectfifo @memB(%tile_0_1 toStream [<size = 8, stride = 128>, <size = 16, stride = 2>, <size = 4, stride = 32>, <size = 2, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xi16>>
    aie.objectfifo.link [@inB] -> [@memB]()
    aie.objectfifo @memC0(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi16>>
    aie.objectfifo @memC1(%tile_0_3, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi16>>
    aie.objectfifo @memC2(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi16>>
    aie.objectfifo @memC3(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi16>>
    aie.objectfifo @outC(%tile_0_1 toStream [<size = 16, stride = 128>, <size = 4, stride = 2>, <size = 16, stride = 8>, <size = 2, stride = 1>], {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16384xi16>>
    aie.objectfifo.link [@memC0, @memC1, @memC2, @memC3] -> [@outC]()
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC0(Produce, 1) : !aie.objectfifosubview<memref<64x64xi16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xi16>> -> memref<64x64xi16>
          // func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA0(Consume, 1) : !aie.objectfifosubview<memref<64x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x32xi16>> -> memref<64x32xi16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x64xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi16>> -> memref<32x64xi16>
            // func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.objectfifo.release @memA0(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC0(Produce, 1)
        }
      }
      aie.end
    } // {link_with = "mm.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC1(Produce, 1) : !aie.objectfifosubview<memref<64x64xi16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xi16>> -> memref<64x64xi16>
          // func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA1(Consume, 1) : !aie.objectfifosubview<memref<64x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x32xi16>> -> memref<64x32xi16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x64xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi16>> -> memref<32x64xi16>
            // func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.objectfifo.release @memA1(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC1(Produce, 1)
        }
      }
      aie.end
    } // {link_with = "mm.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC2(Produce, 1) : !aie.objectfifosubview<memref<64x64xi16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xi16>> -> memref<64x64xi16>
          // func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA2(Consume, 1) : !aie.objectfifosubview<memref<64x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x32xi16>> -> memref<64x32xi16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x64xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi16>> -> memref<32x64xi16>
            // func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.objectfifo.release @memA2(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC2(Produce, 1)
        }
      }
      aie.end
    } // {link_with = "mm.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC3(Produce, 1) : !aie.objectfifosubview<memref<64x64xi16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xi16>> -> memref<64x64xi16>
          // func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA3(Consume, 1) : !aie.objectfifosubview<memref<64x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x32xi16>> -> memref<64x32xi16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x64xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi16>> -> memref<32x64xi16>
            // func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            aie.objectfifo.release @memA3(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC3(Produce, 1)
        }
      }
      aie.end
    } // {link_with = "mm.o"}
    func.func @sequence(%arg0: memref<16384xi32>, %arg1: memref<8192xi32>, %arg2: memref<16384xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 2, 256, 32][16384, 32, 64]) {id = 0 : i64, metadata = @outC} : memref<16384xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][2, 4, 256, 16][0, 16, 64]) {id = 1 : i64, metadata = @inA} : memref<16384xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 2, 128, 32][0, 32, 64]) {id = 2 : i64, metadata = @inB} : memref<8192xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}
