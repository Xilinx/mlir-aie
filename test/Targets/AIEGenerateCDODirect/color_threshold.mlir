// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin

module {
  aie.device(ipu) {
    // // func.func private @thresholdLine(memref<512xui8>, memref<512xui8>, i32, i16, i16, i8)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @inOOB_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xui8>>
    aie.objectfifo @inOOB_L2L1_0(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @inOOB_L2L1_1(%tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @inOOB_L2L1_2(%tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @inOOB_L2L1_3(%tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo.link [@inOOB_L3L2] -> [@inOOB_L2L1_0, @inOOB_L2L1_1, @inOOB_L2L1_2, @inOOB_L2L1_3]()
    aie.objectfifo @outOOB_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<2048xui8>>
    aie.objectfifo @outOOB_L1L2_0(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @outOOB_L1L2_1(%tile_0_3, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @outOOB_L1L2_2(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @outOOB_L1L2_3(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo.link [@outOOB_L1L2_0, @outOOB_L1L2_1, @outOOB_L1L2_2, @outOOB_L1L2_3] -> [@outOOB_L2L3]()
    %rtpComputeTile2 = aie.buffer(%tile_0_2) {sym_name = "rtpComputeTile2"} : memref<16xi32>
    %rtpComputeTile3 = aie.buffer(%tile_0_3) {sym_name = "rtpComputeTile3"} : memref<16xi32>
    %rtpComputeTile4 = aie.buffer(%tile_0_4) {sym_name = "rtpComputeTile4"} : memref<16xi32>
    %rtpComputeTile5 = aie.buffer(%tile_0_5) {sym_name = "rtpComputeTile5"} : memref<16xi32>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<512xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_0(Produce, 1) : !aie.objectfifosubview<memref<512xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile2[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile2[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile2[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c512_i32 = arith.constant 512 : i32
        // func.call @thresholdLine(%1, %3, %c512_i32, %5, %7, %9) : (memref<512xui8>, memref<512xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_0(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_0(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<512xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_1(Produce, 1) : !aie.objectfifosubview<memref<512xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile3[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile3[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile3[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c512_i32 = arith.constant 512 : i32
        // func.call @thresholdLine(%1, %3, %c512_i32, %5, %7, %9) : (memref<512xui8>, memref<512xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_1(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_1(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<512xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_2(Produce, 1) : !aie.objectfifosubview<memref<512xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile4[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile4[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile4[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c512_i32 = arith.constant 512 : i32
        // func.call @thresholdLine(%1, %3, %c512_i32, %5, %7, %9) : (memref<512xui8>, memref<512xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_2(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_2(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<512xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_3(Produce, 1) : !aie.objectfifosubview<memref<512xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile5[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile5[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile5[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c512_i32 = arith.constant 512 : i32
        // func.call @thresholdLine(%1, %3, %c512_i32, %5, %7, %9) : (memref<512xui8>, memref<512xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_3(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_3(Produce, 1)
      }
      aie.end
    } // {link_with = "threshold.cc.o"}
    func.func @sequence(%arg0: memref<1152xi32>, %arg1: memref<32xi32>, %arg2: memref<1152xi32>) {
      aiex.ipu.rtp_write(0, 2, 0, 50) {buffer_sym_name = "rtpComputeTile2"}
      aiex.ipu.rtp_write(0, 2, 1, 255) {buffer_sym_name = "rtpComputeTile2"}
      aiex.ipu.rtp_write(0, 2, 2, 0) {buffer_sym_name = "rtpComputeTile2"}
      aiex.ipu.rtp_write(0, 3, 0, 50) {buffer_sym_name = "rtpComputeTile3"}
      aiex.ipu.rtp_write(0, 3, 1, 255) {buffer_sym_name = "rtpComputeTile3"}
      aiex.ipu.rtp_write(0, 3, 2, 0) {buffer_sym_name = "rtpComputeTile3"}
      aiex.ipu.rtp_write(0, 4, 0, 50) {buffer_sym_name = "rtpComputeTile4"}
      aiex.ipu.rtp_write(0, 4, 1, 255) {buffer_sym_name = "rtpComputeTile4"}
      aiex.ipu.rtp_write(0, 4, 2, 0) {buffer_sym_name = "rtpComputeTile4"}
      aiex.ipu.rtp_write(0, 5, 0, 50) {buffer_sym_name = "rtpComputeTile5"}
      aiex.ipu.rtp_write(0, 5, 1, 255) {buffer_sym_name = "rtpComputeTile5"}
      aiex.ipu.rtp_write(0, 5, 2, 0) {buffer_sym_name = "rtpComputeTile5"}
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1152][0, 0, 0]) {id = 1 : i64, metadata = @inOOB_L3L2} : memref<1152xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 1152][0, 0, 0]) {id = 0 : i64, metadata = @outOOB_L2L3} : memref<1152xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}