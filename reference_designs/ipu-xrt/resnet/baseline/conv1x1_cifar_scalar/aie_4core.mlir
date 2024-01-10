//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(ipu) {

  // Declare tile object of the aie class located at position col 1, row 4
  %tile00 = aie.tile(0, 0)
  %tile01 = aie.tile(0, 1)
  %tile02 = aie.tile(0, 2)
  %tile03 = aie.tile(0, 3)
  %tile04 = aie.tile(0, 4)
  %tile05 = aie.tile(0, 5)

  %rtp2 = aie.buffer(%tile02) {sym_name = "rtp2"} : memref<16xi32>
  %rtp3 = aie.buffer(%tile03) {sym_name = "rtp3"} : memref<16xi32>
  %rtp4 = aie.buffer(%tile04) {sym_name = "rtp4"} : memref<16xi32>
  %rtp5 = aie.buffer(%tile05) {sym_name = "rtp5"} : memref<16xi32>


  //initial activation for 1x1
  aie.objectfifo @inOF_act_L3L2(%tile00, {%tile01}, 2: i32): !aie.objectfifo<memref<8192xi8>> // link the skip buffer in memtile to conv1_skip in tile4
  aie.objectfifo @act_L2_02(%tile01, {%tile02}, 2: i32): !aie.objectfifo<memref<2048xi8>> // link the skip buffer in memtile to conv1_skip in tile4
  aie.objectfifo @act_L2_03(%tile01, {%tile03}, 2: i32): !aie.objectfifo<memref<2048xi8>> // link the skip buffer in memtile to conv1_skip in tile4
  aie.objectfifo @act_L2_04(%tile01, {%tile04}, 2: i32): !aie.objectfifo<memref<2048xi8>> // link the skip buffer in memtile to conv1_skip in tile4
  aie.objectfifo @act_L2_05(%tile01, {%tile05}, 2: i32): !aie.objectfifo<memref<2048xi8>> // link the skip buffer in memtile to conv1_skip in tile4

  aie.objectfifo.link[@inOF_act_L3L2]-> [@act_L2_02,@act_L2_03,@act_L2_04,@act_L2_05] ()

   //wts
  aie.objectfifo @inOF_wts_0_L3L2(%tile00, {%tile02,%tile03,%tile04,%tile05}, 1 : i32) : !aie.objectfifo<memref<4096xi8>> // total buffer for weights


  // Final output OF
  aie.objectfifo @out_02_L2(%tile02, {%tile01}, 2 : i32) : !aie.objectfifo<memref<2048xui8>> //32x1x32
  aie.objectfifo @out_03_L2(%tile03, {%tile01}, 2 : i32) : !aie.objectfifo<memref<2048xui8>> //32x1x32
  aie.objectfifo @out_04_L2(%tile04, {%tile01}, 2 : i32) : !aie.objectfifo<memref<2048xui8>> //32x1x32
  aie.objectfifo @out_05_L2(%tile05, {%tile01}, 2 : i32) : !aie.objectfifo<memref<2048xui8>> //32x1x32

  aie.objectfifo @outOFL2L3(%tile01, {%tile00}, 2 : i32) : !aie.objectfifo<memref<8192xui8>> //2048
  aie.objectfifo.link[@out_02_L2,@out_03_L2,@out_04_L2,@out_05_L2]-> [@outOFL2L3] ()

  func.func private @conv2dk1(memref<2048xi8>, memref<4096xi8>, memref<2048xui8>,i32,i32,i32,i32) -> ()

  // 1x1 conv
  aie.core(%tile02) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %x_dim = arith.constant 32 : i32
    %y_dim = arith.constant 8 : index
 
    %ci = arith.constant 64 : i32
    %co = arith.constant 64 : i32

    %intmax = arith.constant 0xFFFFFFFF : index
    scf.for %arg3 = %c0 to %intmax step %c1 {
      // acquire wts once
      %subviewWts = aie.objectfifo.acquire @inOF_wts_0_L3L2(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
      %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
      %scale = memref.load %rtp2[%c0] : memref<16xi32>
      
      scf.for %n = %c0 to %y_dim step %c1 {
        %subviewIn = aie.objectfifo.acquire @act_L2_02(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>      

        %subviewOut = aie.objectfifo.acquire @out_02_L2(Produce, 1) : !aie.objectfifosubview<memref<2048xui8>>
        %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xui8>> -> memref<2048xui8>

        
        func.call @conv2dk1(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<2048xi8>,memref<4096xi8>, memref<2048xui8>,i32,i32,i32,i32) -> ()

        aie.objectfifo.release @act_L2_02(Consume, 1)
        aie.objectfifo.release @out_02_L2(Produce, 1)
    
      }
      aie.objectfifo.release @inOF_wts_0_L3L2(Consume, 1)
    }
    aie.end
  } { link_with="conv2dk1.o" }

  
  // 1x1 conv
  aie.core(%tile03) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %x_dim = arith.constant 32 : i32
    %y_dim = arith.constant 8 : index
 
    %ci = arith.constant 64 : i32
    %co = arith.constant 64 : i32

    %intmax = arith.constant 0xFFFFFFFF : index
    scf.for %arg3 = %c0 to %intmax step %c1 {
      // acquire wts once
      %subviewWts = aie.objectfifo.acquire @inOF_wts_0_L3L2(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
      %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
      %scale = memref.load %rtp3[%c0] : memref<16xi32>
      
      scf.for %n = %c0 to %y_dim step %c1 {
        %subviewIn = aie.objectfifo.acquire @act_L2_03(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>      

        %subviewOut = aie.objectfifo.acquire @out_03_L2(Produce, 1) : !aie.objectfifosubview<memref<2048xui8>>
        %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xui8>> -> memref<2048xui8>

        
        func.call @conv2dk1(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<2048xi8>,memref<4096xi8>, memref<2048xui8>,i32,i32,i32,i32) -> ()

        aie.objectfifo.release @act_L2_03(Consume, 1)
        aie.objectfifo.release @out_03_L2(Produce, 1)
    
      }
      aie.objectfifo.release @inOF_wts_0_L3L2(Consume, 1)
    }
    aie.end
  } { link_with="conv2dk1.o" }


  
  // 1x1 conv
  aie.core(%tile04) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %x_dim = arith.constant 32 : i32
    %y_dim = arith.constant 8 : index
 
    %ci = arith.constant 64 : i32
    %co = arith.constant 64 : i32

    %intmax = arith.constant 0xFFFFFFFF : index
    scf.for %arg3 = %c0 to %intmax step %c1 {
      // acquire wts once
      %subviewWts = aie.objectfifo.acquire @inOF_wts_0_L3L2(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
      %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
      %scale = memref.load %rtp4[%c0] : memref<16xi32>
      
      scf.for %n = %c0 to %y_dim step %c1 {
        %subviewIn = aie.objectfifo.acquire @act_L2_04(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>      

        %subviewOut = aie.objectfifo.acquire @out_04_L2(Produce, 1) : !aie.objectfifosubview<memref<2048xui8>>
        %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xui8>> -> memref<2048xui8>

        
        func.call @conv2dk1(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<2048xi8>,memref<4096xi8>, memref<2048xui8>,i32,i32,i32,i32) -> ()

        aie.objectfifo.release @act_L2_04(Consume, 1)
        aie.objectfifo.release @out_04_L2(Produce, 1)
    
      }
      aie.objectfifo.release @inOF_wts_0_L3L2(Consume, 1)
    }
    aie.end
  } { link_with="conv2dk1.o" }


  
  // 1x1 conv
  aie.core(%tile05) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %x_dim = arith.constant 32 : i32
    %y_dim = arith.constant 8 : index
 
    %ci = arith.constant 64 : i32
    %co = arith.constant 64 : i32

    %intmax = arith.constant 0xFFFFFFFF : index
    scf.for %arg3 = %c0 to %intmax step %c1 {
      // acquire wts once
      %subviewWts = aie.objectfifo.acquire @inOF_wts_0_L3L2(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
      %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
      %scale = memref.load %rtp5[%c0] : memref<16xi32>
      
      scf.for %n = %c0 to %y_dim step %c1 {
        %subviewIn = aie.objectfifo.acquire @act_L2_05(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>      

        %subviewOut = aie.objectfifo.acquire @out_05_L2(Produce, 1) : !aie.objectfifosubview<memref<2048xui8>>
        %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<2048xui8>> -> memref<2048xui8>

        
        func.call @conv2dk1(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<2048xi8>,memref<4096xi8>, memref<2048xui8>,i32,i32,i32,i32) -> ()

        aie.objectfifo.release @act_L2_05(Consume, 1)
        aie.objectfifo.release @out_05_L2(Produce, 1)
    
      }
      aie.objectfifo.release @inOF_wts_0_L3L2(Consume, 1)
    }
    aie.end
  } { link_with="conv2dk1.o" }


  func.func @sequence(%in0 : memref<16384xi32>, %wts0 : memref<1024xi32>, %out : memref<16384xi32>) {
    aiex.ipu.rtp_write(0, 2, 0,  9) { buffer_sym_name = "rtp2" }  // scale 11 || 6
    aiex.ipu.rtp_write(0, 3, 0,  9) { buffer_sym_name = "rtp3" }  // scale 11 || 6
    aiex.ipu.rtp_write(0, 4, 0,  9) { buffer_sym_name = "rtp4" }  // scale 11 || 6
    aiex.ipu.rtp_write(0, 5, 0,  9) { buffer_sym_name = "rtp5" }  // scale 11 || 6

      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32

      %act_in= arith.constant  16384 : i64 
      %act_out= arith.constant  16384 : i64 
      %total_wts = arith.constant  1024 : i64 

      //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
      // aiex.ipu.dma_memcpy_nd (%c0, %c0, %in0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%act_in][%c0,%c0,%c0]) { metadata = @inOF_act_L3L2, id = 0 : i32 } : (i32, i32, memref<16384xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      // aiex.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%act_out][%c0,%c0,%c0]) { metadata = @outOFL2L3, id = 2 : i32 } : (i32, i32, memref<16384xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      // aiex.ipu.dma_memcpy_nd (%c0, %c0, %wts0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%total_wts][%c0,%c0,%c0]) { metadata = @inOF_wts_0_L3L2, id = 1 : i32 } : (i32, i32, memref<1024xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      // // aiex.ipu.dma_memcpy_nd (%c0, %c0, %wts0[%c0,%c0,%c0,%Ci1_Co1_align1][%c1,%c1,%c1,%Ci2_Co2_align1][%c0,%c0,%c0]) { metadata = @inOF_wts_0_L3L2, id = 1 : i32 } : (i32, i32, memref<13312xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

      aiex.ipu.dma_memcpy_nd(0, 0, %in0[0, 0, 0, 0][1, 1, 1, %act_in][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<16384xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %out[0, 0, 0, 0][1, 1, 1, %act_out][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<16384xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %wts0[0, 0, 0, 0][1, 1, 1, %total_wts][0, 0, 0]) {id = 2 : i64, metadata = @inOF_wts_0_L3L2} : memref<1024xi32>

      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }

    }
}
