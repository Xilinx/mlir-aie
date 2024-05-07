//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
aie.device(npu) {

  //shim
  %tile00 = aie.tile(0, 0)
  %tile10 = aie.tile(1, 0)
  %tile20 = aie.tile(2, 0)

  //memtiles
  %tile01 = aie.tile(0, 1)
  %tile11 = aie.tile(1, 1)
  %tile21 = aie.tile(2, 1)

  %tile02 = aie.tile(0, 2)
  %tile03 = aie.tile(0, 3)
  %tile04 = aie.tile(0, 5)
  %tile05 = aie.tile(0, 4)
 
  %tile12 = aie.tile(1, 2)
  %tile13 = aie.tile(1, 3)
  %tile14 = aie.tile(1, 4)
  %tile15 = aie.tile(1, 5)

  %tile22 = aie.tile(2, 2)
  %tile23 = aie.tile(2, 3)
  %tile24 = aie.tile(2, 4)
  %tile25 = aie.tile(2, 5)
  //Trace: add flow 
  aie.flow(%tile24, "Trace" : 0, %tile00, "DMA" : 1)

  %rtp2 = aie.buffer(%tile02) {sym_name = "rtp2"} : memref<16xi32>
  %rtp3 = aie.buffer(%tile03) {sym_name = "rtp3"} : memref<16xi32>
  %rtp4 = aie.buffer(%tile04) {sym_name = "rtp4"} : memref<16xi32>
  %rtp5 = aie.buffer(%tile05) {sym_name = "rtp5"} : memref<16xi32>

  %rtp12 = aie.buffer(%tile12) {sym_name = "rtp12"} : memref<16xi32>
  %rtp13 = aie.buffer(%tile13) {sym_name = "rtp13"} : memref<16xi32>
  %rtp14 = aie.buffer(%tile14) {sym_name = "rtp14"} : memref<16xi32>
  %rtp15 = aie.buffer(%tile15) {sym_name = "rtp15"} : memref<16xi32>

  %rtp22 = aie.buffer(%tile22) {sym_name = "rtp22"} : memref<16xi32>
  %rtp23 = aie.buffer(%tile23) {sym_name = "rtp23"} : memref<16xi32>
  %rtp24 = aie.buffer(%tile24) {sym_name = "rtp24"} : memref<16xi32>
  %rtp25 = aie.buffer(%tile25) {sym_name = "rtp25"} : memref<16xi32>

  // ___________________________Bottleneck 1___________________________
    //initial activation for 1x1
    aie.objectfifo @inOF_act_L3L2(%tile00, {%tile02,%tile01},[2,2,4]): !aie.objectfifo<memref<32x1x64xi8>> // from shim broadcast to core2 and memtile
    aie.objectfifo @skip_buf(%tile01, {%tile05}, 2: i32): !aie.objectfifo<memref<32x1x64xi8>> // link the skip buffer in memtile to conv1_skip in tile4
    aie.objectfifo.link[@inOF_act_L3L2]-> [@skip_buf] ()
    
    //wts
    aie.objectfifo @inOF_wts_0_L3L2(%tile00, {%tile01}, 1 : i32) : !aie.objectfifo<memref<73728xi8>> // total buffer for weights
    aie.objectfifo @wts_buf_00(%tile01, {%tile02}, 1 : i32) : !aie.objectfifo<memref<4096xi8>> // L1 buffer for first conv1x1 weights 256x64x1x1= 16384
    aie.objectfifo @wts_buf_01(%tile01, {%tile03,%tile04}, 1 : i32) : !aie.objectfifo<memref<36864xi8>> // L1 buffer for middle conv3x3 weights 64x64x3x3= 36864
    aie.objectfifo @wts_buf_02(%tile01, {%tile05}, 1 : i32) : !aie.objectfifo<memref<32768xi8>> // L1 buffer for final conv1x1 weights 64x256x1x1= 16384
    aie.objectfifo.link[@inOF_wts_0_L3L2]-> [@wts_buf_00,@wts_buf_01,@wts_buf_02] ()
    
    // OF for intermediate ofm between 1x1 and 3x3
    aie.objectfifo @act_2_3_4(%tile02, {%tile03,%tile04}, 4 : i32) : !aie.objectfifo<memref<32x1x64xui8>> //32x1x32
    // OF for intermediate ofm between 3x3 and 1x1
    aie.objectfifo @act_3_5(%tile03, {%tile05}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> //32x1x32
    aie.objectfifo @act_4_5(%tile04, {%tile05}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> //32x1x32

  // ___________________________Bottleneck 2___________________________
    //wts
    aie.objectfifo @inOF_wts_1_L3L2(%tile10, {%tile11}, 1 : i32) : !aie.objectfifo<memref<69632xi8>> // total buffer for weights
    aie.objectfifo @wts_buf_10(%tile11, {%tile15}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> // L1 buffer for first conv1x1 weights 256x64x1x1= 16384
    aie.objectfifo @wts_buf_11(%tile11, {%tile12,%tile14}, 1 : i32) : !aie.objectfifo<memref<36864xi8>> // L1 buffer for middle conv3x3 weights 64x64x3x3= 36864
    aie.objectfifo @wts_buf_12(%tile11, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> // L1 buffer for final conv1x1 weights 64x256x1x1= 16384
    aie.objectfifo.link[@inOF_wts_1_L3L2]-> [@wts_buf_10,@wts_buf_11,@wts_buf_12] ()

    //initial activation for 1x1
    aie.objectfifo @act_05_15(%tile05, {%tile15,%tile01},[2,2,4]): !aie.objectfifo<memref<32x1x256xui8>> // from shim broadcast to core2 and memtile
    aie.objectfifo @skip_buf2(%tile01, {%tile13}, 2: i32): !aie.objectfifo<memref<32x1x256xui8>> // link the skip buffer in memtile to conv1_skip in tile4
    aie.objectfifo.link[@act_05_15]-> [@skip_buf2] ()
    
    // OF for intermediate ofm between 1x1 and 3x3
    aie.objectfifo @act_15_12_14(%tile15, {%tile12,%tile14}, 4 : i32) : !aie.objectfifo<memref<32x1x64xui8>> //32x1x32
    
    // OF for intermediate ofm between 3x3 and 1x1
    aie.objectfifo @act_12_13(%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> //32x1x32
    aie.objectfifo @act_14_13(%tile14, {%tile13}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> //32x1x32


  // ___________________________Bottleneck 3___________________________
    //wts
    aie.objectfifo @inOF_wts_2_L3L2(%tile20, {%tile21}, 1 : i32) : !aie.objectfifo<memref<69632xi8>> // total buffer for weights
    aie.objectfifo @wts_buf_20(%tile21, {%tile22}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> // L1 buffer for first conv1x1 weights 256x64x1x1= 16384
    aie.objectfifo @wts_buf_21(%tile21, {%tile23,%tile25}, 1 : i32) : !aie.objectfifo<memref<36864xi8>> // L1 buffer for middle conv3x3 weights 64x64x3x3= 36864
    aie.objectfifo @wts_buf_22(%tile21, {%tile24}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> // L1 buffer for final conv1x1 weights 64x256x1x1= 16384
    aie.objectfifo.link[@inOF_wts_2_L3L2]-> [@wts_buf_20,@wts_buf_21,@wts_buf_22] ()

    //initial activation for 1x1
    aie.objectfifo @act_13_22(%tile13, {%tile22,%tile21},[2,2,4]): !aie.objectfifo<memref<32x1x256xui8>> // from shim broadcast to core2 and memtile
    aie.objectfifo @skip_buf3(%tile21, {%tile24}, 2: i32): !aie.objectfifo<memref<32x1x256xui8>> // link the skip buffer in memtile to conv1_skip in tile4
    aie.objectfifo.link[@act_13_22]-> [@skip_buf3] ()
    
    // OF for intermediate ofm between 1x1 and 3x3
    aie.objectfifo @act_22_23_25(%tile22, {%tile23,%tile25}, 4 : i32) : !aie.objectfifo<memref<32x1x64xui8>> //32x1x32
    
    // OF for intermediate ofm between 3x3 and 1x1
    aie.objectfifo @act_23_24(%tile23, {%tile24}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> //32x1x32
    aie.objectfifo @act_25_24(%tile25, {%tile24}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> //32x1x32

    // Final output OF
    aie.objectfifo @outOFL2L3(%tile24, {%tile10}, 2 : i32) : !aie.objectfifo<memref<32x1x256xui8>> //32x1x64

  // ___________________________Kernel Call___________________________
    func.func private @conv2dk1_i8(memref<32x1x64xi8>, memref<4096xi8>, memref<32x1x64xui8>,i32,i32,i32,i32) -> ()
    func.func private @conv2dk3_ui8(memref<32x1x64xui8>,memref<32x1x64xui8>, memref<32x1x64xui8>,  memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
    func.func private @conv2dk1_skip_init_i8(memref<32x1x32xui8>,memref<32x1x32xui8>, memref<32768xi8>,memref<32x1x256xui8>,memref<32x1x64xi8>,i32,i32,i32,i32,i32,i32,i32) -> ()  
    
    func.func private @conv2dk1_ui8(memref<32x1x256xui8>, memref<16384xi8>, memref<32x1x64xui8>,i32,i32,i32,i32) -> ()
    func.func private @conv2dk1_skip_ui8(memref<32x1x32xui8>,memref<32x1x32xui8>, memref<16384xi8>,memref<32x1x256xui8>,memref<32x1x256xui8>,i32,i32,i32,i32,i32) -> ()
  // ___________________________Bottleneck 1___________________________
    // 1x1 conv
    aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim = arith.constant 32 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 64 : i32

      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        %subviewWts = aie.objectfifo.acquire @wts_buf_00(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
        %scale = memref.load %rtp2[%c0] : memref<16xi32>
        
        scf.for %n = %c0 to %y_dim step %c1 {
          %subviewIn = aie.objectfifo.acquire @inOF_act_L3L2(Consume, 1) : !aie.objectfifosubview<memref<32x1x64xi8>>
          %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xi8>> -> memref<32x1x64xi8>      

          %subviewOut = aie.objectfifo.acquire @act_2_3_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          
          func.call @conv2dk1_i8(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<32x1x64xi8>,memref<4096xi8>, memref<32x1x64xui8>,i32,i32,i32,i32) -> ()

          aie.objectfifo.release @inOF_act_L3L2(Consume, 1)
          aie.objectfifo.release @act_2_3_4(Produce, 1)
      
        }
        aie.objectfifo.release @wts_buf_00(Consume, 1)
      }
      aie.end
    } { link_with="conv2dk1_i8.o" }

    // 3x3 conv
    aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim_minus_2 = arith.constant 30 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 32 : i32

      %kx_dim = arith.constant 3 : i32
      %ky_dim = arith.constant 3 : i32
      
      %top = arith.constant 0 : i32
      %middle = arith.constant 1 : i32
      %bottom = arith.constant 2 : i32

      %co_offset = arith.constant 0 : i32
      // acquire wts once
      // %subviewWts = aie.objectfifo.acquire<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1) : !aie.objectfifosubview<memref<32x32x3x3xi32>>
      // %scale = memref.load %rtp3[%c0] : memref<16xi32>

      %scale = arith.constant 1 : i32
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        %subviewWts = aie.objectfifo.acquire @wts_buf_01(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>

          // Preamble : Top Border
    
          %subviewIn = aie.objectfifo.acquire @act_2_3_4(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut = aie.objectfifo.acquire @act_3_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
          
          
          func.call @conv2dk3_ui8(%elemIn0,%elemIn0,%elemIn1,%elemWts, %elemOut,%x_dim,%ci,%co,%kx_dim,%ky_dim,%top,%scale,%co_offset  ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          
      
          aie.objectfifo.release @act_3_5(Produce, 1)
          
          // Middle
          scf.for %n = %c0 to %y_dim_minus_2 step %c1 {
            %subviewIn1 = aie.objectfifo.acquire @act_2_3_4(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
            %elemIn1_0 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_1 = aie.objectfifo.subview.access %subviewIn1[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_2 = aie.objectfifo.subview.access %subviewIn1[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

            %subviewOut1 = aie.objectfifo.acquire @act_3_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemOut1 = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        
            func.call @conv2dk3_ui8(%elemIn1_0,%elemIn1_1,%elemIn1_2,%elemWts, %elemOut1,%x_dim,%ci,%co,%kx_dim,%ky_dim,%middle,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_3_5(Produce, 1)
            aie.objectfifo.release @act_2_3_4(Consume, 1)
      
        }
        // Postamble : Bottom Border
          %subviewIn2 = aie.objectfifo.acquire @act_2_3_4(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn2_0 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn2_1 = aie.objectfifo.subview.access %subviewIn2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut2 = aie.objectfifo.acquire @act_3_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut2 = aie.objectfifo.subview.access %subviewOut2[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
      
          func.call @conv2dk3_ui8(%elemIn2_0,%elemIn2_1,%elemIn2_1,%elemWts, %elemOut2,%x_dim,%ci,%co,%kx_dim,%ky_dim,%bottom,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          

          aie.objectfifo.release @act_3_5(Produce, 1)
          aie.objectfifo.release @act_2_3_4(Consume, 2)
          
          //release weights
          aie.objectfifo.release @wts_buf_01(Consume, 1)
      }
        // aie.objectfifo.release<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1)
      aie.end
    } { link_with="conv2dk3.o" }

    // 3x3 conv
    aie.core(%tile04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim_minus_2 = arith.constant 30 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 32 : i32

      %kx_dim = arith.constant 3 : i32
      %ky_dim = arith.constant 3 : i32
      
      %top = arith.constant 0 : i32
      %middle = arith.constant 1 : i32
      %bottom = arith.constant 2 : i32

      %co_offset = arith.constant 32 : i32
      %intmax = arith.constant 0xFFFFFFFF : index
      // %scale = memref.load %rtp4[%c0] : memref<16xi32>
      %scale = arith.constant 1 : i32
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        // %subviewWts = aie.objectfifo.acquire<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1) : !aie.objectfifosubview<memref<32x32x3x3xi32>>
        %subviewWts = aie.objectfifo.acquire @wts_buf_01(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>

          // Preamble : Top Border
    
          %subviewIn = aie.objectfifo.acquire @act_2_3_4(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut = aie.objectfifo.acquire @act_4_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
          
          func.call @conv2dk3_ui8(%elemIn0,%elemIn0,%elemIn1,%elemWts, %elemOut,%x_dim,%ci,%co,%kx_dim,%ky_dim,%top,%scale,%co_offset  ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          
      
          aie.objectfifo.release @act_4_5(Produce, 1)
          
          // Middle
          scf.for %n = %c0 to %y_dim_minus_2 step %c1 {
            %subviewIn1 = aie.objectfifo.acquire @act_2_3_4(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
            %elemIn1_0 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_1 = aie.objectfifo.subview.access %subviewIn1[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_2 = aie.objectfifo.subview.access %subviewIn1[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

            %subviewOut1 = aie.objectfifo.acquire @act_4_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemOut1 = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
      
          func.call @conv2dk3_ui8(%elemIn1_0,%elemIn1_1,%elemIn1_2,%elemWts, %elemOut1,%x_dim,%ci,%co,%kx_dim,%ky_dim,%middle,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_4_5(Produce, 1)
            aie.objectfifo.release @act_2_3_4(Consume, 1)
      
        }
        // Postamble : Bottom Border
          %subviewIn2 = aie.objectfifo.acquire @act_2_3_4(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn2_0 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn2_1 = aie.objectfifo.subview.access %subviewIn2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut2 = aie.objectfifo.acquire @act_4_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut2 = aie.objectfifo.subview.access %subviewOut2[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
    
          func.call @conv2dk3_ui8(%elemIn2_0,%elemIn2_1,%elemIn2_1,%elemWts, %elemOut2,%x_dim,%ci,%co,%kx_dim,%ky_dim,%bottom,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          

          aie.objectfifo.release @act_4_5(Produce, 1)
          aie.objectfifo.release @act_2_3_4(Consume, 2)
          
          //release weights
          aie.objectfifo.release @wts_buf_01(Consume, 1)
          // aie.objectfifo.release<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1)
        }
        aie.end
    
    } { link_with="conv2dk3.o" }

    // 1x1 conv with skip
    aie.core(%tile05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      %x_dim = arith.constant 32 : i32
      %y_dim = arith.constant 32 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 256 : i32
      %ci_skip = arith.constant 64 : i32

      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        %subviewWts = aie.objectfifo.acquire @wts_buf_02(Consume, 1) : !aie.objectfifosubview<memref<32768xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<32768xi8>> -> memref<32768xi8>

        %scale = memref.load %rtp5[%c0] : memref<16xi32>
        %skip_scale = memref.load %rtp5[%c1] : memref<16xi32>
        %skip_conv_scale = memref.load %rtp5[%c2] : memref<16xi32>

        // %skip_scale = arith.constant 0 : i32
        scf.for %n = %c0 to %y_dim step %c1 {
          %subviewIn0 = aie.objectfifo.acquire @act_3_5(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn0[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>      

          %subviewIn1 = aie.objectfifo.acquire @act_4_5(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>   

          %subviewOut = aie.objectfifo.acquire @act_05_15(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>

          %subviewSkip = aie.objectfifo.acquire @skip_buf(Consume, 1) : !aie.objectfifosubview<memref<32x1x64xi8>>
          %elemSkip = aie.objectfifo.subview.access %subviewSkip[0] : !aie.objectfifosubview<memref<32x1x64xi8>> -> memref<32x1x64xi8>    

          
          // %skip_scale = arith.constant 0 : i32
          func.call @conv2dk1_skip_init_i8(%elemIn0,%elemIn1,%elemWts, %elemOut0,%elemSkip,%x_dim,%ci,%co,%ci_skip,%scale,%skip_scale,%skip_conv_scale) : (memref<32x1x32xui8>,memref<32x1x32xui8>,  memref<32768xi8>,memref<32x1x256xui8>,memref<32x1x64xi8>,i32,i32,i32,i32,i32,i32,i32) -> ()

          aie.objectfifo.release @act_05_15(Produce, 1)
          aie.objectfifo.release @act_3_5(Consume, 1)
          aie.objectfifo.release @act_4_5(Consume, 1)
          aie.objectfifo.release @skip_buf(Consume, 1)
      
        }
        aie.objectfifo.release @wts_buf_02(Consume, 1)
      }
      aie.end
    } { link_with="conv2dk1_skip_init.o" }
  // ___________________________Bottleneck 2___________________________
    // 1x1 conv
    aie.core(%tile15) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim = arith.constant 32 : index
  
      %ci = arith.constant 256 : i32
      %co = arith.constant 64 : i32

      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        %subviewWts = aie.objectfifo.acquire @wts_buf_10(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %scale = memref.load %rtp15[%c0] : memref<16xi32>
        
        scf.for %n = %c0 to %y_dim step %c1 {
          %subviewIn = aie.objectfifo.acquire @act_05_15(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>      

          %subviewOut = aie.objectfifo.acquire @act_15_12_14(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          
          func.call @conv2dk1_ui8(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<32x1x256xui8>,memref<16384xi8>, memref<32x1x64xui8>,i32,i32,i32,i32) -> ()
          
          aie.objectfifo.release @act_05_15(Consume, 1)
          aie.objectfifo.release @act_15_12_14(Produce, 1)
      
        }
        aie.objectfifo.release @wts_buf_10(Consume, 1)
      }
      aie.end
    } { link_with="conv2dk1_ui8.o" }

    // 3x3 conv
    aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim_minus_2 = arith.constant 30 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 32 : i32

      %kx_dim = arith.constant 3 : i32
      %ky_dim = arith.constant 3 : i32
      
      %top = arith.constant 0 : i32
      %middle = arith.constant 1 : i32
      %bottom = arith.constant 2 : i32

      %co_offset = arith.constant 0 : i32
      // acquire wts once
      // %subviewWts = aie.objectfifo.acquire<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1) : !aie.objectfifosubview<memref<32x32x3x3xi32>>
      // %scale = memref.load %rtp3[%c0] : memref<16xi32>

      %scale = arith.constant 1 : i32
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        %subviewWts = aie.objectfifo.acquire @wts_buf_11(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>

          // Preamble : Top Border
    
          %subviewIn = aie.objectfifo.acquire @act_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut = aie.objectfifo.acquire @act_12_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
          
          
          func.call @conv2dk3_ui8(%elemIn0,%elemIn0,%elemIn1,%elemWts, %elemOut,%x_dim,%ci,%co,%kx_dim,%ky_dim,%top,%scale,%co_offset  ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          
      
          aie.objectfifo.release @act_12_13(Produce, 1)
          
          // Middle
          scf.for %n = %c0 to %y_dim_minus_2 step %c1 {
            %subviewIn1 = aie.objectfifo.acquire @act_15_12_14(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
            %elemIn1_0 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_1 = aie.objectfifo.subview.access %subviewIn1[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_2 = aie.objectfifo.subview.access %subviewIn1[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

            %subviewOut1 = aie.objectfifo.acquire @act_12_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemOut1 = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        
            func.call @conv2dk3_ui8(%elemIn1_0,%elemIn1_1,%elemIn1_2,%elemWts, %elemOut1,%x_dim,%ci,%co,%kx_dim,%ky_dim,%middle,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_12_13(Produce, 1)
            aie.objectfifo.release @act_15_12_14(Consume, 1)
      
        }
        // Postamble : Bottom Border
          %subviewIn2 = aie.objectfifo.acquire @act_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn2_0 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn2_1 = aie.objectfifo.subview.access %subviewIn2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut2 = aie.objectfifo.acquire @act_12_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut2 = aie.objectfifo.subview.access %subviewOut2[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
      
          func.call @conv2dk3_ui8(%elemIn2_0,%elemIn2_1,%elemIn2_1,%elemWts, %elemOut2,%x_dim,%ci,%co,%kx_dim,%ky_dim,%bottom,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          

          aie.objectfifo.release @act_12_13(Produce, 1)
          aie.objectfifo.release @act_15_12_14(Consume, 2)
          
          //release weights
          aie.objectfifo.release @wts_buf_11(Consume, 1)
      }
        // aie.objectfifo.release<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1)
      aie.end
    } { link_with="conv2dk3.o" }

    // 3x3 conv
    aie.core(%tile14) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim_minus_2 = arith.constant 30 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 32 : i32

      %kx_dim = arith.constant 3 : i32
      %ky_dim = arith.constant 3 : i32
      
      %top = arith.constant 0 : i32
      %middle = arith.constant 1 : i32
      %bottom = arith.constant 2 : i32

      %co_offset = arith.constant 32 : i32
      %intmax = arith.constant 0xFFFFFFFF : index
      // %scale = memref.load %rtp4[%c0] : memref<16xi32>
      %scale = arith.constant 1 : i32
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        // %subviewWts = aie.objectfifo.acquire<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1) : !aie.objectfifosubview<memref<32x32x3x3xi32>>
        %subviewWts = aie.objectfifo.acquire @wts_buf_11(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>

          // Preamble : Top Border
    
          %subviewIn = aie.objectfifo.acquire @act_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut = aie.objectfifo.acquire @act_14_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
          
          func.call @conv2dk3_ui8(%elemIn0,%elemIn0,%elemIn1,%elemWts, %elemOut,%x_dim,%ci,%co,%kx_dim,%ky_dim,%top,%scale,%co_offset  ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          
      
          aie.objectfifo.release @act_14_13(Produce, 1)
          
          // Middle
          scf.for %n = %c0 to %y_dim_minus_2 step %c1 {
            %subviewIn1 = aie.objectfifo.acquire @act_15_12_14(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
            %elemIn1_0 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_1 = aie.objectfifo.subview.access %subviewIn1[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_2 = aie.objectfifo.subview.access %subviewIn1[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

            %subviewOut1 = aie.objectfifo.acquire @act_14_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemOut1 = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
      
          func.call @conv2dk3_ui8(%elemIn1_0,%elemIn1_1,%elemIn1_2,%elemWts, %elemOut1,%x_dim,%ci,%co,%kx_dim,%ky_dim,%middle,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_14_13(Produce, 1)
            aie.objectfifo.release @act_15_12_14(Consume, 1)
      
        }
        // Postamble : Bottom Border
          %subviewIn2 = aie.objectfifo.acquire @act_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn2_0 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn2_1 = aie.objectfifo.subview.access %subviewIn2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut2 = aie.objectfifo.acquire @act_14_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut2 = aie.objectfifo.subview.access %subviewOut2[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
    
          func.call @conv2dk3_ui8(%elemIn2_0,%elemIn2_1,%elemIn2_1,%elemWts, %elemOut2,%x_dim,%ci,%co,%kx_dim,%ky_dim,%bottom,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          

          aie.objectfifo.release @act_14_13(Produce, 1)
          aie.objectfifo.release @act_15_12_14(Consume, 2)
          
          //release weights
          aie.objectfifo.release @wts_buf_11(Consume, 1)
          // aie.objectfifo.release<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1)
        }
        aie.end
    
    } { link_with="conv2dk3.o" }

    // 1x1 conv with skip
    aie.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index

        %x_dim = arith.constant 32 : i32
        %y_dim = arith.constant 32 : index
    
        %ci = arith.constant 64 : i32
        %co = arith.constant 256 : i32

        %intmax = arith.constant 0xFFFFFFFF : index
        scf.for %arg3 = %c0 to %intmax step %c1 {
          // acquire wts once
          %subviewWts = aie.objectfifo.acquire @wts_buf_12(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
          %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>

          %scale = memref.load %rtp13[%c0] : memref<16xi32>
          %skip_scale = memref.load %rtp13[%c1] : memref<16xi32>
          // %skip_scale = arith.constant 0 : i32
          scf.for %n = %c0 to %y_dim step %c1 {
            %subviewIn0 = aie.objectfifo.acquire @act_12_13(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemIn0 = aie.objectfifo.subview.access %subviewIn0[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>      

            %subviewIn1 = aie.objectfifo.acquire @act_14_13(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemIn1 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>   

            %subviewOut = aie.objectfifo.acquire @act_13_22(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
            %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>

            %subviewSkip = aie.objectfifo.acquire @skip_buf2(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
            %elemSkip = aie.objectfifo.subview.access %subviewSkip[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>    

            
            // %skip_scale = arith.constant 0 : i32
            func.call @conv2dk1_skip_ui8(%elemIn0,%elemIn1,%elemWts, %elemOut0,%elemSkip,%x_dim,%ci,%co,%scale,%skip_scale) : (memref<32x1x32xui8>,memref<32x1x32xui8>,  memref<16384xi8>,memref<32x1x256xui8>,memref<32x1x256xui8>,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_13_22(Produce, 1)
            aie.objectfifo.release @act_12_13(Consume, 1)
            aie.objectfifo.release @act_14_13(Consume, 1)
            aie.objectfifo.release @skip_buf2(Consume, 1)
        
          }
          aie.objectfifo.release @wts_buf_12(Consume, 1)
        }
        aie.end
      } { link_with="conv2dk1_skip.o" }

            
  // ___________________________Bottleneck 3___________________________
    // 1x1 conv
    aie.core(%tile22) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim = arith.constant 32 : index

      %ci = arith.constant 256 : i32
      %co = arith.constant 64 : i32

      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        %subviewWts = aie.objectfifo.acquire @wts_buf_20(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %scale = memref.load %rtp22[%c0] : memref<16xi32>
        
        scf.for %n = %c0 to %y_dim step %c1 {
          %subviewIn = aie.objectfifo.acquire @act_13_22(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>      

          %subviewOut = aie.objectfifo.acquire @act_22_23_25(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          
          func.call @conv2dk1_ui8(%elemIn,%elemWts,  %elemOut0,%x_dim,%ci,%co,%scale) : (memref<32x1x256xui8>,memref<16384xi8>, memref<32x1x64xui8>,i32,i32,i32,i32) -> ()
          
          aie.objectfifo.release @act_13_22(Consume, 1)
          aie.objectfifo.release @act_22_23_25(Produce, 1)
      
        }
        aie.objectfifo.release @wts_buf_20(Consume, 1)
      }
      aie.end
    } { link_with="conv2dk1_ui8.o" }

    // 3x3 conv
    aie.core(%tile23) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim_minus_2 = arith.constant 30 : index
  
      %ci = arith.constant 64 : i32
      %co = arith.constant 32 : i32

      %kx_dim = arith.constant 3 : i32
      %ky_dim = arith.constant 3 : i32
      
      %top = arith.constant 0 : i32
      %middle = arith.constant 1 : i32
      %bottom = arith.constant 2 : i32

      %co_offset = arith.constant 0 : i32
      // acquire wts once
      // %subviewWts = aie.objectfifo.acquire<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1) : !aie.objectfifosubview<memref<32x32x3x3xi32>>
      // %scale = memref.load %rtp3[%c0] : memref<16xi32>

      %scale = arith.constant 1 : i32
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        %subviewWts = aie.objectfifo.acquire @wts_buf_21(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>

          // Preamble : Top Border
    
          %subviewIn = aie.objectfifo.acquire @act_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut = aie.objectfifo.acquire @act_23_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
          
          
          func.call @conv2dk3_ui8(%elemIn0,%elemIn0,%elemIn1,%elemWts, %elemOut,%x_dim,%ci,%co,%kx_dim,%ky_dim,%top,%scale,%co_offset  ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          
      
          aie.objectfifo.release @act_23_24(Produce, 1)
          
          // Middle
          scf.for %n = %c0 to %y_dim_minus_2 step %c1 {
            %subviewIn1 = aie.objectfifo.acquire @act_22_23_25(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
            %elemIn1_0 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_1 = aie.objectfifo.subview.access %subviewIn1[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_2 = aie.objectfifo.subview.access %subviewIn1[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

            %subviewOut1 = aie.objectfifo.acquire @act_23_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemOut1 = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        
            func.call @conv2dk3_ui8(%elemIn1_0,%elemIn1_1,%elemIn1_2,%elemWts, %elemOut1,%x_dim,%ci,%co,%kx_dim,%ky_dim,%middle,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_23_24(Produce, 1)
            aie.objectfifo.release @act_22_23_25(Consume, 1)
      
        }
        // Postamble : Bottom Border
          %subviewIn2 = aie.objectfifo.acquire @act_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn2_0 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn2_1 = aie.objectfifo.subview.access %subviewIn2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut2 = aie.objectfifo.acquire @act_23_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut2 = aie.objectfifo.subview.access %subviewOut2[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
      
          func.call @conv2dk3_ui8(%elemIn2_0,%elemIn2_1,%elemIn2_1,%elemWts, %elemOut2,%x_dim,%ci,%co,%kx_dim,%ky_dim,%bottom,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          

          aie.objectfifo.release @act_23_24(Produce, 1)
          aie.objectfifo.release @act_22_23_25(Consume, 2)
          
          //release weights
          aie.objectfifo.release @wts_buf_21(Consume, 1)
      }
        // aie.objectfifo.release<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1)
      aie.end
    } { link_with="conv2dk3.o" }
      
    // 3x3 conv
    aie.core(%tile25) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %x_dim = arith.constant 32 : i32
      %y_dim_minus_2 = arith.constant 30 : index

      %ci = arith.constant 64 : i32
      %co = arith.constant 32 : i32

      %kx_dim = arith.constant 3 : i32
      %ky_dim = arith.constant 3 : i32
      
      %top = arith.constant 0 : i32
      %middle = arith.constant 1 : i32
      %bottom = arith.constant 2 : i32

      %co_offset = arith.constant 32 : i32
      %intmax = arith.constant 0xFFFFFFFF : index
      // %scale = memref.load %rtp4[%c0] : memref<16xi32>
      %scale = arith.constant 1 : i32
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        // %subviewWts = aie.objectfifo.acquire<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1) : !aie.objectfifosubview<memref<32x32x3x3xi32>>
        %subviewWts = aie.objectfifo.acquire @wts_buf_21(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>

          // Preamble : Top Border

          %subviewIn = aie.objectfifo.acquire @act_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut = aie.objectfifo.acquire @act_25_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
          
          func.call @conv2dk3_ui8(%elemIn0,%elemIn0,%elemIn1,%elemWts, %elemOut,%x_dim,%ci,%co,%kx_dim,%ky_dim,%top,%scale,%co_offset  ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          
      
          aie.objectfifo.release @act_25_24(Produce, 1)
          
          // Middle
          scf.for %n = %c0 to %y_dim_minus_2 step %c1 {
            %subviewIn1 = aie.objectfifo.acquire @act_22_23_25(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
            %elemIn1_0 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_1 = aie.objectfifo.subview.access %subviewIn1[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
            %elemIn1_2 = aie.objectfifo.subview.access %subviewIn1[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

            %subviewOut1 = aie.objectfifo.acquire @act_25_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
            %elemOut1 = aie.objectfifo.subview.access %subviewOut1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        
          func.call @conv2dk3_ui8(%elemIn1_0,%elemIn1_1,%elemIn1_2,%elemWts, %elemOut1,%x_dim,%ci,%co,%kx_dim,%ky_dim,%middle,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()

            aie.objectfifo.release @act_25_24(Produce, 1)
            aie.objectfifo.release @act_22_23_25(Consume, 1)
      
        }
        // Postamble : Bottom Border
          %subviewIn2 = aie.objectfifo.acquire @act_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %elemIn2_0 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %elemIn2_1 = aie.objectfifo.subview.access %subviewIn2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>

          %subviewOut2 = aie.objectfifo.acquire @act_25_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemOut2 = aie.objectfifo.subview.access %subviewOut2[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          
      
          func.call @conv2dk3_ui8(%elemIn2_0,%elemIn2_1,%elemIn2_1,%elemWts, %elemOut2,%x_dim,%ci,%co,%kx_dim,%ky_dim,%bottom,%scale,%co_offset ) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>,memref<32x1x32xui8>,i32,i32,i32,i32,i32,i32,i32,i32) -> ()
          

          aie.objectfifo.release @act_25_24(Produce, 1)
          aie.objectfifo.release @act_22_23_25(Consume, 2)
          
          //release weights
          aie.objectfifo.release @wts_buf_21(Consume, 1)
          // aie.objectfifo.release<Consume>(%inOF_wts_0_L3L2 : !aie.objectfifo<memref<32x32x3x3xi32>>, 1)
          }
        aie.end
      
    } { link_with="conv2dk3.o" }
    
    // 1x1 conv with skip
    aie.core(%tile24) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      %x_dim = arith.constant 32 : i32
      %y_dim = arith.constant 32 : index

      %ci = arith.constant 64 : i32
      %co = arith.constant 256 : i32

      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
        // acquire wts once
        %subviewWts = aie.objectfifo.acquire @wts_buf_22(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %elemWts = aie.objectfifo.subview.access %subviewWts[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>

        %scale = memref.load %rtp24[%c0] : memref<16xi32>
        %skip_scale = memref.load %rtp24[%c1] : memref<16xi32>
        // %skip_scale = arith.constant 0 : i32
        scf.for %n = %c0 to %y_dim step %c1 {
          %subviewIn0 = aie.objectfifo.acquire @act_23_24(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemIn0 = aie.objectfifo.subview.access %subviewIn0[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>      

          %subviewIn1 = aie.objectfifo.acquire @act_25_24(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %elemIn1 = aie.objectfifo.subview.access %subviewIn1[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>   

          %subviewOut = aie.objectfifo.acquire @outOFL2L3(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %elemOut0 = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>

          %subviewSkip = aie.objectfifo.acquire @skip_buf3(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %elemSkip = aie.objectfifo.subview.access %subviewSkip[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>    

          
          // %skip_scale = arith.constant 0 : i32
          func.call @conv2dk1_skip_ui8(%elemIn0,%elemIn1,%elemWts, %elemOut0,%elemSkip,%x_dim,%ci,%co,%scale,%skip_scale) : (memref<32x1x32xui8>,memref<32x1x32xui8>,  memref<16384xi8>,memref<32x1x256xui8>,memref<32x1x256xui8>,i32,i32,i32,i32,i32) -> ()

          aie.objectfifo.release @outOFL2L3(Produce, 1)
          aie.objectfifo.release @act_23_24(Consume, 1)
          aie.objectfifo.release @act_25_24(Consume, 1)
          aie.objectfifo.release @skip_buf3(Consume, 1)
      
        }
        aie.objectfifo.release @wts_buf_22(Consume, 1)
      }
      aie.end
    } { link_with="conv2dk1_skip.o" }

    
  func.func @sequence(%in0 : memref<16384xi32>, %wts0 : memref<53248xi32>, %out : memref<65536xi32>) {
                  // Trace output

      // Trace_Event0, Trace_Event1: Select which events to trace.
      // Note that the event buffers only appear to be transferred to DDR in
      // bursts of 256 bytes. If less than 256 bytes are written, you may not
      // see trace output, or only see it on the next iteration of your 
      // kernel invocation, as the buffer gets filled up. Note that, even
      // though events are encoded as 4 byte words, it may take more than 64 
      // events to fill the buffer to 256 bytes and cause a flush, since
      // multiple repeating events can be 'compressed' by the trace mechanism.
      // In order to always generate sufficient events, we add the "assert 
      // TRUE" event to one slot, which fires every cycle, and thus fills our
      // buffer quickly.

      // Some events:
      // TRUE                       (0x01)
      // STREAM_STALL               (0x18)
      // LOCK_STALL                 (0x1A)
      // EVENTS_CORE_INSTR_EVENT_1  (0x22)
      // EVENTS_CORE_INSTR_EVENT_0  (0x21)
      // INSTR_VECTOR               (0x25)  Core executes a vecotr MAC, ADD or compare instruction
      // INSTR_LOCK_ACQUIRE_REQ     (0x2C)  Core executes a lock acquire instruction
      // INSTR_LOCK_RELEASE_REQ     (0x2D)  Core executes a lock release instruction
      // EVENTS_CORE_PORT_RUNNING_1 (0x4F)
      // EVENTS_CORE_PORT_RUNNING_0 (0x4B)


      // Trace_Event0  (4 slots)
      aiex.npu.write32 { column = 2 : i32, row = 4 : i32, address = 0x340E0 : ui32, value = 0x4B222125 : ui32 }
      // Trace_Event1  (4 slots)
      aiex.npu.write32 { column = 2 : i32, row = 4 : i32, address = 0x340E4 : ui32, value = 0x2D2C1A4F : ui32 }

      // Event slots as configured above:
      // 0: Kernel executes vector instruction
      // 1: Event 0 -- Kernel starts
      // 2: Event 1 -- Kernel done
      // 3: Port_Running_0
      // 4: Port_Running_1
      // 5: Lock Stall
      // 6: Lock Acquire Instr
      // 7: Lock Release Instr

      // Stream_Switch_Event_Port_Selection_0
      // This is necessary to capture the Port_Running_0 and Port_Running_1 events
      aiex.npu.write32 { column = 2 : i32, row = 4 : i32, address = 0x3FF00 : ui32, value = 0x121 : ui32 }

      // Trace_Control0: Define trace start and stop triggers. Set start event TRUE.
      aiex.npu.write32 { column = 2 : i32, row = 4 : i32, address = 0x340D0 : ui32, value = 0x10000 : ui32 }

      // Start trace copy out.
      aiex.npu.writebd_shimtile { bd_id = 3 : i32,
                                  buffer_length = 16384 : i32,
                                  buffer_offset = 262144 : i32,
                                  enable_packet = 0 : i32,
                                  out_of_order_id = 0 : i32,
                                  packet_id = 0 : i32,
                                  packet_type = 0 : i32,
                                  column = 0 : i32,
                                  column_num = 1 : i32,
                                  d0_stepsize = 0 : i32,
                                  d0_size = 0 : i32,
                                  d0_stride = 0 : i32, 
                                  d0_wrap = 0 : i32,
                                  d1_stepsize = 0 : i32,
                                  d1_wrap = 0 : i32,
                                  d1_size = 0 : i32,
                                  d1_stride = 0 : i32, 
                                  d2_stepsize = 0 : i32,
                                  d2_size = 0 : i32,
                                  d2_stride = 0 : i32, 
                                  ddr_id = 2 : i32,
                                  iteration_current = 0 : i32,
                                  iteration_stepsize = 0 : i32,
                                  iteration_wrap = 0 : i32,
                                  iteration_size = 0 : i32,
                                  iteration_stride = 0 : i32,
                                  lock_acq_enable = 0 : i32,
                                  lock_acq_id = 0 : i32,
                                  lock_acq_val = 0 : i32,
                                  lock_rel_id = 0 : i32,
                                  lock_rel_val = 0 : i32,
                                  next_bd = 0 : i32,
                                  use_next_bd = 0 : i32,
                                  valid_bd = 1 : i32}
      aiex.npu.write32 { column = 0 : i32, row = 0 : i32, address = 0x1D20C : ui32, value = 0x3 : ui32 }

    //End trace dump

      

      aiex.npu.rtp_write(0, 2, 0,  1) { buffer_sym_name = "rtp2" }  
      aiex.npu.rtp_write(0, 3, 0,  1) { buffer_sym_name = "rtp3" } 
      aiex.npu.rtp_write(0, 5, 0,  1) { buffer_sym_name = "rtp4" }  
      aiex.npu.rtp_write(0, 4, 0,  1)  { buffer_sym_name = "rtp5" }  
      aiex.npu.rtp_write(0, 4, 1,  0)  { buffer_sym_name = "rtp5" }  
      aiex.npu.rtp_write(0, 4, 2,  1)  { buffer_sym_name = "rtp5" }  

      aiex.npu.rtp_write(1, 5, 0,  1) { buffer_sym_name = "rtp15" }  
      aiex.npu.rtp_write(1, 4, 0,  1) { buffer_sym_name = "rtp14" }  
      aiex.npu.rtp_write(1, 2, 0,  1) { buffer_sym_name = "rtp12" }  
      aiex.npu.rtp_write(1, 3, 0,  1)  { buffer_sym_name = "rtp13" }  
      aiex.npu.rtp_write(1, 3, 1,  0)  { buffer_sym_name = "rtp13" }  

      aiex.npu.rtp_write(2, 2, 0,  1) { buffer_sym_name = "rtp22" }  
      aiex.npu.rtp_write(2, 3, 0,  1) { buffer_sym_name = "rtp23" }  
      aiex.npu.rtp_write(2, 5, 0,  1) { buffer_sym_name = "rtp25" }  
      aiex.npu.rtp_write(2, 4, 0,  1)  { buffer_sym_name = "rtp24" }  
      aiex.npu.rtp_write(2, 4, 1,  0)  { buffer_sym_name = "rtp24" } 

      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %act_in= arith.constant  16384 : i64 
      %act_out= arith.constant  65536 : i64 
      %total_wts = arith.constant  18432 : i64 
      %total_wts_2 = arith.constant  17408 : i64 
      %total_wts_3 = arith.constant  17408 : i64 
      %total_wts_3_off = arith.constant  35840 : i64 

      //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
      aiex.npu.dma_memcpy_nd(0, 0, %in0[0, 0, 0, 0][1, 1, 1, %act_in][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<16384xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %out[0, 0, 0, 0][1, 1, 1, %act_out][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %wts0[0, 0, 0, 0][1, 1, 1, %total_wts][0, 0, 0]) {id = 1 : i64, metadata = @inOF_wts_0_L3L2} : memref<53248xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %wts0[0, 0, 0, %total_wts][1, 1, 1, %total_wts_2][0, 0, 0]) {id = 1 : i64, metadata = @inOF_wts_1_L3L2} : memref<53248xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %wts0[0, 0, 0, %total_wts_3_off][1, 1, 1, %total_wts_3][0, 0, 0]) {id = 1 : i64, metadata = @inOF_wts_2_L3L2} : memref<53248xi32>

      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }

    }
}