module {
  aie.device(npu1_1col) {
    func.func private @conv2dk1_i8(memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32)
    func.func private @conv2dk3_ui8(memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @conv2dk1_ui8(memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @inOF_act_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @OF_bneck_10_memtile_layer1_act(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo.link [@inOF_act_L3L2] -> [@OF_bneck_10_memtile_layer1_act]()
    aie.objectfifo @OF_bneck_10_wts_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<96480xi8>>
    aie.objectfifo @OF_bneck_10_wts_memtile_layer1(%tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<38400xi8>>
    aie.objectfifo @OF_bneck_10_wts_memtile_layer2(%tile_0_1, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<4320xi8>>
    aie.objectfifo @OF_bneck_10_wts_memtile_layer3(%tile_0_1, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<53760xi8>>
    aie.objectfifo.link [@OF_bneck_10_wts_L3L2] -> [@OF_bneck_10_wts_memtile_layer1, @OF_bneck_10_wts_memtile_layer2, @OF_bneck_10_wts_memtile_layer3]()
    aie.objectfifo @OF_bneck_10_act_layer1_layer2(%tile_0_2, {%tile_0_3}, 4 : i32) {via_DMA = true} : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @OF_bneck_10_act_layer2_layer3(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @OF_bneck_10_layer3_final(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo @outOFL2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo.link [@OF_bneck_10_layer3_final] -> [@outOFL2L3]()
    %rtp2 = aie.buffer(%tile_0_2) {sym_name = "rtp2"} : memref<16xi32> 
    %rtp3 = aie.buffer(%tile_0_3) {sym_name = "rtp3"} : memref<16xi32> 
    %rtp4 = aie.buffer(%tile_0_4) {sym_name = "rtp4"} : memref<16xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_bneck_10_wts_memtile_layer1(Consume, 1) : !aie.objectfifosubview<memref<38400xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<38400xi8>> -> memref<38400xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtp2[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c14 step %c1_2 {
          %3 = aie.objectfifo.acquire @OF_bneck_10_memtile_layer1_act(Consume, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %5 = aie.objectfifo.acquire @OF_bneck_10_act_layer1_layer2(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %c14_i32 = arith.constant 14 : i32
          %c80_i32 = arith.constant 80 : i32
          %c480_i32 = arith.constant 480 : i32
          func.call @conv2dk1_i8(%4, %1, %6, %c14_i32, %c80_i32, %c480_i32, %2) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_bneck_10_memtile_layer1_act(Consume, 1)
          aie.objectfifo.release @OF_bneck_10_act_layer1_layer2(Produce, 1)
        }
        aie.objectfifo.release @OF_bneck_10_wts_memtile_layer1(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_fused_relu.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_bneck_10_wts_memtile_layer2(Consume, 1) : !aie.objectfifosubview<memref<4320xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4320xi8>> -> memref<4320xi8>
        %2 = aie.objectfifo.acquire @OF_bneck_10_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %5 = aie.objectfifo.acquire @OF_bneck_10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c480_i32 = arith.constant 480 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_0 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_1 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c14_i32, %c1_i32, %c480_i32, %c3_i32, %c3_i32_0, %c0_i32, %c8_i32, %c0_i32_1) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_bneck_10_act_layer2_layer3(Produce, 1)
        %c0_2 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c12 step %c1_3 {
          %12 = aie.objectfifo.acquire @OF_bneck_10_act_layer1_layer2(Consume, 3) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %16 = aie.objectfifo.acquire @OF_bneck_10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %c14_i32_11 = arith.constant 14 : i32
          %c1_i32_12 = arith.constant 1 : i32
          %c480_i32_13 = arith.constant 480 : i32
          %c3_i32_14 = arith.constant 3 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c1_i32_16 = arith.constant 1 : i32
          %c8_i32_17 = arith.constant 8 : i32
          %c0_i32_18 = arith.constant 0 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c14_i32_11, %c1_i32_12, %c480_i32_13, %c3_i32_14, %c3_i32_15, %c1_i32_16, %c8_i32_17, %c0_i32_18) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_bneck_10_act_layer1_layer2(Consume, 1)
          aie.objectfifo.release @OF_bneck_10_act_layer2_layer3(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @OF_bneck_10_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %10 = aie.objectfifo.acquire @OF_bneck_10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %c14_i32_4 = arith.constant 14 : i32
        %c1_i32_5 = arith.constant 1 : i32
        %c480_i32_6 = arith.constant 480 : i32
        %c3_i32_7 = arith.constant 3 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c8_i32_9 = arith.constant 8 : i32
        %c0_i32_10 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c14_i32_4, %c1_i32_5, %c480_i32_6, %c3_i32_7, %c3_i32_8, %c2_i32, %c8_i32_9, %c0_i32_10) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_bneck_10_act_layer1_layer2(Consume, 2)
        aie.objectfifo.release @OF_bneck_10_act_layer2_layer3(Produce, 1)
        aie.objectfifo.release @OF_bneck_10_wts_memtile_layer2(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3_dw.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @OF_bneck_10_wts_memtile_layer3(Consume, 1) : !aie.objectfifosubview<memref<53760xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<53760xi8>> -> memref<53760xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtp4[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c14 step %c1_2 {
          %3 = aie.objectfifo.acquire @OF_bneck_10_act_layer2_layer3(Consume, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %5 = aie.objectfifo.acquire @OF_bneck_10_layer3_final(Produce, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %c14_i32 = arith.constant 14 : i32
          %c480_i32 = arith.constant 480 : i32
          %c112_i32 = arith.constant 112 : i32
          func.call @conv2dk1_ui8(%4, %1, %6, %c14_i32, %c480_i32, %c112_i32, %2) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_bneck_10_act_layer2_layer3(Consume, 1)
          aie.objectfifo.release @OF_bneck_10_layer3_final(Produce, 1)
        }
        aie.objectfifo.release @OF_bneck_10_wts_memtile_layer3(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_ui8.o"}
    func.func @sequence(%arg0: memref<3920xi32>, %arg1: memref<24120xi32>, %arg2: memref<5488xi32>) {
      aiex.npu.rtp_write(0, 2, 0, 9) {buffer_sym_name = "rtp2"}
      aiex.npu.rtp_write(0, 3, 0, 8) {buffer_sym_name = "rtp3"}
      aiex.npu.rtp_write(0, 4, 0, 12) {buffer_sym_name = "rtp4"}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 3920][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<3920xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 5488][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<5488xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 24120][0, 0, 0]) {id = 1 : i64, metadata = @OF_bneck_10_wts_L3L2} : memref<24120xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

