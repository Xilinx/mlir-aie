module {
  aie.device(npu1_3col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    aie.objectfifo @act_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @act_out(%tile_2_2, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<7x1x80xi8>>
    aie.objectfifo @wts_b10_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<96480xi8>>
    aie.objectfifo @weightsInBN10_layer1(%tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<38400xi8>>
    aie.objectfifo @weightsInBN10_layer2(%tile_0_1, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<4320xi8>>
    aie.objectfifo @weightsInBN10_layer3(%tile_0_1, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<53760xi8>>
    aie.objectfifo.link [@wts_b10_L3L2] -> [@weightsInBN10_layer1, @weightsInBN10_layer2, @weightsInBN10_layer3]([] [0, 38400, 42720])
    aie.objectfifo @wts_b11_L3L2(%tile_1_0, {%tile_1_1}, 1 : i32) : !aie.objectfifo<memref<78288xi8>>
    aie.objectfifo @weightsInBN11_layer1(%tile_1_1, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<37632xi8>>
    aie.objectfifo @weightsInBN11_layer2(%tile_1_1, {%tile_1_5}, 1 : i32) : !aie.objectfifo<memref<3024xi8>>
    aie.objectfifo @weightsInBN11_layer3(%tile_1_1, {%tile_1_4}, 1 : i32) : !aie.objectfifo<memref<37632xi8>>
    aie.objectfifo.link [@wts_b11_L3L2] -> [@weightsInBN11_layer1, @weightsInBN11_layer2, @weightsInBN11_layer3]([] [0, 37632, 40656])
    aie.objectfifo @wts_b12_L3L2(%tile_1_0, {%tile_2_1}, 1 : i32) : !aie.objectfifo<memref<67536xi8>>
    aie.objectfifo @weightsInBN12_layer1(%tile_2_1, {%tile_1_3}, 1 : i32) : !aie.objectfifo<memref<37632xi8>>
    aie.objectfifo @weightsInBN12_layer2(%tile_2_1, {%tile_1_2}, 1 : i32) : !aie.objectfifo<memref<3024xi8>>
    aie.objectfifo @weightsInBN12_layer3(%tile_2_1, {%tile_2_2}, 1 : i32) : !aie.objectfifo<memref<26880xi8>>
    aie.objectfifo.link [@wts_b12_L3L2] -> [@weightsInBN12_layer1, @weightsInBN12_layer2, @weightsInBN12_layer3]([] [0, 37632, 40656])
    %bn10_1_rtp = aie.buffer(%tile_0_2) {sym_name = "bn10_1_rtp"} : memref<16xi32> 
    %bn10_2_rtp = aie.buffer(%tile_0_3) {sym_name = "bn10_2_rtp"} : memref<16xi32> 
    %bn10_3_rtp = aie.buffer(%tile_0_4) {sym_name = "bn10_3_rtp"} : memref<16xi32> 
    %bn11_1_rtp = aie.buffer(%tile_0_5) {sym_name = "bn11_1_rtp"} : memref<16xi32> 
    %bn11_2_rtp = aie.buffer(%tile_1_5) {sym_name = "bn11_2_rtp"} : memref<16xi32> 
    %bn11_3_rtp = aie.buffer(%tile_1_4) {sym_name = "bn11_3_rtp"} : memref<16xi32> 
    %bn12_1_rtp = aie.buffer(%tile_1_3) {sym_name = "bn12_1_rtp"} : memref<16xi32> 
    %bn12_2_rtp = aie.buffer(%tile_1_2) {sym_name = "bn12_2_rtp"} : memref<16xi32> 
    %bn12_3_rtp = aie.buffer(%tile_2_2) {sym_name = "bn12_3_rtp"} : memref<16xi32> 
    func.func private @bn10_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32)
    func.func private @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn10_conv2dk1_ui8_i8(memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_relu_i8_ui8(memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_skip_ui8_i8_i8(memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32)
    func.func private @bn12_conv2dk1_relu_i8_ui8(memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32)
    func.func private @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn12_conv2dk1_ui8_i8(memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32)
    aie.objectfifo @OF_b10_act_layer1_layer2(%tile_0_2, {%tile_0_3}, 4 : i32) {via_DMA = true} : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @OF_b10_act_layer2_layer3(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @OF_b10_layer3_bn_11_layer1(%tile_0_4, {%tile_0_5, %tile_0_1}, [2 : i32, 2 : i32, 6 : i32]) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo @OF_b11_skip(%tile_0_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo.link [@OF_b10_layer3_bn_11_layer1] -> [@OF_b11_skip]([] [])
    aie.objectfifo @OF_b11_act_layer1_layer2(%tile_0_5, {%tile_1_5}, 4 : i32) {via_DMA = true} : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @OF_b11_act_layer2_layer3(%tile_1_5, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @OF_b11_layer3_bn_12_layer1(%tile_1_4, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo @OF_b12_act_layer1_layer2(%tile_1_3, {%tile_1_2}, 4 : i32) {via_DMA = true} : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @OF_b12_act_layer2_layer3(%tile_1_2, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<7x1x336xui8>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @weightsInBN10_layer1(Consume, 1) : !aie.objectfifosubview<memref<38400xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<38400xi8>> -> memref<38400xi8>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @act_in(Consume, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %4 = aie.objectfifo.acquire @OF_b10_act_layer1_layer2(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %c14_i32 = arith.constant 14 : i32
          %c80_i32 = arith.constant 80 : i32
          %c480_i32 = arith.constant 480 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @bn10_conv2dk1_relu_i8_ui8(%3, %1, %5, %c14_i32, %c80_i32, %c480_i32, %c9_i32) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_in(Consume, 1)
          aie.objectfifo.release @OF_b10_act_layer1_layer2(Produce, 1)
        }
      }
      aie.objectfifo.release @weightsInBN10_layer1(Consume, 1)
      aie.end
    } {link_with = "bn10_conv2dk1_fused_relu.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @weightsInBN10_layer2(Consume, 1) : !aie.objectfifosubview<memref<4320xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4320xi8>> -> memref<4320xi8>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %2 = aie.objectfifo.acquire @OF_b10_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %5 = aie.objectfifo.acquire @OF_b10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c480_i32 = arith.constant 480 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_0 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_1 = arith.constant 0 : i32
        func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%3, %3, %4, %1, %6, %c14_i32, %c1_i32, %c480_i32, %c3_i32, %c3_i32_0, %c0_i32, %c8_i32, %c0_i32_1) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_b10_act_layer2_layer3(Produce, 1)
        %c0_2 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c12 step %c1_3 {
          %12 = aie.objectfifo.acquire @OF_b10_act_layer1_layer2(Consume, 3) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %16 = aie.objectfifo.acquire @OF_b10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %c14_i32_11 = arith.constant 14 : i32
          %c1_i32_12 = arith.constant 1 : i32
          %c480_i32_13 = arith.constant 480 : i32
          %c3_i32_14 = arith.constant 3 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c1_i32_16 = arith.constant 1 : i32
          %c8_i32_17 = arith.constant 8 : i32
          %c0_i32_18 = arith.constant 0 : i32
          func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %14, %15, %1, %17, %c14_i32_11, %c1_i32_12, %c480_i32_13, %c3_i32_14, %c3_i32_15, %c1_i32_16, %c8_i32_17, %c0_i32_18) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b10_act_layer1_layer2(Consume, 1)
          aie.objectfifo.release @OF_b10_act_layer2_layer3(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @OF_b10_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %10 = aie.objectfifo.acquire @OF_b10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %c14_i32_4 = arith.constant 14 : i32
        %c1_i32_5 = arith.constant 1 : i32
        %c480_i32_6 = arith.constant 480 : i32
        %c3_i32_7 = arith.constant 3 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c8_i32_9 = arith.constant 8 : i32
        %c0_i32_10 = arith.constant 0 : i32
        func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%8, %9, %9, %1, %11, %c14_i32_4, %c1_i32_5, %c480_i32_6, %c3_i32_7, %c3_i32_8, %c2_i32, %c8_i32_9, %c0_i32_10) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_b10_act_layer1_layer2(Consume, 2)
        aie.objectfifo.release @OF_b10_act_layer2_layer3(Produce, 1)
      }
      aie.objectfifo.release @weightsInBN10_layer2(Consume, 1)
      aie.end
    } {link_with = "bn10_conv2dk3_dw.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @weightsInBN10_layer3(Consume, 1) : !aie.objectfifosubview<memref<53760xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<53760xi8>> -> memref<53760xi8>
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @OF_b10_act_layer2_layer3(Consume, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %4 = aie.objectfifo.acquire @OF_b10_layer3_bn_11_layer1(Produce, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %c14_i32 = arith.constant 14 : i32
          %c480_i32 = arith.constant 480 : i32
          %c112_i32 = arith.constant 112 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @bn10_conv2dk1_ui8_i8(%3, %1, %5, %c14_i32, %c480_i32, %c112_i32, %c9_i32) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b10_act_layer2_layer3(Consume, 1)
          aie.objectfifo.release @OF_b10_layer3_bn_11_layer1(Produce, 1)
        }
      }
      aie.objectfifo.release @weightsInBN10_layer3(Consume, 1)
      aie.end
    } {link_with = "bn10_conv2dk1_ui8.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %0 = aie.objectfifo.acquire @weightsInBN11_layer1(Consume, 1) : !aie.objectfifosubview<memref<37632xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<37632xi8>> -> memref<37632xi8>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @OF_b10_layer3_bn_11_layer1(Consume, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %4 = aie.objectfifo.acquire @OF_b11_act_layer1_layer2(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %c14_i32 = arith.constant 14 : i32
          %c112_i32 = arith.constant 112 : i32
          %c336_i32 = arith.constant 336 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @bn11_conv2dk1_relu_i8_ui8(%3, %1, %5, %c14_i32, %c112_i32, %c336_i32, %c9_i32) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b10_layer3_bn_11_layer1(Consume, 1)
          aie.objectfifo.release @OF_b11_act_layer1_layer2(Produce, 1)
        }
      }
      aie.objectfifo.release @weightsInBN11_layer1(Consume, 1)
      aie.end
    } {link_with = "bn11_conv2dk1_fused_relu.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %0 = aie.objectfifo.acquire @weightsInBN11_layer2(Consume, 1) : !aie.objectfifosubview<memref<3024xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<3024xi8>> -> memref<3024xi8>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %2 = aie.objectfifo.acquire @OF_b11_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %5 = aie.objectfifo.acquire @OF_b11_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c336_i32 = arith.constant 336 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_0 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_1 = arith.constant 0 : i32
        func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%3, %3, %4, %1, %6, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_0, %c0_i32, %c8_i32, %c0_i32_1) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_b11_act_layer2_layer3(Produce, 1)
        %c0_2 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c12 step %c1_3 {
          %12 = aie.objectfifo.acquire @OF_b11_act_layer1_layer2(Consume, 3) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %16 = aie.objectfifo.acquire @OF_b11_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %c14_i32_11 = arith.constant 14 : i32
          %c1_i32_12 = arith.constant 1 : i32
          %c336_i32_13 = arith.constant 336 : i32
          %c3_i32_14 = arith.constant 3 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c1_i32_16 = arith.constant 1 : i32
          %c8_i32_17 = arith.constant 8 : i32
          %c0_i32_18 = arith.constant 0 : i32
          func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %14, %15, %1, %17, %c14_i32_11, %c1_i32_12, %c336_i32_13, %c3_i32_14, %c3_i32_15, %c1_i32_16, %c8_i32_17, %c0_i32_18) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b11_act_layer1_layer2(Consume, 1)
          aie.objectfifo.release @OF_b11_act_layer2_layer3(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @OF_b11_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %10 = aie.objectfifo.acquire @OF_b11_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %c14_i32_4 = arith.constant 14 : i32
        %c1_i32_5 = arith.constant 1 : i32
        %c336_i32_6 = arith.constant 336 : i32
        %c3_i32_7 = arith.constant 3 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c8_i32_9 = arith.constant 8 : i32
        %c0_i32_10 = arith.constant 0 : i32
        func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%8, %9, %9, %1, %11, %c14_i32_4, %c1_i32_5, %c336_i32_6, %c3_i32_7, %c3_i32_8, %c2_i32, %c8_i32_9, %c0_i32_10) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_b11_act_layer1_layer2(Consume, 2)
        aie.objectfifo.release @OF_b11_act_layer2_layer3(Produce, 1)
      }
      aie.objectfifo.release @weightsInBN11_layer2(Consume, 1)
      aie.end
    } {link_with = "bn11_conv2dk3_dw.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %0 = aie.objectfifo.acquire @weightsInBN11_layer3(Consume, 1) : !aie.objectfifosubview<memref<37632xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<37632xi8>> -> memref<37632xi8>
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @OF_b11_act_layer2_layer3(Consume, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %4 = aie.objectfifo.acquire @OF_b11_layer3_bn_12_layer1(Produce, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %6 = aie.objectfifo.acquire @OF_b11_skip(Consume, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %c14_i32 = arith.constant 14 : i32
          %c336_i32 = arith.constant 336 : i32
          %c112_i32 = arith.constant 112 : i32
          %c12_i32 = arith.constant 12 : i32
          %c1_i32 = arith.constant 1 : i32
          func.call @bn11_conv2dk1_skip_ui8_i8_i8(%3, %1, %5, %7, %c14_i32, %c336_i32, %c112_i32, %c12_i32, %c1_i32) : (memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b11_act_layer2_layer3(Consume, 1)
          aie.objectfifo.release @OF_b11_layer3_bn_12_layer1(Produce, 1)
          aie.objectfifo.release @OF_b11_skip(Consume, 1)
        }
      }
      aie.objectfifo.release @weightsInBN11_layer3(Consume, 1)
      aie.end
    } {link_with = "bn11_conv2dk1_skip.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %0 = aie.objectfifo.acquire @weightsInBN12_layer1(Consume, 1) : !aie.objectfifosubview<memref<37632xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<37632xi8>> -> memref<37632xi8>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @OF_b11_layer3_bn_12_layer1(Consume, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %4 = aie.objectfifo.acquire @OF_b12_act_layer1_layer2(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %c14_i32 = arith.constant 14 : i32
          %c112_i32 = arith.constant 112 : i32
          %c336_i32 = arith.constant 336 : i32
          %c8_i32 = arith.constant 8 : i32
          func.call @bn12_conv2dk1_relu_i8_ui8(%3, %1, %5, %c14_i32, %c112_i32, %c336_i32, %c8_i32) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b11_layer3_bn_12_layer1(Consume, 1)
          aie.objectfifo.release @OF_b12_act_layer1_layer2(Produce, 1)
        }
      }
      aie.objectfifo.release @weightsInBN12_layer1(Consume, 1)
      aie.end
    } {link_with = "bn12_conv2dk1_fused_relu.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %0 = aie.objectfifo.acquire @weightsInBN12_layer2(Consume, 1) : !aie.objectfifosubview<memref<3024xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<3024xi8>> -> memref<3024xi8>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %2 = aie.objectfifo.acquire @OF_b12_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %5 = aie.objectfifo.acquire @OF_b12_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<7x1x336xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<7x1x336xui8>> -> memref<7x1x336xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c336_i32 = arith.constant 336 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_0 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_1 = arith.constant 0 : i32
        func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%3, %3, %4, %1, %6, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_0, %c0_i32, %c7_i32, %c0_i32_1) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @OF_b12_act_layer2_layer3(Produce, 1)
        aie.objectfifo.release @OF_b12_act_layer1_layer2(Consume, 1)
        %c0_2 = arith.constant 0 : index
        %c6 = arith.constant 6 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c6 step %c1_3 {
          %7 = aie.objectfifo.acquire @OF_b12_act_layer1_layer2(Consume, 3) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %10 = aie.objectfifo.subview.access %7[2] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %11 = aie.objectfifo.acquire @OF_b12_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<7x1x336xui8>>
          %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<7x1x336xui8>> -> memref<7x1x336xui8>
          %c14_i32_4 = arith.constant 14 : i32
          %c1_i32_5 = arith.constant 1 : i32
          %c336_i32_6 = arith.constant 336 : i32
          %c3_i32_7 = arith.constant 3 : i32
          %c3_i32_8 = arith.constant 3 : i32
          %c1_i32_9 = arith.constant 1 : i32
          %c7_i32_10 = arith.constant 7 : i32
          %c0_i32_11 = arith.constant 0 : i32
          func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%8, %9, %10, %1, %12, %c14_i32_4, %c1_i32_5, %c336_i32_6, %c3_i32_7, %c3_i32_8, %c1_i32_9, %c7_i32_10, %c0_i32_11) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b12_act_layer1_layer2(Consume, 2)
          aie.objectfifo.release @OF_b12_act_layer2_layer3(Produce, 1)
        }
        aie.objectfifo.release @OF_b12_act_layer1_layer2(Consume, 1)
      }
      aie.objectfifo.release @weightsInBN12_layer2(Consume, 1)
      aie.end
    } {link_with = "bn12_conv2dk3_dw_stride2.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %0 = aie.objectfifo.acquire @weightsInBN12_layer3(Consume, 1) : !aie.objectfifosubview<memref<26880xi8>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<26880xi8>> -> memref<26880xi8>
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c7 step %c1_1 {
          %2 = aie.objectfifo.acquire @OF_b12_act_layer2_layer3(Consume, 1) : !aie.objectfifosubview<memref<7x1x336xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<7x1x336xui8>> -> memref<7x1x336xui8>
          %4 = aie.objectfifo.acquire @act_out(Produce, 1) : !aie.objectfifosubview<memref<7x1x80xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<7x1x80xi8>> -> memref<7x1x80xi8>
          %c7_i32 = arith.constant 7 : i32
          %c336_i32 = arith.constant 336 : i32
          %c80_i32 = arith.constant 80 : i32
          %c10_i32 = arith.constant 10 : i32
          func.call @bn12_conv2dk1_ui8_i8(%3, %1, %5, %c7_i32, %c336_i32, %c80_i32, %c10_i32) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @OF_b12_act_layer2_layer3(Consume, 1)
          aie.objectfifo.release @act_out(Produce, 1)
        }
      }
      aie.objectfifo.release @weightsInBN12_layer3(Consume, 1)
      aie.end
    } {link_with = "bn12_conv2dk1_ui8.o"}
    func.func @sequence(%arg0: memref<3920xi32>, %arg1: memref<60576xi32>, %arg2: memref<980xi32>) {
      aiex.npu.rtp_write(@bn10_1_rtp, 0, 9)
      aiex.npu.rtp_write(@bn10_2_rtp, 0, 8)
      aiex.npu.rtp_write(@bn10_3_rtp, 0, 9)
      aiex.npu.rtp_write(@bn11_1_rtp, 0, 9)
      aiex.npu.rtp_write(@bn11_2_rtp, 0, 8)
      aiex.npu.rtp_write(@bn11_3_rtp, 0, 12)
      aiex.npu.rtp_write(@bn11_3_rtp, 1, 1)
      aiex.npu.rtp_write(@bn12_1_rtp, 0, 8)
      aiex.npu.rtp_write(@bn12_2_rtp, 0, 7)
      aiex.npu.rtp_write(@bn12_3_rtp, 0, 10)
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 3920][0, 0, 0, 1]) {id = 0 : i64, metadata = @act_in} : memref<3920xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 980][0, 0, 0, 1]) {id = 2 : i64, metadata = @act_out} : memref<980xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 24120][0, 0, 0, 1]) {id = 1 : i64, metadata = @wts_b10_L3L2} : memref<60576xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24120][1, 1, 1, 19572][0, 0, 0, 1]) {id = 1 : i64, metadata = @wts_b11_L3L2} : memref<60576xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 43692][1, 1, 1, 16884][0, 0, 0, 1]) {id = 1 : i64, metadata = @wts_b12_L3L2} : memref<60576xi32>
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

