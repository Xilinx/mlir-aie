module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_2 = aie.tile(3, 2)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_2_1 = aie.tile(2, 1)
    %tile_3_1 = aie.tile(3, 1)
    aie.objectfifo @wts_b10_L3L2(%tile_2_0, {%tile_2_1}, 1 : i32) : !aie.objectfifo<memref<96480xi8>>
    aie.objectfifo @weightsInBN10_layer1(%tile_2_1, {%tile_2_4}, 1 : i32) : !aie.objectfifo<memref<38400xi8>>
    aie.objectfifo @weightsInBN10_layer2(%tile_2_1, {%tile_2_5}, 1 : i32) : !aie.objectfifo<memref<4320xi8>>
    aie.objectfifo @weightsInBN10_layer3(%tile_2_1, {%tile_3_5}, 1 : i32) : !aie.objectfifo<memref<53760xi8>>
    aie.objectfifo.link [@wts_b10_L3L2] -> [@weightsInBN10_layer1, @weightsInBN10_layer2, @weightsInBN10_layer3]([] [0, 38400, 42720])
    aie.objectfifo @wts_b11_L3L2(%tile_3_0, {%tile_3_1}, 1 : i32) : !aie.objectfifo<memref<78288xi8>>
    aie.objectfifo @weightsInBN11_layer1(%tile_3_1, {%tile_3_4}, 1 : i32) : !aie.objectfifo<memref<37632xi8>>
    aie.objectfifo @weightsInBN11_layer2(%tile_3_1, {%tile_3_3}, 1 : i32) : !aie.objectfifo<memref<3024xi8>>
    aie.objectfifo @weightsInBN11_layer3(%tile_3_1, {%tile_3_2}, 1 : i32) : !aie.objectfifo<memref<37632xi8>>
    aie.objectfifo.link [@wts_b11_L3L2] -> [@weightsInBN11_layer1, @weightsInBN11_layer2, @weightsInBN11_layer3]([] [0, 37632, 40656])
    %bn10_1_rtp = aie.buffer(%tile_2_4) {sym_name = "bn10_1_rtp"} : memref<16xi32> 
    %bn10_2_rtp = aie.buffer(%tile_2_5) {sym_name = "bn10_2_rtp"} : memref<16xi32> 
    %bn10_3_rtp = aie.buffer(%tile_3_5) {sym_name = "bn10_3_rtp"} : memref<16xi32> 
    %bn11_1_rtp = aie.buffer(%tile_3_4) {sym_name = "bn11_1_rtp"} : memref<16xi32> 
    %bn11_2_rtp = aie.buffer(%tile_3_3) {sym_name = "bn11_2_rtp"} : memref<16xi32> 
    %bn11_3_rtp = aie.buffer(%tile_3_2) {sym_name = "bn11_3_rtp"} : memref<16xi32> 
    %rtp03 = aie.buffer(%tile_0_3) {sym_name = "rtp03"} : memref<16xi32> 
    %rtp04 = aie.buffer(%tile_0_4) {sym_name = "rtp04"} : memref<16xi32> 
    %rtp05 = aie.buffer(%tile_0_5) {sym_name = "rtp05"} : memref<16xi32> 
    %rtp15 = aie.buffer(%tile_1_5) {sym_name = "rtp15"} : memref<16xi32> 
    %rtp14 = aie.buffer(%tile_1_4) {sym_name = "rtp14"} : memref<16xi32> 
    %rtp12 = aie.buffer(%tile_1_2) {sym_name = "rtp12"} : memref<16xi32> 
    %rtp13 = aie.buffer(%tile_1_3) {sym_name = "rtp13"} : memref<16xi32> 
    %rtp22 = aie.buffer(%tile_2_2) {sym_name = "rtp22"} : memref<16xi32> 
    %rtp23 = aie.buffer(%tile_2_3) {sym_name = "rtp23"} : memref<16xi32> 
    aie.objectfifo @act_in(%tile_0_0, {%tile_0_3}, [3 : i32, 3 : i32]) : !aie.objectfifo<memref<112x1x16xui8>>
    aie.objectfifo @wts_OF_01_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<34256xi8>>
    aie.objectfifo @bn0_1_wts_OF_L2L1(%tile_0_1, {%tile_0_3}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<3536xi8>>
    aie.objectfifo @bn2_wts_OF_L2L1(%tile_0_1, {%tile_0_4}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<4104xi8>>
    aie.objectfifo @bn3_wts_OF_L2L1(%tile_0_1, {%tile_0_5}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<5256xi8>>
    aie.objectfifo @bn4_wts_OF_L2L1(%tile_0_1, {%tile_1_5}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<10680xi8>>
    aie.objectfifo @bn5_wts_OF_L2L1(%tile_0_1, {%tile_1_4}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<10680xi8>>
    aie.objectfifo.link [@wts_OF_01_L3L2] -> [@bn0_1_wts_OF_L2L1, @bn2_wts_OF_L2L1, @bn3_wts_OF_L2L1, @bn4_wts_OF_L2L1, @bn5_wts_OF_L2L1]([] [0, 3536, 7640, 12896, 23576])
    aie.objectfifo @wts_OF_11_L3L2(%tile_1_0, {%tile_1_1}, 1 : i32) : !aie.objectfifo<memref<126952xi8>>
    aie.objectfifo @bn6_wts_OF_L2L1(%tile_1_1, {%tile_1_2}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<30960xi8>>
    aie.objectfifo @bn7_wts_OF_L2L1(%tile_1_1, {%tile_1_3}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<33800xi8>>
    aie.objectfifo @bn8_wts_OF_L2L1(%tile_1_1, {%tile_2_2}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<31096xi8>>
    aie.objectfifo @bn9_wts_OF_L2L1(%tile_1_1, {%tile_2_3}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<31096xi8>>
    aie.objectfifo.link [@wts_OF_11_L3L2] -> [@bn6_wts_OF_L2L1, @bn7_wts_OF_L2L1, @bn8_wts_OF_L2L1, @bn9_wts_OF_L2L1]([] [0, 30960, 64760, 95856])
    func.func private @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn0_conv2dk1_skip_ui8_ui8_i8(memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32)
    func.func private @bn1_conv2dk1_relu_i8_ui8(memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32)
    func.func private @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn1_conv2dk1_ui8_i8(memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn01_bn2(%tile_0_3, {%tile_0_4}, [3 : i32, 2 : i32]) : !aie.objectfifo<memref<56x1x24xi8>>
    aie.objectfifo @bn01_act_bn0_2_3(%tile_0_3, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<112x1x16xui8>>
    aie.objectfifo @bn01_act_bn0_bn1(%tile_0_3, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<112x1x16xi8>>
    aie.objectfifo @bn01_act_bn1_1_2(%tile_0_3, {%tile_0_3}, 3 : i32) : !aie.objectfifo<memref<112x1x64xui8>>
    aie.objectfifo @bn01_act_bn1_2_3(%tile_0_3, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<56x1x64xui8>>
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn0_1_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<3536xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<3536xi8>> -> memref<3536xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<3536xi8> to memref<144xi8>
        %c144 = arith.constant 144 : index
        %view_2 = memref.view %1[%c144][] : memref<3536xi8> to memref<256xi8>
        %c400 = arith.constant 400 : index
        %view_3 = memref.view %1[%c400][] : memref<3536xi8> to memref<1024xi8>
        %c1424 = arith.constant 1424 : index
        %view_4 = memref.view %1[%c1424][] : memref<3536xi8> to memref<576xi8>
        %c2000 = arith.constant 2000 : index
        %view_5 = memref.view %1[%c2000][] : memref<3536xi8> to memref<1536xi8>
        %c0_6 = arith.constant 0 : index
        %2 = memref.load %rtp03[%c0_6] : memref<16xi32>
        %c1_7 = arith.constant 1 : index
        %3 = memref.load %rtp03[%c1_7] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp03[%c2] : memref<16xi32>
        %c3 = arith.constant 3 : index
        %5 = memref.load %rtp03[%c3] : memref<16xi32>
        %c4 = arith.constant 4 : index
        %6 = memref.load %rtp03[%c4] : memref<16xi32>
        %c5 = arith.constant 5 : index
        %7 = memref.load %rtp03[%c5] : memref<16xi32>
        %8 = aie.objectfifo.acquire @act_in(Consume, 2) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %11 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %c112_i32 = arith.constant 112 : i32
        %c1_i32 = arith.constant 1 : i32
        %c16_i32 = arith.constant 16 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_9 = arith.constant 0 : i32
        func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%9, %9, %10, %view, %12, %c112_i32, %c1_i32, %c16_i32, %c3_i32, %c3_i32_8, %c0_i32, %2, %c0_i32_9) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_2_3(Produce, 1)
        %13 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %15 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %c112_i32_10 = arith.constant 112 : i32
        %c16_i32_11 = arith.constant 16 : i32
        %c16_i32_12 = arith.constant 16 : i32
        func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%14, %view_2, %16, %9, %c112_i32_10, %c16_i32_11, %c16_i32_12, %3, %4) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_2_3(Consume, 1)
        aie.objectfifo.release @bn01_act_bn0_bn1(Produce, 1)
        %17 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %19 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Produce, 1) : !aie.objectfifosubview<memref<112x1x64xui8>>
        %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %c112_i32_13 = arith.constant 112 : i32
        %c16_i32_14 = arith.constant 16 : i32
        %c64_i32 = arith.constant 64 : i32
        func.call @bn1_conv2dk1_relu_i8_ui8(%18, %view_3, %20, %c112_i32_13, %c16_i32_14, %c64_i32, %5) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_bn1(Consume, 1)
        aie.objectfifo.release @bn01_act_bn1_1_2(Produce, 1)
        %21 = aie.objectfifo.acquire @act_in(Consume, 3) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %23 = aie.objectfifo.subview.access %21[1] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %24 = aie.objectfifo.subview.access %21[2] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %25 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %c112_i32_15 = arith.constant 112 : i32
        %c1_i32_16 = arith.constant 1 : i32
        %c16_i32_17 = arith.constant 16 : i32
        %c3_i32_18 = arith.constant 3 : i32
        %c3_i32_19 = arith.constant 3 : i32
        %c1_i32_20 = arith.constant 1 : i32
        %c0_i32_21 = arith.constant 0 : i32
        func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%22, %23, %24, %view, %26, %c112_i32_15, %c1_i32_16, %c16_i32_17, %c3_i32_18, %c3_i32_19, %c1_i32_20, %2, %c0_i32_21) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_2_3(Produce, 1)
        %27 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %29 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %c112_i32_22 = arith.constant 112 : i32
        %c16_i32_23 = arith.constant 16 : i32
        %c16_i32_24 = arith.constant 16 : i32
        func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%28, %view_2, %30, %23, %c112_i32_22, %c16_i32_23, %c16_i32_24, %3, %4) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_in(Consume, 1)
        aie.objectfifo.release @bn01_act_bn0_2_3(Consume, 1)
        aie.objectfifo.release @bn01_act_bn0_bn1(Produce, 1)
        %31 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %33 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Produce, 1) : !aie.objectfifosubview<memref<112x1x64xui8>>
        %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %c112_i32_25 = arith.constant 112 : i32
        %c16_i32_26 = arith.constant 16 : i32
        %c64_i32_27 = arith.constant 64 : i32
        func.call @bn1_conv2dk1_relu_i8_ui8(%32, %view_3, %34, %c112_i32_25, %c16_i32_26, %c64_i32_27, %5) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_bn1(Consume, 1)
        aie.objectfifo.release @bn01_act_bn1_1_2(Produce, 1)
        %35 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Consume, 2) : !aie.objectfifosubview<memref<112x1x64xui8>>
        %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %37 = aie.objectfifo.subview.access %35[1] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %38 = aie.objectfifo.acquire @bn01_act_bn1_2_3(Produce, 1) : !aie.objectfifosubview<memref<56x1x64xui8>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<56x1x64xui8>> -> memref<56x1x64xui8>
        %c112_i32_28 = arith.constant 112 : i32
        %c1_i32_29 = arith.constant 1 : i32
        %c64_i32_30 = arith.constant 64 : i32
        %c3_i32_31 = arith.constant 3 : i32
        %c3_i32_32 = arith.constant 3 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c0_i32_34 = arith.constant 0 : i32
        func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%36, %36, %37, %view_4, %39, %c112_i32_28, %c1_i32_29, %c64_i32_30, %c3_i32_31, %c3_i32_32, %c0_i32_33, %6, %c0_i32_34) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn1_1_2(Consume, 1)
        aie.objectfifo.release @bn01_act_bn1_2_3(Produce, 1)
        %40 = aie.objectfifo.acquire @bn01_act_bn1_2_3(Consume, 1) : !aie.objectfifosubview<memref<56x1x64xui8>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<56x1x64xui8>> -> memref<56x1x64xui8>
        %42 = aie.objectfifo.acquire @act_bn01_bn2(Produce, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %c56_i32 = arith.constant 56 : i32
        %c64_i32_35 = arith.constant 64 : i32
        %c24_i32 = arith.constant 24 : i32
        func.call @bn1_conv2dk1_ui8_i8(%41, %view_5, %43, %c56_i32, %c64_i32_35, %c24_i32, %7) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn1_2_3(Consume, 1)
        aie.objectfifo.release @act_bn01_bn2(Produce, 1)
        %c0_36 = arith.constant 0 : index
        %c54 = arith.constant 54 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_36 to %c54 step %c1_37 {
          %c0_73 = arith.constant 0 : index
          %c2_74 = arith.constant 2 : index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_73 to %c2_74 step %c1_75 {
            %91 = aie.objectfifo.acquire @act_in(Consume, 3) : !aie.objectfifosubview<memref<112x1x16xui8>>
            %92 = aie.objectfifo.subview.access %91[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
            %93 = aie.objectfifo.subview.access %91[1] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
            %94 = aie.objectfifo.subview.access %91[2] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
            %95 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
            %96 = aie.objectfifo.subview.access %95[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
            %c112_i32_86 = arith.constant 112 : i32
            %c1_i32_87 = arith.constant 1 : i32
            %c16_i32_88 = arith.constant 16 : i32
            %c3_i32_89 = arith.constant 3 : i32
            %c3_i32_90 = arith.constant 3 : i32
            %c1_i32_91 = arith.constant 1 : i32
            %c0_i32_92 = arith.constant 0 : i32
            func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%92, %93, %94, %view, %96, %c112_i32_86, %c1_i32_87, %c16_i32_88, %c3_i32_89, %c3_i32_90, %c1_i32_91, %2, %c0_i32_92) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn01_act_bn0_2_3(Produce, 1)
            %97 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
            %98 = aie.objectfifo.subview.access %97[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
            %99 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
            %100 = aie.objectfifo.subview.access %99[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
            %c112_i32_93 = arith.constant 112 : i32
            %c16_i32_94 = arith.constant 16 : i32
            %c16_i32_95 = arith.constant 16 : i32
            func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%98, %view_2, %100, %93, %c112_i32_93, %c16_i32_94, %c16_i32_95, %3, %4) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @act_in(Consume, 1)
            aie.objectfifo.release @bn01_act_bn0_2_3(Consume, 1)
            aie.objectfifo.release @bn01_act_bn0_bn1(Produce, 1)
            %101 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
            %102 = aie.objectfifo.subview.access %101[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
            %103 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Produce, 1) : !aie.objectfifosubview<memref<112x1x64xui8>>
            %104 = aie.objectfifo.subview.access %103[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
            %c112_i32_96 = arith.constant 112 : i32
            %c16_i32_97 = arith.constant 16 : i32
            %c64_i32_98 = arith.constant 64 : i32
            func.call @bn1_conv2dk1_relu_i8_ui8(%102, %view_3, %104, %c112_i32_96, %c16_i32_97, %c64_i32_98, %5) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn01_act_bn0_bn1(Consume, 1)
            aie.objectfifo.release @bn01_act_bn1_1_2(Produce, 1)
          }
          %81 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Consume, 3) : !aie.objectfifosubview<memref<112x1x64xui8>>
          %82 = aie.objectfifo.subview.access %81[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
          %83 = aie.objectfifo.subview.access %81[1] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
          %84 = aie.objectfifo.subview.access %81[2] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
          %85 = aie.objectfifo.acquire @bn01_act_bn1_2_3(Produce, 1) : !aie.objectfifosubview<memref<56x1x64xui8>>
          %86 = aie.objectfifo.subview.access %85[0] : !aie.objectfifosubview<memref<56x1x64xui8>> -> memref<56x1x64xui8>
          %c112_i32_76 = arith.constant 112 : i32
          %c1_i32_77 = arith.constant 1 : i32
          %c64_i32_78 = arith.constant 64 : i32
          %c3_i32_79 = arith.constant 3 : i32
          %c3_i32_80 = arith.constant 3 : i32
          %c1_i32_81 = arith.constant 1 : i32
          %c0_i32_82 = arith.constant 0 : i32
          func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%82, %83, %84, %view_4, %86, %c112_i32_76, %c1_i32_77, %c64_i32_78, %c3_i32_79, %c3_i32_80, %c1_i32_81, %6, %c0_i32_82) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn01_act_bn1_1_2(Consume, 2)
          aie.objectfifo.release @bn01_act_bn1_2_3(Produce, 1)
          %87 = aie.objectfifo.acquire @bn01_act_bn1_2_3(Consume, 1) : !aie.objectfifosubview<memref<56x1x64xui8>>
          %88 = aie.objectfifo.subview.access %87[0] : !aie.objectfifosubview<memref<56x1x64xui8>> -> memref<56x1x64xui8>
          %89 = aie.objectfifo.acquire @act_bn01_bn2(Produce, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
          %90 = aie.objectfifo.subview.access %89[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
          %c56_i32_83 = arith.constant 56 : i32
          %c64_i32_84 = arith.constant 64 : i32
          %c24_i32_85 = arith.constant 24 : i32
          func.call @bn1_conv2dk1_ui8_i8(%88, %view_5, %90, %c56_i32_83, %c64_i32_84, %c24_i32_85, %7) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn01_act_bn1_2_3(Consume, 1)
          aie.objectfifo.release @act_bn01_bn2(Produce, 1)
        }
        %44 = aie.objectfifo.acquire @act_in(Consume, 3) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %46 = aie.objectfifo.subview.access %44[1] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %47 = aie.objectfifo.subview.access %44[2] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %48 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %c112_i32_38 = arith.constant 112 : i32
        %c1_i32_39 = arith.constant 1 : i32
        %c16_i32_40 = arith.constant 16 : i32
        %c3_i32_41 = arith.constant 3 : i32
        %c3_i32_42 = arith.constant 3 : i32
        %c1_i32_43 = arith.constant 1 : i32
        %c0_i32_44 = arith.constant 0 : i32
        func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%45, %46, %47, %view, %49, %c112_i32_38, %c1_i32_39, %c16_i32_40, %c3_i32_41, %c3_i32_42, %c1_i32_43, %2, %c0_i32_44) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_2_3(Produce, 1)
        %50 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %52 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %c112_i32_45 = arith.constant 112 : i32
        %c16_i32_46 = arith.constant 16 : i32
        %c16_i32_47 = arith.constant 16 : i32
        func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%51, %view_2, %53, %46, %c112_i32_45, %c16_i32_46, %c16_i32_47, %3, %4) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_in(Consume, 1)
        aie.objectfifo.release @bn01_act_bn0_2_3(Consume, 1)
        aie.objectfifo.release @bn01_act_bn0_bn1(Produce, 1)
        %54 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %56 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Produce, 1) : !aie.objectfifosubview<memref<112x1x64xui8>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %c112_i32_48 = arith.constant 112 : i32
        %c16_i32_49 = arith.constant 16 : i32
        %c64_i32_50 = arith.constant 64 : i32
        func.call @bn1_conv2dk1_relu_i8_ui8(%55, %view_3, %57, %c112_i32_48, %c16_i32_49, %c64_i32_50, %5) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_bn1(Consume, 1)
        aie.objectfifo.release @bn01_act_bn1_1_2(Produce, 1)
        %58 = aie.objectfifo.acquire @act_in(Consume, 2) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %60 = aie.objectfifo.subview.access %58[1] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %61 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %62 = aie.objectfifo.subview.access %61[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %c112_i32_51 = arith.constant 112 : i32
        %c1_i32_52 = arith.constant 1 : i32
        %c16_i32_53 = arith.constant 16 : i32
        %c3_i32_54 = arith.constant 3 : i32
        %c3_i32_55 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_56 = arith.constant 0 : i32
        func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%59, %60, %60, %view, %62, %c112_i32_51, %c1_i32_52, %c16_i32_53, %c3_i32_54, %c3_i32_55, %c2_i32, %2, %c0_i32_56) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_2_3(Produce, 1)
        %63 = aie.objectfifo.acquire @bn01_act_bn0_2_3(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xui8>>
        %64 = aie.objectfifo.subview.access %63[0] : !aie.objectfifosubview<memref<112x1x16xui8>> -> memref<112x1x16xui8>
        %65 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Produce, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %66 = aie.objectfifo.subview.access %65[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %c112_i32_57 = arith.constant 112 : i32
        %c16_i32_58 = arith.constant 16 : i32
        %c16_i32_59 = arith.constant 16 : i32
        func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%64, %view_2, %66, %60, %c112_i32_57, %c16_i32_58, %c16_i32_59, %3, %4) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_in(Consume, 2)
        aie.objectfifo.release @bn01_act_bn0_2_3(Consume, 1)
        aie.objectfifo.release @bn01_act_bn0_bn1(Produce, 1)
        %67 = aie.objectfifo.acquire @bn01_act_bn0_bn1(Consume, 1) : !aie.objectfifosubview<memref<112x1x16xi8>>
        %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<112x1x16xi8>> -> memref<112x1x16xi8>
        %69 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Produce, 1) : !aie.objectfifosubview<memref<112x1x64xui8>>
        %70 = aie.objectfifo.subview.access %69[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %c112_i32_60 = arith.constant 112 : i32
        %c16_i32_61 = arith.constant 16 : i32
        %c64_i32_62 = arith.constant 64 : i32
        func.call @bn1_conv2dk1_relu_i8_ui8(%68, %view_3, %70, %c112_i32_60, %c16_i32_61, %c64_i32_62, %5) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn0_bn1(Consume, 1)
        aie.objectfifo.release @bn01_act_bn1_1_2(Produce, 1)
        %71 = aie.objectfifo.acquire @bn01_act_bn1_1_2(Consume, 3) : !aie.objectfifosubview<memref<112x1x64xui8>>
        %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %73 = aie.objectfifo.subview.access %71[1] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %74 = aie.objectfifo.subview.access %71[2] : !aie.objectfifosubview<memref<112x1x64xui8>> -> memref<112x1x64xui8>
        %75 = aie.objectfifo.acquire @bn01_act_bn1_2_3(Produce, 1) : !aie.objectfifosubview<memref<56x1x64xui8>>
        %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<56x1x64xui8>> -> memref<56x1x64xui8>
        %c112_i32_63 = arith.constant 112 : i32
        %c1_i32_64 = arith.constant 1 : i32
        %c64_i32_65 = arith.constant 64 : i32
        %c3_i32_66 = arith.constant 3 : i32
        %c3_i32_67 = arith.constant 3 : i32
        %c1_i32_68 = arith.constant 1 : i32
        %c0_i32_69 = arith.constant 0 : i32
        func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%72, %73, %74, %view_4, %76, %c112_i32_63, %c1_i32_64, %c64_i32_65, %c3_i32_66, %c3_i32_67, %c1_i32_68, %6, %c0_i32_69) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn1_1_2(Consume, 3)
        aie.objectfifo.release @bn01_act_bn1_2_3(Produce, 1)
        %77 = aie.objectfifo.acquire @bn01_act_bn1_2_3(Consume, 1) : !aie.objectfifosubview<memref<56x1x64xui8>>
        %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<56x1x64xui8>> -> memref<56x1x64xui8>
        %79 = aie.objectfifo.acquire @act_bn01_bn2(Produce, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %c56_i32_70 = arith.constant 56 : i32
        %c64_i32_71 = arith.constant 64 : i32
        %c24_i32_72 = arith.constant 24 : i32
        func.call @bn1_conv2dk1_ui8_i8(%78, %view_5, %80, %c56_i32_70, %c64_i32_71, %c24_i32_72, %7) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn01_act_bn1_2_3(Consume, 1)
        aie.objectfifo.release @act_bn01_bn2(Produce, 1)
        aie.objectfifo.release @bn0_1_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "fused_bn0_bn1.a"}
    func.func private @bn2_conv2dk1_relu_i8_ui8(memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32)
    func.func private @bn2_conv2dk3_dw_stride2_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn2_conv2dk1_skip_ui8_i8_i8(memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32)
    func.func private @bn2_conv2dk1_ui8_i8(memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn2_bn3(%tile_0_4, {%tile_0_5}, [3 : i32, 2 : i32]) : !aie.objectfifo<memref<56x1x24xi8>>
    aie.objectfifo @bn2_act_1_2(%tile_0_4, {%tile_0_4}, 3 : i32) : !aie.objectfifo<memref<56x1x72xui8>>
    aie.objectfifo @bn2_act_2_3(%tile_0_4, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<56x1x72xui8>>
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn2_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<4104xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4104xi8>> -> memref<4104xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<4104xi8> to memref<1728xi8>
        %c1728 = arith.constant 1728 : index
        %view_2 = memref.view %1[%c1728][] : memref<4104xi8> to memref<648xi8>
        %c2376 = arith.constant 2376 : index
        %view_3 = memref.view %1[%c2376][] : memref<4104xi8> to memref<1728xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp04[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp04[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp04[%c2] : memref<16xi32>
        %c3 = arith.constant 3 : index
        %5 = memref.load %rtp04[%c3] : memref<16xi32>
        %6 = aie.objectfifo.acquire @act_bn01_bn2(Consume, 2) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %9 = aie.objectfifo.acquire @bn2_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %11 = aie.objectfifo.subview.access %9[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %c56_i32 = arith.constant 56 : i32
        %c24_i32 = arith.constant 24 : i32
        %c72_i32 = arith.constant 72 : i32
        func.call @bn2_conv2dk1_relu_i8_ui8(%7, %view, %10, %c56_i32, %c24_i32, %c72_i32, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
        %c56_i32_6 = arith.constant 56 : i32
        %c24_i32_7 = arith.constant 24 : i32
        %c72_i32_8 = arith.constant 72 : i32
        func.call @bn2_conv2dk1_relu_i8_ui8(%8, %view, %11, %c56_i32_6, %c24_i32_7, %c72_i32_8, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn2_act_1_2(Produce, 2)
        %12 = aie.objectfifo.acquire @bn2_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %15 = aie.objectfifo.acquire @bn2_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %c56_i32_9 = arith.constant 56 : i32
        %c1_i32 = arith.constant 1 : i32
        %c72_i32_10 = arith.constant 72 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %13, %14, %view_2, %16, %c56_i32_9, %c1_i32, %c72_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn2_act_2_3(Produce, 1)
        %17 = aie.objectfifo.acquire @bn2_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %19 = aie.objectfifo.acquire @act_bn2_bn3(Produce, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %c56_i32_13 = arith.constant 56 : i32
        %c72_i32_14 = arith.constant 72 : i32
        %c24_i32_15 = arith.constant 24 : i32
        func.call @bn2_conv2dk1_skip_ui8_i8_i8(%18, %view_3, %20, %7, %c56_i32_13, %c72_i32_14, %c24_i32_15, %4, %5) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn01_bn2(Consume, 1)
        aie.objectfifo.release @bn2_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn2_bn3(Produce, 1)
        %c0_16 = arith.constant 0 : index
        %c54 = arith.constant 54 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c54 step %c1_17 {
          %32 = aie.objectfifo.acquire @act_bn01_bn2(Consume, 2) : !aie.objectfifosubview<memref<56x1x24xi8>>
          %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
          %34 = aie.objectfifo.subview.access %32[1] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
          %35 = aie.objectfifo.acquire @bn2_act_1_2(Produce, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
          %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %c56_i32_27 = arith.constant 56 : i32
          %c24_i32_28 = arith.constant 24 : i32
          %c72_i32_29 = arith.constant 72 : i32
          func.call @bn2_conv2dk1_relu_i8_ui8(%34, %view, %36, %c56_i32_27, %c24_i32_28, %c72_i32_29, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn2_act_1_2(Produce, 1)
          %37 = aie.objectfifo.acquire @bn2_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<56x1x72xui8>>
          %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %39 = aie.objectfifo.subview.access %37[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %40 = aie.objectfifo.subview.access %37[2] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %41 = aie.objectfifo.acquire @bn2_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
          %42 = aie.objectfifo.subview.access %41[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %c56_i32_30 = arith.constant 56 : i32
          %c1_i32_31 = arith.constant 1 : i32
          %c72_i32_32 = arith.constant 72 : i32
          %c3_i32_33 = arith.constant 3 : i32
          %c3_i32_34 = arith.constant 3 : i32
          %c1_i32_35 = arith.constant 1 : i32
          %c0_i32_36 = arith.constant 0 : i32
          func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%38, %39, %40, %view_2, %42, %c56_i32_30, %c1_i32_31, %c72_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %3, %c0_i32_36) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn2_act_1_2(Consume, 1)
          aie.objectfifo.release @bn2_act_2_3(Produce, 1)
          %43 = aie.objectfifo.acquire @bn2_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
          %44 = aie.objectfifo.subview.access %43[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %45 = aie.objectfifo.acquire @act_bn2_bn3(Produce, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
          %46 = aie.objectfifo.subview.access %45[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
          %c56_i32_37 = arith.constant 56 : i32
          %c72_i32_38 = arith.constant 72 : i32
          %c24_i32_39 = arith.constant 24 : i32
          func.call @bn2_conv2dk1_skip_ui8_i8_i8(%44, %view_3, %46, %33, %c56_i32_37, %c72_i32_38, %c24_i32_39, %4, %5) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_bn01_bn2(Consume, 1)
          aie.objectfifo.release @bn2_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn2_bn3(Produce, 1)
        }
        %21 = aie.objectfifo.acquire @bn2_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %23 = aie.objectfifo.subview.access %21[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %24 = aie.objectfifo.acquire @bn2_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %c56_i32_18 = arith.constant 56 : i32
        %c1_i32_19 = arith.constant 1 : i32
        %c72_i32_20 = arith.constant 72 : i32
        %c3_i32_21 = arith.constant 3 : i32
        %c3_i32_22 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_23 = arith.constant 0 : i32
        func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%22, %23, %23, %view_2, %25, %c56_i32_18, %c1_i32_19, %c72_i32_20, %c3_i32_21, %c3_i32_22, %c2_i32, %3, %c0_i32_23) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn2_act_1_2(Consume, 2)
        aie.objectfifo.release @bn2_act_2_3(Produce, 1)
        %26 = aie.objectfifo.acquire @bn2_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %28 = aie.objectfifo.acquire @act_bn2_bn3(Produce, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %30 = aie.objectfifo.acquire @act_bn01_bn2(Consume, 1) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %c56_i32_24 = arith.constant 56 : i32
        %c72_i32_25 = arith.constant 72 : i32
        %c24_i32_26 = arith.constant 24 : i32
        func.call @bn2_conv2dk1_skip_ui8_i8_i8(%27, %view_3, %29, %31, %c56_i32_24, %c72_i32_25, %c24_i32_26, %4, %5) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn01_bn2(Consume, 1)
        aie.objectfifo.release @bn2_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn2_bn3(Produce, 1)
        aie.objectfifo.release @bn2_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn3_conv2dk1_relu_i8_ui8(memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32)
    func.func private @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn3_conv2dk3_dw_stride1_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn3_conv2dk1_skip_ui8_i8_i8(memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32)
    func.func private @bn3_conv2dk1_ui8_i8(memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn3_bn4(%tile_0_5, {%tile_1_5}, [3 : i32, 2 : i32]) : !aie.objectfifo<memref<28x1x40xi8>>
    aie.objectfifo @bn3_act_1_2(%tile_0_5, {%tile_0_5}, 3 : i32) : !aie.objectfifo<memref<56x1x72xui8>>
    aie.objectfifo @bn3_act_2_3(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<28x1x72xui8>>
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn3_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<5256xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<5256xi8>> -> memref<5256xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<5256xi8> to memref<1728xi8>
        %c1728 = arith.constant 1728 : index
        %view_2 = memref.view %1[%c1728][] : memref<5256xi8> to memref<648xi8>
        %c2376 = arith.constant 2376 : index
        %view_3 = memref.view %1[%c2376][] : memref<5256xi8> to memref<2880xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp05[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp05[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp05[%c2] : memref<16xi32>
        %5 = aie.objectfifo.acquire @act_bn2_bn3(Consume, 2) : !aie.objectfifosubview<memref<56x1x24xi8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
        %8 = aie.objectfifo.acquire @bn3_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %c56_i32 = arith.constant 56 : i32
        %c24_i32 = arith.constant 24 : i32
        %c72_i32 = arith.constant 72 : i32
        func.call @bn3_conv2dk1_relu_i8_ui8(%6, %view, %9, %c56_i32, %c24_i32, %c72_i32, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
        %c56_i32_6 = arith.constant 56 : i32
        %c24_i32_7 = arith.constant 24 : i32
        %c72_i32_8 = arith.constant 72 : i32
        func.call @bn3_conv2dk1_relu_i8_ui8(%7, %view, %10, %c56_i32_6, %c24_i32_7, %c72_i32_8, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn3_act_1_2(Produce, 2)
        aie.objectfifo.release @act_bn2_bn3(Consume, 2)
        %11 = aie.objectfifo.acquire @bn3_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<56x1x72xui8>>
        %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
        %14 = aie.objectfifo.acquire @bn3_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x72xui8>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<28x1x72xui8>> -> memref<28x1x72xui8>
        %c56_i32_9 = arith.constant 56 : i32
        %c1_i32 = arith.constant 1 : i32
        %c72_i32_10 = arith.constant 72 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%12, %12, %13, %view_2, %15, %c56_i32_9, %c1_i32, %c72_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn3_act_1_2(Consume, 1)
        aie.objectfifo.release @bn3_act_2_3(Produce, 1)
        %16 = aie.objectfifo.acquire @bn3_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x72xui8>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<28x1x72xui8>> -> memref<28x1x72xui8>
        %18 = aie.objectfifo.acquire @act_bn3_bn4(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %c28_i32 = arith.constant 28 : i32
        %c72_i32_13 = arith.constant 72 : i32
        %c40_i32 = arith.constant 40 : i32
        func.call @bn3_conv2dk1_ui8_i8(%17, %view_3, %19, %c28_i32, %c72_i32_13, %c40_i32, %4) : (memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn3_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn3_bn4(Produce, 1)
        %c0_14 = arith.constant 0 : index
        %c27 = arith.constant 27 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c27 step %c1_15 {
          %20 = aie.objectfifo.acquire @act_bn2_bn3(Consume, 2) : !aie.objectfifosubview<memref<56x1x24xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
          %22 = aie.objectfifo.subview.access %20[1] : !aie.objectfifosubview<memref<56x1x24xi8>> -> memref<56x1x24xi8>
          %23 = aie.objectfifo.acquire @bn3_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<56x1x72xui8>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %25 = aie.objectfifo.subview.access %23[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %c56_i32_16 = arith.constant 56 : i32
          %c24_i32_17 = arith.constant 24 : i32
          %c72_i32_18 = arith.constant 72 : i32
          func.call @bn3_conv2dk1_relu_i8_ui8(%21, %view, %24, %c56_i32_16, %c24_i32_17, %c72_i32_18, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
          %c56_i32_19 = arith.constant 56 : i32
          %c24_i32_20 = arith.constant 24 : i32
          %c72_i32_21 = arith.constant 72 : i32
          func.call @bn3_conv2dk1_relu_i8_ui8(%22, %view, %25, %c56_i32_19, %c24_i32_20, %c72_i32_21, %2) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn3_act_1_2(Produce, 2)
          aie.objectfifo.release @act_bn2_bn3(Consume, 2)
          %26 = aie.objectfifo.acquire @bn3_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<56x1x72xui8>>
          %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %28 = aie.objectfifo.subview.access %26[1] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %29 = aie.objectfifo.subview.access %26[2] : !aie.objectfifosubview<memref<56x1x72xui8>> -> memref<56x1x72xui8>
          %30 = aie.objectfifo.acquire @bn3_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x72xui8>>
          %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<28x1x72xui8>> -> memref<28x1x72xui8>
          %c56_i32_22 = arith.constant 56 : i32
          %c1_i32_23 = arith.constant 1 : i32
          %c72_i32_24 = arith.constant 72 : i32
          %c3_i32_25 = arith.constant 3 : i32
          %c3_i32_26 = arith.constant 3 : i32
          %c1_i32_27 = arith.constant 1 : i32
          %c0_i32_28 = arith.constant 0 : i32
          func.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%27, %28, %29, %view_2, %31, %c56_i32_22, %c1_i32_23, %c72_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %3, %c0_i32_28) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn3_act_1_2(Consume, 2)
          aie.objectfifo.release @bn3_act_2_3(Produce, 1)
          %32 = aie.objectfifo.acquire @bn3_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x72xui8>>
          %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<28x1x72xui8>> -> memref<28x1x72xui8>
          %34 = aie.objectfifo.acquire @act_bn3_bn4(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
          %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %c28_i32_29 = arith.constant 28 : i32
          %c72_i32_30 = arith.constant 72 : i32
          %c40_i32_31 = arith.constant 40 : i32
          func.call @bn3_conv2dk1_ui8_i8(%33, %view_3, %35, %c28_i32_29, %c72_i32_30, %c40_i32_31, %4) : (memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn3_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn3_bn4(Produce, 1)
        }
        aie.objectfifo.release @bn3_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a"}
    func.func private @bn4_conv2dk1_relu_i8_ui8(memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32)
    func.func private @bn4_conv2dk3_dw_stride2_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn4_conv2dk1_skip_ui8_i8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32)
    func.func private @bn4_conv2dk1_ui8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn4_bn5(%tile_1_5, {%tile_1_4}, [3 : i32, 2 : i32]) : !aie.objectfifo<memref<28x1x40xi8>>
    aie.objectfifo @bn4_act_1_2(%tile_1_5, {%tile_1_5}, 3 : i32) : !aie.objectfifo<memref<28x1x120xui8>>
    aie.objectfifo @bn4_act_2_3(%tile_1_5, {%tile_1_5}, 1 : i32) : !aie.objectfifo<memref<28x1x120xui8>>
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn4_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<10680xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10680xi8>> -> memref<10680xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<10680xi8> to memref<4800xi8>
        %c4800 = arith.constant 4800 : index
        %view_2 = memref.view %1[%c4800][] : memref<10680xi8> to memref<1080xi8>
        %c5880 = arith.constant 5880 : index
        %view_3 = memref.view %1[%c5880][] : memref<10680xi8> to memref<4800xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp15[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp15[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp15[%c2] : memref<16xi32>
        %c3 = arith.constant 3 : index
        %5 = memref.load %rtp15[%c3] : memref<16xi32>
        %6 = aie.objectfifo.acquire @act_bn3_bn4(Consume, 2) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %9 = aie.objectfifo.acquire @bn4_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %11 = aie.objectfifo.subview.access %9[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %c28_i32 = arith.constant 28 : i32
        %c40_i32 = arith.constant 40 : i32
        %c120_i32 = arith.constant 120 : i32
        func.call @bn4_conv2dk1_relu_i8_ui8(%7, %view, %10, %c28_i32, %c40_i32, %c120_i32, %2) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
        %c28_i32_6 = arith.constant 28 : i32
        %c40_i32_7 = arith.constant 40 : i32
        %c120_i32_8 = arith.constant 120 : i32
        func.call @bn4_conv2dk1_relu_i8_ui8(%8, %view, %11, %c28_i32_6, %c40_i32_7, %c120_i32_8, %2) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn4_act_1_2(Produce, 2)
        %12 = aie.objectfifo.acquire @bn4_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %15 = aie.objectfifo.acquire @bn4_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %c28_i32_9 = arith.constant 28 : i32
        %c1_i32 = arith.constant 1 : i32
        %c120_i32_10 = arith.constant 120 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %13, %14, %view_2, %16, %c28_i32_9, %c1_i32, %c120_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn4_act_2_3(Produce, 1)
        %17 = aie.objectfifo.acquire @bn4_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %19 = aie.objectfifo.acquire @act_bn4_bn5(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %c28_i32_13 = arith.constant 28 : i32
        %c120_i32_14 = arith.constant 120 : i32
        %c40_i32_15 = arith.constant 40 : i32
        func.call @bn4_conv2dk1_skip_ui8_i8_i8(%18, %view_3, %20, %7, %c28_i32_13, %c120_i32_14, %c40_i32_15, %4, %5) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn3_bn4(Consume, 1)
        aie.objectfifo.release @bn4_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn4_bn5(Produce, 1)
        %c0_16 = arith.constant 0 : index
        %c26 = arith.constant 26 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c26 step %c1_17 {
          %32 = aie.objectfifo.acquire @act_bn3_bn4(Consume, 2) : !aie.objectfifosubview<memref<28x1x40xi8>>
          %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %34 = aie.objectfifo.subview.access %32[1] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %35 = aie.objectfifo.acquire @bn4_act_1_2(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %c28_i32_27 = arith.constant 28 : i32
          %c40_i32_28 = arith.constant 40 : i32
          %c120_i32_29 = arith.constant 120 : i32
          func.call @bn4_conv2dk1_relu_i8_ui8(%34, %view, %36, %c28_i32_27, %c40_i32_28, %c120_i32_29, %2) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn4_act_1_2(Produce, 1)
          %37 = aie.objectfifo.acquire @bn4_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %39 = aie.objectfifo.subview.access %37[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %40 = aie.objectfifo.subview.access %37[2] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %41 = aie.objectfifo.acquire @bn4_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %42 = aie.objectfifo.subview.access %41[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %c28_i32_30 = arith.constant 28 : i32
          %c1_i32_31 = arith.constant 1 : i32
          %c120_i32_32 = arith.constant 120 : i32
          %c3_i32_33 = arith.constant 3 : i32
          %c3_i32_34 = arith.constant 3 : i32
          %c1_i32_35 = arith.constant 1 : i32
          %c0_i32_36 = arith.constant 0 : i32
          func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%38, %39, %40, %view_2, %42, %c28_i32_30, %c1_i32_31, %c120_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %3, %c0_i32_36) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn4_act_1_2(Consume, 1)
          aie.objectfifo.release @bn4_act_2_3(Produce, 1)
          %43 = aie.objectfifo.acquire @bn4_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %44 = aie.objectfifo.subview.access %43[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %45 = aie.objectfifo.acquire @act_bn4_bn5(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
          %46 = aie.objectfifo.subview.access %45[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %c28_i32_37 = arith.constant 28 : i32
          %c120_i32_38 = arith.constant 120 : i32
          %c40_i32_39 = arith.constant 40 : i32
          func.call @bn4_conv2dk1_skip_ui8_i8_i8(%44, %view_3, %46, %33, %c28_i32_37, %c120_i32_38, %c40_i32_39, %4, %5) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_bn3_bn4(Consume, 1)
          aie.objectfifo.release @bn4_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn4_bn5(Produce, 1)
        }
        %21 = aie.objectfifo.acquire @bn4_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %23 = aie.objectfifo.subview.access %21[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %24 = aie.objectfifo.acquire @bn4_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %c28_i32_18 = arith.constant 28 : i32
        %c1_i32_19 = arith.constant 1 : i32
        %c120_i32_20 = arith.constant 120 : i32
        %c3_i32_21 = arith.constant 3 : i32
        %c3_i32_22 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_23 = arith.constant 0 : i32
        func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%22, %23, %23, %view_2, %25, %c28_i32_18, %c1_i32_19, %c120_i32_20, %c3_i32_21, %c3_i32_22, %c2_i32, %3, %c0_i32_23) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn4_act_1_2(Consume, 2)
        aie.objectfifo.release @bn4_act_2_3(Produce, 1)
        %26 = aie.objectfifo.acquire @bn4_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %28 = aie.objectfifo.acquire @act_bn4_bn5(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %30 = aie.objectfifo.acquire @act_bn3_bn4(Consume, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %c28_i32_24 = arith.constant 28 : i32
        %c120_i32_25 = arith.constant 120 : i32
        %c40_i32_26 = arith.constant 40 : i32
        func.call @bn4_conv2dk1_skip_ui8_i8_i8(%27, %view_3, %29, %31, %c28_i32_24, %c120_i32_25, %c40_i32_26, %4, %5) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn3_bn4(Consume, 1)
        aie.objectfifo.release @bn4_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn4_bn5(Produce, 1)
        aie.objectfifo.release @bn4_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn5_conv2dk1_relu_i8_ui8(memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32)
    func.func private @bn5_conv2dk3_dw_stride2_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn5_conv2dk1_skip_ui8_i8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32)
    func.func private @bn5_conv2dk1_ui8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn5_bn6(%tile_1_4, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<28x1x40xi8>>
    aie.objectfifo @bn5_act_1_2(%tile_1_4, {%tile_1_4}, 3 : i32) : !aie.objectfifo<memref<28x1x120xui8>>
    aie.objectfifo @bn5_act_2_3(%tile_1_4, {%tile_1_4}, 1 : i32) : !aie.objectfifo<memref<28x1x120xui8>>
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn5_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<10680xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10680xi8>> -> memref<10680xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<10680xi8> to memref<4800xi8>
        %c4800 = arith.constant 4800 : index
        %view_2 = memref.view %1[%c4800][] : memref<10680xi8> to memref<1080xi8>
        %c5880 = arith.constant 5880 : index
        %view_3 = memref.view %1[%c5880][] : memref<10680xi8> to memref<4800xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp14[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp14[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp14[%c2] : memref<16xi32>
        %5 = aie.objectfifo.acquire @act_bn4_bn5(Consume, 2) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %8 = aie.objectfifo.acquire @bn5_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %c28_i32 = arith.constant 28 : i32
        %c40_i32 = arith.constant 40 : i32
        %c120_i32 = arith.constant 120 : i32
        func.call @bn5_conv2dk1_relu_i8_ui8(%6, %view, %9, %c28_i32, %c40_i32, %c120_i32, %2) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
        %c28_i32_6 = arith.constant 28 : i32
        %c40_i32_7 = arith.constant 40 : i32
        %c120_i32_8 = arith.constant 120 : i32
        func.call @bn5_conv2dk1_relu_i8_ui8(%7, %view, %10, %c28_i32_6, %c40_i32_7, %c120_i32_8, %2) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn5_act_1_2(Produce, 2)
        aie.objectfifo.release @act_bn4_bn5(Consume, 2)
        %11 = aie.objectfifo.acquire @bn5_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %14 = aie.objectfifo.acquire @bn5_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %c28_i32_9 = arith.constant 28 : i32
        %c1_i32 = arith.constant 1 : i32
        %c120_i32_10 = arith.constant 120 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%12, %12, %13, %view_2, %15, %c28_i32_9, %c1_i32, %c120_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn5_act_2_3(Produce, 1)
        %16 = aie.objectfifo.acquire @bn5_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %18 = aie.objectfifo.acquire @act_bn5_bn6(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %c28_i32_13 = arith.constant 28 : i32
        %c120_i32_14 = arith.constant 120 : i32
        %c40_i32_15 = arith.constant 40 : i32
        func.call @bn5_conv2dk1_ui8_i8(%17, %view_3, %19, %c28_i32_13, %c120_i32_14, %c40_i32_15, %4) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn5_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn5_bn6(Produce, 1)
        %c0_16 = arith.constant 0 : index
        %c26 = arith.constant 26 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c26 step %c1_17 {
          %29 = aie.objectfifo.acquire @act_bn4_bn5(Consume, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
          %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %31 = aie.objectfifo.acquire @bn5_act_1_2(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %c28_i32_27 = arith.constant 28 : i32
          %c40_i32_28 = arith.constant 40 : i32
          %c120_i32_29 = arith.constant 120 : i32
          func.call @bn5_conv2dk1_relu_i8_ui8(%30, %view, %32, %c28_i32_27, %c40_i32_28, %c120_i32_29, %2) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn5_act_1_2(Produce, 1)
          aie.objectfifo.release @act_bn4_bn5(Consume, 1)
          %33 = aie.objectfifo.acquire @bn5_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %35 = aie.objectfifo.subview.access %33[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %36 = aie.objectfifo.subview.access %33[2] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %37 = aie.objectfifo.acquire @bn5_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %c28_i32_30 = arith.constant 28 : i32
          %c1_i32_31 = arith.constant 1 : i32
          %c120_i32_32 = arith.constant 120 : i32
          %c3_i32_33 = arith.constant 3 : i32
          %c3_i32_34 = arith.constant 3 : i32
          %c1_i32_35 = arith.constant 1 : i32
          %c0_i32_36 = arith.constant 0 : i32
          func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%34, %35, %36, %view_2, %38, %c28_i32_30, %c1_i32_31, %c120_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %3, %c0_i32_36) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn5_act_1_2(Consume, 1)
          aie.objectfifo.release @bn5_act_2_3(Produce, 1)
          %39 = aie.objectfifo.acquire @bn5_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
          %40 = aie.objectfifo.subview.access %39[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
          %41 = aie.objectfifo.acquire @act_bn5_bn6(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
          %42 = aie.objectfifo.subview.access %41[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %c28_i32_37 = arith.constant 28 : i32
          %c120_i32_38 = arith.constant 120 : i32
          %c40_i32_39 = arith.constant 40 : i32
          func.call @bn5_conv2dk1_ui8_i8(%40, %view_3, %42, %c28_i32_37, %c120_i32_38, %c40_i32_39, %4) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn5_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn5_bn6(Produce, 1)
        }
        %20 = aie.objectfifo.acquire @bn5_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %22 = aie.objectfifo.subview.access %20[1] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %23 = aie.objectfifo.acquire @bn5_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %c28_i32_18 = arith.constant 28 : i32
        %c1_i32_19 = arith.constant 1 : i32
        %c120_i32_20 = arith.constant 120 : i32
        %c3_i32_21 = arith.constant 3 : i32
        %c3_i32_22 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_23 = arith.constant 0 : i32
        func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%21, %22, %22, %view_2, %24, %c28_i32_18, %c1_i32_19, %c120_i32_20, %c3_i32_21, %c3_i32_22, %c2_i32, %3, %c0_i32_23) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn5_act_1_2(Consume, 2)
        aie.objectfifo.release @bn5_act_2_3(Produce, 1)
        %25 = aie.objectfifo.acquire @bn5_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<28x1x120xui8>>
        %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<28x1x120xui8>> -> memref<28x1x120xui8>
        %27 = aie.objectfifo.acquire @act_bn5_bn6(Produce, 1) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %c28_i32_24 = arith.constant 28 : i32
        %c120_i32_25 = arith.constant 120 : i32
        %c40_i32_26 = arith.constant 40 : i32
        func.call @bn5_conv2dk1_ui8_i8(%26, %view_3, %28, %c28_i32_24, %c120_i32_25, %c40_i32_26, %4) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn5_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn5_bn6(Produce, 1)
        aie.objectfifo.release @bn5_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a"}
    func.func private @bn6_conv2dk1_relu_i8_ui8(memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32)
    func.func private @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn6_conv2dk3_dw_stride1_relu_ui8_ui8(memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn6_conv2dk1_skip_ui8_i8_i8(memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn6_conv2dk1_ui8_i8(memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn6_bn7(%tile_1_2, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @bn6_act_1_2(%tile_1_2, {%tile_1_2}, 3 : i32) : !aie.objectfifo<memref<28x1x240xui8>>
    aie.objectfifo @bn6_act_2_3(%tile_1_2, {%tile_1_2}, 1 : i32) : !aie.objectfifo<memref<14x1x240xui8>>
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn6_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<30960xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<30960xi8>> -> memref<30960xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<30960xi8> to memref<9600xi8>
        %c9600 = arith.constant 9600 : index
        %view_2 = memref.view %1[%c9600][] : memref<30960xi8> to memref<2160xi8>
        %c11760 = arith.constant 11760 : index
        %view_3 = memref.view %1[%c11760][] : memref<30960xi8> to memref<19200xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp12[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp12[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp12[%c2] : memref<16xi32>
        %5 = aie.objectfifo.acquire @act_bn5_bn6(Consume, 2) : !aie.objectfifosubview<memref<28x1x40xi8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
        %8 = aie.objectfifo.acquire @bn6_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<28x1x240xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
        %c28_i32 = arith.constant 28 : i32
        %c40_i32 = arith.constant 40 : i32
        %c240_i32 = arith.constant 240 : i32
        func.call @bn6_conv2dk1_relu_i8_ui8(%6, %view, %9, %c28_i32, %c40_i32, %c240_i32, %2) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
        %c28_i32_6 = arith.constant 28 : i32
        %c40_i32_7 = arith.constant 40 : i32
        %c240_i32_8 = arith.constant 240 : i32
        func.call @bn6_conv2dk1_relu_i8_ui8(%7, %view, %10, %c28_i32_6, %c40_i32_7, %c240_i32_8, %2) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn6_act_1_2(Produce, 2)
        aie.objectfifo.release @act_bn5_bn6(Consume, 2)
        %11 = aie.objectfifo.acquire @bn6_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<28x1x240xui8>>
        %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
        %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
        %14 = aie.objectfifo.acquire @bn6_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x240xui8>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<14x1x240xui8>> -> memref<14x1x240xui8>
        %c28_i32_9 = arith.constant 28 : i32
        %c1_i32 = arith.constant 1 : i32
        %c240_i32_10 = arith.constant 240 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%12, %12, %13, %view_2, %15, %c28_i32_9, %c1_i32, %c240_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn6_act_1_2(Consume, 1)
        aie.objectfifo.release @bn6_act_2_3(Produce, 1)
        %16 = aie.objectfifo.acquire @bn6_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x240xui8>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<14x1x240xui8>> -> memref<14x1x240xui8>
        %18 = aie.objectfifo.acquire @act_bn6_bn7(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32 = arith.constant 14 : i32
        %c240_i32_13 = arith.constant 240 : i32
        %c80_i32 = arith.constant 80 : i32
        func.call @bn6_conv2dk1_ui8_i8(%17, %view_3, %19, %c14_i32, %c240_i32_13, %c80_i32, %4) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn6_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn6_bn7(Produce, 1)
        %c0_14 = arith.constant 0 : index
        %c13 = arith.constant 13 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c13 step %c1_15 {
          %20 = aie.objectfifo.acquire @act_bn5_bn6(Consume, 2) : !aie.objectfifosubview<memref<28x1x40xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %22 = aie.objectfifo.subview.access %20[1] : !aie.objectfifosubview<memref<28x1x40xi8>> -> memref<28x1x40xi8>
          %23 = aie.objectfifo.acquire @bn6_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<28x1x240xui8>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
          %25 = aie.objectfifo.subview.access %23[1] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
          %c28_i32_16 = arith.constant 28 : i32
          %c40_i32_17 = arith.constant 40 : i32
          %c240_i32_18 = arith.constant 240 : i32
          func.call @bn6_conv2dk1_relu_i8_ui8(%21, %view, %24, %c28_i32_16, %c40_i32_17, %c240_i32_18, %2) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
          %c28_i32_19 = arith.constant 28 : i32
          %c40_i32_20 = arith.constant 40 : i32
          %c240_i32_21 = arith.constant 240 : i32
          func.call @bn6_conv2dk1_relu_i8_ui8(%22, %view, %25, %c28_i32_19, %c40_i32_20, %c240_i32_21, %2) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn6_act_1_2(Produce, 2)
          aie.objectfifo.release @act_bn5_bn6(Consume, 2)
          %26 = aie.objectfifo.acquire @bn6_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<28x1x240xui8>>
          %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
          %28 = aie.objectfifo.subview.access %26[1] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
          %29 = aie.objectfifo.subview.access %26[2] : !aie.objectfifosubview<memref<28x1x240xui8>> -> memref<28x1x240xui8>
          %30 = aie.objectfifo.acquire @bn6_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x240xui8>>
          %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<14x1x240xui8>> -> memref<14x1x240xui8>
          %c28_i32_22 = arith.constant 28 : i32
          %c1_i32_23 = arith.constant 1 : i32
          %c240_i32_24 = arith.constant 240 : i32
          %c3_i32_25 = arith.constant 3 : i32
          %c3_i32_26 = arith.constant 3 : i32
          %c1_i32_27 = arith.constant 1 : i32
          %c0_i32_28 = arith.constant 0 : i32
          func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%27, %28, %29, %view_2, %31, %c28_i32_22, %c1_i32_23, %c240_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %3, %c0_i32_28) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn6_act_1_2(Consume, 2)
          aie.objectfifo.release @bn6_act_2_3(Produce, 1)
          %32 = aie.objectfifo.acquire @bn6_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x240xui8>>
          %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<14x1x240xui8>> -> memref<14x1x240xui8>
          %34 = aie.objectfifo.acquire @act_bn6_bn7(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %c14_i32_29 = arith.constant 14 : i32
          %c240_i32_30 = arith.constant 240 : i32
          %c80_i32_31 = arith.constant 80 : i32
          func.call @bn6_conv2dk1_ui8_i8(%33, %view_3, %35, %c14_i32_29, %c240_i32_30, %c80_i32_31, %4) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn6_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn6_bn7(Produce, 1)
        }
        aie.objectfifo.release @bn6_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a"}
    func.func private @bn7_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32)
    func.func private @bn7_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn7_conv2dk1_skip_ui8_i8_i8(memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn7_conv2dk1_ui8_i8(memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn7_bn8(%tile_1_3, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @bn7_act_1_2(%tile_1_3, {%tile_1_3}, 3 : i32) : !aie.objectfifo<memref<14x1x200xui8>>
    aie.objectfifo @bn7_act_2_3(%tile_1_3, {%tile_1_3}, 1 : i32) : !aie.objectfifo<memref<14x1x200xui8>>
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn7_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<33800xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<33800xi8>> -> memref<33800xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<33800xi8> to memref<16000xi8>
        %c16000 = arith.constant 16000 : index
        %view_2 = memref.view %1[%c16000][] : memref<33800xi8> to memref<1800xi8>
        %c17800 = arith.constant 17800 : index
        %view_3 = memref.view %1[%c17800][] : memref<33800xi8> to memref<16000xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp13[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp13[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp13[%c2] : memref<16xi32>
        %c3 = arith.constant 3 : index
        %5 = memref.load %rtp13[%c3] : memref<16xi32>
        %6 = aie.objectfifo.acquire @act_bn6_bn7(Consume, 2) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %9 = aie.objectfifo.acquire @bn7_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %11 = aie.objectfifo.subview.access %9[1] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %c14_i32 = arith.constant 14 : i32
        %c80_i32 = arith.constant 80 : i32
        %c200_i32 = arith.constant 200 : i32
        func.call @bn7_conv2dk1_relu_i8_ui8(%7, %view, %10, %c14_i32, %c80_i32, %c200_i32, %2) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
        %c14_i32_6 = arith.constant 14 : i32
        %c80_i32_7 = arith.constant 80 : i32
        %c200_i32_8 = arith.constant 200 : i32
        func.call @bn7_conv2dk1_relu_i8_ui8(%8, %view, %11, %c14_i32_6, %c80_i32_7, %c200_i32_8, %2) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn7_act_1_2(Produce, 2)
        %12 = aie.objectfifo.acquire @bn7_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %15 = aie.objectfifo.acquire @bn7_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %c14_i32_9 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c200_i32_10 = arith.constant 200 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %13, %14, %view_2, %16, %c14_i32_9, %c1_i32, %c200_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn7_act_2_3(Produce, 1)
        %17 = aie.objectfifo.acquire @bn7_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %19 = aie.objectfifo.acquire @act_bn7_bn8(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32_13 = arith.constant 14 : i32
        %c200_i32_14 = arith.constant 200 : i32
        %c80_i32_15 = arith.constant 80 : i32
        func.call @bn7_conv2dk1_skip_ui8_i8_i8(%18, %view_3, %20, %7, %c14_i32_13, %c200_i32_14, %c80_i32_15, %4, %5) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn6_bn7(Consume, 1)
        aie.objectfifo.release @bn7_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn7_bn8(Produce, 1)
        %c0_16 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c12 step %c1_17 {
          %32 = aie.objectfifo.acquire @act_bn6_bn7(Consume, 2) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %34 = aie.objectfifo.subview.access %32[1] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %35 = aie.objectfifo.acquire @bn7_act_1_2(Produce, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
          %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
          %c14_i32_27 = arith.constant 14 : i32
          %c80_i32_28 = arith.constant 80 : i32
          %c200_i32_29 = arith.constant 200 : i32
          func.call @bn7_conv2dk1_relu_i8_ui8(%34, %view, %36, %c14_i32_27, %c80_i32_28, %c200_i32_29, %2) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn7_act_1_2(Produce, 1)
          %37 = aie.objectfifo.acquire @bn7_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<14x1x200xui8>>
          %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
          %39 = aie.objectfifo.subview.access %37[1] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
          %40 = aie.objectfifo.subview.access %37[2] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
          %41 = aie.objectfifo.acquire @bn7_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
          %42 = aie.objectfifo.subview.access %41[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
          %c14_i32_30 = arith.constant 14 : i32
          %c1_i32_31 = arith.constant 1 : i32
          %c200_i32_32 = arith.constant 200 : i32
          %c3_i32_33 = arith.constant 3 : i32
          %c3_i32_34 = arith.constant 3 : i32
          %c1_i32_35 = arith.constant 1 : i32
          %c0_i32_36 = arith.constant 0 : i32
          func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%38, %39, %40, %view_2, %42, %c14_i32_30, %c1_i32_31, %c200_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %3, %c0_i32_36) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn7_act_1_2(Consume, 1)
          aie.objectfifo.release @bn7_act_2_3(Produce, 1)
          %43 = aie.objectfifo.acquire @bn7_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
          %44 = aie.objectfifo.subview.access %43[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
          %45 = aie.objectfifo.acquire @act_bn7_bn8(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %46 = aie.objectfifo.subview.access %45[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %c14_i32_37 = arith.constant 14 : i32
          %c200_i32_38 = arith.constant 200 : i32
          %c80_i32_39 = arith.constant 80 : i32
          func.call @bn7_conv2dk1_skip_ui8_i8_i8(%44, %view_3, %46, %33, %c14_i32_37, %c200_i32_38, %c80_i32_39, %4, %5) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_bn6_bn7(Consume, 1)
          aie.objectfifo.release @bn7_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn7_bn8(Produce, 1)
        }
        %21 = aie.objectfifo.acquire @bn7_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %23 = aie.objectfifo.subview.access %21[1] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %24 = aie.objectfifo.acquire @bn7_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %c14_i32_18 = arith.constant 14 : i32
        %c1_i32_19 = arith.constant 1 : i32
        %c200_i32_20 = arith.constant 200 : i32
        %c3_i32_21 = arith.constant 3 : i32
        %c3_i32_22 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_23 = arith.constant 0 : i32
        func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%22, %23, %23, %view_2, %25, %c14_i32_18, %c1_i32_19, %c200_i32_20, %c3_i32_21, %c3_i32_22, %c2_i32, %3, %c0_i32_23) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn7_act_1_2(Consume, 2)
        aie.objectfifo.release @bn7_act_2_3(Produce, 1)
        %26 = aie.objectfifo.acquire @bn7_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x200xui8>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<14x1x200xui8>> -> memref<14x1x200xui8>
        %28 = aie.objectfifo.acquire @act_bn7_bn8(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %30 = aie.objectfifo.acquire @act_bn6_bn7(Consume, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32_24 = arith.constant 14 : i32
        %c200_i32_25 = arith.constant 200 : i32
        %c80_i32_26 = arith.constant 80 : i32
        func.call @bn7_conv2dk1_skip_ui8_i8_i8(%27, %view_3, %29, %31, %c14_i32_24, %c200_i32_25, %c80_i32_26, %4, %5) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn6_bn7(Consume, 1)
        aie.objectfifo.release @bn7_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn7_bn8(Produce, 1)
        aie.objectfifo.release @bn7_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn8_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32)
    func.func private @bn8_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn8_conv2dk1_skip_ui8_i8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn8_conv2dk1_ui8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    aie.objectfifo @act_bn8_bn9(%tile_2_2, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @bn8_act_1_2(%tile_2_2, {%tile_2_2}, 3 : i32) : !aie.objectfifo<memref<14x1x184xui8>>
    aie.objectfifo @bn8_act_2_3(%tile_2_2, {%tile_2_2}, 1 : i32) : !aie.objectfifo<memref<14x1x184xui8>>
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn8_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<31096xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<31096xi8>> -> memref<31096xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<31096xi8> to memref<14720xi8>
        %c14720 = arith.constant 14720 : index
        %view_2 = memref.view %1[%c14720][] : memref<31096xi8> to memref<1656xi8>
        %c16376 = arith.constant 16376 : index
        %view_3 = memref.view %1[%c16376][] : memref<31096xi8> to memref<14720xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp22[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp22[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp22[%c2] : memref<16xi32>
        %5 = aie.objectfifo.acquire @act_bn7_bn8(Consume, 2) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %8 = aie.objectfifo.acquire @bn8_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %c14_i32 = arith.constant 14 : i32
        %c80_i32 = arith.constant 80 : i32
        %c184_i32 = arith.constant 184 : i32
        func.call @bn8_conv2dk1_relu_i8_ui8(%6, %view, %9, %c14_i32, %c80_i32, %c184_i32, %2) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
        %c14_i32_6 = arith.constant 14 : i32
        %c80_i32_7 = arith.constant 80 : i32
        %c184_i32_8 = arith.constant 184 : i32
        func.call @bn8_conv2dk1_relu_i8_ui8(%7, %view, %10, %c14_i32_6, %c80_i32_7, %c184_i32_8, %2) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn8_act_1_2(Produce, 2)
        aie.objectfifo.release @act_bn7_bn8(Consume, 2)
        %11 = aie.objectfifo.acquire @bn8_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %14 = aie.objectfifo.acquire @bn8_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %c14_i32_9 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c184_i32_10 = arith.constant 184 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%12, %12, %13, %view_2, %15, %c14_i32_9, %c1_i32, %c184_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn8_act_2_3(Produce, 1)
        %16 = aie.objectfifo.acquire @bn8_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %18 = aie.objectfifo.acquire @act_bn8_bn9(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32_13 = arith.constant 14 : i32
        %c184_i32_14 = arith.constant 184 : i32
        %c80_i32_15 = arith.constant 80 : i32
        func.call @bn8_conv2dk1_ui8_i8(%17, %view_3, %19, %c14_i32_13, %c184_i32_14, %c80_i32_15, %4) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn8_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn8_bn9(Produce, 1)
        %c0_16 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c12 step %c1_17 {
          %29 = aie.objectfifo.acquire @act_bn7_bn8(Consume, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %31 = aie.objectfifo.acquire @bn8_act_1_2(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %c14_i32_27 = arith.constant 14 : i32
          %c80_i32_28 = arith.constant 80 : i32
          %c184_i32_29 = arith.constant 184 : i32
          func.call @bn8_conv2dk1_relu_i8_ui8(%30, %view, %32, %c14_i32_27, %c80_i32_28, %c184_i32_29, %2) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn8_act_1_2(Produce, 1)
          aie.objectfifo.release @act_bn7_bn8(Consume, 1)
          %33 = aie.objectfifo.acquire @bn8_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %35 = aie.objectfifo.subview.access %33[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %36 = aie.objectfifo.subview.access %33[2] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %37 = aie.objectfifo.acquire @bn8_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %c14_i32_30 = arith.constant 14 : i32
          %c1_i32_31 = arith.constant 1 : i32
          %c184_i32_32 = arith.constant 184 : i32
          %c3_i32_33 = arith.constant 3 : i32
          %c3_i32_34 = arith.constant 3 : i32
          %c1_i32_35 = arith.constant 1 : i32
          %c0_i32_36 = arith.constant 0 : i32
          func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%34, %35, %36, %view_2, %38, %c14_i32_30, %c1_i32_31, %c184_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %3, %c0_i32_36) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn8_act_1_2(Consume, 1)
          aie.objectfifo.release @bn8_act_2_3(Produce, 1)
          %39 = aie.objectfifo.acquire @bn8_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %40 = aie.objectfifo.subview.access %39[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %41 = aie.objectfifo.acquire @act_bn8_bn9(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %42 = aie.objectfifo.subview.access %41[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %c14_i32_37 = arith.constant 14 : i32
          %c184_i32_38 = arith.constant 184 : i32
          %c80_i32_39 = arith.constant 80 : i32
          func.call @bn8_conv2dk1_ui8_i8(%40, %view_3, %42, %c14_i32_37, %c184_i32_38, %c80_i32_39, %4) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn8_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn8_bn9(Produce, 1)
        }
        %20 = aie.objectfifo.acquire @bn8_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %22 = aie.objectfifo.subview.access %20[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %23 = aie.objectfifo.acquire @bn8_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %c14_i32_18 = arith.constant 14 : i32
        %c1_i32_19 = arith.constant 1 : i32
        %c184_i32_20 = arith.constant 184 : i32
        %c3_i32_21 = arith.constant 3 : i32
        %c3_i32_22 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_23 = arith.constant 0 : i32
        func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%21, %22, %22, %view_2, %24, %c14_i32_18, %c1_i32_19, %c184_i32_20, %c3_i32_21, %c3_i32_22, %c2_i32, %3, %c0_i32_23) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn8_act_1_2(Consume, 2)
        aie.objectfifo.release @bn8_act_2_3(Produce, 1)
        %25 = aie.objectfifo.acquire @bn8_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %27 = aie.objectfifo.acquire @act_bn8_bn9(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32_24 = arith.constant 14 : i32
        %c184_i32_25 = arith.constant 184 : i32
        %c80_i32_26 = arith.constant 80 : i32
        func.call @bn8_conv2dk1_ui8_i8(%26, %view_3, %28, %c14_i32_24, %c184_i32_25, %c80_i32_26, %4) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn8_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn8_bn9(Produce, 1)
        aie.objectfifo.release @bn8_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn8_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a"}
    aie.objectfifo @act_bn9_bn10(%tile_2_3, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    func.func private @bn9_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32)
    func.func private @bn9_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn9_conv2dk1_skip_ui8_i8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn9_conv2dk1_ui8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    aie.objectfifo @bn9_act_1_2(%tile_2_3, {%tile_2_3}, 3 : i32) : !aie.objectfifo<memref<14x1x184xui8>>
    aie.objectfifo @bn9_act_2_3(%tile_2_3, {%tile_2_3}, 1 : i32) : !aie.objectfifo<memref<14x1x184xui8>>
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %0 = aie.objectfifo.acquire @bn9_wts_OF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<31096xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<31096xi8>> -> memref<31096xi8>
        %c0_1 = arith.constant 0 : index
        %view = memref.view %1[%c0_1][] : memref<31096xi8> to memref<14720xi8>
        %c14720 = arith.constant 14720 : index
        %view_2 = memref.view %1[%c14720][] : memref<31096xi8> to memref<1656xi8>
        %c16376 = arith.constant 16376 : index
        %view_3 = memref.view %1[%c16376][] : memref<31096xi8> to memref<14720xi8>
        %c0_4 = arith.constant 0 : index
        %2 = memref.load %rtp23[%c0_4] : memref<16xi32>
        %c1_5 = arith.constant 1 : index
        %3 = memref.load %rtp23[%c1_5] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp23[%c2] : memref<16xi32>
        %c3 = arith.constant 3 : index
        %5 = memref.load %rtp23[%c3] : memref<16xi32>
        %6 = aie.objectfifo.acquire @act_bn8_bn9(Consume, 2) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %9 = aie.objectfifo.acquire @bn9_act_1_2(Produce, 2) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %11 = aie.objectfifo.subview.access %9[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %c14_i32 = arith.constant 14 : i32
        %c80_i32 = arith.constant 80 : i32
        %c184_i32 = arith.constant 184 : i32
        func.call @bn9_conv2dk1_relu_i8_ui8(%7, %view, %10, %c14_i32, %c80_i32, %c184_i32, %2) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
        %c14_i32_6 = arith.constant 14 : i32
        %c80_i32_7 = arith.constant 80 : i32
        %c184_i32_8 = arith.constant 184 : i32
        func.call @bn9_conv2dk1_relu_i8_ui8(%8, %view, %11, %c14_i32_6, %c80_i32_7, %c184_i32_8, %2) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn9_act_1_2(Produce, 2)
        %12 = aie.objectfifo.acquire @bn9_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %15 = aie.objectfifo.acquire @bn9_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %c14_i32_9 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c184_i32_10 = arith.constant 184 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_11 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_12 = arith.constant 0 : i32
        func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %13, %14, %view_2, %16, %c14_i32_9, %c1_i32, %c184_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %3, %c0_i32_12) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn9_act_2_3(Produce, 1)
        %17 = aie.objectfifo.acquire @bn9_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %19 = aie.objectfifo.acquire @act_bn9_bn10(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32_13 = arith.constant 14 : i32
        %c184_i32_14 = arith.constant 184 : i32
        %c80_i32_15 = arith.constant 80 : i32
        func.call @bn9_conv2dk1_skip_ui8_i8_i8(%18, %view_3, %20, %7, %c14_i32_13, %c184_i32_14, %c80_i32_15, %4, %5) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn8_bn9(Consume, 1)
        aie.objectfifo.release @bn9_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn9_bn10(Produce, 1)
        %c0_16 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c12 step %c1_17 {
          %32 = aie.objectfifo.acquire @act_bn8_bn9(Consume, 2) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %34 = aie.objectfifo.subview.access %32[1] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %35 = aie.objectfifo.acquire @bn9_act_1_2(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %c14_i32_27 = arith.constant 14 : i32
          %c80_i32_28 = arith.constant 80 : i32
          %c184_i32_29 = arith.constant 184 : i32
          func.call @bn9_conv2dk1_relu_i8_ui8(%34, %view, %36, %c14_i32_27, %c80_i32_28, %c184_i32_29, %2) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn9_act_1_2(Produce, 1)
          %37 = aie.objectfifo.acquire @bn9_act_1_2(Consume, 3) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %39 = aie.objectfifo.subview.access %37[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %40 = aie.objectfifo.subview.access %37[2] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %41 = aie.objectfifo.acquire @bn9_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %42 = aie.objectfifo.subview.access %41[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %c14_i32_30 = arith.constant 14 : i32
          %c1_i32_31 = arith.constant 1 : i32
          %c184_i32_32 = arith.constant 184 : i32
          %c3_i32_33 = arith.constant 3 : i32
          %c3_i32_34 = arith.constant 3 : i32
          %c1_i32_35 = arith.constant 1 : i32
          %c0_i32_36 = arith.constant 0 : i32
          func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%38, %39, %40, %view_2, %42, %c14_i32_30, %c1_i32_31, %c184_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %3, %c0_i32_36) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @bn9_act_1_2(Consume, 1)
          aie.objectfifo.release @bn9_act_2_3(Produce, 1)
          %43 = aie.objectfifo.acquire @bn9_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
          %44 = aie.objectfifo.subview.access %43[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
          %45 = aie.objectfifo.acquire @act_bn9_bn10(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %46 = aie.objectfifo.subview.access %45[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %c14_i32_37 = arith.constant 14 : i32
          %c184_i32_38 = arith.constant 184 : i32
          %c80_i32_39 = arith.constant 80 : i32
          func.call @bn9_conv2dk1_skip_ui8_i8_i8(%44, %view_3, %46, %33, %c14_i32_37, %c184_i32_38, %c80_i32_39, %4, %5) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_bn8_bn9(Consume, 1)
          aie.objectfifo.release @bn9_act_2_3(Consume, 1)
          aie.objectfifo.release @act_bn9_bn10(Produce, 1)
        }
        %21 = aie.objectfifo.acquire @bn9_act_1_2(Consume, 2) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %23 = aie.objectfifo.subview.access %21[1] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %24 = aie.objectfifo.acquire @bn9_act_2_3(Produce, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %c14_i32_18 = arith.constant 14 : i32
        %c1_i32_19 = arith.constant 1 : i32
        %c184_i32_20 = arith.constant 184 : i32
        %c3_i32_21 = arith.constant 3 : i32
        %c3_i32_22 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c0_i32_23 = arith.constant 0 : i32
        func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%22, %23, %23, %view_2, %25, %c14_i32_18, %c1_i32_19, %c184_i32_20, %c3_i32_21, %c3_i32_22, %c2_i32, %3, %c0_i32_23) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @bn9_act_1_2(Consume, 2)
        aie.objectfifo.release @bn9_act_2_3(Produce, 1)
        %26 = aie.objectfifo.acquire @bn9_act_2_3(Consume, 1) : !aie.objectfifosubview<memref<14x1x184xui8>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<14x1x184xui8>> -> memref<14x1x184xui8>
        %28 = aie.objectfifo.acquire @act_bn9_bn10(Produce, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %30 = aie.objectfifo.acquire @act_bn8_bn9(Consume, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
        %c14_i32_24 = arith.constant 14 : i32
        %c184_i32_25 = arith.constant 184 : i32
        %c80_i32_26 = arith.constant 80 : i32
        func.call @bn9_conv2dk1_skip_ui8_i8_i8(%27, %view_3, %29, %31, %c14_i32_24, %c184_i32_25, %c80_i32_26, %4, %5) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_bn8_bn9(Consume, 1)
        aie.objectfifo.release @bn9_act_2_3(Consume, 1)
        aie.objectfifo.release @act_bn9_bn10(Produce, 1)
        aie.objectfifo.release @bn9_wts_OF_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "bn9_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    aie.objectfifo @act_out(%tile_3_2, {%tile_3_0}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    func.func private @bn10_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32)
    func.func private @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn10_conv2dk1_ui8_i8(memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_relu_i8_ui8(memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_skip_ui8_i8_i8(memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32)
    aie.objectfifo @B_OF_b10_act_layer1_layer2(%tile_2_4, {%tile_2_5}, 4 : i32) {via_DMA = true} : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @B_OF_b10_act_layer2_layer3(%tile_2_5, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @B_OF_b10_layer3_bn_11_layer1(%tile_3_5, {%tile_3_4, %tile_2_1}, [2 : i32, 2 : i32, 6 : i32]) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo @OF_b11_skip(%tile_2_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo.link [@B_OF_b10_layer3_bn_11_layer1] -> [@OF_b11_skip]([] [])
    aie.objectfifo @B_OF_b11_act_layer1_layer2(%tile_3_4, {%tile_3_3}, 4 : i32) {via_DMA = true} : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @B_OF_b11_act_layer2_layer3(%tile_3_3, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<14x1x336xui8>>
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @weightsInBN10_layer1(Consume, 1) : !aie.objectfifosubview<memref<38400xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<38400xi8>> -> memref<38400xi8>
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @act_bn9_bn10(Consume, 1) : !aie.objectfifosubview<memref<14x1x80xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x80xi8>> -> memref<14x1x80xi8>
          %4 = aie.objectfifo.acquire @B_OF_b10_act_layer1_layer2(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %c14_i32 = arith.constant 14 : i32
          %c80_i32 = arith.constant 80 : i32
          %c480_i32 = arith.constant 480 : i32
          %c8_i32 = arith.constant 8 : i32
          func.call @bn10_conv2dk1_relu_i8_ui8(%3, %1, %5, %c14_i32, %c80_i32, %c480_i32, %c8_i32) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_bn9_bn10(Consume, 1)
          aie.objectfifo.release @B_OF_b10_act_layer1_layer2(Produce, 1)
        }
        aie.objectfifo.release @weightsInBN10_layer1(Consume, 1)
      }
      aie.end
    } {link_with = "bn10_conv2dk1_fused_relu.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @weightsInBN10_layer2(Consume, 1) : !aie.objectfifosubview<memref<4320xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4320xi8>> -> memref<4320xi8>
        %2 = aie.objectfifo.acquire @B_OF_b10_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %5 = aie.objectfifo.acquire @B_OF_b10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c480_i32 = arith.constant 480 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_0 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_1 = arith.constant 0 : i32
        func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%3, %3, %4, %1, %6, %c14_i32, %c1_i32, %c480_i32, %c3_i32, %c3_i32_0, %c0_i32, %c7_i32, %c0_i32_1) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @B_OF_b10_act_layer2_layer3(Produce, 1)
        %c0_2 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c12 step %c1_3 {
          %12 = aie.objectfifo.acquire @B_OF_b10_act_layer1_layer2(Consume, 3) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %16 = aie.objectfifo.acquire @B_OF_b10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %c14_i32_11 = arith.constant 14 : i32
          %c1_i32_12 = arith.constant 1 : i32
          %c480_i32_13 = arith.constant 480 : i32
          %c3_i32_14 = arith.constant 3 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c1_i32_16 = arith.constant 1 : i32
          %c7_i32_17 = arith.constant 7 : i32
          %c0_i32_18 = arith.constant 0 : i32
          func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%13, %14, %15, %1, %17, %c14_i32_11, %c1_i32_12, %c480_i32_13, %c3_i32_14, %c3_i32_15, %c1_i32_16, %c7_i32_17, %c0_i32_18) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @B_OF_b10_act_layer1_layer2(Consume, 1)
          aie.objectfifo.release @B_OF_b10_act_layer2_layer3(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @B_OF_b10_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %10 = aie.objectfifo.acquire @B_OF_b10_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
        %c14_i32_4 = arith.constant 14 : i32
        %c1_i32_5 = arith.constant 1 : i32
        %c480_i32_6 = arith.constant 480 : i32
        %c3_i32_7 = arith.constant 3 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c7_i32_9 = arith.constant 7 : i32
        %c0_i32_10 = arith.constant 0 : i32
        func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%8, %9, %9, %1, %11, %c14_i32_4, %c1_i32_5, %c480_i32_6, %c3_i32_7, %c3_i32_8, %c2_i32, %c7_i32_9, %c0_i32_10) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @B_OF_b10_act_layer1_layer2(Consume, 2)
        aie.objectfifo.release @B_OF_b10_act_layer2_layer3(Produce, 1)
        aie.objectfifo.release @weightsInBN10_layer2(Consume, 1)
      }
      aie.end
    } {link_with = "bn10_conv2dk3_dw.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @weightsInBN10_layer3(Consume, 1) : !aie.objectfifosubview<memref<53760xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<53760xi8>> -> memref<53760xi8>
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @B_OF_b10_act_layer2_layer3(Consume, 1) : !aie.objectfifosubview<memref<14x1x480xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x480xui8>> -> memref<14x1x480xui8>
          %4 = aie.objectfifo.acquire @B_OF_b10_layer3_bn_11_layer1(Produce, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %c14_i32 = arith.constant 14 : i32
          %c480_i32 = arith.constant 480 : i32
          %c112_i32 = arith.constant 112 : i32
          %c10_i32 = arith.constant 10 : i32
          func.call @bn10_conv2dk1_ui8_i8(%3, %1, %5, %c14_i32, %c480_i32, %c112_i32, %c10_i32) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @B_OF_b10_act_layer2_layer3(Consume, 1)
          aie.objectfifo.release @B_OF_b10_layer3_bn_11_layer1(Produce, 1)
        }
        aie.objectfifo.release @weightsInBN10_layer3(Consume, 1)
      }
      aie.end
    } {link_with = "bn10_conv2dk1_ui8.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @weightsInBN11_layer1(Consume, 1) : !aie.objectfifosubview<memref<37632xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<37632xi8>> -> memref<37632xi8>
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @B_OF_b10_layer3_bn_11_layer1(Consume, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %4 = aie.objectfifo.acquire @B_OF_b11_act_layer1_layer2(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %c14_i32 = arith.constant 14 : i32
          %c112_i32 = arith.constant 112 : i32
          %c336_i32 = arith.constant 336 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @bn11_conv2dk1_relu_i8_ui8(%3, %1, %5, %c14_i32, %c112_i32, %c336_i32, %c9_i32) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @B_OF_b10_layer3_bn_11_layer1(Consume, 1)
          aie.objectfifo.release @B_OF_b11_act_layer1_layer2(Produce, 1)
        }
        aie.objectfifo.release @weightsInBN11_layer1(Consume, 1)
      }
      aie.end
    } {link_with = "bn11_conv2dk1_fused_relu.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @weightsInBN11_layer2(Consume, 1) : !aie.objectfifosubview<memref<3024xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<3024xi8>> -> memref<3024xi8>
        %2 = aie.objectfifo.acquire @B_OF_b11_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %5 = aie.objectfifo.acquire @B_OF_b11_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
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
        aie.objectfifo.release @B_OF_b11_act_layer2_layer3(Produce, 1)
        %c0_2 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c12 step %c1_3 {
          %12 = aie.objectfifo.acquire @B_OF_b11_act_layer1_layer2(Consume, 3) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %16 = aie.objectfifo.acquire @B_OF_b11_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
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
          aie.objectfifo.release @B_OF_b11_act_layer1_layer2(Consume, 1)
          aie.objectfifo.release @B_OF_b11_act_layer2_layer3(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @B_OF_b11_act_layer1_layer2(Consume, 2) : !aie.objectfifosubview<memref<14x1x336xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
        %10 = aie.objectfifo.acquire @B_OF_b11_act_layer2_layer3(Produce, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
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
        aie.objectfifo.release @B_OF_b11_act_layer1_layer2(Consume, 2)
        aie.objectfifo.release @B_OF_b11_act_layer2_layer3(Produce, 1)
        aie.objectfifo.release @weightsInBN11_layer2(Consume, 1)
      }
      aie.end
    } {link_with = "bn11_conv2dk3_dw.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @weightsInBN11_layer3(Consume, 1) : !aie.objectfifosubview<memref<37632xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<37632xi8>> -> memref<37632xi8>
        %c0_0 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c14 step %c1_1 {
          %2 = aie.objectfifo.acquire @B_OF_b11_act_layer2_layer3(Consume, 1) : !aie.objectfifosubview<memref<14x1x336xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<14x1x336xui8>> -> memref<14x1x336xui8>
          %4 = aie.objectfifo.acquire @act_out(Produce, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %6 = aie.objectfifo.acquire @OF_b11_skip(Consume, 1) : !aie.objectfifosubview<memref<14x1x112xi8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<14x1x112xi8>> -> memref<14x1x112xi8>
          %c14_i32 = arith.constant 14 : i32
          %c336_i32 = arith.constant 336 : i32
          %c112_i32 = arith.constant 112 : i32
          %c12_i32 = arith.constant 12 : i32
          %c1_i32 = arith.constant 1 : i32
          func.call @bn11_conv2dk1_skip_ui8_i8_i8(%3, %1, %5, %7, %c14_i32, %c336_i32, %c112_i32, %c12_i32, %c1_i32) : (memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @B_OF_b11_act_layer2_layer3(Consume, 1)
          aie.objectfifo.release @act_out(Produce, 1)
          aie.objectfifo.release @OF_b11_skip(Consume, 1)
        }
        aie.objectfifo.release @weightsInBN11_layer3(Consume, 1)
      }
      aie.end
    } {link_with = "bn11_conv2dk1_skip.o"}
    func.func @sequence(%arg0: memref<50176xi32>, %arg1: memref<83994xi32>, %arg2: memref<5488xi32>) {
      aiex.npu.rtp_write(0, 3, 0, 9) {buffer_sym_name = "rtp03"}
      aiex.npu.rtp_write(0, 3, 1, 8) {buffer_sym_name = "rtp03"}
      aiex.npu.rtp_write(0, 3, 2, 2) {buffer_sym_name = "rtp03"}
      aiex.npu.rtp_write(0, 3, 3, 8) {buffer_sym_name = "rtp03"}
      aiex.npu.rtp_write(0, 3, 4, 8) {buffer_sym_name = "rtp03"}
      aiex.npu.rtp_write(0, 3, 5, 8) {buffer_sym_name = "rtp03"}
      aiex.npu.rtp_write(0, 4, 0, 8) {buffer_sym_name = "rtp04"}
      aiex.npu.rtp_write(0, 4, 1, 8) {buffer_sym_name = "rtp04"}
      aiex.npu.rtp_write(0, 4, 2, 11) {buffer_sym_name = "rtp04"}
      aiex.npu.rtp_write(0, 4, 3, 1) {buffer_sym_name = "rtp04"}
      aiex.npu.rtp_write(0, 5, 0, 8) {buffer_sym_name = "rtp05"}
      aiex.npu.rtp_write(0, 5, 1, 6) {buffer_sym_name = "rtp05"}
      aiex.npu.rtp_write(0, 5, 2, 8) {buffer_sym_name = "rtp05"}
      aiex.npu.rtp_write(0, 5, 3, 0) {buffer_sym_name = "rtp05"}
      aiex.npu.rtp_write(1, 5, 0, 8) {buffer_sym_name = "rtp15"}
      aiex.npu.rtp_write(1, 5, 1, 8) {buffer_sym_name = "rtp15"}
      aiex.npu.rtp_write(1, 5, 2, 11) {buffer_sym_name = "rtp15"}
      aiex.npu.rtp_write(1, 5, 3, 1) {buffer_sym_name = "rtp15"}
      aiex.npu.rtp_write(1, 4, 0, 7) {buffer_sym_name = "rtp14"}
      aiex.npu.rtp_write(1, 4, 1, 8) {buffer_sym_name = "rtp14"}
      aiex.npu.rtp_write(1, 4, 2, 9) {buffer_sym_name = "rtp14"}
      aiex.npu.rtp_write(1, 4, 3, 0) {buffer_sym_name = "rtp14"}
      aiex.npu.rtp_write(1, 2, 0, 7) {buffer_sym_name = "rtp12"}
      aiex.npu.rtp_write(1, 2, 1, 7) {buffer_sym_name = "rtp12"}
      aiex.npu.rtp_write(1, 2, 2, 8) {buffer_sym_name = "rtp12"}
      aiex.npu.rtp_write(1, 2, 3, 0) {buffer_sym_name = "rtp12"}
      aiex.npu.rtp_write(1, 3, 0, 9) {buffer_sym_name = "rtp13"}
      aiex.npu.rtp_write(1, 3, 1, 7) {buffer_sym_name = "rtp13"}
      aiex.npu.rtp_write(1, 3, 2, 12) {buffer_sym_name = "rtp13"}
      aiex.npu.rtp_write(1, 3, 3, 1) {buffer_sym_name = "rtp13"}
      aiex.npu.rtp_write(2, 2, 0, 8) {buffer_sym_name = "rtp22"}
      aiex.npu.rtp_write(2, 2, 1, 8) {buffer_sym_name = "rtp22"}
      aiex.npu.rtp_write(2, 2, 2, 8) {buffer_sym_name = "rtp22"}
      aiex.npu.rtp_write(2, 2, 3, 0) {buffer_sym_name = "rtp22"}
      aiex.npu.rtp_write(2, 3, 0, 9) {buffer_sym_name = "rtp23"}
      aiex.npu.rtp_write(2, 3, 1, 8) {buffer_sym_name = "rtp23"}
      aiex.npu.rtp_write(2, 3, 2, 11) {buffer_sym_name = "rtp23"}
      aiex.npu.rtp_write(2, 3, 3, 1) {buffer_sym_name = "rtp23"}
      aiex.npu.rtp_write(2, 4, 0, 8) {buffer_sym_name = "bn10_1_rtp"}
      aiex.npu.rtp_write(2, 5, 0, 7) {buffer_sym_name = "bn10_2_rtp"}
      aiex.npu.rtp_write(3, 5, 0, 10) {buffer_sym_name = "bn10_3_rtp"}
      aiex.npu.rtp_write(3, 4, 0, 9) {buffer_sym_name = "bn11_1_rtp"}
      aiex.npu.rtp_write(3, 3, 0, 8) {buffer_sym_name = "bn11_2_rtp"}
      aiex.npu.rtp_write(3, 2, 0, 12) {buffer_sym_name = "bn11_3_rtp"}
      aiex.npu.rtp_write(3, 2, 1, 1) {buffer_sym_name = "bn11_3_rtp"}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 50176][0, 0, 0]) {id = 0 : i64, metadata = @act_in} : memref<50176xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 5488][0, 0, 0]) {id = 2 : i64, metadata = @act_out} : memref<5488xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 8564][0, 0, 0]) {id = 1 : i64, metadata = @wts_OF_01_L3L2} : memref<83994xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8564][1, 1, 1, 31738][0, 0, 0]) {id = 1 : i64, metadata = @wts_OF_11_L3L2} : memref<83994xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 40302][1, 1, 1, 24120][0, 0, 0]) {id = 1 : i64, metadata = @wts_b10_L3L2} : memref<83994xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 64422][1, 1, 1, 19572][0, 0, 0]) {id = 1 : i64, metadata = @wts_b11_L3L2} : memref<83994xi32>
      aiex.npu.sync {channel = 0 : i32, column = 3 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

