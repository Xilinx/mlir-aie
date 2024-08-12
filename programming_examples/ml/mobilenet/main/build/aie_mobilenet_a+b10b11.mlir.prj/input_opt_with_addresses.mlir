module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @rtp23() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp22() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp13() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp12() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp14() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp15() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp05() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp04() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @rtp03() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @bn11_3_rtp() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @bn11_2_rtp() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @bn11_1_rtp() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @bn10_3_rtp() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @bn10_2_rtp() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @bn10_1_rtp() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @wts_b10_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<96480 x i8>
  llvm.mlir.global external @weightsInBN10_layer1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<38400 x i8>
  llvm.mlir.global external @weightsInBN10_layer2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<4320 x i8>
  llvm.mlir.global external @weightsInBN10_layer3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<53760 x i8>
  llvm.mlir.global external @wts_b11_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<78288 x i8>
  llvm.mlir.global external @weightsInBN11_layer1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<37632 x i8>
  llvm.mlir.global external @weightsInBN11_layer2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<3024 x i8>
  llvm.mlir.global external @weightsInBN11_layer3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<37632 x i8>
  llvm.mlir.global external @act_in_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @act_in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @act_in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @wts_OF_01_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<34256 x i8>
  llvm.mlir.global external @bn0_1_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<3536 x i8>
  llvm.mlir.global external @bn2_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<4104 x i8>
  llvm.mlir.global external @bn3_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<5256 x i8>
  llvm.mlir.global external @bn4_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<10680 x i8>
  llvm.mlir.global external @bn5_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<10680 x i8>
  llvm.mlir.global external @wts_OF_11_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<126952 x i8>
  llvm.mlir.global external @bn6_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<30960 x i8>
  llvm.mlir.global external @bn7_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<33800 x i8>
  llvm.mlir.global external @bn8_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<31096 x i8>
  llvm.mlir.global external @bn9_wts_OF_L2L1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<31096 x i8>
  llvm.mlir.global external @act_bn01_bn2_buff_2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @act_bn01_bn2_buff_1() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @act_bn01_bn2_buff_0() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @bn01_act_bn0_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @bn01_act_bn0_bn1_buff_0() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @bn01_act_bn1_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @bn01_act_bn1_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @bn01_act_bn1_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @bn01_act_bn1_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @act_bn2_bn3_buff_2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @act_bn2_bn3_buff_1() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @act_bn2_bn3_buff_0() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @bn2_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn2_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn2_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn2_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @act_bn3_bn4_buff_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn3_bn4_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn3_bn4_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @bn3_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn3_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn3_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn3_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @act_bn4_bn5_buff_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn4_bn5_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn4_bn5_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @bn4_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn4_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn4_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn4_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @act_bn5_bn6_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn5_bn6_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn5_bn6_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn5_bn6_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn5_bn6_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @bn5_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn5_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn5_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn5_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @act_bn6_bn7_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn6_bn7_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn6_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<240 x i8>>>
  llvm.mlir.global external @bn6_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<240 x i8>>>
  llvm.mlir.global external @bn6_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<240 x i8>>>
  llvm.mlir.global external @bn6_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<240 x i8>>>
  llvm.mlir.global external @act_bn7_bn8_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn7_bn8_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn7_bn8_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn7_bn8_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn7_bn8_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn7_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<200 x i8>>>
  llvm.mlir.global external @bn7_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<200 x i8>>>
  llvm.mlir.global external @bn7_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<200 x i8>>>
  llvm.mlir.global external @bn7_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<200 x i8>>>
  llvm.mlir.global external @act_bn8_bn9_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn8_bn9_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn8_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn8_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn8_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn8_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @act_bn9_bn10_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn9_bn10_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn9_act_1_2_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn9_act_1_2_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn9_act_1_2_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn9_act_2_3_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @act_out_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @act_out_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer2_layer3_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer2_layer3_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons_buff_5() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons_buff_4() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @OF_b11_skip_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @OF_b11_skip_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_cons_buff_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_cons_buff_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer2_layer3_buff_1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer2_layer3_buff_0() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @B_OF_b11_act_layer2_layer3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @B_OF_b11_act_layer1_layer2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<336 x i8>>>
  llvm.mlir.global external @OF_b11_skip_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @OF_b11_skip() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_0_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1_1_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_layer3_bn_11_layer1() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer2_layer3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @B_OF_b10_act_layer1_layer2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<480 x i8>>>
  llvm.mlir.global external @act_out_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @act_out() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<112 x i8>>>
  llvm.mlir.global external @bn9_act_2_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn9_act_1_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @act_bn9_bn10() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn8_act_2_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @bn8_act_1_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<184 x i8>>>
  llvm.mlir.global external @act_bn8_bn9() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn7_act_2_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<200 x i8>>>
  llvm.mlir.global external @bn7_act_1_2() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<200 x i8>>>
  llvm.mlir.global external @act_bn7_bn8_cons() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @act_bn7_bn8() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn6_act_2_3() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<240 x i8>>>
  llvm.mlir.global external @bn6_act_1_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<240 x i8>>>
  llvm.mlir.global external @act_bn6_bn7() {addr_space = 0 : i32} : !llvm.array<14 x array<1 x array<80 x i8>>>
  llvm.mlir.global external @bn5_act_2_3() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn5_act_1_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @act_bn5_bn6_cons() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @act_bn5_bn6() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @bn4_act_2_3() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @bn4_act_1_2() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<120 x i8>>>
  llvm.mlir.global external @act_bn4_bn5() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @bn3_act_2_3() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn3_act_1_2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @act_bn3_bn4() {addr_space = 0 : i32} : !llvm.array<28 x array<1 x array<40 x i8>>>
  llvm.mlir.global external @bn2_act_2_3() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @bn2_act_1_2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<72 x i8>>>
  llvm.mlir.global external @act_bn2_bn3() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @bn01_act_bn1_2_3() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @bn01_act_bn1_1_2() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @bn01_act_bn0_bn1() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @bn01_act_bn0_2_3() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @act_bn01_bn2() {addr_space = 0 : i32} : !llvm.array<56 x array<1 x array<24 x i8>>>
  llvm.mlir.global external @bn9_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<31096 x i8>
  llvm.mlir.global external @bn9_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<31096 x i8>
  llvm.mlir.global external @bn8_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<31096 x i8>
  llvm.mlir.global external @bn8_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<31096 x i8>
  llvm.mlir.global external @bn7_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<33800 x i8>
  llvm.mlir.global external @bn7_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<33800 x i8>
  llvm.mlir.global external @bn6_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<30960 x i8>
  llvm.mlir.global external @bn6_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<30960 x i8>
  llvm.mlir.global external @wts_OF_11_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<126952 x i8>
  llvm.mlir.global external @wts_OF_11_L3L2() {addr_space = 0 : i32} : !llvm.array<126952 x i8>
  llvm.mlir.global external @bn5_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<10680 x i8>
  llvm.mlir.global external @bn5_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<10680 x i8>
  llvm.mlir.global external @bn4_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<10680 x i8>
  llvm.mlir.global external @bn4_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<10680 x i8>
  llvm.mlir.global external @bn3_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<5256 x i8>
  llvm.mlir.global external @bn3_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<5256 x i8>
  llvm.mlir.global external @bn2_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<4104 x i8>
  llvm.mlir.global external @bn2_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<4104 x i8>
  llvm.mlir.global external @bn0_1_wts_OF_L2L1_cons() {addr_space = 0 : i32} : !llvm.array<3536 x i8>
  llvm.mlir.global external @bn0_1_wts_OF_L2L1() {addr_space = 0 : i32} : !llvm.array<3536 x i8>
  llvm.mlir.global external @wts_OF_01_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<34256 x i8>
  llvm.mlir.global external @wts_OF_01_L3L2() {addr_space = 0 : i32} : !llvm.array<34256 x i8>
  llvm.mlir.global external @act_in_cons() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @act_in() {addr_space = 0 : i32} : !llvm.array<112 x array<1 x array<16 x i8>>>
  llvm.mlir.global external @weightsInBN11_layer3_cons() {addr_space = 0 : i32} : !llvm.array<37632 x i8>
  llvm.mlir.global external @weightsInBN11_layer3() {addr_space = 0 : i32} : !llvm.array<37632 x i8>
  llvm.mlir.global external @weightsInBN11_layer2_cons() {addr_space = 0 : i32} : !llvm.array<3024 x i8>
  llvm.mlir.global external @weightsInBN11_layer2() {addr_space = 0 : i32} : !llvm.array<3024 x i8>
  llvm.mlir.global external @weightsInBN11_layer1_cons() {addr_space = 0 : i32} : !llvm.array<37632 x i8>
  llvm.mlir.global external @weightsInBN11_layer1() {addr_space = 0 : i32} : !llvm.array<37632 x i8>
  llvm.mlir.global external @wts_b11_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<78288 x i8>
  llvm.mlir.global external @wts_b11_L3L2() {addr_space = 0 : i32} : !llvm.array<78288 x i8>
  llvm.mlir.global external @weightsInBN10_layer3_cons() {addr_space = 0 : i32} : !llvm.array<53760 x i8>
  llvm.mlir.global external @weightsInBN10_layer3() {addr_space = 0 : i32} : !llvm.array<53760 x i8>
  llvm.mlir.global external @weightsInBN10_layer2_cons() {addr_space = 0 : i32} : !llvm.array<4320 x i8>
  llvm.mlir.global external @weightsInBN10_layer2() {addr_space = 0 : i32} : !llvm.array<4320 x i8>
  llvm.mlir.global external @weightsInBN10_layer1_cons() {addr_space = 0 : i32} : !llvm.array<38400 x i8>
  llvm.mlir.global external @weightsInBN10_layer1() {addr_space = 0 : i32} : !llvm.array<38400 x i8>
  llvm.mlir.global external @wts_b10_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<96480 x i8>
  llvm.mlir.global external @wts_b10_L3L2() {addr_space = 0 : i32} : !llvm.array<96480 x i8>
  llvm.func @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn0_conv2dk1_skip_ui8_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn1_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn1_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn2_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn2_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn2_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn2_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn3_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn3_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn3_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn3_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn4_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn4_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn4_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn4_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn5_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn5_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn5_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn5_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn6_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn6_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn6_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn6_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn7_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn7_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn7_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn7_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn8_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn8_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn8_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn8_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn9_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn9_conv2dk3_dw_stride2_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn9_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn9_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn10_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn10_conv2dk1_ui8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn11_conv2dk1_relu_i8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @bn11_conv2dk1_skip_ui8_i8_i8(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @sequence(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.return
  }
  llvm.func @core_3_2() {
    %0 = llvm.mlir.addressof @act_out_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @OF_b11_skip_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_OF_b11_act_layer2_layer3_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @weightsInBN11_layer3_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @act_out_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @OF_b11_skip_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(31 : index) : i64
    %7 = llvm.mlir.addressof @B_OF_b11_act_layer2_layer3_buff_0 : !llvm.ptr
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(4294967295 : index) : i64
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(52 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(36 : i32) : i32
    %14 = llvm.mlir.constant(53 : i32) : i32
    %15 = llvm.mlir.constant(50 : i32) : i32
    %16 = llvm.mlir.constant(37 : i32) : i32
    %17 = llvm.mlir.constant(49 : i32) : i32
    %18 = llvm.mlir.constant(1 : i32) : i32
    %19 = llvm.mlir.constant(12 : i32) : i32
    %20 = llvm.mlir.constant(112 : i32) : i32
    %21 = llvm.mlir.constant(336 : i32) : i32
    %22 = llvm.mlir.constant(14 : i32) : i32
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.mlir.constant(14 : index) : i64
    %25 = llvm.mlir.constant(-1 : i32) : i32
    %26 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%26 : i64)
  ^bb1(%27: i64):  // 2 preds: ^bb0, ^bb5
    %28 = llvm.icmp "slt" %27, %9 : i64
    llvm.cond_br %28, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%17, %25) : (i32, i32) -> ()
    llvm.br ^bb3(%26 : i64)
  ^bb3(%29: i64):  // 2 preds: ^bb2, ^bb4
    %30 = llvm.icmp "slt" %29, %24 : i64
    llvm.cond_br %30, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%16, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    %31 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %6  : i64
    %34 = llvm.icmp "eq" %33, %26 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %6  : i64
    %38 = llvm.icmp "eq" %37, %26 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    %39 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %6  : i64
    %42 = llvm.icmp "eq" %41, %26 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37632 x i8>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %6  : i64
    %46 = llvm.icmp "eq" %45, %26 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @bn11_conv2dk1_skip_ui8_i8_i8(%31, %43, %39, %35, %22, %21, %20, %19, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%16, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    %47 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.and %48, %6  : i64
    %50 = llvm.icmp "eq" %49, %26 : i64
    "llvm.intr.assume"(%50) : (i1) -> ()
    %51 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.and %52, %6  : i64
    %54 = llvm.icmp "eq" %53, %26 : i64
    "llvm.intr.assume"(%54) : (i1) -> ()
    %55 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %6  : i64
    %58 = llvm.icmp "eq" %57, %26 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @bn11_conv2dk1_skip_ui8_i8_i8(%47, %43, %55, %51, %22, %21, %20, %19, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%13, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %18) : (i32, i32) -> ()
    %59 = llvm.add %29, %23 : i64
    llvm.br ^bb3(%59 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%10, %18) : (i32, i32) -> ()
    %60 = llvm.add %27, %8 : i64
    llvm.br ^bb1(%60 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_3_3() {
    %0 = llvm.mlir.addressof @B_OF_b11_act_layer1_layer2_cons_buff_3 : !llvm.ptr
    %1 = llvm.mlir.addressof @B_OF_b11_act_layer1_layer2_cons_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_OF_b11_act_layer2_layer3_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @weightsInBN11_layer2_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_OF_b11_act_layer1_layer2_cons_buff_1 : !llvm.ptr
    %5 = llvm.mlir.addressof @B_OF_b11_act_layer1_layer2_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(31 : index) : i64
    %7 = llvm.mlir.addressof @B_OF_b11_act_layer2_layer3_buff_0 : !llvm.ptr
    %8 = llvm.mlir.constant(9223372036854775804 : index) : i64
    %9 = llvm.mlir.constant(48 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(53 : i32) : i32
    %12 = llvm.mlir.constant(52 : i32) : i32
    %13 = llvm.mlir.constant(51 : i32) : i32
    %14 = llvm.mlir.constant(49 : i32) : i32
    %15 = llvm.mlir.constant(2 : i32) : i32
    %16 = llvm.mlir.constant(12 : index) : i64
    %17 = llvm.mlir.constant(8 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(3 : i32) : i32
    %20 = llvm.mlir.constant(336 : i32) : i32
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(14 : i32) : i32
    %23 = llvm.mlir.constant(-2 : i32) : i32
    %24 = llvm.mlir.constant(-1 : i32) : i32
    %25 = llvm.mlir.constant(4 : index) : i64
    %26 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%26 : i64)
  ^bb1(%27: i64):  // 2 preds: ^bb0, ^bb14
    %28 = llvm.icmp "slt" %27, %8 : i64
    llvm.cond_br %28, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    %29 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.and %30, %6  : i64
    %32 = llvm.icmp "eq" %31, %26 : i64
    "llvm.intr.assume"(%32) : (i1) -> ()
    %33 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %6  : i64
    %36 = llvm.icmp "eq" %35, %26 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.and %38, %6  : i64
    %40 = llvm.icmp "eq" %39, %26 : i64
    "llvm.intr.assume"(%40) : (i1) -> ()
    %41 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3024 x i8>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %6  : i64
    %44 = llvm.icmp "eq" %43, %26 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %33, %37, %41, %29, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb3(%26 : i64)
  ^bb3(%45: i64):  // 2 preds: ^bb2, ^bb4
    %46 = llvm.icmp "slt" %45, %16 : i64
    llvm.cond_br %46, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    %47 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.and %48, %6  : i64
    %50 = llvm.icmp "eq" %49, %26 : i64
    "llvm.intr.assume"(%50) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    %51 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.and %52, %6  : i64
    %54 = llvm.icmp "eq" %53, %26 : i64
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %51, %41, %47, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    %55 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %6  : i64
    %58 = llvm.icmp "eq" %57, %26 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %51, %55, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%50) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%51, %55, %33, %41, %47, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%55, %33, %37, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %59 = llvm.add %45, %25 : i64
    llvm.br ^bb3(%59 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    %60 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.and %61, %6  : i64
    %63 = llvm.icmp "eq" %62, %26 : i64
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %37, %41, %60, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    %64 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.and %65, %6  : i64
    %67 = llvm.icmp "eq" %66, %26 : i64
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    %68 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.and %69, %6  : i64
    %71 = llvm.icmp "eq" %70, %26 : i64
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %64, %68, %41, %29, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb6(%26 : i64)
  ^bb6(%72: i64):  // 2 preds: ^bb5, ^bb7
    %73 = llvm.icmp "slt" %72, %16 : i64
    llvm.cond_br %73, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %33, %41, %60, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%68, %33, %37, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %64, %41, %60, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %64, %68, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %74 = llvm.add %72, %25 : i64
    llvm.br ^bb6(%74 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %68, %41, %60, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %33, %37, %41, %29, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb9(%26 : i64)
  ^bb9(%75: i64):  // 2 preds: ^bb8, ^bb10
    %76 = llvm.icmp "slt" %75, %16 : i64
    llvm.cond_br %76, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %64, %41, %60, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %64, %68, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %33, %41, %60, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%68, %33, %37, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %77 = llvm.add %75, %25 : i64
    llvm.br ^bb9(%77 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %37, %41, %60, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %64, %68, %41, %29, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb12(%26 : i64)
  ^bb12(%78: i64):  // 2 preds: ^bb11, ^bb13
    %79 = llvm.icmp "slt" %78, %16 : i64
    llvm.cond_br %79, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %33, %41, %60, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%68, %33, %37, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %64, %41, %60, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %64, %68, %41, %29, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %80 = llvm.add %78, %25 : i64
    llvm.br ^bb12(%80 : i64)
  ^bb14:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %68, %41, %60, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    %81 = llvm.add %27, %25 : i64
    llvm.br ^bb1(%81 : i64)
  ^bb15:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    %82 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.and %83, %6  : i64
    %85 = llvm.icmp "eq" %84, %26 : i64
    "llvm.intr.assume"(%85) : (i1) -> ()
    %86 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %6  : i64
    %89 = llvm.icmp "eq" %88, %26 : i64
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    %90 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %92 = llvm.and %91, %6  : i64
    %93 = llvm.icmp "eq" %92, %26 : i64
    "llvm.intr.assume"(%93) : (i1) -> ()
    %94 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3024 x i8>
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.and %95, %6  : i64
    %97 = llvm.icmp "eq" %96, %26 : i64
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %86, %90, %94, %82, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb16(%26 : i64)
  ^bb16(%98: i64):  // 2 preds: ^bb15, ^bb17
    %99 = llvm.icmp "slt" %98, %16 : i64
    llvm.cond_br %99, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    %100 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.and %101, %6  : i64
    %103 = llvm.icmp "eq" %102, %26 : i64
    "llvm.intr.assume"(%103) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    %104 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %105 = llvm.ptrtoint %104 : !llvm.ptr to i64
    %106 = llvm.and %105, %6  : i64
    %107 = llvm.icmp "eq" %106, %26 : i64
    "llvm.intr.assume"(%107) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %104, %94, %100, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%107) : (i1) -> ()
    %108 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.and %109, %6  : i64
    %111 = llvm.icmp "eq" %110, %26 : i64
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%90, %104, %108, %94, %82, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%103) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%107) : (i1) -> ()
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%104, %108, %86, %94, %100, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%108, %86, %90, %94, %82, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %112 = llvm.add %98, %25 : i64
    llvm.br ^bb16(%112 : i64)
  ^bb18:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    %113 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.and %114, %6  : i64
    %116 = llvm.icmp "eq" %115, %26 : i64
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %90, %94, %113, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    %117 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.and %118, %6  : i64
    %120 = llvm.icmp "eq" %119, %26 : i64
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    %121 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %122 = llvm.ptrtoint %121 : !llvm.ptr to i64
    %123 = llvm.and %122, %6  : i64
    %124 = llvm.icmp "eq" %123, %26 : i64
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %117, %121, %94, %82, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb19(%26 : i64)
  ^bb19(%125: i64):  // 2 preds: ^bb18, ^bb20
    %126 = llvm.icmp "slt" %125, %16 : i64
    llvm.cond_br %126, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %121, %86, %94, %113, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%121, %86, %90, %94, %82, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %117, %94, %113, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%90, %117, %121, %94, %82, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %127 = llvm.add %125, %25 : i64
    llvm.br ^bb19(%127 : i64)
  ^bb21:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %121, %121, %94, %113, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %23) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %86, %90, %94, %82, %22, %21, %20, %19, %19, %18, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.br ^bb22(%26 : i64)
  ^bb22(%128: i64):  // 2 preds: ^bb21, ^bb23
    %129 = llvm.icmp "slt" %128, %16 : i64
    llvm.cond_br %129, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %117, %94, %113, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%90, %117, %121, %94, %82, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %121, %86, %94, %113, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%121, %86, %90, %94, %82, %22, %21, %20, %19, %19, %21, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    %130 = llvm.add %128, %25 : i64
    llvm.br ^bb22(%130 : i64)
  ^bb24:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%12, %24) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %90, %94, %113, %22, %21, %20, %19, %19, %15, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %21) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %21) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_3_4() {
    %0 = llvm.mlir.addressof @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @B_OF_b11_act_layer1_layer2_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @weightsInBN11_layer1_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(31 : index) : i64
    %5 = llvm.mlir.addressof @B_OF_b11_act_layer1_layer2_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(53 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(52 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(9 : i32) : i32
    %16 = llvm.mlir.constant(336 : i32) : i32
    %17 = llvm.mlir.constant(112 : i32) : i32
    %18 = llvm.mlir.constant(14 : i32) : i32
    %19 = llvm.mlir.constant(14 : index) : i64
    %20 = llvm.mlir.constant(-1 : i32) : i32
    %21 = llvm.mlir.constant(2 : index) : i64
    %22 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%22 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb5
    %24 = llvm.icmp "slt" %23, %6 : i64
    llvm.cond_br %24, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%13, %20) : (i32, i32) -> ()
    llvm.br ^bb3(%22 : i64)
  ^bb3(%25: i64):  // 2 preds: ^bb2, ^bb4
    %26 = llvm.icmp "slt" %25, %19 : i64
    llvm.cond_br %26, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%12, %20) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %20) : (i32, i32) -> ()
    %27 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.and %28, %4  : i64
    %30 = llvm.icmp "eq" %29, %22 : i64
    "llvm.intr.assume"(%30) : (i1) -> ()
    %31 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %4  : i64
    %34 = llvm.icmp "eq" %33, %22 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37632 x i8>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %4  : i64
    %38 = llvm.icmp "eq" %37, %22 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @bn11_conv2dk1_relu_i8_ui8(%31, %35, %27, %18, %17, %16, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %20) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %20) : (i32, i32) -> ()
    %39 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<336 x i8>>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %4  : i64
    %42 = llvm.icmp "eq" %41, %22 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %4  : i64
    %46 = llvm.icmp "eq" %45, %22 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @bn11_conv2dk1_relu_i8_ui8(%43, %35, %39, %18, %17, %16, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %47 = llvm.add %25, %21 : i64
    llvm.br ^bb3(%47 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    %48 = llvm.add %23, %7 : i64
    llvm.br ^bb1(%48 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_3_5() {
    %0 = llvm.mlir.addressof @B_OF_b10_act_layer2_layer3_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @B_OF_b10_layer3_bn_11_layer1_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @weightsInBN10_layer3_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @B_OF_b10_act_layer2_layer3_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(31 : index) : i64
    %5 = llvm.mlir.addressof @B_OF_b10_layer3_bn_11_layer1_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(4294967295 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(20 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(21 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(10 : i32) : i32
    %16 = llvm.mlir.constant(112 : i32) : i32
    %17 = llvm.mlir.constant(480 : i32) : i32
    %18 = llvm.mlir.constant(14 : i32) : i32
    %19 = llvm.mlir.constant(14 : index) : i64
    %20 = llvm.mlir.constant(-1 : i32) : i32
    %21 = llvm.mlir.constant(2 : index) : i64
    %22 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%22 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb5
    %24 = llvm.icmp "slt" %23, %6 : i64
    llvm.cond_br %24, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%13, %20) : (i32, i32) -> ()
    llvm.br ^bb3(%22 : i64)
  ^bb3(%25: i64):  // 2 preds: ^bb2, ^bb4
    %26 = llvm.icmp "slt" %25, %19 : i64
    llvm.cond_br %26, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%12, %20) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %20) : (i32, i32) -> ()
    %27 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.and %28, %4  : i64
    %30 = llvm.icmp "eq" %29, %22 : i64
    "llvm.intr.assume"(%30) : (i1) -> ()
    %31 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %4  : i64
    %34 = llvm.icmp "eq" %33, %22 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53760 x i8>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %4  : i64
    %38 = llvm.icmp "eq" %37, %22 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @bn10_conv2dk1_ui8_i8(%31, %35, %27, %18, %17, %16, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %20) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %20) : (i32, i32) -> ()
    %39 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<112 x i8>>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %4  : i64
    %42 = llvm.icmp "eq" %41, %22 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %4  : i64
    %46 = llvm.icmp "eq" %45, %22 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @bn10_conv2dk1_ui8_i8(%43, %35, %39, %18, %17, %16, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %47 = llvm.add %25, %21 : i64
    llvm.br ^bb3(%47 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    %48 = llvm.add %23, %7 : i64
    llvm.br ^bb1(%48 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_2_5() {
    %0 = llvm.mlir.addressof @B_OF_b10_act_layer1_layer2_cons_buff_3 : !llvm.ptr
    %1 = llvm.mlir.addressof @B_OF_b10_act_layer1_layer2_cons_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_OF_b10_act_layer2_layer3_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @weightsInBN10_layer2_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @B_OF_b10_act_layer1_layer2_cons_buff_1 : !llvm.ptr
    %5 = llvm.mlir.addressof @B_OF_b10_act_layer1_layer2_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(31 : index) : i64
    %7 = llvm.mlir.addressof @B_OF_b10_act_layer2_layer3_buff_0 : !llvm.ptr
    %8 = llvm.mlir.constant(4 : index) : i64
    %9 = llvm.mlir.constant(9223372036854775804 : index) : i64
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(53 : i32) : i32
    %13 = llvm.mlir.constant(52 : i32) : i32
    %14 = llvm.mlir.constant(51 : i32) : i32
    %15 = llvm.mlir.constant(49 : i32) : i32
    %16 = llvm.mlir.constant(2 : i32) : i32
    %17 = llvm.mlir.constant(12 : index) : i64
    %18 = llvm.mlir.constant(7 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(3 : i32) : i32
    %21 = llvm.mlir.constant(480 : i32) : i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.mlir.constant(14 : i32) : i32
    %24 = llvm.mlir.constant(-2 : i32) : i32
    %25 = llvm.mlir.constant(-1 : i32) : i32
    %26 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%26 : i64)
  ^bb1(%27: i64):  // 2 preds: ^bb0, ^bb14
    %28 = llvm.icmp "slt" %27, %9 : i64
    llvm.cond_br %28, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    %29 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.and %30, %6  : i64
    %32 = llvm.icmp "eq" %31, %26 : i64
    "llvm.intr.assume"(%32) : (i1) -> ()
    %33 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %6  : i64
    %36 = llvm.icmp "eq" %35, %26 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.and %38, %6  : i64
    %40 = llvm.icmp "eq" %39, %26 : i64
    "llvm.intr.assume"(%40) : (i1) -> ()
    %41 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4320 x i8>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %6  : i64
    %44 = llvm.icmp "eq" %43, %26 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %33, %37, %41, %29, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb3(%26 : i64)
  ^bb3(%45: i64):  // 2 preds: ^bb2, ^bb4
    %46 = llvm.icmp "slt" %45, %17 : i64
    llvm.cond_br %46, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    %47 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.and %48, %6  : i64
    %50 = llvm.icmp "eq" %49, %26 : i64
    "llvm.intr.assume"(%50) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    %51 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.and %52, %6  : i64
    %54 = llvm.icmp "eq" %53, %26 : i64
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %51, %41, %47, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    %55 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %6  : i64
    %58 = llvm.icmp "eq" %57, %26 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %51, %55, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%50) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%51, %55, %33, %41, %47, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%55, %33, %37, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %59 = llvm.add %45, %8 : i64
    llvm.br ^bb3(%59 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    %60 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.and %61, %6  : i64
    %63 = llvm.icmp "eq" %62, %26 : i64
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %37, %41, %60, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    %64 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.and %65, %6  : i64
    %67 = llvm.icmp "eq" %66, %26 : i64
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    %68 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.and %69, %6  : i64
    %71 = llvm.icmp "eq" %70, %26 : i64
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %64, %68, %41, %29, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb6(%26 : i64)
  ^bb6(%72: i64):  // 2 preds: ^bb5, ^bb7
    %73 = llvm.icmp "slt" %72, %17 : i64
    llvm.cond_br %73, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %33, %41, %60, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%68, %33, %37, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %64, %41, %60, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %64, %68, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %74 = llvm.add %72, %8 : i64
    llvm.br ^bb6(%74 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %68, %41, %60, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %33, %37, %41, %29, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb9(%26 : i64)
  ^bb9(%75: i64):  // 2 preds: ^bb8, ^bb10
    %76 = llvm.icmp "slt" %75, %17 : i64
    llvm.cond_br %76, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %64, %41, %60, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %64, %68, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %33, %41, %60, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%68, %33, %37, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %77 = llvm.add %75, %8 : i64
    llvm.br ^bb9(%77 : i64)
  ^bb11:  // pred: ^bb9
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %37, %41, %60, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %64, %68, %41, %29, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb12(%26 : i64)
  ^bb12(%78: i64):  // 2 preds: ^bb11, ^bb13
    %79 = llvm.icmp "slt" %78, %17 : i64
    llvm.cond_br %79, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %33, %41, %60, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%68, %33, %37, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%33, %37, %64, %41, %60, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%32) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%37, %64, %68, %41, %29, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %80 = llvm.add %78, %8 : i64
    llvm.br ^bb12(%80 : i64)
  ^bb14:  // pred: ^bb12
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%44) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%64, %68, %68, %41, %60, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    %81 = llvm.add %27, %8 : i64
    llvm.br ^bb1(%81 : i64)
  ^bb15:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    %82 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.and %83, %6  : i64
    %85 = llvm.icmp "eq" %84, %26 : i64
    "llvm.intr.assume"(%85) : (i1) -> ()
    %86 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %6  : i64
    %89 = llvm.icmp "eq" %88, %26 : i64
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    %90 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %92 = llvm.and %91, %6  : i64
    %93 = llvm.icmp "eq" %92, %26 : i64
    "llvm.intr.assume"(%93) : (i1) -> ()
    %94 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4320 x i8>
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.and %95, %6  : i64
    %97 = llvm.icmp "eq" %96, %26 : i64
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %86, %90, %94, %82, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb16(%26 : i64)
  ^bb16(%98: i64):  // 2 preds: ^bb15, ^bb17
    %99 = llvm.icmp "slt" %98, %17 : i64
    llvm.cond_br %99, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    %100 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.and %101, %6  : i64
    %103 = llvm.icmp "eq" %102, %26 : i64
    "llvm.intr.assume"(%103) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    %104 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %105 = llvm.ptrtoint %104 : !llvm.ptr to i64
    %106 = llvm.and %105, %6  : i64
    %107 = llvm.icmp "eq" %106, %26 : i64
    "llvm.intr.assume"(%107) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %104, %94, %100, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%107) : (i1) -> ()
    %108 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.and %109, %6  : i64
    %111 = llvm.icmp "eq" %110, %26 : i64
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%90, %104, %108, %94, %82, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%103) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%107) : (i1) -> ()
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%104, %108, %86, %94, %100, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%108, %86, %90, %94, %82, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %112 = llvm.add %98, %8 : i64
    llvm.br ^bb16(%112 : i64)
  ^bb18:  // pred: ^bb16
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    %113 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.and %114, %6  : i64
    %116 = llvm.icmp "eq" %115, %26 : i64
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %90, %94, %113, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    %117 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.and %118, %6  : i64
    %120 = llvm.icmp "eq" %119, %26 : i64
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    %121 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %122 = llvm.ptrtoint %121 : !llvm.ptr to i64
    %123 = llvm.and %122, %6  : i64
    %124 = llvm.icmp "eq" %123, %26 : i64
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %117, %121, %94, %82, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb19(%26 : i64)
  ^bb19(%125: i64):  // 2 preds: ^bb18, ^bb20
    %126 = llvm.icmp "slt" %125, %17 : i64
    llvm.cond_br %126, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %121, %86, %94, %113, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%121, %86, %90, %94, %82, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %117, %94, %113, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%90, %117, %121, %94, %82, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %127 = llvm.add %125, %8 : i64
    llvm.br ^bb19(%127 : i64)
  ^bb21:  // pred: ^bb19
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %121, %121, %94, %113, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%15, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %24) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %86, %90, %94, %82, %23, %22, %21, %20, %20, %19, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.br ^bb22(%26 : i64)
  ^bb22(%128: i64):  // 2 preds: ^bb21, ^bb23
    %129 = llvm.icmp "slt" %128, %17 : i64
    llvm.cond_br %129, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %117, %94, %113, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%90, %117, %121, %94, %82, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%120) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%117, %121, %86, %94, %113, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %25) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%124) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%121, %86, %90, %94, %82, %23, %22, %21, %20, %20, %22, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    %130 = llvm.add %128, %8 : i64
    llvm.br ^bb22(%130 : i64)
  ^bb24:  // pred: ^bb22
    llvm.call @llvm.aie2.acquire(%13, %25) : (i32, i32) -> ()
    "llvm.intr.assume"(%116) : (i1) -> ()
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%86, %90, %90, %94, %113, %23, %22, %21, %20, %20, %16, %18, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%12, %22) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %22) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_2_4() {
    %0 = llvm.mlir.addressof @act_bn9_bn10_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @B_OF_b10_act_layer1_layer2_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @weightsInBN10_layer1_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @act_bn9_bn10_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(31 : index) : i64
    %5 = llvm.mlir.addressof @B_OF_b10_act_layer1_layer2_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(3 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(8 : i32) : i32
    %16 = llvm.mlir.constant(480 : i32) : i32
    %17 = llvm.mlir.constant(80 : i32) : i32
    %18 = llvm.mlir.constant(14 : i32) : i32
    %19 = llvm.mlir.constant(14 : index) : i64
    %20 = llvm.mlir.constant(-1 : i32) : i32
    %21 = llvm.mlir.constant(2 : index) : i64
    %22 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%22 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb5
    %24 = llvm.icmp "slt" %23, %6 : i64
    llvm.cond_br %24, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%13, %20) : (i32, i32) -> ()
    llvm.br ^bb3(%22 : i64)
  ^bb3(%25: i64):  // 2 preds: ^bb2, ^bb4
    %26 = llvm.icmp "slt" %25, %19 : i64
    llvm.cond_br %26, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%12, %20) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %20) : (i32, i32) -> ()
    %27 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.and %28, %4  : i64
    %30 = llvm.icmp "eq" %29, %22 : i64
    "llvm.intr.assume"(%30) : (i1) -> ()
    %31 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.and %32, %4  : i64
    %34 = llvm.icmp "eq" %33, %22 : i64
    "llvm.intr.assume"(%34) : (i1) -> ()
    %35 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38400 x i8>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %4  : i64
    %38 = llvm.icmp "eq" %37, %22 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @bn10_conv2dk1_relu_i8_ui8(%31, %35, %27, %18, %17, %16, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %20) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %20) : (i32, i32) -> ()
    %39 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<480 x i8>>>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %4  : i64
    %42 = llvm.icmp "eq" %41, %22 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %4  : i64
    %46 = llvm.icmp "eq" %45, %22 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    "llvm.intr.assume"(%38) : (i1) -> ()
    llvm.call @bn10_conv2dk1_relu_i8_ui8(%43, %35, %39, %18, %17, %16, %15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %47 = llvm.add %25, %21 : i64
    llvm.br ^bb3(%47 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    %48 = llvm.add %23, %7 : i64
    llvm.br ^bb1(%48 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_2_3() {
    %0 = llvm.mlir.addressof @act_bn9_bn10_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @bn9_act_1_2_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @act_bn9_bn10_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @bn9_act_2_3_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @act_bn8_bn9_buff_1 : !llvm.ptr
    %5 = llvm.mlir.addressof @bn9_act_1_2_buff_1 : !llvm.ptr
    %6 = llvm.mlir.addressof @act_bn8_bn9_buff_0 : !llvm.ptr
    %7 = llvm.mlir.addressof @bn9_act_1_2_buff_0 : !llvm.ptr
    %8 = llvm.mlir.addressof @rtp23 : !llvm.ptr
    %9 = llvm.mlir.constant(31 : index) : i64
    %10 = llvm.mlir.addressof @bn9_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %11 = llvm.mlir.constant(49 : i32) : i32
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(6 : index) : i64
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(48 : i32) : i32
    %16 = llvm.mlir.constant(51 : i32) : i32
    %17 = llvm.mlir.constant(4 : i32) : i32
    %18 = llvm.mlir.constant(50 : i32) : i32
    %19 = llvm.mlir.constant(55 : i32) : i32
    %20 = llvm.mlir.constant(54 : i32) : i32
    %21 = llvm.mlir.constant(53 : i32) : i32
    %22 = llvm.mlir.constant(52 : i32) : i32
    %23 = llvm.mlir.constant(5 : i32) : i32
    %24 = llvm.mlir.constant(12 : index) : i64
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(3 : i32) : i32
    %27 = llvm.mlir.constant(1 : i32) : i32
    %28 = llvm.mlir.constant(2 : i32) : i32
    %29 = llvm.mlir.constant(184 : i32) : i32
    %30 = llvm.mlir.constant(80 : i32) : i32
    %31 = llvm.mlir.constant(14 : i32) : i32
    %32 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %33 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31096 x i8>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %9  : i64
    %36 = llvm.icmp "eq" %35, %12 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.getelementptr %33[14720] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%36) : (i1) -> ()
    %38 = llvm.getelementptr %33[16376] : (!llvm.ptr) -> !llvm.ptr, i8
    %39 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %9  : i64
    %42 = llvm.icmp "eq" %41, %12 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.load %39 : !llvm.ptr -> i32
    "llvm.intr.assume"(%42) : (i1) -> ()
    %44 = llvm.getelementptr %39[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %45 = llvm.load %44 : !llvm.ptr -> i32
    "llvm.intr.assume"(%42) : (i1) -> ()
    %46 = llvm.getelementptr %39[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %47 = llvm.load %46 : !llvm.ptr -> i32
    "llvm.intr.assume"(%42) : (i1) -> ()
    %48 = llvm.getelementptr %39[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.load %48 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%23, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %32) : (i32, i32) -> ()
    %50 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.and %51, %9  : i64
    %53 = llvm.icmp "eq" %52, %12 : i64
    "llvm.intr.assume"(%53) : (i1) -> ()
    %54 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %9  : i64
    %57 = llvm.icmp "eq" %56, %12 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%54, %33, %50, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %58 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %9  : i64
    %61 = llvm.icmp "eq" %60, %12 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    %62 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %9  : i64
    %65 = llvm.icmp "eq" %64, %12 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%62, %33, %58, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    %66 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %9  : i64
    %69 = llvm.icmp "eq" %68, %12 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %50, %58, %37, %66, %31, %27, %29, %26, %26, %25, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.and %71, %9  : i64
    %73 = llvm.icmp "eq" %72, %12 : i64
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.br ^bb1(%12 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb2
    %75 = llvm.icmp "slt" %74, %24 : i64
    llvm.cond_br %75, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    %76 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.and %77, %9  : i64
    %79 = llvm.icmp "eq" %78, %12 : i64
    "llvm.intr.assume"(%79) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%54, %33, %76, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %58, %76, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %80 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.and %81, %9  : i64
    %83 = llvm.icmp "eq" %82, %12 : i64
    "llvm.intr.assume"(%83) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %80, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%62, %33, %50, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%58, %76, %50, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%54, %33, %58, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%76, %50, %58, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%83) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %80, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%62, %33, %76, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %58, %76, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%54, %33, %50, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%58, %76, %50, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%83) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %80, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_relu_i8_ui8(%62, %33, %58, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%76, %50, %58, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    %84 = llvm.add %74, %13 : i64
    llvm.br ^bb1(%84 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %58, %58, %37, %66, %31, %27, %29, %26, %26, %28, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %85 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.and %86, %9  : i64
    %88 = llvm.icmp "eq" %87, %12 : i64
    "llvm.intr.assume"(%88) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn9_conv2dk1_skip_ui8_i8_i8(%66, %38, %85, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%15, %27) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_2_2() {
    %0 = llvm.mlir.addressof @act_bn8_bn9_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn7_bn8_cons_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @bn8_act_1_2_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @act_bn8_bn9_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @bn8_act_2_3_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @act_bn7_bn8_cons_buff_1 : !llvm.ptr
    %6 = llvm.mlir.addressof @bn8_act_1_2_buff_1 : !llvm.ptr
    %7 = llvm.mlir.addressof @act_bn7_bn8_cons_buff_0 : !llvm.ptr
    %8 = llvm.mlir.addressof @bn8_act_1_2_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @rtp22 : !llvm.ptr
    %10 = llvm.mlir.constant(31 : index) : i64
    %11 = llvm.mlir.addressof @bn8_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(48 : i32) : i32
    %16 = llvm.mlir.constant(53 : i32) : i32
    %17 = llvm.mlir.constant(52 : i32) : i32
    %18 = llvm.mlir.constant(57 : i32) : i32
    %19 = llvm.mlir.constant(56 : i32) : i32
    %20 = llvm.mlir.constant(50 : i32) : i32
    %21 = llvm.mlir.constant(55 : i32) : i32
    %22 = llvm.mlir.constant(54 : i32) : i32
    %23 = llvm.mlir.constant(51 : i32) : i32
    %24 = llvm.mlir.constant(6 : index) : i64
    %25 = llvm.mlir.constant(12 : index) : i64
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(3 : i32) : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.mlir.constant(2 : i32) : i32
    %30 = llvm.mlir.constant(184 : i32) : i32
    %31 = llvm.mlir.constant(80 : i32) : i32
    %32 = llvm.mlir.constant(14 : i32) : i32
    %33 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %34 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31096 x i8>
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.and %35, %10  : i64
    %37 = llvm.icmp "eq" %36, %13 : i64
    "llvm.intr.assume"(%37) : (i1) -> ()
    "llvm.intr.assume"(%37) : (i1) -> ()
    %38 = llvm.getelementptr %34[14720] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%37) : (i1) -> ()
    %39 = llvm.getelementptr %34[16376] : (!llvm.ptr) -> !llvm.ptr, i8
    %40 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.and %41, %10  : i64
    %43 = llvm.icmp "eq" %42, %13 : i64
    "llvm.intr.assume"(%43) : (i1) -> ()
    %44 = llvm.load %40 : !llvm.ptr -> i32
    "llvm.intr.assume"(%43) : (i1) -> ()
    %45 = llvm.getelementptr %40[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %46 = llvm.load %45 : !llvm.ptr -> i32
    "llvm.intr.assume"(%43) : (i1) -> ()
    %47 = llvm.getelementptr %40[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %48 = llvm.load %47 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%23, %33) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %33) : (i32, i32) -> ()
    %49 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.and %50, %10  : i64
    %52 = llvm.icmp "eq" %51, %13 : i64
    "llvm.intr.assume"(%52) : (i1) -> ()
    %53 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.and %54, %10  : i64
    %56 = llvm.icmp "eq" %55, %13 : i64
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %57 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.and %58, %10  : i64
    %60 = llvm.icmp "eq" %59, %13 : i64
    "llvm.intr.assume"(%60) : (i1) -> ()
    %61 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %10  : i64
    %64 = llvm.icmp "eq" %63, %13 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%61, %34, %57, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %33) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    %65 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.and %66, %10  : i64
    %68 = llvm.icmp "eq" %67, %13 : i64
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %49, %57, %38, %65, %32, %28, %30, %27, %27, %26, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    %69 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.and %70, %10  : i64
    %72 = llvm.icmp "eq" %71, %13 : i64
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.br ^bb1(%13 : i64)
  ^bb1(%73: i64):  // 2 preds: ^bb0, ^bb2
    %74 = llvm.icmp "slt" %73, %25 : i64
    llvm.cond_br %74, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    %75 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<184 x i8>>>
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.and %76, %10  : i64
    %78 = llvm.icmp "eq" %77, %13 : i64
    "llvm.intr.assume"(%78) : (i1) -> ()
    %79 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
    %81 = llvm.and %80, %10  : i64
    %82 = llvm.icmp "eq" %81, %13 : i64
    "llvm.intr.assume"(%82) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%79, %34, %75, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %57, %75, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    %83 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.and %84, %10  : i64
    %86 = llvm.icmp "eq" %85, %13 : i64
    "llvm.intr.assume"(%86) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %83, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%57, %75, %49, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%61, %34, %57, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%75, %49, %57, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%86) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %83, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    "llvm.intr.assume"(%82) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%79, %34, %75, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %57, %75, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%57, %75, %49, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%86) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %83, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @bn8_conv2dk1_relu_i8_ui8(%61, %34, %57, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%75, %49, %57, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    %87 = llvm.add %73, %24 : i64
    llvm.br ^bb1(%87 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    llvm.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %57, %57, %38, %65, %32, %28, %30, %27, %27, %29, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    %88 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
    %90 = llvm.and %89, %10  : i64
    %91 = llvm.icmp "eq" %90, %13 : i64
    "llvm.intr.assume"(%91) : (i1) -> ()
    llvm.call @bn8_conv2dk1_ui8_i8(%65, %39, %88, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%15, %28) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_1_3() {
    %0 = llvm.mlir.addressof @act_bn7_bn8_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @bn7_act_1_2_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @act_bn7_bn8_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @bn7_act_2_3_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @act_bn6_bn7_buff_1 : !llvm.ptr
    %5 = llvm.mlir.addressof @bn7_act_1_2_buff_1 : !llvm.ptr
    %6 = llvm.mlir.addressof @act_bn6_bn7_buff_0 : !llvm.ptr
    %7 = llvm.mlir.addressof @bn7_act_1_2_buff_0 : !llvm.ptr
    %8 = llvm.mlir.addressof @rtp13 : !llvm.ptr
    %9 = llvm.mlir.constant(31 : index) : i64
    %10 = llvm.mlir.addressof @bn7_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %11 = llvm.mlir.constant(49 : i32) : i32
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(6 : index) : i64
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(48 : i32) : i32
    %16 = llvm.mlir.constant(51 : i32) : i32
    %17 = llvm.mlir.constant(4 : i32) : i32
    %18 = llvm.mlir.constant(50 : i32) : i32
    %19 = llvm.mlir.constant(55 : i32) : i32
    %20 = llvm.mlir.constant(54 : i32) : i32
    %21 = llvm.mlir.constant(53 : i32) : i32
    %22 = llvm.mlir.constant(52 : i32) : i32
    %23 = llvm.mlir.constant(5 : i32) : i32
    %24 = llvm.mlir.constant(12 : index) : i64
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(3 : i32) : i32
    %27 = llvm.mlir.constant(1 : i32) : i32
    %28 = llvm.mlir.constant(2 : i32) : i32
    %29 = llvm.mlir.constant(200 : i32) : i32
    %30 = llvm.mlir.constant(80 : i32) : i32
    %31 = llvm.mlir.constant(14 : i32) : i32
    %32 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %33 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<33800 x i8>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %9  : i64
    %36 = llvm.icmp "eq" %35, %12 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.getelementptr %33[16000] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%36) : (i1) -> ()
    %38 = llvm.getelementptr %33[17800] : (!llvm.ptr) -> !llvm.ptr, i8
    %39 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.and %40, %9  : i64
    %42 = llvm.icmp "eq" %41, %12 : i64
    "llvm.intr.assume"(%42) : (i1) -> ()
    %43 = llvm.load %39 : !llvm.ptr -> i32
    "llvm.intr.assume"(%42) : (i1) -> ()
    %44 = llvm.getelementptr %39[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %45 = llvm.load %44 : !llvm.ptr -> i32
    "llvm.intr.assume"(%42) : (i1) -> ()
    %46 = llvm.getelementptr %39[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %47 = llvm.load %46 : !llvm.ptr -> i32
    "llvm.intr.assume"(%42) : (i1) -> ()
    %48 = llvm.getelementptr %39[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.load %48 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%23, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %32) : (i32, i32) -> ()
    %50 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<200 x i8>>>
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.and %51, %9  : i64
    %53 = llvm.icmp "eq" %52, %12 : i64
    "llvm.intr.assume"(%53) : (i1) -> ()
    %54 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %9  : i64
    %57 = llvm.icmp "eq" %56, %12 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%54, %33, %50, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %58 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<200 x i8>>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %9  : i64
    %61 = llvm.icmp "eq" %60, %12 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    %62 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %9  : i64
    %65 = llvm.icmp "eq" %64, %12 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%62, %33, %58, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    %66 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<200 x i8>>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %9  : i64
    %69 = llvm.icmp "eq" %68, %12 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %50, %58, %37, %66, %31, %27, %29, %26, %26, %25, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.and %71, %9  : i64
    %73 = llvm.icmp "eq" %72, %12 : i64
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.br ^bb1(%12 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb2
    %75 = llvm.icmp "slt" %74, %24 : i64
    llvm.cond_br %75, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    %76 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<200 x i8>>>
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.and %77, %9  : i64
    %79 = llvm.icmp "eq" %78, %12 : i64
    "llvm.intr.assume"(%79) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%54, %33, %76, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %58, %76, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %80 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.and %81, %9  : i64
    %83 = llvm.icmp "eq" %82, %12 : i64
    "llvm.intr.assume"(%83) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %80, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%62, %33, %50, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%58, %76, %50, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%54, %33, %58, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%76, %50, %58, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%83) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %80, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%62, %33, %76, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %58, %76, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%54, %33, %50, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%58, %76, %50, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%83) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %80, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_relu_i8_ui8(%62, %33, %58, %31, %30, %29, %43) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%76, %50, %58, %37, %66, %31, %27, %29, %26, %26, %27, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %70, %54, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    %84 = llvm.add %74, %13 : i64
    llvm.br ^bb1(%84 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%20, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%50, %58, %58, %37, %66, %31, %27, %29, %26, %26, %28, %45, %25) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %85 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.and %86, %9  : i64
    %88 = llvm.icmp "eq" %87, %12 : i64
    "llvm.intr.assume"(%88) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn7_conv2dk1_skip_ui8_i8_i8(%66, %38, %85, %62, %31, %29, %30, %47, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %27) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%15, %27) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_1_2() {
    %0 = llvm.mlir.addressof @act_bn6_bn7_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn5_bn6_cons_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @bn6_act_1_2_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @act_bn6_bn7_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @bn6_act_2_3_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @act_bn5_bn6_cons_buff_1 : !llvm.ptr
    %6 = llvm.mlir.addressof @bn6_act_1_2_buff_1 : !llvm.ptr
    %7 = llvm.mlir.addressof @act_bn5_bn6_cons_buff_0 : !llvm.ptr
    %8 = llvm.mlir.addressof @bn6_act_1_2_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @rtp12 : !llvm.ptr
    %10 = llvm.mlir.constant(31 : index) : i64
    %11 = llvm.mlir.addressof @bn6_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(48 : i32) : i32
    %16 = llvm.mlir.constant(53 : i32) : i32
    %17 = llvm.mlir.constant(52 : i32) : i32
    %18 = llvm.mlir.constant(57 : i32) : i32
    %19 = llvm.mlir.constant(56 : i32) : i32
    %20 = llvm.mlir.constant(50 : i32) : i32
    %21 = llvm.mlir.constant(55 : i32) : i32
    %22 = llvm.mlir.constant(54 : i32) : i32
    %23 = llvm.mlir.constant(51 : i32) : i32
    %24 = llvm.mlir.constant(6 : index) : i64
    %25 = llvm.mlir.constant(12 : index) : i64
    %26 = llvm.mlir.constant(80 : i32) : i32
    %27 = llvm.mlir.constant(14 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(3 : i32) : i32
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.mlir.constant(2 : i32) : i32
    %32 = llvm.mlir.constant(240 : i32) : i32
    %33 = llvm.mlir.constant(40 : i32) : i32
    %34 = llvm.mlir.constant(28 : i32) : i32
    %35 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %36 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<30960 x i8>
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.and %37, %10  : i64
    %39 = llvm.icmp "eq" %38, %13 : i64
    "llvm.intr.assume"(%39) : (i1) -> ()
    "llvm.intr.assume"(%39) : (i1) -> ()
    %40 = llvm.getelementptr %36[9600] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%39) : (i1) -> ()
    %41 = llvm.getelementptr %36[11760] : (!llvm.ptr) -> !llvm.ptr, i8
    %42 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.and %43, %10  : i64
    %45 = llvm.icmp "eq" %44, %13 : i64
    "llvm.intr.assume"(%45) : (i1) -> ()
    %46 = llvm.load %42 : !llvm.ptr -> i32
    "llvm.intr.assume"(%45) : (i1) -> ()
    %47 = llvm.getelementptr %42[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %48 = llvm.load %47 : !llvm.ptr -> i32
    "llvm.intr.assume"(%45) : (i1) -> ()
    %49 = llvm.getelementptr %42[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %50 = llvm.load %49 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    %51 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<240 x i8>>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.and %52, %10  : i64
    %54 = llvm.icmp "eq" %53, %13 : i64
    "llvm.intr.assume"(%54) : (i1) -> ()
    %55 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %10  : i64
    %58 = llvm.icmp "eq" %57, %13 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%55, %36, %51, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %59 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<240 x i8>>>
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.and %60, %10  : i64
    %62 = llvm.icmp "eq" %61, %13 : i64
    "llvm.intr.assume"(%62) : (i1) -> ()
    %63 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.and %64, %10  : i64
    %66 = llvm.icmp "eq" %65, %13 : i64
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%63, %36, %59, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    %67 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<240 x i8>>>
    %68 = llvm.ptrtoint %67 : !llvm.ptr to i64
    %69 = llvm.and %68, %10  : i64
    %70 = llvm.icmp "eq" %69, %13 : i64
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%51, %51, %59, %40, %67, %34, %30, %32, %29, %29, %28, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    %71 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.and %72, %10  : i64
    %74 = llvm.icmp "eq" %73, %13 : i64
    "llvm.intr.assume"(%74) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %71, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.br ^bb1(%13 : i64)
  ^bb1(%75: i64):  // 2 preds: ^bb0, ^bb2
    %76 = llvm.icmp "slt" %75, %25 : i64
    llvm.cond_br %76, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    %77 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<240 x i8>>>
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.and %78, %10  : i64
    %80 = llvm.icmp "eq" %79, %13 : i64
    "llvm.intr.assume"(%80) : (i1) -> ()
    %81 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %82 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %83 = llvm.and %82, %10  : i64
    %84 = llvm.icmp "eq" %83, %13 : i64
    "llvm.intr.assume"(%84) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%81, %36, %77, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%55, %36, %51, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%59, %77, %51, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    %85 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.and %86, %10  : i64
    %88 = llvm.icmp "eq" %87, %13 : i64
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %85, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%63, %36, %59, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%81, %36, %77, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%51, %59, %77, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %71, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%55, %36, %51, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%63, %36, %59, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%77, %51, %59, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %85, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%81, %36, %77, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%55, %36, %51, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%59, %77, %51, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %71, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%63, %36, %59, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%81, %36, %77, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%51, %59, %77, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %85, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%55, %36, %51, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%63, %36, %59, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%77, %51, %59, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%74) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %71, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    %89 = llvm.add %75, %24 : i64
    llvm.br ^bb1(%89 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%23, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %35) : (i32, i32) -> ()
    %90 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<240 x i8>>>
    %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %92 = llvm.and %91, %10  : i64
    %93 = llvm.icmp "eq" %92, %13 : i64
    "llvm.intr.assume"(%93) : (i1) -> ()
    %94 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.and %95, %10  : i64
    %97 = llvm.icmp "eq" %96, %13 : i64
    "llvm.intr.assume"(%97) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%94, %36, %90, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    llvm.call @bn6_conv2dk1_relu_i8_ui8(%55, %36, %51, %34, %33, %32, %46) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %35) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    "llvm.intr.assume"(%54) : (i1) -> ()
    "llvm.intr.assume"(%62) : (i1) -> ()
    "llvm.intr.assume"(%93) : (i1) -> ()
    llvm.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%59, %90, %51, %40, %67, %34, %30, %32, %29, %29, %30, %48, %28) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %31) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%17, %14) : (i32, i32) -> ()
    "llvm.intr.assume"(%70) : (i1) -> ()
    %98 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x array<1 x array<80 x i8>>>
    %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
    %100 = llvm.and %99, %10  : i64
    %101 = llvm.icmp "eq" %100, %13 : i64
    "llvm.intr.assume"(%101) : (i1) -> ()
    llvm.call @bn6_conv2dk1_ui8_i8(%67, %41, %98, %27, %32, %26, %50) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%15, %30) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_1_4() {
    %0 = llvm.mlir.addressof @act_bn5_bn6_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn4_bn5_buff_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @bn5_act_1_2_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @act_bn5_bn6_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @bn5_act_2_3_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @act_bn4_bn5_buff_1 : !llvm.ptr
    %6 = llvm.mlir.addressof @bn5_act_1_2_buff_1 : !llvm.ptr
    %7 = llvm.mlir.addressof @act_bn4_bn5_buff_0 : !llvm.ptr
    %8 = llvm.mlir.addressof @bn5_act_1_2_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @rtp14 : !llvm.ptr
    %10 = llvm.mlir.constant(31 : index) : i64
    %11 = llvm.mlir.addressof @bn5_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(6 : index) : i64
    %15 = llvm.mlir.constant(-1 : i32) : i32
    %16 = llvm.mlir.constant(48 : i32) : i32
    %17 = llvm.mlir.constant(51 : i32) : i32
    %18 = llvm.mlir.constant(50 : i32) : i32
    %19 = llvm.mlir.constant(55 : i32) : i32
    %20 = llvm.mlir.constant(54 : i32) : i32
    %21 = llvm.mlir.constant(34 : i32) : i32
    %22 = llvm.mlir.constant(53 : i32) : i32
    %23 = llvm.mlir.constant(52 : i32) : i32
    %24 = llvm.mlir.constant(35 : i32) : i32
    %25 = llvm.mlir.constant(24 : index) : i64
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(3 : i32) : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.mlir.constant(2 : i32) : i32
    %30 = llvm.mlir.constant(120 : i32) : i32
    %31 = llvm.mlir.constant(40 : i32) : i32
    %32 = llvm.mlir.constant(28 : i32) : i32
    %33 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%12, %15) : (i32, i32) -> ()
    %34 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10680 x i8>
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.and %35, %10  : i64
    %37 = llvm.icmp "eq" %36, %13 : i64
    "llvm.intr.assume"(%37) : (i1) -> ()
    "llvm.intr.assume"(%37) : (i1) -> ()
    %38 = llvm.getelementptr %34[4800] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%37) : (i1) -> ()
    %39 = llvm.getelementptr %34[5880] : (!llvm.ptr) -> !llvm.ptr, i8
    %40 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.and %41, %10  : i64
    %43 = llvm.icmp "eq" %42, %13 : i64
    "llvm.intr.assume"(%43) : (i1) -> ()
    %44 = llvm.load %40 : !llvm.ptr -> i32
    "llvm.intr.assume"(%43) : (i1) -> ()
    %45 = llvm.getelementptr %40[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %46 = llvm.load %45 : !llvm.ptr -> i32
    "llvm.intr.assume"(%43) : (i1) -> ()
    %47 = llvm.getelementptr %40[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %48 = llvm.load %47 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%24, %33) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %33) : (i32, i32) -> ()
    %49 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.and %50, %10  : i64
    %52 = llvm.icmp "eq" %51, %13 : i64
    "llvm.intr.assume"(%52) : (i1) -> ()
    %53 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.and %54, %10  : i64
    %56 = llvm.icmp "eq" %55, %13 : i64
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %57 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.and %58, %10  : i64
    %60 = llvm.icmp "eq" %59, %13 : i64
    "llvm.intr.assume"(%60) : (i1) -> ()
    %61 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %10  : i64
    %64 = llvm.icmp "eq" %63, %13 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%61, %34, %57, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %33) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    %65 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.and %66, %10  : i64
    %68 = llvm.icmp "eq" %67, %13 : i64
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %49, %57, %38, %65, %32, %28, %30, %27, %27, %26, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    %69 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.and %70, %10  : i64
    %72 = llvm.icmp "eq" %71, %13 : i64
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.br ^bb1(%13 : i64)
  ^bb1(%73: i64):  // 2 preds: ^bb0, ^bb2
    %74 = llvm.icmp "slt" %73, %25 : i64
    llvm.cond_br %74, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    %75 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.and %76, %10  : i64
    %78 = llvm.icmp "eq" %77, %13 : i64
    "llvm.intr.assume"(%78) : (i1) -> ()
    %79 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
    %81 = llvm.and %80, %10  : i64
    %82 = llvm.icmp "eq" %81, %13 : i64
    "llvm.intr.assume"(%82) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%79, %34, %75, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %57, %75, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    %83 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.and %84, %10  : i64
    %86 = llvm.icmp "eq" %85, %13 : i64
    "llvm.intr.assume"(%86) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %83, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%57, %75, %49, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%61, %34, %57, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%75, %49, %57, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%86) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %83, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    "llvm.intr.assume"(%82) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%79, %34, %75, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %57, %75, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%57, %75, %49, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%86) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %83, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%61, %34, %57, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%78) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%75, %49, %57, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    %87 = llvm.add %73, %14 : i64
    llvm.br ^bb1(%87 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    %88 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
    %90 = llvm.and %89, %10  : i64
    %91 = llvm.icmp "eq" %90, %13 : i64
    "llvm.intr.assume"(%91) : (i1) -> ()
    %92 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %93 = llvm.ptrtoint %92 : !llvm.ptr to i64
    %94 = llvm.and %93, %10  : i64
    %95 = llvm.icmp "eq" %94, %13 : i64
    "llvm.intr.assume"(%95) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%92, %34, %88, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%91) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%49, %57, %88, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    %96 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.and %97, %10  : i64
    %99 = llvm.icmp "eq" %98, %13 : i64
    "llvm.intr.assume"(%99) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %96, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%56) : (i1) -> ()
    llvm.call @bn5_conv2dk1_relu_i8_ui8(%53, %34, %49, %32, %31, %30, %44) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%91) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%57, %88, %49, %38, %65, %32, %28, %30, %27, %27, %28, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %69, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%52) : (i1) -> ()
    "llvm.intr.assume"(%91) : (i1) -> ()
    llvm.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%88, %49, %49, %38, %65, %32, %28, %30, %27, %27, %29, %46, %26) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%18, %15) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%99) : (i1) -> ()
    llvm.call @bn5_conv2dk1_ui8_i8(%65, %39, %96, %32, %30, %31, %48) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %28) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%16, %28) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_1_5() {
    %0 = llvm.mlir.addressof @act_bn4_bn5_buff_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn4_bn5_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @act_bn3_bn4_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @bn4_act_1_2_buff_2 : !llvm.ptr
    %4 = llvm.mlir.addressof @act_bn4_bn5_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @bn4_act_2_3_buff_0 : !llvm.ptr
    %6 = llvm.mlir.addressof @act_bn3_bn4_buff_1 : !llvm.ptr
    %7 = llvm.mlir.addressof @bn4_act_1_2_buff_1 : !llvm.ptr
    %8 = llvm.mlir.addressof @act_bn3_bn4_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @bn4_act_1_2_buff_0 : !llvm.ptr
    %10 = llvm.mlir.addressof @rtp15 : !llvm.ptr
    %11 = llvm.mlir.constant(31 : index) : i64
    %12 = llvm.mlir.addressof @bn4_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(3 : index) : i64
    %16 = llvm.mlir.constant(-1 : i32) : i32
    %17 = llvm.mlir.constant(48 : i32) : i32
    %18 = llvm.mlir.constant(51 : i32) : i32
    %19 = llvm.mlir.constant(18 : i32) : i32
    %20 = llvm.mlir.constant(50 : i32) : i32
    %21 = llvm.mlir.constant(55 : i32) : i32
    %22 = llvm.mlir.constant(54 : i32) : i32
    %23 = llvm.mlir.constant(53 : i32) : i32
    %24 = llvm.mlir.constant(52 : i32) : i32
    %25 = llvm.mlir.constant(19 : i32) : i32
    %26 = llvm.mlir.constant(24 : index) : i64
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(3 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.mlir.constant(2 : i32) : i32
    %31 = llvm.mlir.constant(120 : i32) : i32
    %32 = llvm.mlir.constant(40 : i32) : i32
    %33 = llvm.mlir.constant(28 : i32) : i32
    %34 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    %35 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10680 x i8>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %11  : i64
    %38 = llvm.icmp "eq" %37, %14 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    "llvm.intr.assume"(%38) : (i1) -> ()
    %39 = llvm.getelementptr %35[4800] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%38) : (i1) -> ()
    %40 = llvm.getelementptr %35[5880] : (!llvm.ptr) -> !llvm.ptr, i8
    %41 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %11  : i64
    %44 = llvm.icmp "eq" %43, %14 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    %45 = llvm.load %41 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %46 = llvm.getelementptr %41[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %47 = llvm.load %46 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %48 = llvm.getelementptr %41[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.load %48 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %50 = llvm.getelementptr %41[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %51 = llvm.load %50 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%25, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %34) : (i32, i32) -> ()
    %52 = llvm.getelementptr %9[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.and %53, %11  : i64
    %55 = llvm.icmp "eq" %54, %14 : i64
    "llvm.intr.assume"(%55) : (i1) -> ()
    %56 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr to i64
    %58 = llvm.and %57, %11  : i64
    %59 = llvm.icmp "eq" %58, %14 : i64
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%56, %35, %52, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %60 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.and %61, %11  : i64
    %63 = llvm.icmp "eq" %62, %14 : i64
    "llvm.intr.assume"(%63) : (i1) -> ()
    %64 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.and %65, %11  : i64
    %67 = llvm.icmp "eq" %66, %14 : i64
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%64, %35, %60, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    %68 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.and %69, %11  : i64
    %71 = llvm.icmp "eq" %70, %14 : i64
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%52, %52, %60, %39, %68, %33, %29, %31, %28, %28, %27, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %72 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.and %73, %11  : i64
    %75 = llvm.icmp "eq" %74, %14 : i64
    "llvm.intr.assume"(%75) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %72, %56, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.br ^bb1(%14 : i64)
  ^bb1(%76: i64):  // 2 preds: ^bb0, ^bb2
    %77 = llvm.icmp "slt" %76, %26 : i64
    llvm.cond_br %77, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%25, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %16) : (i32, i32) -> ()
    %78 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.and %79, %11  : i64
    %81 = llvm.icmp "eq" %80, %14 : i64
    "llvm.intr.assume"(%81) : (i1) -> ()
    %82 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.and %83, %11  : i64
    %85 = llvm.icmp "eq" %84, %14 : i64
    "llvm.intr.assume"(%85) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%82, %35, %78, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%81) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%52, %60, %78, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %86 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %11  : i64
    %89 = llvm.icmp "eq" %88, %14 : i64
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %86, %64, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%56, %35, %52, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%81) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%60, %78, %52, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %90 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %92 = llvm.and %91, %11  : i64
    %93 = llvm.icmp "eq" %92, %14 : i64
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %90, %82, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%64, %35, %60, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%81) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%78, %52, %60, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%75) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %72, %56, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    %94 = llvm.add %76, %15 : i64
    llvm.br ^bb1(%94 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%25, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %16) : (i32, i32) -> ()
    %95 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<120 x i8>>>
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.and %96, %11  : i64
    %98 = llvm.icmp "eq" %97, %14 : i64
    "llvm.intr.assume"(%98) : (i1) -> ()
    %99 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.and %100, %11  : i64
    %102 = llvm.icmp "eq" %101, %14 : i64
    "llvm.intr.assume"(%102) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%99, %35, %95, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%98) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%52, %60, %95, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %103 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %104 = llvm.ptrtoint %103 : !llvm.ptr to i64
    %105 = llvm.and %104, %11  : i64
    %106 = llvm.icmp "eq" %105, %14 : i64
    "llvm.intr.assume"(%106) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %103, %64, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn4_conv2dk1_relu_i8_ui8(%56, %35, %52, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%98) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%60, %95, %52, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %107 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %108 = llvm.ptrtoint %107 : !llvm.ptr to i64
    %109 = llvm.and %108, %11  : i64
    %110 = llvm.icmp "eq" %109, %14 : i64
    "llvm.intr.assume"(%110) : (i1) -> ()
    "llvm.intr.assume"(%102) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %107, %99, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%98) : (i1) -> ()
    llvm.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%95, %52, %52, %39, %68, %33, %29, %31, %28, %28, %30, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%75) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn4_conv2dk1_skip_ui8_i8_i8(%68, %40, %72, %56, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%17, %29) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_5() {
    %0 = llvm.mlir.addressof @act_bn3_bn4_buff_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn3_bn4_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @act_bn2_bn3_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @bn3_act_1_2_buff_2 : !llvm.ptr
    %4 = llvm.mlir.addressof @act_bn3_bn4_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @bn3_act_2_3_buff_0 : !llvm.ptr
    %6 = llvm.mlir.addressof @act_bn2_bn3_buff_1 : !llvm.ptr
    %7 = llvm.mlir.addressof @bn3_act_1_2_buff_1 : !llvm.ptr
    %8 = llvm.mlir.addressof @act_bn2_bn3_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @bn3_act_1_2_buff_0 : !llvm.ptr
    %10 = llvm.mlir.addressof @rtp05 : !llvm.ptr
    %11 = llvm.mlir.constant(31 : index) : i64
    %12 = llvm.mlir.addressof @bn3_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(3 : index) : i64
    %16 = llvm.mlir.constant(-1 : i32) : i32
    %17 = llvm.mlir.constant(48 : i32) : i32
    %18 = llvm.mlir.constant(51 : i32) : i32
    %19 = llvm.mlir.constant(50 : i32) : i32
    %20 = llvm.mlir.constant(55 : i32) : i32
    %21 = llvm.mlir.constant(54 : i32) : i32
    %22 = llvm.mlir.constant(53 : i32) : i32
    %23 = llvm.mlir.constant(52 : i32) : i32
    %24 = llvm.mlir.constant(27 : index) : i64
    %25 = llvm.mlir.constant(40 : i32) : i32
    %26 = llvm.mlir.constant(28 : i32) : i32
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(3 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.mlir.constant(2 : i32) : i32
    %31 = llvm.mlir.constant(72 : i32) : i32
    %32 = llvm.mlir.constant(24 : i32) : i32
    %33 = llvm.mlir.constant(56 : i32) : i32
    %34 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%13, %16) : (i32, i32) -> ()
    %35 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5256 x i8>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %11  : i64
    %38 = llvm.icmp "eq" %37, %14 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    "llvm.intr.assume"(%38) : (i1) -> ()
    %39 = llvm.getelementptr %35[1728] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%38) : (i1) -> ()
    %40 = llvm.getelementptr %35[2376] : (!llvm.ptr) -> !llvm.ptr, i8
    %41 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %11  : i64
    %44 = llvm.icmp "eq" %43, %14 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    %45 = llvm.load %41 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %46 = llvm.getelementptr %41[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %47 = llvm.load %46 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %48 = llvm.getelementptr %41[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.load %48 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%28, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %34) : (i32, i32) -> ()
    %50 = llvm.getelementptr %9[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.and %51, %11  : i64
    %53 = llvm.icmp "eq" %52, %14 : i64
    "llvm.intr.assume"(%53) : (i1) -> ()
    %54 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.and %55, %11  : i64
    %57 = llvm.icmp "eq" %56, %14 : i64
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%54, %35, %50, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %58 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %11  : i64
    %61 = llvm.icmp "eq" %60, %14 : i64
    "llvm.intr.assume"(%61) : (i1) -> ()
    %62 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %11  : i64
    %65 = llvm.icmp "eq" %64, %14 : i64
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%62, %35, %58, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    %66 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<72 x i8>>>
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.and %67, %11  : i64
    %69 = llvm.icmp "eq" %68, %14 : i64
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    llvm.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%50, %50, %58, %39, %66, %33, %29, %31, %28, %28, %27, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.and %71, %11  : i64
    %73 = llvm.icmp "eq" %72, %14 : i64
    "llvm.intr.assume"(%73) : (i1) -> ()
    llvm.call @bn3_conv2dk1_ui8_i8(%66, %40, %70, %26, %31, %25, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.br ^bb1(%14 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb2
    %75 = llvm.icmp "slt" %74, %24 : i64
    llvm.cond_br %75, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%28, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %34) : (i32, i32) -> ()
    %76 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.and %77, %11  : i64
    %79 = llvm.icmp "eq" %78, %14 : i64
    "llvm.intr.assume"(%79) : (i1) -> ()
    %80 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.and %81, %11  : i64
    %83 = llvm.icmp "eq" %82, %14 : i64
    "llvm.intr.assume"(%83) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%80, %35, %76, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%54, %35, %50, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%58, %76, %50, %39, %66, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %84 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.and %85, %11  : i64
    %87 = llvm.icmp "eq" %86, %14 : i64
    "llvm.intr.assume"(%87) : (i1) -> ()
    llvm.call @bn3_conv2dk1_ui8_i8(%66, %40, %84, %26, %31, %25, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %34) : (i32, i32) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%62, %35, %58, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    "llvm.intr.assume"(%83) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%80, %35, %76, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%50, %58, %76, %39, %66, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    %88 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x array<1 x array<40 x i8>>>
    %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
    %90 = llvm.and %89, %11  : i64
    %91 = llvm.icmp "eq" %90, %14 : i64
    "llvm.intr.assume"(%91) : (i1) -> ()
    llvm.call @bn3_conv2dk1_ui8_i8(%66, %40, %88, %26, %31, %25, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %34) : (i32, i32) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%57) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%54, %35, %50, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%65) : (i1) -> ()
    llvm.call @bn3_conv2dk1_relu_i8_ui8(%62, %35, %58, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%53) : (i1) -> ()
    "llvm.intr.assume"(%61) : (i1) -> ()
    "llvm.intr.assume"(%79) : (i1) -> ()
    llvm.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%76, %50, %58, %39, %66, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%20, %16) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%19, %16) : (i32, i32) -> ()
    "llvm.intr.assume"(%69) : (i1) -> ()
    "llvm.intr.assume"(%73) : (i1) -> ()
    llvm.call @bn3_conv2dk1_ui8_i8(%66, %40, %70, %26, %31, %25, %49) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    %92 = llvm.add %74, %15 : i64
    llvm.br ^bb1(%92 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.release(%17, %29) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_4() {
    %0 = llvm.mlir.addressof @act_bn2_bn3_buff_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn2_bn3_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @act_bn01_bn2_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @bn2_act_1_2_buff_2 : !llvm.ptr
    %4 = llvm.mlir.addressof @act_bn2_bn3_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @bn2_act_2_3_buff_0 : !llvm.ptr
    %6 = llvm.mlir.addressof @act_bn01_bn2_buff_1 : !llvm.ptr
    %7 = llvm.mlir.addressof @bn2_act_1_2_buff_1 : !llvm.ptr
    %8 = llvm.mlir.addressof @act_bn01_bn2_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @bn2_act_1_2_buff_0 : !llvm.ptr
    %10 = llvm.mlir.addressof @rtp04 : !llvm.ptr
    %11 = llvm.mlir.constant(31 : index) : i64
    %12 = llvm.mlir.addressof @bn2_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(3 : index) : i64
    %16 = llvm.mlir.constant(54 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(48 : i32) : i32
    %19 = llvm.mlir.constant(51 : i32) : i32
    %20 = llvm.mlir.constant(4 : i32) : i32
    %21 = llvm.mlir.constant(50 : i32) : i32
    %22 = llvm.mlir.constant(55 : i32) : i32
    %23 = llvm.mlir.constant(54 : i32) : i32
    %24 = llvm.mlir.constant(53 : i32) : i32
    %25 = llvm.mlir.constant(52 : i32) : i32
    %26 = llvm.mlir.constant(5 : i32) : i32
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(3 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.mlir.constant(2 : i32) : i32
    %31 = llvm.mlir.constant(72 : i32) : i32
    %32 = llvm.mlir.constant(24 : i32) : i32
    %33 = llvm.mlir.constant(56 : i32) : i32
    %34 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%13, %17) : (i32, i32) -> ()
    %35 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4104 x i8>
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.and %36, %11  : i64
    %38 = llvm.icmp "eq" %37, %14 : i64
    "llvm.intr.assume"(%38) : (i1) -> ()
    "llvm.intr.assume"(%38) : (i1) -> ()
    %39 = llvm.getelementptr %35[1728] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%38) : (i1) -> ()
    %40 = llvm.getelementptr %35[2376] : (!llvm.ptr) -> !llvm.ptr, i8
    %41 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %11  : i64
    %44 = llvm.icmp "eq" %43, %14 : i64
    "llvm.intr.assume"(%44) : (i1) -> ()
    %45 = llvm.load %41 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %46 = llvm.getelementptr %41[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %47 = llvm.load %46 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %48 = llvm.getelementptr %41[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.load %48 : !llvm.ptr -> i32
    "llvm.intr.assume"(%44) : (i1) -> ()
    %50 = llvm.getelementptr %41[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %51 = llvm.load %50 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%26, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %34) : (i32, i32) -> ()
    %52 = llvm.getelementptr %9[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.and %53, %11  : i64
    %55 = llvm.icmp "eq" %54, %14 : i64
    "llvm.intr.assume"(%55) : (i1) -> ()
    %56 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr to i64
    %58 = llvm.and %57, %11  : i64
    %59 = llvm.icmp "eq" %58, %14 : i64
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn2_conv2dk1_relu_i8_ui8(%56, %35, %52, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    %60 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.and %61, %11  : i64
    %63 = llvm.icmp "eq" %62, %14 : i64
    "llvm.intr.assume"(%63) : (i1) -> ()
    %64 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.and %65, %11  : i64
    %67 = llvm.icmp "eq" %66, %14 : i64
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn2_conv2dk1_relu_i8_ui8(%64, %35, %60, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %34) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %17) : (i32, i32) -> ()
    %68 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.and %69, %11  : i64
    %71 = llvm.icmp "eq" %70, %14 : i64
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    llvm.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%52, %52, %60, %39, %68, %33, %29, %31, %28, %28, %27, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %72 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.and %73, %11  : i64
    %75 = llvm.icmp "eq" %74, %14 : i64
    "llvm.intr.assume"(%75) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn2_conv2dk1_skip_ui8_i8_i8(%68, %40, %72, %56, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.br ^bb1(%14 : i64)
  ^bb1(%76: i64):  // 2 preds: ^bb0, ^bb2
    %77 = llvm.icmp "slt" %76, %16 : i64
    llvm.cond_br %77, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%26, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %17) : (i32, i32) -> ()
    %78 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<72 x i8>>>
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.and %79, %11  : i64
    %81 = llvm.icmp "eq" %80, %14 : i64
    "llvm.intr.assume"(%81) : (i1) -> ()
    %82 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.and %83, %11  : i64
    %85 = llvm.icmp "eq" %84, %14 : i64
    "llvm.intr.assume"(%85) : (i1) -> ()
    llvm.call @bn2_conv2dk1_relu_i8_ui8(%82, %35, %78, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%81) : (i1) -> ()
    llvm.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%52, %60, %78, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %86 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %11  : i64
    %89 = llvm.icmp "eq" %88, %14 : i64
    "llvm.intr.assume"(%89) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn2_conv2dk1_skip_ui8_i8_i8(%68, %40, %86, %64, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn2_conv2dk1_relu_i8_ui8(%56, %35, %52, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%81) : (i1) -> ()
    llvm.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%60, %78, %52, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %90 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
    %92 = llvm.and %91, %11  : i64
    %93 = llvm.icmp "eq" %92, %14 : i64
    "llvm.intr.assume"(%93) : (i1) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    llvm.call @bn2_conv2dk1_skip_ui8_i8_i8(%68, %40, %90, %82, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%25, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn2_conv2dk1_relu_i8_ui8(%64, %35, %60, %33, %32, %31, %45) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%81) : (i1) -> ()
    llvm.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%78, %52, %60, %39, %68, %33, %29, %31, %28, %28, %29, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%75) : (i1) -> ()
    "llvm.intr.assume"(%59) : (i1) -> ()
    llvm.call @bn2_conv2dk1_skip_ui8_i8_i8(%68, %40, %72, %56, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    %94 = llvm.add %76, %15 : i64
    llvm.br ^bb1(%94 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%23, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    "llvm.intr.assume"(%55) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    "llvm.intr.assume"(%63) : (i1) -> ()
    llvm.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%52, %60, %60, %39, %68, %33, %29, %31, %28, %28, %30, %47, %27) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %30) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%22, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%21, %17) : (i32, i32) -> ()
    "llvm.intr.assume"(%71) : (i1) -> ()
    %95 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.and %96, %11  : i64
    %98 = llvm.icmp "eq" %97, %14 : i64
    "llvm.intr.assume"(%98) : (i1) -> ()
    "llvm.intr.assume"(%67) : (i1) -> ()
    llvm.call @bn2_conv2dk1_skip_ui8_i8_i8(%68, %40, %95, %64, %33, %31, %32, %49, %51) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%19, %29) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%18, %29) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_3() {
    %0 = llvm.mlir.addressof @act_bn01_bn2_buff_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @act_bn01_bn2_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @bn01_act_bn1_1_2_buff_2 : !llvm.ptr
    %3 = llvm.mlir.addressof @act_bn01_bn2_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @bn01_act_bn1_2_3_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @bn01_act_bn1_1_2_buff_1 : !llvm.ptr
    %6 = llvm.mlir.addressof @act_in_cons_buff_2 : !llvm.ptr
    %7 = llvm.mlir.addressof @bn01_act_bn1_1_2_buff_0 : !llvm.ptr
    %8 = llvm.mlir.addressof @bn01_act_bn0_bn1_buff_0 : !llvm.ptr
    %9 = llvm.mlir.addressof @act_in_cons_buff_1 : !llvm.ptr
    %10 = llvm.mlir.addressof @act_in_cons_buff_0 : !llvm.ptr
    %11 = llvm.mlir.addressof @bn01_act_bn0_2_3_buff_0 : !llvm.ptr
    %12 = llvm.mlir.addressof @rtp03 : !llvm.ptr
    %13 = llvm.mlir.constant(31 : index) : i64
    %14 = llvm.mlir.addressof @bn0_1_wts_OF_L2L1_cons_buff_0 : !llvm.ptr
    %15 = llvm.mlir.constant(51 : i32) : i32
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(54 : index) : i64
    %18 = llvm.mlir.constant(-1 : i32) : i32
    %19 = llvm.mlir.constant(3 : index) : i64
    %20 = llvm.mlir.constant(50 : i32) : i32
    %21 = llvm.mlir.constant(53 : i32) : i32
    %22 = llvm.mlir.constant(52 : i32) : i32
    %23 = llvm.mlir.constant(61 : i32) : i32
    %24 = llvm.mlir.constant(60 : i32) : i32
    %25 = llvm.mlir.constant(48 : i32) : i32
    %26 = llvm.mlir.constant(59 : i32) : i32
    %27 = llvm.mlir.constant(58 : i32) : i32
    %28 = llvm.mlir.constant(57 : i32) : i32
    %29 = llvm.mlir.constant(55 : i32) : i32
    %30 = llvm.mlir.constant(54 : i32) : i32
    %31 = llvm.mlir.constant(49 : i32) : i32
    %32 = llvm.mlir.constant(2 : i32) : i32
    %33 = llvm.mlir.constant(24 : i32) : i32
    %34 = llvm.mlir.constant(56 : i32) : i32
    %35 = llvm.mlir.constant(64 : i32) : i32
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(3 : i32) : i32
    %38 = llvm.mlir.constant(16 : i32) : i32
    %39 = llvm.mlir.constant(1 : i32) : i32
    %40 = llvm.mlir.constant(112 : i32) : i32
    %41 = llvm.mlir.constant(-2 : i32) : i32
    llvm.call @llvm.aie2.acquire(%15, %18) : (i32, i32) -> ()
    %42 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3536 x i8>
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.and %43, %13  : i64
    %45 = llvm.icmp "eq" %44, %16 : i64
    "llvm.intr.assume"(%45) : (i1) -> ()
    "llvm.intr.assume"(%45) : (i1) -> ()
    %46 = llvm.getelementptr %42[144] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%45) : (i1) -> ()
    %47 = llvm.getelementptr %42[400] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%45) : (i1) -> ()
    %48 = llvm.getelementptr %42[1424] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.assume"(%45) : (i1) -> ()
    %49 = llvm.getelementptr %42[2000] : (!llvm.ptr) -> !llvm.ptr, i8
    %50 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.and %51, %13  : i64
    %53 = llvm.icmp "eq" %52, %16 : i64
    "llvm.intr.assume"(%53) : (i1) -> ()
    %54 = llvm.load %50 : !llvm.ptr -> i32
    "llvm.intr.assume"(%53) : (i1) -> ()
    %55 = llvm.getelementptr %50[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %56 = llvm.load %55 : !llvm.ptr -> i32
    "llvm.intr.assume"(%53) : (i1) -> ()
    %57 = llvm.getelementptr %50[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %58 = llvm.load %57 : !llvm.ptr -> i32
    "llvm.intr.assume"(%53) : (i1) -> ()
    %59 = llvm.getelementptr %50[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %60 = llvm.load %59 : !llvm.ptr -> i32
    "llvm.intr.assume"(%53) : (i1) -> ()
    %61 = llvm.getelementptr %50[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %62 = llvm.load %61 : !llvm.ptr -> i32
    "llvm.intr.assume"(%53) : (i1) -> ()
    %63 = llvm.getelementptr %50[5] : (!llvm.ptr) -> !llvm.ptr, i32
    %64 = llvm.load %63 : !llvm.ptr -> i32
    llvm.call @llvm.aie2.acquire(%31, %41) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    %65 = llvm.getelementptr %11[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<16 x i8>>>
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.and %66, %13  : i64
    %68 = llvm.icmp "eq" %67, %16 : i64
    "llvm.intr.assume"(%68) : (i1) -> ()
    %69 = llvm.getelementptr %10[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<16 x i8>>>
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.and %70, %13  : i64
    %72 = llvm.icmp "eq" %71, %16 : i64
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    %73 = llvm.getelementptr %9[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<16 x i8>>>
    %74 = llvm.ptrtoint %73 : !llvm.ptr to i64
    %75 = llvm.and %74, %13  : i64
    %76 = llvm.icmp "eq" %75, %16 : i64
    "llvm.intr.assume"(%76) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%69, %69, %73, %42, %65, %40, %39, %38, %37, %37, %36, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    %77 = llvm.getelementptr %8[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<16 x i8>>>
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.and %78, %13  : i64
    %80 = llvm.icmp "eq" %79, %16 : i64
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %69, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    %81 = llvm.getelementptr %7[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<64 x i8>>>
    %82 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %83 = llvm.and %82, %13  : i64
    %84 = llvm.icmp "eq" %83, %16 : i64
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %81, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    %85 = llvm.getelementptr %6[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<16 x i8>>>
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.and %86, %13  : i64
    %88 = llvm.icmp "eq" %87, %16 : i64
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%69, %73, %85, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %73, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    %89 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<64 x i8>>>
    %90 = llvm.ptrtoint %89 : !llvm.ptr to i64
    %91 = llvm.and %90, %13  : i64
    %92 = llvm.icmp "eq" %91, %16 : i64
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %89, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %41) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %18) : (i32, i32) -> ()
    %93 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<64 x i8>>>
    %94 = llvm.ptrtoint %93 : !llvm.ptr to i64
    %95 = llvm.and %94, %13  : i64
    %96 = llvm.icmp "eq" %95, %16 : i64
    "llvm.intr.assume"(%96) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    llvm.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%81, %81, %89, %48, %93, %40, %39, %35, %37, %37, %36, %62, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%27, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    %97 = llvm.getelementptr %3[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
    %99 = llvm.and %98, %13  : i64
    %100 = llvm.icmp "eq" %99, %16 : i64
    "llvm.intr.assume"(%100) : (i1) -> ()
    llvm.call @bn1_conv2dk1_ui8_i8(%93, %49, %97, %34, %35, %33, %64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %39) : (i32, i32) -> ()
    llvm.br ^bb1(%16 : i64)
  ^bb1(%101: i64):  // 2 preds: ^bb0, ^bb2
    %102 = llvm.icmp "slt" %101, %17 : i64
    llvm.cond_br %102, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%73, %85, %69, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %85, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    %103 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<64 x i8>>>
    %104 = llvm.ptrtoint %103 : !llvm.ptr to i64
    %105 = llvm.and %104, %13  : i64
    %106 = llvm.icmp "eq" %105, %16 : i64
    "llvm.intr.assume"(%106) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %103, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%85, %69, %73, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %69, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %81, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %41) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%106) : (i1) -> ()
    llvm.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%89, %103, %81, %48, %93, %40, %39, %35, %37, %37, %39, %62, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%27, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    %107 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %108 = llvm.ptrtoint %107 : !llvm.ptr to i64
    %109 = llvm.and %108, %13  : i64
    %110 = llvm.icmp "eq" %109, %16 : i64
    "llvm.intr.assume"(%110) : (i1) -> ()
    llvm.call @bn1_conv2dk1_ui8_i8(%93, %49, %107, %34, %35, %33, %64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%69, %73, %85, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %73, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %89, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%73, %85, %69, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %85, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%106) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %103, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %41) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%106) : (i1) -> ()
    llvm.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%81, %89, %103, %48, %93, %40, %39, %35, %37, %37, %39, %62, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%27, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    %111 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %112 = llvm.ptrtoint %111 : !llvm.ptr to i64
    %113 = llvm.and %112, %13  : i64
    %114 = llvm.icmp "eq" %113, %16 : i64
    "llvm.intr.assume"(%114) : (i1) -> ()
    llvm.call @bn1_conv2dk1_ui8_i8(%93, %49, %111, %34, %35, %33, %64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%85, %69, %73, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %69, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %81, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%69, %73, %85, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %73, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %89, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %41) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%106) : (i1) -> ()
    llvm.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%103, %81, %89, %48, %93, %40, %39, %35, %37, %37, %39, %62, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%27, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    "llvm.intr.assume"(%100) : (i1) -> ()
    llvm.call @bn1_conv2dk1_ui8_i8(%93, %49, %97, %34, %35, %33, %64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %39) : (i32, i32) -> ()
    %115 = llvm.add %101, %19 : i64
    llvm.br ^bb1(%115 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%31, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%73, %85, %69, %42, %65, %40, %39, %38, %37, %37, %39, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %85, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    %116 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<112 x array<1 x array<64 x i8>>>
    %117 = llvm.ptrtoint %116 : !llvm.ptr to i64
    %118 = llvm.and %117, %13  : i64
    %119 = llvm.icmp "eq" %118, %16 : i64
    "llvm.intr.assume"(%119) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %116, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%30, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%88) : (i1) -> ()
    llvm.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%85, %69, %69, %42, %65, %40, %39, %38, %37, %37, %32, %54, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%29, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%29, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%34, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    "llvm.intr.assume"(%68) : (i1) -> ()
    "llvm.intr.assume"(%72) : (i1) -> ()
    llvm.call @bn0_conv2dk1_skip_ui8_ui8_i8(%65, %46, %77, %69, %40, %38, %38, %56, %58) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%25, %32) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%30, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%28, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%28, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%27, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%80) : (i1) -> ()
    llvm.call @bn1_conv2dk1_relu_i8_ui8(%77, %47, %81, %40, %38, %35, %60) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%34, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%26, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%26, %41) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%24, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    "llvm.intr.assume"(%84) : (i1) -> ()
    "llvm.intr.assume"(%92) : (i1) -> ()
    "llvm.intr.assume"(%119) : (i1) -> ()
    llvm.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%89, %116, %81, %48, %93, %40, %39, %35, %37, %37, %39, %62, %36) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%27, %37) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%23, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%23, %18) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%22, %18) : (i32, i32) -> ()
    "llvm.intr.assume"(%96) : (i1) -> ()
    %120 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x array<1 x array<24 x i8>>>
    %121 = llvm.ptrtoint %120 : !llvm.ptr to i64
    %122 = llvm.and %121, %13  : i64
    %123 = llvm.icmp "eq" %122, %16 : i64
    "llvm.intr.assume"(%123) : (i1) -> ()
    llvm.call @bn1_conv2dk1_ui8_i8(%93, %49, %120, %34, %35, %33, %64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%24, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%21, %39) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%20, %39) : (i32, i32) -> ()
    llvm.return
  }
}

