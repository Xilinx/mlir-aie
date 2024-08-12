module {
  aie.device(npu1_4col) {
    memref.global "public" @B_OF_b11_act_layer2_layer3 : memref<14x1x336xui8>
    memref.global "public" @B_OF_b11_act_layer1_layer2_cons : memref<14x1x336xui8>
    memref.global "public" @B_OF_b11_act_layer1_layer2 : memref<14x1x336xui8>
    memref.global "public" @OF_b11_skip_cons : memref<14x1x112xi8>
    memref.global "public" @OF_b11_skip : memref<14x1x112xi8>
    memref.global "public" @B_OF_b10_layer3_bn_11_layer1_0_cons : memref<14x1x112xi8>
    memref.global "public" @B_OF_b10_layer3_bn_11_layer1_1_cons : memref<14x1x112xi8>
    memref.global "public" @B_OF_b10_layer3_bn_11_layer1 : memref<14x1x112xi8>
    memref.global "public" @B_OF_b10_act_layer2_layer3 : memref<14x1x480xui8>
    memref.global "public" @B_OF_b10_act_layer1_layer2_cons : memref<14x1x480xui8>
    memref.global "public" @B_OF_b10_act_layer1_layer2 : memref<14x1x480xui8>
    memref.global "public" @act_out_cons : memref<14x1x112xi8>
    memref.global "public" @act_out : memref<14x1x112xi8>
    memref.global "public" @bn9_act_2_3 : memref<14x1x184xui8>
    memref.global "public" @bn9_act_1_2 : memref<14x1x184xui8>
    memref.global "public" @act_bn9_bn10 : memref<14x1x80xi8>
    memref.global "public" @bn8_act_2_3 : memref<14x1x184xui8>
    memref.global "public" @bn8_act_1_2 : memref<14x1x184xui8>
    memref.global "public" @act_bn8_bn9 : memref<14x1x80xi8>
    memref.global "public" @bn7_act_2_3 : memref<14x1x200xui8>
    memref.global "public" @bn7_act_1_2 : memref<14x1x200xui8>
    memref.global "public" @act_bn7_bn8_cons : memref<14x1x80xi8>
    memref.global "public" @act_bn7_bn8 : memref<14x1x80xi8>
    memref.global "public" @bn6_act_2_3 : memref<14x1x240xui8>
    memref.global "public" @bn6_act_1_2 : memref<28x1x240xui8>
    memref.global "public" @act_bn6_bn7 : memref<14x1x80xi8>
    memref.global "public" @bn5_act_2_3 : memref<28x1x120xui8>
    memref.global "public" @bn5_act_1_2 : memref<28x1x120xui8>
    memref.global "public" @act_bn5_bn6_cons : memref<28x1x40xi8>
    memref.global "public" @act_bn5_bn6 : memref<28x1x40xi8>
    memref.global "public" @bn4_act_2_3 : memref<28x1x120xui8>
    memref.global "public" @bn4_act_1_2 : memref<28x1x120xui8>
    memref.global "public" @act_bn4_bn5 : memref<28x1x40xi8>
    memref.global "public" @bn3_act_2_3 : memref<28x1x72xui8>
    memref.global "public" @bn3_act_1_2 : memref<56x1x72xui8>
    memref.global "public" @act_bn3_bn4 : memref<28x1x40xi8>
    memref.global "public" @bn2_act_2_3 : memref<56x1x72xui8>
    memref.global "public" @bn2_act_1_2 : memref<56x1x72xui8>
    memref.global "public" @act_bn2_bn3 : memref<56x1x24xi8>
    memref.global "public" @bn01_act_bn1_2_3 : memref<56x1x64xui8>
    memref.global "public" @bn01_act_bn1_1_2 : memref<112x1x64xui8>
    memref.global "public" @bn01_act_bn0_bn1 : memref<112x1x16xi8>
    memref.global "public" @bn01_act_bn0_2_3 : memref<112x1x16xui8>
    memref.global "public" @act_bn01_bn2 : memref<56x1x24xi8>
    memref.global "public" @bn9_wts_OF_L2L1_cons : memref<31096xi8>
    memref.global "public" @bn9_wts_OF_L2L1 : memref<31096xi8>
    memref.global "public" @bn8_wts_OF_L2L1_cons : memref<31096xi8>
    memref.global "public" @bn8_wts_OF_L2L1 : memref<31096xi8>
    memref.global "public" @bn7_wts_OF_L2L1_cons : memref<33800xi8>
    memref.global "public" @bn7_wts_OF_L2L1 : memref<33800xi8>
    memref.global "public" @bn6_wts_OF_L2L1_cons : memref<30960xi8>
    memref.global "public" @bn6_wts_OF_L2L1 : memref<30960xi8>
    memref.global "public" @wts_OF_11_L3L2_cons : memref<126952xi8>
    memref.global "public" @wts_OF_11_L3L2 : memref<126952xi8>
    memref.global "public" @bn5_wts_OF_L2L1_cons : memref<10680xi8>
    memref.global "public" @bn5_wts_OF_L2L1 : memref<10680xi8>
    memref.global "public" @bn4_wts_OF_L2L1_cons : memref<10680xi8>
    memref.global "public" @bn4_wts_OF_L2L1 : memref<10680xi8>
    memref.global "public" @bn3_wts_OF_L2L1_cons : memref<5256xi8>
    memref.global "public" @bn3_wts_OF_L2L1 : memref<5256xi8>
    memref.global "public" @bn2_wts_OF_L2L1_cons : memref<4104xi8>
    memref.global "public" @bn2_wts_OF_L2L1 : memref<4104xi8>
    memref.global "public" @bn0_1_wts_OF_L2L1_cons : memref<3536xi8>
    memref.global "public" @bn0_1_wts_OF_L2L1 : memref<3536xi8>
    memref.global "public" @wts_OF_01_L3L2_cons : memref<34256xi8>
    memref.global "public" @wts_OF_01_L3L2 : memref<34256xi8>
    memref.global "public" @act_in_cons : memref<112x1x16xui8>
    memref.global "public" @act_in : memref<112x1x16xui8>
    memref.global "public" @weightsInBN11_layer3_cons : memref<37632xi8>
    memref.global "public" @weightsInBN11_layer3 : memref<37632xi8>
    memref.global "public" @weightsInBN11_layer2_cons : memref<3024xi8>
    memref.global "public" @weightsInBN11_layer2 : memref<3024xi8>
    memref.global "public" @weightsInBN11_layer1_cons : memref<37632xi8>
    memref.global "public" @weightsInBN11_layer1 : memref<37632xi8>
    memref.global "public" @wts_b11_L3L2_cons : memref<78288xi8>
    memref.global "public" @wts_b11_L3L2 : memref<78288xi8>
    memref.global "public" @weightsInBN10_layer3_cons : memref<53760xi8>
    memref.global "public" @weightsInBN10_layer3 : memref<53760xi8>
    memref.global "public" @weightsInBN10_layer2_cons : memref<4320xi8>
    memref.global "public" @weightsInBN10_layer2 : memref<4320xi8>
    memref.global "public" @weightsInBN10_layer1_cons : memref<38400xi8>
    memref.global "public" @weightsInBN10_layer1 : memref<38400xi8>
    memref.global "public" @wts_b10_L3L2_cons : memref<96480xi8>
    memref.global "public" @wts_b10_L3L2 : memref<96480xi8>
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
    %B_OF_b11_act_layer2_layer3_buff_0 = aie.buffer(%tile_3_3) {address = 1024 : i32, sym_name = "B_OF_b11_act_layer2_layer3_buff_0"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer2_layer3_buff_1 = aie.buffer(%tile_3_3) {address = 5728 : i32, sym_name = "B_OF_b11_act_layer2_layer3_buff_1"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer2_layer3_prod_lock = aie.lock(%tile_3_3, 4) {init = 2 : i32, sym_name = "B_OF_b11_act_layer2_layer3_prod_lock"}
    %B_OF_b11_act_layer2_layer3_cons_lock = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "B_OF_b11_act_layer2_layer3_cons_lock"}
    %B_OF_b11_act_layer1_layer2_cons_buff_0 = aie.buffer(%tile_3_3) {address = 10432 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_buff_0"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer1_layer2_cons_buff_1 = aie.buffer(%tile_3_3) {address = 15136 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_buff_1"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer1_layer2_cons_buff_2 = aie.buffer(%tile_3_3) {address = 19840 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_buff_2"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer1_layer2_cons_buff_3 = aie.buffer(%tile_3_3) {address = 24544 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_buff_3"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer1_layer2_cons_prod_lock = aie.lock(%tile_3_3, 2) {init = 4 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_prod_lock"}
    %B_OF_b11_act_layer1_layer2_cons_cons_lock = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_cons_lock"}
    %B_OF_b11_act_layer1_layer2_buff_0 = aie.buffer(%tile_3_4) {address = 38656 : i32, sym_name = "B_OF_b11_act_layer1_layer2_buff_0"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer1_layer2_buff_1 = aie.buffer(%tile_3_4) {address = 43360 : i32, sym_name = "B_OF_b11_act_layer1_layer2_buff_1"} : memref<14x1x336xui8> 
    %B_OF_b11_act_layer1_layer2_prod_lock = aie.lock(%tile_3_4, 4) {init = 2 : i32, sym_name = "B_OF_b11_act_layer1_layer2_prod_lock"}
    %B_OF_b11_act_layer1_layer2_cons_lock = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "B_OF_b11_act_layer1_layer2_cons_lock"}
    %OF_b11_skip_cons_buff_0 = aie.buffer(%tile_3_2) {address = 38656 : i32, sym_name = "OF_b11_skip_cons_buff_0"} : memref<14x1x112xi8> 
    %OF_b11_skip_cons_buff_1 = aie.buffer(%tile_3_2) {address = 40224 : i32, sym_name = "OF_b11_skip_cons_buff_1"} : memref<14x1x112xi8> 
    %OF_b11_skip_cons_prod_lock = aie.lock(%tile_3_2, 4) {init = 2 : i32, sym_name = "OF_b11_skip_cons_prod_lock"}
    %OF_b11_skip_cons_cons_lock = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "OF_b11_skip_cons_cons_lock"}
    %B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0 = aie.buffer(%tile_3_4) {address = 48064 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1 = aie.buffer(%tile_3_4) {address = 49632 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_0_cons_prod_lock = aie.lock(%tile_3_4, 2) {init = 2 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_0_cons_prod_lock"}
    %B_OF_b10_layer3_bn_11_layer1_0_cons_cons_lock = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_0_cons_cons_lock"}
    %B_OF_b10_layer3_bn_11_layer1_1_cons_buff_0 = aie.buffer(%tile_2_1) {address = 96480 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_buff_0"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_1_cons_buff_1 = aie.buffer(%tile_2_1) {address = 98048 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_buff_1"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_1_cons_buff_2 = aie.buffer(%tile_2_1) {address = 99616 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_buff_2"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_1_cons_buff_3 = aie.buffer(%tile_2_1) {address = 101184 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_buff_3"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_1_cons_buff_4 = aie.buffer(%tile_2_1) {address = 102752 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_buff_4"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_1_cons_buff_5 = aie.buffer(%tile_2_1) {address = 104320 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_buff_5"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock = aie.lock(%tile_2_1, 2) {init = 6 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock"}
    %B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock = aie.lock(%tile_2_1, 3) {init = 0 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock"}
    %B_OF_b10_layer3_bn_11_layer1_buff_0 = aie.buffer(%tile_3_5) {address = 54784 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_buff_0"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_buff_1 = aie.buffer(%tile_3_5) {address = 56352 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_buff_1"} : memref<14x1x112xi8> 
    %B_OF_b10_layer3_bn_11_layer1_prod_lock = aie.lock(%tile_3_5, 2) {init = 2 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_prod_lock"}
    %B_OF_b10_layer3_bn_11_layer1_cons_lock = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "B_OF_b10_layer3_bn_11_layer1_cons_lock"}
    %B_OF_b10_act_layer2_layer3_buff_0 = aie.buffer(%tile_2_5) {address = 1024 : i32, sym_name = "B_OF_b10_act_layer2_layer3_buff_0"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer2_layer3_buff_1 = aie.buffer(%tile_2_5) {address = 7744 : i32, sym_name = "B_OF_b10_act_layer2_layer3_buff_1"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer2_layer3_prod_lock = aie.lock(%tile_2_5, 4) {init = 2 : i32, sym_name = "B_OF_b10_act_layer2_layer3_prod_lock"}
    %B_OF_b10_act_layer2_layer3_cons_lock = aie.lock(%tile_2_5, 5) {init = 0 : i32, sym_name = "B_OF_b10_act_layer2_layer3_cons_lock"}
    %B_OF_b10_act_layer1_layer2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 14464 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_buff_0"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer1_layer2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 21184 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_buff_1"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer1_layer2_cons_buff_2 = aie.buffer(%tile_2_5) {address = 27904 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_buff_2"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer1_layer2_cons_buff_3 = aie.buffer(%tile_2_5) {address = 34624 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_buff_3"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer1_layer2_cons_prod_lock = aie.lock(%tile_2_5, 2) {init = 4 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_prod_lock"}
    %B_OF_b10_act_layer1_layer2_cons_cons_lock = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_cons_lock"}
    %B_OF_b10_act_layer1_layer2_buff_0 = aie.buffer(%tile_2_4) {address = 39424 : i32, sym_name = "B_OF_b10_act_layer1_layer2_buff_0"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer1_layer2_buff_1 = aie.buffer(%tile_2_4) {address = 46144 : i32, sym_name = "B_OF_b10_act_layer1_layer2_buff_1"} : memref<14x1x480xui8> 
    %B_OF_b10_act_layer1_layer2_prod_lock = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "B_OF_b10_act_layer1_layer2_prod_lock"}
    %B_OF_b10_act_layer1_layer2_cons_lock = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "B_OF_b10_act_layer1_layer2_cons_lock"}
    %act_out_cons_prod_lock = aie.lock(%tile_3_0, 2) {init = 1 : i32, sym_name = "act_out_cons_prod_lock"}
    %act_out_cons_cons_lock = aie.lock(%tile_3_0, 3) {init = 0 : i32, sym_name = "act_out_cons_cons_lock"}
    %act_out_buff_0 = aie.buffer(%tile_3_2) {address = 41792 : i32, sym_name = "act_out_buff_0"} : memref<14x1x112xi8> 
    %act_out_buff_1 = aie.buffer(%tile_3_2) {address = 43360 : i32, sym_name = "act_out_buff_1"} : memref<14x1x112xi8> 
    %act_out_prod_lock = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "act_out_prod_lock"}
    %act_out_cons_lock = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "act_out_cons_lock"}
    %bn9_act_2_3_buff_0 = aie.buffer(%tile_2_3) {address = 32120 : i32, sym_name = "bn9_act_2_3_buff_0"} : memref<14x1x184xui8> 
    %bn9_act_2_3_prod_lock = aie.lock(%tile_2_3, 6) {init = 1 : i32, sym_name = "bn9_act_2_3_prod_lock"}
    %bn9_act_2_3_cons_lock = aie.lock(%tile_2_3, 7) {init = 0 : i32, sym_name = "bn9_act_2_3_cons_lock"}
    %bn9_act_1_2_buff_0 = aie.buffer(%tile_2_3) {address = 34696 : i32, sym_name = "bn9_act_1_2_buff_0"} : memref<14x1x184xui8> 
    %bn9_act_1_2_buff_1 = aie.buffer(%tile_2_3) {address = 37272 : i32, sym_name = "bn9_act_1_2_buff_1"} : memref<14x1x184xui8> 
    %bn9_act_1_2_buff_2 = aie.buffer(%tile_2_3) {address = 39848 : i32, sym_name = "bn9_act_1_2_buff_2"} : memref<14x1x184xui8> 
    %bn9_act_1_2_prod_lock = aie.lock(%tile_2_3, 4) {init = 3 : i32, sym_name = "bn9_act_1_2_prod_lock"}
    %bn9_act_1_2_cons_lock = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "bn9_act_1_2_cons_lock"}
    %act_bn9_bn10_buff_0 = aie.buffer(%tile_2_3) {address = 42424 : i32, sym_name = "act_bn9_bn10_buff_0"} : memref<14x1x80xi8> 
    %act_bn9_bn10_buff_1 = aie.buffer(%tile_2_3) {address = 43544 : i32, sym_name = "act_bn9_bn10_buff_1"} : memref<14x1x80xi8> 
    %act_bn9_bn10_prod_lock = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "act_bn9_bn10_prod_lock"}
    %act_bn9_bn10_cons_lock = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "act_bn9_bn10_cons_lock"}
    %bn8_act_2_3_buff_0 = aie.buffer(%tile_2_2) {address = 32120 : i32, sym_name = "bn8_act_2_3_buff_0"} : memref<14x1x184xui8> 
    %bn8_act_2_3_prod_lock = aie.lock(%tile_2_2, 8) {init = 1 : i32, sym_name = "bn8_act_2_3_prod_lock"}
    %bn8_act_2_3_cons_lock = aie.lock(%tile_2_2, 9) {init = 0 : i32, sym_name = "bn8_act_2_3_cons_lock"}
    %bn8_act_1_2_buff_0 = aie.buffer(%tile_2_2) {address = 34696 : i32, sym_name = "bn8_act_1_2_buff_0"} : memref<14x1x184xui8> 
    %bn8_act_1_2_buff_1 = aie.buffer(%tile_2_2) {address = 37272 : i32, sym_name = "bn8_act_1_2_buff_1"} : memref<14x1x184xui8> 
    %bn8_act_1_2_buff_2 = aie.buffer(%tile_2_2) {address = 39848 : i32, sym_name = "bn8_act_1_2_buff_2"} : memref<14x1x184xui8> 
    %bn8_act_1_2_prod_lock = aie.lock(%tile_2_2, 6) {init = 3 : i32, sym_name = "bn8_act_1_2_prod_lock"}
    %bn8_act_1_2_cons_lock = aie.lock(%tile_2_2, 7) {init = 0 : i32, sym_name = "bn8_act_1_2_cons_lock"}
    %act_bn8_bn9_buff_0 = aie.buffer(%tile_2_2) {address = 42424 : i32, sym_name = "act_bn8_bn9_buff_0"} : memref<14x1x80xi8> 
    %act_bn8_bn9_buff_1 = aie.buffer(%tile_2_2) {address = 43544 : i32, sym_name = "act_bn8_bn9_buff_1"} : memref<14x1x80xi8> 
    %act_bn8_bn9_prod_lock = aie.lock(%tile_2_2, 4) {init = 2 : i32, sym_name = "act_bn8_bn9_prod_lock"}
    %act_bn8_bn9_cons_lock = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "act_bn8_bn9_cons_lock"}
    %bn7_act_2_3_buff_0 = aie.buffer(%tile_1_3) {address = 34824 : i32, sym_name = "bn7_act_2_3_buff_0"} : memref<14x1x200xui8> 
    %bn7_act_2_3_prod_lock = aie.lock(%tile_1_3, 6) {init = 1 : i32, sym_name = "bn7_act_2_3_prod_lock"}
    %bn7_act_2_3_cons_lock = aie.lock(%tile_1_3, 7) {init = 0 : i32, sym_name = "bn7_act_2_3_cons_lock"}
    %bn7_act_1_2_buff_0 = aie.buffer(%tile_1_3) {address = 37624 : i32, sym_name = "bn7_act_1_2_buff_0"} : memref<14x1x200xui8> 
    %bn7_act_1_2_buff_1 = aie.buffer(%tile_1_3) {address = 40424 : i32, sym_name = "bn7_act_1_2_buff_1"} : memref<14x1x200xui8> 
    %bn7_act_1_2_buff_2 = aie.buffer(%tile_1_3) {address = 43224 : i32, sym_name = "bn7_act_1_2_buff_2"} : memref<14x1x200xui8> 
    %bn7_act_1_2_prod_lock = aie.lock(%tile_1_3, 4) {init = 3 : i32, sym_name = "bn7_act_1_2_prod_lock"}
    %bn7_act_1_2_cons_lock = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "bn7_act_1_2_cons_lock"}
    %act_bn7_bn8_cons_buff_0 = aie.buffer(%tile_2_2) {address = 44664 : i32, sym_name = "act_bn7_bn8_cons_buff_0"} : memref<14x1x80xi8> 
    %act_bn7_bn8_cons_buff_1 = aie.buffer(%tile_2_2) {address = 45784 : i32, sym_name = "act_bn7_bn8_cons_buff_1"} : memref<14x1x80xi8> 
    %act_bn7_bn8_cons_buff_2 = aie.buffer(%tile_2_2) {address = 46904 : i32, sym_name = "act_bn7_bn8_cons_buff_2"} : memref<14x1x80xi8> 
    %act_bn7_bn8_cons_prod_lock = aie.lock(%tile_2_2, 2) {init = 3 : i32, sym_name = "act_bn7_bn8_cons_prod_lock"}
    %act_bn7_bn8_cons_cons_lock = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "act_bn7_bn8_cons_cons_lock"}
    %act_bn7_bn8_buff_0 = aie.buffer(%tile_1_3) {address = 46024 : i32, sym_name = "act_bn7_bn8_buff_0"} : memref<14x1x80xi8> 
    %act_bn7_bn8_buff_1 = aie.buffer(%tile_1_3) {address = 47144 : i32, sym_name = "act_bn7_bn8_buff_1"} : memref<14x1x80xi8> 
    %act_bn7_bn8_prod_lock = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "act_bn7_bn8_prod_lock"}
    %act_bn7_bn8_cons_lock = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "act_bn7_bn8_cons_lock"}
    %bn6_act_2_3_buff_0 = aie.buffer(%tile_1_2) {address = 52144 : i32, sym_name = "bn6_act_2_3_buff_0"} : memref<14x1x240xui8> 
    %bn6_act_2_3_prod_lock = aie.lock(%tile_1_2, 8) {init = 1 : i32, sym_name = "bn6_act_2_3_prod_lock"}
    %bn6_act_2_3_cons_lock = aie.lock(%tile_1_2, 9) {init = 0 : i32, sym_name = "bn6_act_2_3_cons_lock"}
    %bn6_act_1_2_buff_0 = aie.buffer(%tile_1_2) {address = 31984 : i32, sym_name = "bn6_act_1_2_buff_0"} : memref<28x1x240xui8> 
    %bn6_act_1_2_buff_1 = aie.buffer(%tile_1_2) {address = 38704 : i32, sym_name = "bn6_act_1_2_buff_1"} : memref<28x1x240xui8> 
    %bn6_act_1_2_buff_2 = aie.buffer(%tile_1_2) {address = 45424 : i32, sym_name = "bn6_act_1_2_buff_2"} : memref<28x1x240xui8> 
    %bn6_act_1_2_prod_lock = aie.lock(%tile_1_2, 6) {init = 3 : i32, sym_name = "bn6_act_1_2_prod_lock"}
    %bn6_act_1_2_cons_lock = aie.lock(%tile_1_2, 7) {init = 0 : i32, sym_name = "bn6_act_1_2_cons_lock"}
    %act_bn6_bn7_buff_0 = aie.buffer(%tile_1_2) {address = 55504 : i32, sym_name = "act_bn6_bn7_buff_0"} : memref<14x1x80xi8> 
    %act_bn6_bn7_buff_1 = aie.buffer(%tile_1_2) {address = 56624 : i32, sym_name = "act_bn6_bn7_buff_1"} : memref<14x1x80xi8> 
    %act_bn6_bn7_prod_lock = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "act_bn6_bn7_prod_lock"}
    %act_bn6_bn7_cons_lock = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "act_bn6_bn7_cons_lock"}
    %bn5_act_2_3_buff_0 = aie.buffer(%tile_1_4) {address = 11704 : i32, sym_name = "bn5_act_2_3_buff_0"} : memref<28x1x120xui8> 
    %bn5_act_2_3_prod_lock = aie.lock(%tile_1_4, 6) {init = 1 : i32, sym_name = "bn5_act_2_3_prod_lock"}
    %bn5_act_2_3_cons_lock = aie.lock(%tile_1_4, 7) {init = 0 : i32, sym_name = "bn5_act_2_3_cons_lock"}
    %bn5_act_1_2_buff_0 = aie.buffer(%tile_1_4) {address = 15064 : i32, sym_name = "bn5_act_1_2_buff_0"} : memref<28x1x120xui8> 
    %bn5_act_1_2_buff_1 = aie.buffer(%tile_1_4) {address = 18424 : i32, sym_name = "bn5_act_1_2_buff_1"} : memref<28x1x120xui8> 
    %bn5_act_1_2_buff_2 = aie.buffer(%tile_1_4) {address = 21784 : i32, sym_name = "bn5_act_1_2_buff_2"} : memref<28x1x120xui8> 
    %bn5_act_1_2_prod_lock = aie.lock(%tile_1_4, 4) {init = 3 : i32, sym_name = "bn5_act_1_2_prod_lock"}
    %bn5_act_1_2_cons_lock = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "bn5_act_1_2_cons_lock"}
    %act_bn5_bn6_cons_buff_0 = aie.buffer(%tile_1_2) {address = 57744 : i32, sym_name = "act_bn5_bn6_cons_buff_0"} : memref<28x1x40xi8> 
    %act_bn5_bn6_cons_buff_1 = aie.buffer(%tile_1_2) {address = 58864 : i32, sym_name = "act_bn5_bn6_cons_buff_1"} : memref<28x1x40xi8> 
    %act_bn5_bn6_cons_buff_2 = aie.buffer(%tile_1_2) {address = 59984 : i32, sym_name = "act_bn5_bn6_cons_buff_2"} : memref<28x1x40xi8> 
    %act_bn5_bn6_cons_prod_lock = aie.lock(%tile_1_2, 2) {init = 3 : i32, sym_name = "act_bn5_bn6_cons_prod_lock"}
    %act_bn5_bn6_cons_cons_lock = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "act_bn5_bn6_cons_cons_lock"}
    %act_bn5_bn6_buff_0 = aie.buffer(%tile_1_4) {address = 25144 : i32, sym_name = "act_bn5_bn6_buff_0"} : memref<28x1x40xi8> 
    %act_bn5_bn6_buff_1 = aie.buffer(%tile_1_4) {address = 26264 : i32, sym_name = "act_bn5_bn6_buff_1"} : memref<28x1x40xi8> 
    %act_bn5_bn6_prod_lock = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "act_bn5_bn6_prod_lock"}
    %act_bn5_bn6_cons_lock = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "act_bn5_bn6_cons_lock"}
    %bn4_act_2_3_buff_0 = aie.buffer(%tile_1_5) {address = 11704 : i32, sym_name = "bn4_act_2_3_buff_0"} : memref<28x1x120xui8> 
    %bn4_act_2_3_prod_lock = aie.lock(%tile_1_5, 6) {init = 1 : i32, sym_name = "bn4_act_2_3_prod_lock"}
    %bn4_act_2_3_cons_lock = aie.lock(%tile_1_5, 7) {init = 0 : i32, sym_name = "bn4_act_2_3_cons_lock"}
    %bn4_act_1_2_buff_0 = aie.buffer(%tile_1_5) {address = 15064 : i32, sym_name = "bn4_act_1_2_buff_0"} : memref<28x1x120xui8> 
    %bn4_act_1_2_buff_1 = aie.buffer(%tile_1_5) {address = 18424 : i32, sym_name = "bn4_act_1_2_buff_1"} : memref<28x1x120xui8> 
    %bn4_act_1_2_buff_2 = aie.buffer(%tile_1_5) {address = 21784 : i32, sym_name = "bn4_act_1_2_buff_2"} : memref<28x1x120xui8> 
    %bn4_act_1_2_prod_lock = aie.lock(%tile_1_5, 4) {init = 3 : i32, sym_name = "bn4_act_1_2_prod_lock"}
    %bn4_act_1_2_cons_lock = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "bn4_act_1_2_cons_lock"}
    %act_bn4_bn5_buff_0 = aie.buffer(%tile_1_5) {address = 25144 : i32, sym_name = "act_bn4_bn5_buff_0"} : memref<28x1x40xi8> 
    %act_bn4_bn5_buff_1 = aie.buffer(%tile_1_5) {address = 26264 : i32, sym_name = "act_bn4_bn5_buff_1"} : memref<28x1x40xi8> 
    %act_bn4_bn5_buff_2 = aie.buffer(%tile_1_5) {address = 27384 : i32, sym_name = "act_bn4_bn5_buff_2"} : memref<28x1x40xi8> 
    %act_bn4_bn5_prod_lock = aie.lock(%tile_1_5, 2) {init = 3 : i32, sym_name = "act_bn4_bn5_prod_lock"}
    %act_bn4_bn5_cons_lock = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "act_bn4_bn5_cons_lock"}
    %bn3_act_2_3_buff_0 = aie.buffer(%tile_0_5) {address = 18376 : i32, sym_name = "bn3_act_2_3_buff_0"} : memref<28x1x72xui8> 
    %bn3_act_2_3_prod_lock = aie.lock(%tile_0_5, 6) {init = 1 : i32, sym_name = "bn3_act_2_3_prod_lock"}
    %bn3_act_2_3_cons_lock = aie.lock(%tile_0_5, 7) {init = 0 : i32, sym_name = "bn3_act_2_3_cons_lock"}
    %bn3_act_1_2_buff_0 = aie.buffer(%tile_0_5) {address = 6280 : i32, sym_name = "bn3_act_1_2_buff_0"} : memref<56x1x72xui8> 
    %bn3_act_1_2_buff_1 = aie.buffer(%tile_0_5) {address = 10312 : i32, sym_name = "bn3_act_1_2_buff_1"} : memref<56x1x72xui8> 
    %bn3_act_1_2_buff_2 = aie.buffer(%tile_0_5) {address = 14344 : i32, sym_name = "bn3_act_1_2_buff_2"} : memref<56x1x72xui8> 
    %bn3_act_1_2_prod_lock = aie.lock(%tile_0_5, 4) {init = 3 : i32, sym_name = "bn3_act_1_2_prod_lock"}
    %bn3_act_1_2_cons_lock = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "bn3_act_1_2_cons_lock"}
    %act_bn3_bn4_buff_0 = aie.buffer(%tile_0_5) {address = 20392 : i32, sym_name = "act_bn3_bn4_buff_0"} : memref<28x1x40xi8> 
    %act_bn3_bn4_buff_1 = aie.buffer(%tile_0_5) {address = 21512 : i32, sym_name = "act_bn3_bn4_buff_1"} : memref<28x1x40xi8> 
    %act_bn3_bn4_buff_2 = aie.buffer(%tile_0_5) {address = 22632 : i32, sym_name = "act_bn3_bn4_buff_2"} : memref<28x1x40xi8> 
    %act_bn3_bn4_prod_lock = aie.lock(%tile_0_5, 2) {init = 3 : i32, sym_name = "act_bn3_bn4_prod_lock"}
    %act_bn3_bn4_cons_lock = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "act_bn3_bn4_cons_lock"}
    %bn2_act_2_3_buff_0 = aie.buffer(%tile_0_4) {address = 5128 : i32, sym_name = "bn2_act_2_3_buff_0"} : memref<56x1x72xui8> 
    %bn2_act_2_3_prod_lock = aie.lock(%tile_0_4, 6) {init = 1 : i32, sym_name = "bn2_act_2_3_prod_lock"}
    %bn2_act_2_3_cons_lock = aie.lock(%tile_0_4, 7) {init = 0 : i32, sym_name = "bn2_act_2_3_cons_lock"}
    %bn2_act_1_2_buff_0 = aie.buffer(%tile_0_4) {address = 9160 : i32, sym_name = "bn2_act_1_2_buff_0"} : memref<56x1x72xui8> 
    %bn2_act_1_2_buff_1 = aie.buffer(%tile_0_4) {address = 13192 : i32, sym_name = "bn2_act_1_2_buff_1"} : memref<56x1x72xui8> 
    %bn2_act_1_2_buff_2 = aie.buffer(%tile_0_4) {address = 17224 : i32, sym_name = "bn2_act_1_2_buff_2"} : memref<56x1x72xui8> 
    %bn2_act_1_2_prod_lock = aie.lock(%tile_0_4, 4) {init = 3 : i32, sym_name = "bn2_act_1_2_prod_lock"}
    %bn2_act_1_2_cons_lock = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "bn2_act_1_2_cons_lock"}
    %act_bn2_bn3_buff_0 = aie.buffer(%tile_0_4) {address = 21256 : i32, sym_name = "act_bn2_bn3_buff_0"} : memref<56x1x24xi8> 
    %act_bn2_bn3_buff_1 = aie.buffer(%tile_0_4) {address = 22600 : i32, sym_name = "act_bn2_bn3_buff_1"} : memref<56x1x24xi8> 
    %act_bn2_bn3_buff_2 = aie.buffer(%tile_0_4) {address = 23944 : i32, sym_name = "act_bn2_bn3_buff_2"} : memref<56x1x24xi8> 
    %act_bn2_bn3_prod_lock = aie.lock(%tile_0_4, 2) {init = 3 : i32, sym_name = "act_bn2_bn3_prod_lock"}
    %act_bn2_bn3_cons_lock = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "act_bn2_bn3_cons_lock"}
    %bn01_act_bn1_2_3_buff_0 = aie.buffer(%tile_0_3) {address = 22528 : i32, sym_name = "bn01_act_bn1_2_3_buff_0"} : memref<56x1x64xui8> 
    %bn01_act_bn1_2_3_prod_lock = aie.lock(%tile_0_3, 12) {init = 1 : i32, sym_name = "bn01_act_bn1_2_3_prod_lock"}
    %bn01_act_bn1_2_3_cons_lock = aie.lock(%tile_0_3, 13) {init = 0 : i32, sym_name = "bn01_act_bn1_2_3_cons_lock"}
    %bn01_act_bn1_1_2_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "bn01_act_bn1_1_2_buff_0"} : memref<112x1x64xui8> 
    %bn01_act_bn1_1_2_buff_1 = aie.buffer(%tile_0_3) {address = 8192 : i32, sym_name = "bn01_act_bn1_1_2_buff_1"} : memref<112x1x64xui8> 
    %bn01_act_bn1_1_2_buff_2 = aie.buffer(%tile_0_3) {address = 15360 : i32, sym_name = "bn01_act_bn1_1_2_buff_2"} : memref<112x1x64xui8> 
    %bn01_act_bn1_1_2_prod_lock = aie.lock(%tile_0_3, 10) {init = 3 : i32, sym_name = "bn01_act_bn1_1_2_prod_lock"}
    %bn01_act_bn1_1_2_cons_lock = aie.lock(%tile_0_3, 11) {init = 0 : i32, sym_name = "bn01_act_bn1_1_2_cons_lock"}
    %bn01_act_bn0_bn1_buff_0 = aie.buffer(%tile_0_3) {address = 29648 : i32, sym_name = "bn01_act_bn0_bn1_buff_0"} : memref<112x1x16xi8> 
    %bn01_act_bn0_bn1_prod_lock = aie.lock(%tile_0_3, 8) {init = 1 : i32, sym_name = "bn01_act_bn0_bn1_prod_lock"}
    %bn01_act_bn0_bn1_cons_lock = aie.lock(%tile_0_3, 9) {init = 0 : i32, sym_name = "bn01_act_bn0_bn1_cons_lock"}
    %bn01_act_bn0_2_3_buff_0 = aie.buffer(%tile_0_3) {address = 31440 : i32, sym_name = "bn01_act_bn0_2_3_buff_0"} : memref<112x1x16xui8> 
    %bn01_act_bn0_2_3_prod_lock = aie.lock(%tile_0_3, 6) {init = 1 : i32, sym_name = "bn01_act_bn0_2_3_prod_lock"}
    %bn01_act_bn0_2_3_cons_lock = aie.lock(%tile_0_3, 7) {init = 0 : i32, sym_name = "bn01_act_bn0_2_3_cons_lock"}
    %act_bn01_bn2_buff_0 = aie.buffer(%tile_0_3) {address = 38608 : i32, sym_name = "act_bn01_bn2_buff_0"} : memref<56x1x24xi8> 
    %act_bn01_bn2_buff_1 = aie.buffer(%tile_0_3) {address = 39952 : i32, sym_name = "act_bn01_bn2_buff_1"} : memref<56x1x24xi8> 
    %act_bn01_bn2_buff_2 = aie.buffer(%tile_0_3) {address = 41296 : i32, sym_name = "act_bn01_bn2_buff_2"} : memref<56x1x24xi8> 
    %act_bn01_bn2_prod_lock = aie.lock(%tile_0_3, 4) {init = 3 : i32, sym_name = "act_bn01_bn2_prod_lock"}
    %act_bn01_bn2_cons_lock = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "act_bn01_bn2_cons_lock"}
    %bn9_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 1024 : i32, sym_name = "bn9_wts_OF_L2L1_cons_buff_0"} : memref<31096xi8> 
    %bn9_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_2_3, 0) {init = 1 : i32, sym_name = "bn9_wts_OF_L2L1_cons_prod_lock"}
    %bn9_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "bn9_wts_OF_L2L1_cons_cons_lock"}
    %bn8_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, sym_name = "bn8_wts_OF_L2L1_cons_buff_0"} : memref<31096xi8> 
    %bn8_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_2_2, 0) {init = 1 : i32, sym_name = "bn8_wts_OF_L2L1_cons_prod_lock"}
    %bn8_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "bn8_wts_OF_L2L1_cons_cons_lock"}
    %bn7_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, sym_name = "bn7_wts_OF_L2L1_cons_buff_0"} : memref<33800xi8> 
    %bn7_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_1_3, 0) {init = 1 : i32, sym_name = "bn7_wts_OF_L2L1_cons_prod_lock"}
    %bn7_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "bn7_wts_OF_L2L1_cons_cons_lock"}
    %bn6_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "bn6_wts_OF_L2L1_cons_buff_0"} : memref<30960xi8> 
    %bn6_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "bn6_wts_OF_L2L1_cons_prod_lock"}
    %bn6_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "bn6_wts_OF_L2L1_cons_cons_lock"}
    %wts_OF_11_L3L2_cons_buff_0 = aie.buffer(%tile_1_1) {address = 0 : i32, sym_name = "wts_OF_11_L3L2_cons_buff_0"} : memref<126952xi8> 
    %wts_OF_11_L3L2_cons_prod_lock = aie.lock(%tile_1_1, 0) {init = 4 : i32, sym_name = "wts_OF_11_L3L2_cons_prod_lock"}
    %wts_OF_11_L3L2_cons_cons_lock = aie.lock(%tile_1_1, 1) {init = 0 : i32, sym_name = "wts_OF_11_L3L2_cons_cons_lock"}
    %wts_OF_11_L3L2_prod_lock = aie.lock(%tile_1_0, 0) {init = 1 : i32, sym_name = "wts_OF_11_L3L2_prod_lock"}
    %wts_OF_11_L3L2_cons_lock = aie.lock(%tile_1_0, 1) {init = 0 : i32, sym_name = "wts_OF_11_L3L2_cons_lock"}
    %bn5_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, sym_name = "bn5_wts_OF_L2L1_cons_buff_0"} : memref<10680xi8> 
    %bn5_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_1_4, 0) {init = 1 : i32, sym_name = "bn5_wts_OF_L2L1_cons_prod_lock"}
    %bn5_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "bn5_wts_OF_L2L1_cons_cons_lock"}
    %bn4_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, sym_name = "bn4_wts_OF_L2L1_cons_buff_0"} : memref<10680xi8> 
    %bn4_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_1_5, 0) {init = 1 : i32, sym_name = "bn4_wts_OF_L2L1_cons_prod_lock"}
    %bn4_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "bn4_wts_OF_L2L1_cons_cons_lock"}
    %bn3_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "bn3_wts_OF_L2L1_cons_buff_0"} : memref<5256xi8> 
    %bn3_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_0_5, 0) {init = 1 : i32, sym_name = "bn3_wts_OF_L2L1_cons_prod_lock"}
    %bn3_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "bn3_wts_OF_L2L1_cons_cons_lock"}
    %bn2_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "bn2_wts_OF_L2L1_cons_buff_0"} : memref<4104xi8> 
    %bn2_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "bn2_wts_OF_L2L1_cons_prod_lock"}
    %bn2_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "bn2_wts_OF_L2L1_cons_cons_lock"}
    %bn0_1_wts_OF_L2L1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 26112 : i32, sym_name = "bn0_1_wts_OF_L2L1_cons_buff_0"} : memref<3536xi8> 
    %bn0_1_wts_OF_L2L1_cons_prod_lock = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "bn0_1_wts_OF_L2L1_cons_prod_lock"}
    %bn0_1_wts_OF_L2L1_cons_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "bn0_1_wts_OF_L2L1_cons_cons_lock"}
    %wts_OF_01_L3L2_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "wts_OF_01_L3L2_cons_buff_0"} : memref<34256xi8> 
    %wts_OF_01_L3L2_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 5 : i32, sym_name = "wts_OF_01_L3L2_cons_prod_lock"}
    %wts_OF_01_L3L2_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "wts_OF_01_L3L2_cons_cons_lock"}
    %wts_OF_01_L3L2_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "wts_OF_01_L3L2_prod_lock"}
    %wts_OF_01_L3L2_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "wts_OF_01_L3L2_cons_lock"}
    %act_in_cons_buff_0 = aie.buffer(%tile_0_3) {address = 33232 : i32, sym_name = "act_in_cons_buff_0"} : memref<112x1x16xui8> 
    %act_in_cons_buff_1 = aie.buffer(%tile_0_3) {address = 35024 : i32, sym_name = "act_in_cons_buff_1"} : memref<112x1x16xui8> 
    %act_in_cons_buff_2 = aie.buffer(%tile_0_3) {address = 36816 : i32, sym_name = "act_in_cons_buff_2"} : memref<112x1x16xui8> 
    %act_in_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 3 : i32, sym_name = "act_in_cons_prod_lock"}
    %act_in_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "act_in_cons_cons_lock"}
    %act_in_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "act_in_prod_lock"}
    %act_in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "act_in_cons_lock"}
    %weightsInBN11_layer3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, sym_name = "weightsInBN11_layer3_cons_buff_0"} : memref<37632xi8> 
    %weightsInBN11_layer3_cons_prod_lock = aie.lock(%tile_3_2, 0) {init = 1 : i32, sym_name = "weightsInBN11_layer3_cons_prod_lock"}
    %weightsInBN11_layer3_cons_cons_lock = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "weightsInBN11_layer3_cons_cons_lock"}
    %weightsInBN11_layer2_cons_buff_0 = aie.buffer(%tile_3_3) {address = 29248 : i32, sym_name = "weightsInBN11_layer2_cons_buff_0"} : memref<3024xi8> 
    %weightsInBN11_layer2_cons_prod_lock = aie.lock(%tile_3_3, 0) {init = 1 : i32, sym_name = "weightsInBN11_layer2_cons_prod_lock"}
    %weightsInBN11_layer2_cons_cons_lock = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "weightsInBN11_layer2_cons_cons_lock"}
    %weightsInBN11_layer1_cons_buff_0 = aie.buffer(%tile_3_4) {address = 1024 : i32, sym_name = "weightsInBN11_layer1_cons_buff_0"} : memref<37632xi8> 
    %weightsInBN11_layer1_cons_prod_lock = aie.lock(%tile_3_4, 0) {init = 1 : i32, sym_name = "weightsInBN11_layer1_cons_prod_lock"}
    %weightsInBN11_layer1_cons_cons_lock = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "weightsInBN11_layer1_cons_cons_lock"}
    %wts_b11_L3L2_cons_buff_0 = aie.buffer(%tile_3_1) {address = 0 : i32, sym_name = "wts_b11_L3L2_cons_buff_0"} : memref<78288xi8> 
    %wts_b11_L3L2_cons_prod_lock = aie.lock(%tile_3_1, 0) {init = 3 : i32, sym_name = "wts_b11_L3L2_cons_prod_lock"}
    %wts_b11_L3L2_cons_cons_lock = aie.lock(%tile_3_1, 1) {init = 0 : i32, sym_name = "wts_b11_L3L2_cons_cons_lock"}
    %wts_b11_L3L2_prod_lock = aie.lock(%tile_3_0, 0) {init = 1 : i32, sym_name = "wts_b11_L3L2_prod_lock"}
    %wts_b11_L3L2_cons_lock = aie.lock(%tile_3_0, 1) {init = 0 : i32, sym_name = "wts_b11_L3L2_cons_lock"}
    %weightsInBN10_layer3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 1024 : i32, sym_name = "weightsInBN10_layer3_cons_buff_0"} : memref<53760xi8> 
    %weightsInBN10_layer3_cons_prod_lock = aie.lock(%tile_3_5, 0) {init = 1 : i32, sym_name = "weightsInBN10_layer3_cons_prod_lock"}
    %weightsInBN10_layer3_cons_cons_lock = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "weightsInBN10_layer3_cons_cons_lock"}
    %weightsInBN10_layer2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 41344 : i32, sym_name = "weightsInBN10_layer2_cons_buff_0"} : memref<4320xi8> 
    %weightsInBN10_layer2_cons_prod_lock = aie.lock(%tile_2_5, 0) {init = 1 : i32, sym_name = "weightsInBN10_layer2_cons_prod_lock"}
    %weightsInBN10_layer2_cons_cons_lock = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "weightsInBN10_layer2_cons_cons_lock"}
    %weightsInBN10_layer1_cons_buff_0 = aie.buffer(%tile_2_4) {address = 1024 : i32, sym_name = "weightsInBN10_layer1_cons_buff_0"} : memref<38400xi8> 
    %weightsInBN10_layer1_cons_prod_lock = aie.lock(%tile_2_4, 0) {init = 1 : i32, sym_name = "weightsInBN10_layer1_cons_prod_lock"}
    %weightsInBN10_layer1_cons_cons_lock = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "weightsInBN10_layer1_cons_cons_lock"}
    %wts_b10_L3L2_cons_buff_0 = aie.buffer(%tile_2_1) {address = 0 : i32, sym_name = "wts_b10_L3L2_cons_buff_0"} : memref<96480xi8> 
    %wts_b10_L3L2_cons_prod_lock = aie.lock(%tile_2_1, 0) {init = 3 : i32, sym_name = "wts_b10_L3L2_cons_prod_lock"}
    %wts_b10_L3L2_cons_cons_lock = aie.lock(%tile_2_1, 1) {init = 0 : i32, sym_name = "wts_b10_L3L2_cons_cons_lock"}
    %wts_b10_L3L2_prod_lock = aie.lock(%tile_2_0, 0) {init = 1 : i32, sym_name = "wts_b10_L3L2_prod_lock"}
    %wts_b10_L3L2_cons_lock = aie.lock(%tile_2_0, 1) {init = 0 : i32, sym_name = "wts_b10_L3L2_cons_lock"}
    %switchbox_2_0 = aie.switchbox(%tile_2_0) {
      aie.connect<South : 3, North : 0>
    }
    %shim_mux_2_0 = aie.shim_mux(%tile_2_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_2_1 = aie.switchbox(%tile_2_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, North : 0>
      aie.connect<DMA : 1, North : 1>
      aie.connect<DMA : 2, North : 2>
      aie.connect<North : 0, DMA : 1>
      aie.connect<DMA : 3, North : 3>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<West : 0, DMA : 0>
      aie.connect<West : 1, DMA : 1>
      aie.connect<East : 0, South : 0>
      aie.connect<South : 3, East : 0>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<West : 0, DMA : 0>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, North : 0>
      aie.connect<South : 2, North : 1>
      aie.connect<DMA : 0, North : 2>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, East : 0>
      aie.connect<South : 2, DMA : 1>
    }
    %switchbox_3_5 = aie.switchbox(%tile_3_5) {
      aie.connect<West : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_3_0 = aie.switchbox(%tile_3_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_3_0 = aie.shim_mux(%tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, North : 0>
      aie.connect<DMA : 1, North : 1>
      aie.connect<DMA : 2, North : 2>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, DMA : 0>
      aie.connect<DMA : 0, South : 0>
      aie.connect<North : 0, West : 0>
      aie.connect<West : 0, DMA : 1>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, DMA : 0>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, DMA : 1>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<North : 0, DMA : 1>
      aie.connect<North : 0, South : 0>
      aie.connect<DMA : 0, South : 1>
    }
    %bn10_1_rtp = aie.buffer(%tile_2_4) {address = 52864 : i32, sym_name = "bn10_1_rtp"} : memref<16xi32> 
    %bn10_2_rtp = aie.buffer(%tile_2_5) {address = 45664 : i32, sym_name = "bn10_2_rtp"} : memref<16xi32> 
    %bn10_3_rtp = aie.buffer(%tile_3_5) {address = 57920 : i32, sym_name = "bn10_3_rtp"} : memref<16xi32> 
    %bn11_1_rtp = aie.buffer(%tile_3_4) {address = 51200 : i32, sym_name = "bn11_1_rtp"} : memref<16xi32> 
    %bn11_2_rtp = aie.buffer(%tile_3_3) {address = 32272 : i32, sym_name = "bn11_2_rtp"} : memref<16xi32> 
    %bn11_3_rtp = aie.buffer(%tile_3_2) {address = 44928 : i32, sym_name = "bn11_3_rtp"} : memref<16xi32> 
    %rtp03 = aie.buffer(%tile_0_3) {address = 42640 : i32, sym_name = "rtp03"} : memref<16xi32> 
    %rtp04 = aie.buffer(%tile_0_4) {address = 25288 : i32, sym_name = "rtp04"} : memref<16xi32> 
    %rtp05 = aie.buffer(%tile_0_5) {address = 23752 : i32, sym_name = "rtp05"} : memref<16xi32> 
    %rtp15 = aie.buffer(%tile_1_5) {address = 28504 : i32, sym_name = "rtp15"} : memref<16xi32> 
    %rtp14 = aie.buffer(%tile_1_4) {address = 27384 : i32, sym_name = "rtp14"} : memref<16xi32> 
    %rtp12 = aie.buffer(%tile_1_2) {address = 61104 : i32, sym_name = "rtp12"} : memref<16xi32> 
    %rtp13 = aie.buffer(%tile_1_3) {address = 48264 : i32, sym_name = "rtp13"} : memref<16xi32> 
    %rtp22 = aie.buffer(%tile_2_2) {address = 48024 : i32, sym_name = "rtp22"} : memref<16xi32> 
    %rtp23 = aie.buffer(%tile_2_3) {address = 44664 : i32, sym_name = "rtp23"} : memref<16xi32> 
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 2>
      aie.connect<DMA : 2, North : 3>
      aie.connect<DMA : 3, North : 4>
      aie.connect<DMA : 4, North : 5>
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 3, North : 3>
      aie.connect<South : 4, North : 4>
      aie.connect<South : 5, North : 5>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<South : 2, North : 0>
      aie.connect<South : 3, North : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<South : 5, North : 3>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, North : 0>
      aie.connect<South : 2, North : 1>
      aie.connect<South : 3, East : 0>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, East : 0>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<West : 0, DMA : 0>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<West : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%tile_1_0) {
      aie.connect<South : 3, North : 0>
    }
    %shim_mux_1_0 = aie.shim_mux(%tile_1_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_1_1 = aie.switchbox(%tile_1_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, North : 0>
      aie.connect<DMA : 1, North : 1>
      aie.connect<DMA : 2, North : 2>
      aie.connect<DMA : 3, North : 3>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, North : 0>
      aie.connect<South : 2, East : 0>
      aie.connect<South : 3, North : 1>
      aie.connect<North : 0, DMA : 1>
      aie.connect<North : 1, East : 1>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, East : 0>
      aie.connect<North : 0, South : 0>
      aie.connect<DMA : 0, South : 1>
    }
    func.func private @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn0_conv2dk1_skip_ui8_ui8_i8(memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32)
    func.func private @bn1_conv2dk1_relu_i8_ui8(memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32)
    func.func private @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn1_conv2dk1_ui8_i8(memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32)
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn0_1_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn0_1_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<3536xi8> to memref<144xi8>
      %c144 = arith.constant 144 : index
      %view_2 = memref.view %bn0_1_wts_OF_L2L1_cons_buff_0[%c144][] : memref<3536xi8> to memref<256xi8>
      %c400 = arith.constant 400 : index
      %view_3 = memref.view %bn0_1_wts_OF_L2L1_cons_buff_0[%c400][] : memref<3536xi8> to memref<1024xi8>
      %c1424 = arith.constant 1424 : index
      %view_4 = memref.view %bn0_1_wts_OF_L2L1_cons_buff_0[%c1424][] : memref<3536xi8> to memref<576xi8>
      %c2000 = arith.constant 2000 : index
      %view_5 = memref.view %bn0_1_wts_OF_L2L1_cons_buff_0[%c2000][] : memref<3536xi8> to memref<1536xi8>
      %c0_6 = arith.constant 0 : index
      %0 = memref.load %rtp03[%c0_6] : memref<16xi32>
      %c1_7 = arith.constant 1 : index
      %1 = memref.load %rtp03[%c1_7] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp03[%c2] : memref<16xi32>
      %c3 = arith.constant 3 : index
      %3 = memref.load %rtp03[%c3] : memref<16xi32>
      %c4 = arith.constant 4 : index
      %4 = memref.load %rtp03[%c4] : memref<16xi32>
      %c5 = arith.constant 5 : index
      %5 = memref.load %rtp03[%c5] : memref<16xi32>
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32 = arith.constant 112 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_8 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_9 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_0, %act_in_cons_buff_0, %act_in_cons_buff_1, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32, %c1_i32, %c16_i32, %c3_i32, %c3_i32_8, %c0_i32, %0, %c0_i32_9) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_10 = arith.constant 112 : i32
      %c16_i32_11 = arith.constant 16 : i32
      %c16_i32_12 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_0, %c112_i32_10, %c16_i32_11, %c16_i32_12, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_13 = arith.constant 112 : i32
      %c16_i32_14 = arith.constant 16 : i32
      %c64_i32 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_0, %c112_i32_13, %c16_i32_14, %c64_i32, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_15 = arith.constant 112 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %c16_i32_17 = arith.constant 16 : i32
      %c3_i32_18 = arith.constant 3 : i32
      %c3_i32_19 = arith.constant 3 : i32
      %c1_i32_20 = arith.constant 1 : i32
      %c0_i32_21 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_0, %act_in_cons_buff_1, %act_in_cons_buff_2, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_15, %c1_i32_16, %c16_i32_17, %c3_i32_18, %c3_i32_19, %c1_i32_20, %0, %c0_i32_21) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_22 = arith.constant 112 : i32
      %c16_i32_23 = arith.constant 16 : i32
      %c16_i32_24 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_1, %c112_i32_22, %c16_i32_23, %c16_i32_24, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_25 = arith.constant 112 : i32
      %c16_i32_26 = arith.constant 16 : i32
      %c64_i32_27 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_1, %c112_i32_25, %c16_i32_26, %c64_i32_27, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_28 = arith.constant 112 : i32
      %c1_i32_29 = arith.constant 1 : i32
      %c64_i32_30 = arith.constant 64 : i32
      %c3_i32_31 = arith.constant 3 : i32
      %c3_i32_32 = arith.constant 3 : i32
      %c0_i32_33 = arith.constant 0 : i32
      %c0_i32_34 = arith.constant 0 : i32
      func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%bn01_act_bn1_1_2_buff_0, %bn01_act_bn1_1_2_buff_0, %bn01_act_bn1_1_2_buff_1, %view_4, %bn01_act_bn1_2_3_buff_0, %c112_i32_28, %c1_i32_29, %c64_i32_30, %c3_i32_31, %c3_i32_32, %c0_i32_33, %4, %c0_i32_34) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn01_bn2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32 = arith.constant 56 : i32
      %c64_i32_35 = arith.constant 64 : i32
      %c24_i32 = arith.constant 24 : i32
      func.call @bn1_conv2dk1_ui8_i8(%bn01_act_bn1_2_3_buff_0, %view_5, %act_bn01_bn2_buff_0, %c56_i32, %c64_i32_35, %c24_i32, %5) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, Release, 1)
      %c0_36 = arith.constant 0 : index
      %c54 = arith.constant 54 : index
      %c1_37 = arith.constant 1 : index
      %c3_38 = arith.constant 3 : index
      cf.br ^bb1(%c0_36 : index)
    ^bb1(%6: index):  // 2 preds: ^bb0, ^bb2
      %7 = arith.cmpi slt, %6, %c54 : index
      cf.cond_br %7, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %c0_39 = arith.constant 0 : index
      %c2_40 = arith.constant 2 : index
      %c1_41 = arith.constant 1 : index
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_42 = arith.constant 112 : i32
      %c1_i32_43 = arith.constant 1 : i32
      %c16_i32_44 = arith.constant 16 : i32
      %c3_i32_45 = arith.constant 3 : i32
      %c3_i32_46 = arith.constant 3 : i32
      %c1_i32_47 = arith.constant 1 : i32
      %c0_i32_48 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_1, %act_in_cons_buff_2, %act_in_cons_buff_0, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_42, %c1_i32_43, %c16_i32_44, %c3_i32_45, %c3_i32_46, %c1_i32_47, %0, %c0_i32_48) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_49 = arith.constant 112 : i32
      %c16_i32_50 = arith.constant 16 : i32
      %c16_i32_51 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_2, %c112_i32_49, %c16_i32_50, %c16_i32_51, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_52 = arith.constant 112 : i32
      %c16_i32_53 = arith.constant 16 : i32
      %c64_i32_54 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_2, %c112_i32_52, %c16_i32_53, %c64_i32_54, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_55 = arith.constant 112 : i32
      %c1_i32_56 = arith.constant 1 : i32
      %c16_i32_57 = arith.constant 16 : i32
      %c3_i32_58 = arith.constant 3 : i32
      %c3_i32_59 = arith.constant 3 : i32
      %c1_i32_60 = arith.constant 1 : i32
      %c0_i32_61 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_2, %act_in_cons_buff_0, %act_in_cons_buff_1, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_55, %c1_i32_56, %c16_i32_57, %c3_i32_58, %c3_i32_59, %c1_i32_60, %0, %c0_i32_61) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_62 = arith.constant 112 : i32
      %c16_i32_63 = arith.constant 16 : i32
      %c16_i32_64 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_0, %c112_i32_62, %c16_i32_63, %c16_i32_64, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_65 = arith.constant 112 : i32
      %c16_i32_66 = arith.constant 16 : i32
      %c64_i32_67 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_0, %c112_i32_65, %c16_i32_66, %c64_i32_67, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_68 = arith.constant 112 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c64_i32_70 = arith.constant 64 : i32
      %c3_i32_71 = arith.constant 3 : i32
      %c3_i32_72 = arith.constant 3 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c0_i32_74 = arith.constant 0 : i32
      func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%bn01_act_bn1_1_2_buff_1, %bn01_act_bn1_1_2_buff_2, %bn01_act_bn1_1_2_buff_0, %view_4, %bn01_act_bn1_2_3_buff_0, %c112_i32_68, %c1_i32_69, %c64_i32_70, %c3_i32_71, %c3_i32_72, %c1_i32_73, %4, %c0_i32_74) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn01_bn2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_75 = arith.constant 56 : i32
      %c64_i32_76 = arith.constant 64 : i32
      %c24_i32_77 = arith.constant 24 : i32
      func.call @bn1_conv2dk1_ui8_i8(%bn01_act_bn1_2_3_buff_0, %view_5, %act_bn01_bn2_buff_1, %c56_i32_75, %c64_i32_76, %c24_i32_77, %5) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, Release, 1)
      %c0_78 = arith.constant 0 : index
      %c2_79 = arith.constant 2 : index
      %c1_80 = arith.constant 1 : index
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_81 = arith.constant 112 : i32
      %c1_i32_82 = arith.constant 1 : i32
      %c16_i32_83 = arith.constant 16 : i32
      %c3_i32_84 = arith.constant 3 : i32
      %c3_i32_85 = arith.constant 3 : i32
      %c1_i32_86 = arith.constant 1 : i32
      %c0_i32_87 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_0, %act_in_cons_buff_1, %act_in_cons_buff_2, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_81, %c1_i32_82, %c16_i32_83, %c3_i32_84, %c3_i32_85, %c1_i32_86, %0, %c0_i32_87) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_88 = arith.constant 112 : i32
      %c16_i32_89 = arith.constant 16 : i32
      %c16_i32_90 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_1, %c112_i32_88, %c16_i32_89, %c16_i32_90, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_91 = arith.constant 112 : i32
      %c16_i32_92 = arith.constant 16 : i32
      %c64_i32_93 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_1, %c112_i32_91, %c16_i32_92, %c64_i32_93, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_94 = arith.constant 112 : i32
      %c1_i32_95 = arith.constant 1 : i32
      %c16_i32_96 = arith.constant 16 : i32
      %c3_i32_97 = arith.constant 3 : i32
      %c3_i32_98 = arith.constant 3 : i32
      %c1_i32_99 = arith.constant 1 : i32
      %c0_i32_100 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_1, %act_in_cons_buff_2, %act_in_cons_buff_0, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_94, %c1_i32_95, %c16_i32_96, %c3_i32_97, %c3_i32_98, %c1_i32_99, %0, %c0_i32_100) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_101 = arith.constant 112 : i32
      %c16_i32_102 = arith.constant 16 : i32
      %c16_i32_103 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_2, %c112_i32_101, %c16_i32_102, %c16_i32_103, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_104 = arith.constant 112 : i32
      %c16_i32_105 = arith.constant 16 : i32
      %c64_i32_106 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_2, %c112_i32_104, %c16_i32_105, %c64_i32_106, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_107 = arith.constant 112 : i32
      %c1_i32_108 = arith.constant 1 : i32
      %c64_i32_109 = arith.constant 64 : i32
      %c3_i32_110 = arith.constant 3 : i32
      %c3_i32_111 = arith.constant 3 : i32
      %c1_i32_112 = arith.constant 1 : i32
      %c0_i32_113 = arith.constant 0 : i32
      func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%bn01_act_bn1_1_2_buff_0, %bn01_act_bn1_1_2_buff_1, %bn01_act_bn1_1_2_buff_2, %view_4, %bn01_act_bn1_2_3_buff_0, %c112_i32_107, %c1_i32_108, %c64_i32_109, %c3_i32_110, %c3_i32_111, %c1_i32_112, %4, %c0_i32_113) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn01_bn2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_114 = arith.constant 56 : i32
      %c64_i32_115 = arith.constant 64 : i32
      %c24_i32_116 = arith.constant 24 : i32
      func.call @bn1_conv2dk1_ui8_i8(%bn01_act_bn1_2_3_buff_0, %view_5, %act_bn01_bn2_buff_2, %c56_i32_114, %c64_i32_115, %c24_i32_116, %5) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, Release, 1)
      %c0_117 = arith.constant 0 : index
      %c2_118 = arith.constant 2 : index
      %c1_119 = arith.constant 1 : index
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_120 = arith.constant 112 : i32
      %c1_i32_121 = arith.constant 1 : i32
      %c16_i32_122 = arith.constant 16 : i32
      %c3_i32_123 = arith.constant 3 : i32
      %c3_i32_124 = arith.constant 3 : i32
      %c1_i32_125 = arith.constant 1 : i32
      %c0_i32_126 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_2, %act_in_cons_buff_0, %act_in_cons_buff_1, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_120, %c1_i32_121, %c16_i32_122, %c3_i32_123, %c3_i32_124, %c1_i32_125, %0, %c0_i32_126) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_127 = arith.constant 112 : i32
      %c16_i32_128 = arith.constant 16 : i32
      %c16_i32_129 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_0, %c112_i32_127, %c16_i32_128, %c16_i32_129, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_130 = arith.constant 112 : i32
      %c16_i32_131 = arith.constant 16 : i32
      %c64_i32_132 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_0, %c112_i32_130, %c16_i32_131, %c64_i32_132, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_133 = arith.constant 112 : i32
      %c1_i32_134 = arith.constant 1 : i32
      %c16_i32_135 = arith.constant 16 : i32
      %c3_i32_136 = arith.constant 3 : i32
      %c3_i32_137 = arith.constant 3 : i32
      %c1_i32_138 = arith.constant 1 : i32
      %c0_i32_139 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_0, %act_in_cons_buff_1, %act_in_cons_buff_2, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_133, %c1_i32_134, %c16_i32_135, %c3_i32_136, %c3_i32_137, %c1_i32_138, %0, %c0_i32_139) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_140 = arith.constant 112 : i32
      %c16_i32_141 = arith.constant 16 : i32
      %c16_i32_142 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_1, %c112_i32_140, %c16_i32_141, %c16_i32_142, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_143 = arith.constant 112 : i32
      %c16_i32_144 = arith.constant 16 : i32
      %c64_i32_145 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_1, %c112_i32_143, %c16_i32_144, %c64_i32_145, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_146 = arith.constant 112 : i32
      %c1_i32_147 = arith.constant 1 : i32
      %c64_i32_148 = arith.constant 64 : i32
      %c3_i32_149 = arith.constant 3 : i32
      %c3_i32_150 = arith.constant 3 : i32
      %c1_i32_151 = arith.constant 1 : i32
      %c0_i32_152 = arith.constant 0 : i32
      func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%bn01_act_bn1_1_2_buff_2, %bn01_act_bn1_1_2_buff_0, %bn01_act_bn1_1_2_buff_1, %view_4, %bn01_act_bn1_2_3_buff_0, %c112_i32_146, %c1_i32_147, %c64_i32_148, %c3_i32_149, %c3_i32_150, %c1_i32_151, %4, %c0_i32_152) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn01_bn2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_153 = arith.constant 56 : i32
      %c64_i32_154 = arith.constant 64 : i32
      %c24_i32_155 = arith.constant 24 : i32
      func.call @bn1_conv2dk1_ui8_i8(%bn01_act_bn1_2_3_buff_0, %view_5, %act_bn01_bn2_buff_0, %c56_i32_153, %c64_i32_154, %c24_i32_155, %5) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, Release, 1)
      %8 = arith.addi %6, %c3_38 : index
      cf.br ^bb1(%8 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_156 = arith.constant 112 : i32
      %c1_i32_157 = arith.constant 1 : i32
      %c16_i32_158 = arith.constant 16 : i32
      %c3_i32_159 = arith.constant 3 : i32
      %c3_i32_160 = arith.constant 3 : i32
      %c1_i32_161 = arith.constant 1 : i32
      %c0_i32_162 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_1, %act_in_cons_buff_2, %act_in_cons_buff_0, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_156, %c1_i32_157, %c16_i32_158, %c3_i32_159, %c3_i32_160, %c1_i32_161, %0, %c0_i32_162) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_163 = arith.constant 112 : i32
      %c16_i32_164 = arith.constant 16 : i32
      %c16_i32_165 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_2, %c112_i32_163, %c16_i32_164, %c16_i32_165, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_166 = arith.constant 112 : i32
      %c16_i32_167 = arith.constant 16 : i32
      %c64_i32_168 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_2, %c112_i32_166, %c16_i32_167, %c64_i32_168, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_169 = arith.constant 112 : i32
      %c1_i32_170 = arith.constant 1 : i32
      %c16_i32_171 = arith.constant 16 : i32
      %c3_i32_172 = arith.constant 3 : i32
      %c3_i32_173 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_174 = arith.constant 0 : i32
      func.call @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(%act_in_cons_buff_2, %act_in_cons_buff_0, %act_in_cons_buff_0, %view, %bn01_act_bn0_2_3_buff_0, %c112_i32_169, %c1_i32_170, %c16_i32_171, %c3_i32_172, %c3_i32_173, %c2_i32, %0, %c0_i32_174) : (memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>, memref<144xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_175 = arith.constant 112 : i32
      %c16_i32_176 = arith.constant 16 : i32
      %c16_i32_177 = arith.constant 16 : i32
      func.call @bn0_conv2dk1_skip_ui8_ui8_i8(%bn01_act_bn0_2_3_buff_0, %view_2, %bn01_act_bn0_bn1_buff_0, %act_in_cons_buff_0, %c112_i32_175, %c16_i32_176, %c16_i32_177, %1, %2) : (memref<112x1x16xui8>, memref<256xi8>, memref<112x1x16xi8>, memref<112x1x16xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 2)
      aie.use_lock(%bn01_act_bn0_2_3_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn0_bn1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_178 = arith.constant 112 : i32
      %c16_i32_179 = arith.constant 16 : i32
      %c64_i32_180 = arith.constant 64 : i32
      func.call @bn1_conv2dk1_relu_i8_ui8(%bn01_act_bn0_bn1_buff_0, %view_3, %bn01_act_bn1_1_2_buff_0, %c112_i32_178, %c16_i32_179, %c64_i32_180, %3) : (memref<112x1x16xi8>, memref<1024xi8>, memref<112x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn0_bn1_prod_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c112_i32_181 = arith.constant 112 : i32
      %c1_i32_182 = arith.constant 1 : i32
      %c64_i32_183 = arith.constant 64 : i32
      %c3_i32_184 = arith.constant 3 : i32
      %c3_i32_185 = arith.constant 3 : i32
      %c1_i32_186 = arith.constant 1 : i32
      %c0_i32_187 = arith.constant 0 : i32
      func.call @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(%bn01_act_bn1_1_2_buff_1, %bn01_act_bn1_1_2_buff_2, %bn01_act_bn1_1_2_buff_0, %view_4, %bn01_act_bn1_2_3_buff_0, %c112_i32_181, %c1_i32_182, %c64_i32_183, %c3_i32_184, %c3_i32_185, %c1_i32_186, %4, %c0_i32_187) : (memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>, memref<576xi8>, memref<56x1x64xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_1_2_prod_lock, Release, 3)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn01_act_bn1_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn01_bn2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_188 = arith.constant 56 : i32
      %c64_i32_189 = arith.constant 64 : i32
      %c24_i32_190 = arith.constant 24 : i32
      func.call @bn1_conv2dk1_ui8_i8(%bn01_act_bn1_2_3_buff_0, %view_5, %act_bn01_bn2_buff_1, %c56_i32_188, %c64_i32_189, %c24_i32_190, %5) : (memref<56x1x64xui8>, memref<1536xi8>, memref<56x1x24xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn01_act_bn1_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, Release, 1)
      aie.use_lock(%bn0_1_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "fused_bn0_bn1.a"}
    func.func private @bn2_conv2dk1_relu_i8_ui8(memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32)
    func.func private @bn2_conv2dk3_dw_stride2_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn2_conv2dk1_skip_ui8_i8_i8(memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32)
    func.func private @bn2_conv2dk1_ui8_i8(memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, i32, i32, i32, i32)
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn2_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn2_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<4104xi8> to memref<1728xi8>
      %c1728 = arith.constant 1728 : index
      %view_2 = memref.view %bn2_wts_OF_L2L1_cons_buff_0[%c1728][] : memref<4104xi8> to memref<648xi8>
      %c2376 = arith.constant 2376 : index
      %view_3 = memref.view %bn2_wts_OF_L2L1_cons_buff_0[%c2376][] : memref<4104xi8> to memref<1728xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp04[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp04[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp04[%c2] : memref<16xi32>
      %c3 = arith.constant 3 : index
      %3 = memref.load %rtp04[%c3] : memref<16xi32>
      aie.use_lock(%act_bn01_bn2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn2_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c56_i32 = arith.constant 56 : i32
      %c24_i32 = arith.constant 24 : i32
      %c72_i32 = arith.constant 72 : i32
      func.call @bn2_conv2dk1_relu_i8_ui8(%act_bn01_bn2_buff_0, %view, %bn2_act_1_2_buff_0, %c56_i32, %c24_i32, %c72_i32, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      %c56_i32_6 = arith.constant 56 : i32
      %c24_i32_7 = arith.constant 24 : i32
      %c72_i32_8 = arith.constant 72 : i32
      func.call @bn2_conv2dk1_relu_i8_ui8(%act_bn01_bn2_buff_1, %view, %bn2_act_1_2_buff_1, %c56_i32_6, %c24_i32_7, %c72_i32_8, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%bn2_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn2_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_9 = arith.constant 56 : i32
      %c1_i32 = arith.constant 1 : i32
      %c72_i32_10 = arith.constant 72 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%bn2_act_1_2_buff_0, %bn2_act_1_2_buff_0, %bn2_act_1_2_buff_1, %view_2, %bn2_act_2_3_buff_0, %c56_i32_9, %c1_i32, %c72_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn2_bn3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_13 = arith.constant 56 : i32
      %c72_i32_14 = arith.constant 72 : i32
      %c24_i32_15 = arith.constant 24 : i32
      func.call @bn2_conv2dk1_skip_ui8_i8_i8(%bn2_act_2_3_buff_0, %view_3, %act_bn2_bn3_buff_0, %act_bn01_bn2_buff_0, %c56_i32_13, %c72_i32_14, %c24_i32_15, %2, %3) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn01_bn2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, Release, 1)
      %c0_16 = arith.constant 0 : index
      %c54 = arith.constant 54 : index
      %c1_17 = arith.constant 1 : index
      %c3_18 = arith.constant 3 : index
      cf.br ^bb1(%c0_16 : index)
    ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
      %5 = arith.cmpi slt, %4, %c54 : index
      cf.cond_br %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn01_bn2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn2_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_19 = arith.constant 56 : i32
      %c24_i32_20 = arith.constant 24 : i32
      %c72_i32_21 = arith.constant 72 : i32
      func.call @bn2_conv2dk1_relu_i8_ui8(%act_bn01_bn2_buff_2, %view, %bn2_act_1_2_buff_2, %c56_i32_19, %c24_i32_20, %c72_i32_21, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_22 = arith.constant 56 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c72_i32_24 = arith.constant 72 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c3_i32_26 = arith.constant 3 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%bn2_act_1_2_buff_0, %bn2_act_1_2_buff_1, %bn2_act_1_2_buff_2, %view_2, %bn2_act_2_3_buff_0, %c56_i32_22, %c1_i32_23, %c72_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %1, %c0_i32_28) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn2_bn3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_29 = arith.constant 56 : i32
      %c72_i32_30 = arith.constant 72 : i32
      %c24_i32_31 = arith.constant 24 : i32
      func.call @bn2_conv2dk1_skip_ui8_i8_i8(%bn2_act_2_3_buff_0, %view_3, %act_bn2_bn3_buff_1, %act_bn01_bn2_buff_1, %c56_i32_29, %c72_i32_30, %c24_i32_31, %2, %3) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn01_bn2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn2_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_32 = arith.constant 56 : i32
      %c24_i32_33 = arith.constant 24 : i32
      %c72_i32_34 = arith.constant 72 : i32
      func.call @bn2_conv2dk1_relu_i8_ui8(%act_bn01_bn2_buff_0, %view, %bn2_act_1_2_buff_0, %c56_i32_32, %c24_i32_33, %c72_i32_34, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_35 = arith.constant 56 : i32
      %c1_i32_36 = arith.constant 1 : i32
      %c72_i32_37 = arith.constant 72 : i32
      %c3_i32_38 = arith.constant 3 : i32
      %c3_i32_39 = arith.constant 3 : i32
      %c1_i32_40 = arith.constant 1 : i32
      %c0_i32_41 = arith.constant 0 : i32
      func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%bn2_act_1_2_buff_1, %bn2_act_1_2_buff_2, %bn2_act_1_2_buff_0, %view_2, %bn2_act_2_3_buff_0, %c56_i32_35, %c1_i32_36, %c72_i32_37, %c3_i32_38, %c3_i32_39, %c1_i32_40, %1, %c0_i32_41) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn2_bn3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_42 = arith.constant 56 : i32
      %c72_i32_43 = arith.constant 72 : i32
      %c24_i32_44 = arith.constant 24 : i32
      func.call @bn2_conv2dk1_skip_ui8_i8_i8(%bn2_act_2_3_buff_0, %view_3, %act_bn2_bn3_buff_2, %act_bn01_bn2_buff_2, %c56_i32_42, %c72_i32_43, %c24_i32_44, %2, %3) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn01_bn2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, Release, 1)
      aie.use_lock(%act_bn01_bn2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn2_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_45 = arith.constant 56 : i32
      %c24_i32_46 = arith.constant 24 : i32
      %c72_i32_47 = arith.constant 72 : i32
      func.call @bn2_conv2dk1_relu_i8_ui8(%act_bn01_bn2_buff_1, %view, %bn2_act_1_2_buff_1, %c56_i32_45, %c24_i32_46, %c72_i32_47, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_48 = arith.constant 56 : i32
      %c1_i32_49 = arith.constant 1 : i32
      %c72_i32_50 = arith.constant 72 : i32
      %c3_i32_51 = arith.constant 3 : i32
      %c3_i32_52 = arith.constant 3 : i32
      %c1_i32_53 = arith.constant 1 : i32
      %c0_i32_54 = arith.constant 0 : i32
      func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%bn2_act_1_2_buff_2, %bn2_act_1_2_buff_0, %bn2_act_1_2_buff_1, %view_2, %bn2_act_2_3_buff_0, %c56_i32_48, %c1_i32_49, %c72_i32_50, %c3_i32_51, %c3_i32_52, %c1_i32_53, %1, %c0_i32_54) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn2_bn3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_55 = arith.constant 56 : i32
      %c72_i32_56 = arith.constant 72 : i32
      %c24_i32_57 = arith.constant 24 : i32
      func.call @bn2_conv2dk1_skip_ui8_i8_i8(%bn2_act_2_3_buff_0, %view_3, %act_bn2_bn3_buff_0, %act_bn01_bn2_buff_0, %c56_i32_55, %c72_i32_56, %c24_i32_57, %2, %3) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn01_bn2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, Release, 1)
      %6 = arith.addi %4, %c3_18 : index
      cf.br ^bb1(%6 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%bn2_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_58 = arith.constant 56 : i32
      %c1_i32_59 = arith.constant 1 : i32
      %c72_i32_60 = arith.constant 72 : i32
      %c3_i32_61 = arith.constant 3 : i32
      %c3_i32_62 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_63 = arith.constant 0 : i32
      func.call @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(%bn2_act_1_2_buff_0, %bn2_act_1_2_buff_1, %bn2_act_1_2_buff_1, %view_2, %bn2_act_2_3_buff_0, %c56_i32_58, %c1_i32_59, %c72_i32_60, %c3_i32_61, %c3_i32_62, %c2_i32, %1, %c0_i32_63) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<56x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn2_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn2_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn2_bn3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_64 = arith.constant 56 : i32
      %c72_i32_65 = arith.constant 72 : i32
      %c24_i32_66 = arith.constant 24 : i32
      func.call @bn2_conv2dk1_skip_ui8_i8_i8(%bn2_act_2_3_buff_0, %view_3, %act_bn2_bn3_buff_1, %act_bn01_bn2_buff_1, %c56_i32_64, %c72_i32_65, %c24_i32_66, %2, %3) : (memref<56x1x72xui8>, memref<1728xi8>, memref<56x1x24xi8>, memref<56x1x24xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn01_bn2_prod_lock, Release, 1)
      aie.use_lock(%bn2_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, Release, 1)
      aie.use_lock(%bn2_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn3_conv2dk1_relu_i8_ui8(memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32)
    func.func private @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn3_conv2dk3_dw_stride1_relu_ui8_ui8(memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn3_conv2dk1_skip_ui8_i8_i8(memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32)
    func.func private @bn3_conv2dk1_ui8_i8(memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32)
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn3_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn3_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<5256xi8> to memref<1728xi8>
      %c1728 = arith.constant 1728 : index
      %view_2 = memref.view %bn3_wts_OF_L2L1_cons_buff_0[%c1728][] : memref<5256xi8> to memref<648xi8>
      %c2376 = arith.constant 2376 : index
      %view_3 = memref.view %bn3_wts_OF_L2L1_cons_buff_0[%c2376][] : memref<5256xi8> to memref<2880xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp05[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp05[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp05[%c2] : memref<16xi32>
      aie.use_lock(%act_bn2_bn3_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c56_i32 = arith.constant 56 : i32
      %c24_i32 = arith.constant 24 : i32
      %c72_i32 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_0, %view, %bn3_act_1_2_buff_0, %c56_i32, %c24_i32, %c72_i32, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      %c56_i32_6 = arith.constant 56 : i32
      %c24_i32_7 = arith.constant 24 : i32
      %c72_i32_8 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_1, %view, %bn3_act_1_2_buff_1, %c56_i32_6, %c24_i32_7, %c72_i32_8, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn2_bn3_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_9 = arith.constant 56 : i32
      %c1_i32 = arith.constant 1 : i32
      %c72_i32_10 = arith.constant 72 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%bn3_act_1_2_buff_0, %bn3_act_1_2_buff_0, %bn3_act_1_2_buff_1, %view_2, %bn3_act_2_3_buff_0, %c56_i32_9, %c1_i32, %c72_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn3_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn3_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn3_bn4_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32 = arith.constant 28 : i32
      %c72_i32_13 = arith.constant 72 : i32
      %c40_i32 = arith.constant 40 : i32
      func.call @bn3_conv2dk1_ui8_i8(%bn3_act_2_3_buff_0, %view_3, %act_bn3_bn4_buff_0, %c28_i32, %c72_i32_13, %c40_i32, %2) : (memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, Release, 1)
      %c0_14 = arith.constant 0 : index
      %c27 = arith.constant 27 : index
      %c1_15 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      cf.br ^bb1(%c0_14 : index)
    ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
      %4 = arith.cmpi slt, %3, %c27 : index
      cf.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn2_bn3_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c56_i32_16 = arith.constant 56 : i32
      %c24_i32_17 = arith.constant 24 : i32
      %c72_i32_18 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_2, %view, %bn3_act_1_2_buff_2, %c56_i32_16, %c24_i32_17, %c72_i32_18, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      %c56_i32_19 = arith.constant 56 : i32
      %c24_i32_20 = arith.constant 24 : i32
      %c72_i32_21 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_0, %view, %bn3_act_1_2_buff_0, %c56_i32_19, %c24_i32_20, %c72_i32_21, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn2_bn3_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_22 = arith.constant 56 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c72_i32_24 = arith.constant 72 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c3_i32_26 = arith.constant 3 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%bn3_act_1_2_buff_1, %bn3_act_1_2_buff_2, %bn3_act_1_2_buff_0, %view_2, %bn3_act_2_3_buff_0, %c56_i32_22, %c1_i32_23, %c72_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %1, %c0_i32_28) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn3_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn3_bn4_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_29 = arith.constant 28 : i32
      %c72_i32_30 = arith.constant 72 : i32
      %c40_i32_31 = arith.constant 40 : i32
      func.call @bn3_conv2dk1_ui8_i8(%bn3_act_2_3_buff_0, %view_3, %act_bn3_bn4_buff_1, %c28_i32_29, %c72_i32_30, %c40_i32_31, %2) : (memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c56_i32_32 = arith.constant 56 : i32
      %c24_i32_33 = arith.constant 24 : i32
      %c72_i32_34 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_1, %view, %bn3_act_1_2_buff_1, %c56_i32_32, %c24_i32_33, %c72_i32_34, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      %c56_i32_35 = arith.constant 56 : i32
      %c24_i32_36 = arith.constant 24 : i32
      %c72_i32_37 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_2, %view, %bn3_act_1_2_buff_2, %c56_i32_35, %c24_i32_36, %c72_i32_37, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn2_bn3_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_38 = arith.constant 56 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c72_i32_40 = arith.constant 72 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c3_i32_42 = arith.constant 3 : i32
      %c1_i32_43 = arith.constant 1 : i32
      %c0_i32_44 = arith.constant 0 : i32
      func.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%bn3_act_1_2_buff_0, %bn3_act_1_2_buff_1, %bn3_act_1_2_buff_2, %view_2, %bn3_act_2_3_buff_0, %c56_i32_38, %c1_i32_39, %c72_i32_40, %c3_i32_41, %c3_i32_42, %c1_i32_43, %1, %c0_i32_44) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn3_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn3_bn4_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_45 = arith.constant 28 : i32
      %c72_i32_46 = arith.constant 72 : i32
      %c40_i32_47 = arith.constant 40 : i32
      func.call @bn3_conv2dk1_ui8_i8(%bn3_act_2_3_buff_0, %view_3, %act_bn3_bn4_buff_2, %c28_i32_45, %c72_i32_46, %c40_i32_47, %2) : (memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, Release, 1)
      aie.use_lock(%act_bn2_bn3_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c56_i32_48 = arith.constant 56 : i32
      %c24_i32_49 = arith.constant 24 : i32
      %c72_i32_50 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_0, %view, %bn3_act_1_2_buff_0, %c56_i32_48, %c24_i32_49, %c72_i32_50, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      %c56_i32_51 = arith.constant 56 : i32
      %c24_i32_52 = arith.constant 24 : i32
      %c72_i32_53 = arith.constant 72 : i32
      func.call @bn3_conv2dk1_relu_i8_ui8(%act_bn2_bn3_buff_1, %view, %bn3_act_1_2_buff_1, %c56_i32_51, %c24_i32_52, %c72_i32_53, %0) : (memref<56x1x24xi8>, memref<1728xi8>, memref<56x1x72xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn2_bn3_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn3_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c56_i32_54 = arith.constant 56 : i32
      %c1_i32_55 = arith.constant 1 : i32
      %c72_i32_56 = arith.constant 72 : i32
      %c3_i32_57 = arith.constant 3 : i32
      %c3_i32_58 = arith.constant 3 : i32
      %c1_i32_59 = arith.constant 1 : i32
      %c0_i32_60 = arith.constant 0 : i32
      func.call @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(%bn3_act_1_2_buff_2, %bn3_act_1_2_buff_0, %bn3_act_1_2_buff_1, %view_2, %bn3_act_2_3_buff_0, %c56_i32_54, %c1_i32_55, %c72_i32_56, %c3_i32_57, %c3_i32_58, %c1_i32_59, %1, %c0_i32_60) : (memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>, memref<648xi8>, memref<28x1x72xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn3_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn3_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn3_bn4_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_61 = arith.constant 28 : i32
      %c72_i32_62 = arith.constant 72 : i32
      %c40_i32_63 = arith.constant 40 : i32
      func.call @bn3_conv2dk1_ui8_i8(%bn3_act_2_3_buff_0, %view_3, %act_bn3_bn4_buff_0, %c28_i32_61, %c72_i32_62, %c40_i32_63, %2) : (memref<28x1x72xui8>, memref<2880xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn3_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, Release, 1)
      %5 = arith.addi %3, %c3 : index
      cf.br ^bb1(%5 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%bn3_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a"}
    func.func private @bn4_conv2dk1_relu_i8_ui8(memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32)
    func.func private @bn4_conv2dk3_dw_stride2_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn4_conv2dk1_skip_ui8_i8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32)
    func.func private @bn4_conv2dk1_ui8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32)
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn4_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn4_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<10680xi8> to memref<4800xi8>
      %c4800 = arith.constant 4800 : index
      %view_2 = memref.view %bn4_wts_OF_L2L1_cons_buff_0[%c4800][] : memref<10680xi8> to memref<1080xi8>
      %c5880 = arith.constant 5880 : index
      %view_3 = memref.view %bn4_wts_OF_L2L1_cons_buff_0[%c5880][] : memref<10680xi8> to memref<4800xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp15[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp15[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp15[%c2] : memref<16xi32>
      %c3 = arith.constant 3 : index
      %3 = memref.load %rtp15[%c3] : memref<16xi32>
      aie.use_lock(%act_bn3_bn4_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn4_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32 = arith.constant 28 : i32
      %c40_i32 = arith.constant 40 : i32
      %c120_i32 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_0, %view, %bn4_act_1_2_buff_0, %c28_i32, %c40_i32, %c120_i32, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_6 = arith.constant 28 : i32
      %c40_i32_7 = arith.constant 40 : i32
      %c120_i32_8 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_1, %view, %bn4_act_1_2_buff_1, %c28_i32_6, %c40_i32_7, %c120_i32_8, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%bn4_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_9 = arith.constant 28 : i32
      %c1_i32 = arith.constant 1 : i32
      %c120_i32_10 = arith.constant 120 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_0, %bn4_act_1_2_buff_0, %bn4_act_1_2_buff_1, %view_2, %bn4_act_2_3_buff_0, %c28_i32_9, %c1_i32, %c120_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_13 = arith.constant 28 : i32
      %c120_i32_14 = arith.constant 120 : i32
      %c40_i32_15 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_0, %act_bn3_bn4_buff_0, %c28_i32_13, %c120_i32_14, %c40_i32_15, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      %c0_16 = arith.constant 0 : index
      %c26 = arith.constant 26 : index
      %c1_17 = arith.constant 1 : index
      %c24 = arith.constant 24 : index
      %c3_18 = arith.constant 3 : index
      cf.br ^bb1(%c0_16 : index)
    ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
      %5 = arith.cmpi slt, %4, %c24 : index
      cf.cond_br %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn3_bn4_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_19 = arith.constant 28 : i32
      %c40_i32_20 = arith.constant 40 : i32
      %c120_i32_21 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_2, %view, %bn4_act_1_2_buff_2, %c28_i32_19, %c40_i32_20, %c120_i32_21, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_22 = arith.constant 28 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c120_i32_24 = arith.constant 120 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c3_i32_26 = arith.constant 3 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_0, %bn4_act_1_2_buff_1, %bn4_act_1_2_buff_2, %view_2, %bn4_act_2_3_buff_0, %c28_i32_22, %c1_i32_23, %c120_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %1, %c0_i32_28) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_29 = arith.constant 28 : i32
      %c120_i32_30 = arith.constant 120 : i32
      %c40_i32_31 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_1, %act_bn3_bn4_buff_1, %c28_i32_29, %c120_i32_30, %c40_i32_31, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_32 = arith.constant 28 : i32
      %c40_i32_33 = arith.constant 40 : i32
      %c120_i32_34 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_0, %view, %bn4_act_1_2_buff_0, %c28_i32_32, %c40_i32_33, %c120_i32_34, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_35 = arith.constant 28 : i32
      %c1_i32_36 = arith.constant 1 : i32
      %c120_i32_37 = arith.constant 120 : i32
      %c3_i32_38 = arith.constant 3 : i32
      %c3_i32_39 = arith.constant 3 : i32
      %c1_i32_40 = arith.constant 1 : i32
      %c0_i32_41 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_1, %bn4_act_1_2_buff_2, %bn4_act_1_2_buff_0, %view_2, %bn4_act_2_3_buff_0, %c28_i32_35, %c1_i32_36, %c120_i32_37, %c3_i32_38, %c3_i32_39, %c1_i32_40, %1, %c0_i32_41) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_42 = arith.constant 28 : i32
      %c120_i32_43 = arith.constant 120 : i32
      %c40_i32_44 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_2, %act_bn3_bn4_buff_2, %c28_i32_42, %c120_i32_43, %c40_i32_44, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_45 = arith.constant 28 : i32
      %c40_i32_46 = arith.constant 40 : i32
      %c120_i32_47 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_1, %view, %bn4_act_1_2_buff_1, %c28_i32_45, %c40_i32_46, %c120_i32_47, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_48 = arith.constant 28 : i32
      %c1_i32_49 = arith.constant 1 : i32
      %c120_i32_50 = arith.constant 120 : i32
      %c3_i32_51 = arith.constant 3 : i32
      %c3_i32_52 = arith.constant 3 : i32
      %c1_i32_53 = arith.constant 1 : i32
      %c0_i32_54 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_2, %bn4_act_1_2_buff_0, %bn4_act_1_2_buff_1, %view_2, %bn4_act_2_3_buff_0, %c28_i32_48, %c1_i32_49, %c120_i32_50, %c3_i32_51, %c3_i32_52, %c1_i32_53, %1, %c0_i32_54) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_55 = arith.constant 28 : i32
      %c120_i32_56 = arith.constant 120 : i32
      %c40_i32_57 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_0, %act_bn3_bn4_buff_0, %c28_i32_55, %c120_i32_56, %c40_i32_57, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      %6 = arith.addi %4, %c3_18 : index
      cf.br ^bb1(%6 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%act_bn3_bn4_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_58 = arith.constant 28 : i32
      %c40_i32_59 = arith.constant 40 : i32
      %c120_i32_60 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_2, %view, %bn4_act_1_2_buff_2, %c28_i32_58, %c40_i32_59, %c120_i32_60, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_61 = arith.constant 28 : i32
      %c1_i32_62 = arith.constant 1 : i32
      %c120_i32_63 = arith.constant 120 : i32
      %c3_i32_64 = arith.constant 3 : i32
      %c3_i32_65 = arith.constant 3 : i32
      %c1_i32_66 = arith.constant 1 : i32
      %c0_i32_67 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_0, %bn4_act_1_2_buff_1, %bn4_act_1_2_buff_2, %view_2, %bn4_act_2_3_buff_0, %c28_i32_61, %c1_i32_62, %c120_i32_63, %c3_i32_64, %c3_i32_65, %c1_i32_66, %1, %c0_i32_67) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_68 = arith.constant 28 : i32
      %c120_i32_69 = arith.constant 120 : i32
      %c40_i32_70 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_1, %act_bn3_bn4_buff_1, %c28_i32_68, %c120_i32_69, %c40_i32_70, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      aie.use_lock(%act_bn3_bn4_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_71 = arith.constant 28 : i32
      %c40_i32_72 = arith.constant 40 : i32
      %c120_i32_73 = arith.constant 120 : i32
      func.call @bn4_conv2dk1_relu_i8_ui8(%act_bn3_bn4_buff_0, %view, %bn4_act_1_2_buff_0, %c28_i32_71, %c40_i32_72, %c120_i32_73, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_74 = arith.constant 28 : i32
      %c1_i32_75 = arith.constant 1 : i32
      %c120_i32_76 = arith.constant 120 : i32
      %c3_i32_77 = arith.constant 3 : i32
      %c3_i32_78 = arith.constant 3 : i32
      %c1_i32_79 = arith.constant 1 : i32
      %c0_i32_80 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_1, %bn4_act_1_2_buff_2, %bn4_act_1_2_buff_0, %view_2, %bn4_act_2_3_buff_0, %c28_i32_74, %c1_i32_75, %c120_i32_76, %c3_i32_77, %c3_i32_78, %c1_i32_79, %1, %c0_i32_80) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_81 = arith.constant 28 : i32
      %c120_i32_82 = arith.constant 120 : i32
      %c40_i32_83 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_2, %act_bn3_bn4_buff_2, %c28_i32_81, %c120_i32_82, %c40_i32_83, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_84 = arith.constant 28 : i32
      %c1_i32_85 = arith.constant 1 : i32
      %c120_i32_86 = arith.constant 120 : i32
      %c3_i32_87 = arith.constant 3 : i32
      %c3_i32_88 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_89 = arith.constant 0 : i32
      func.call @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(%bn4_act_1_2_buff_2, %bn4_act_1_2_buff_0, %bn4_act_1_2_buff_0, %view_2, %bn4_act_2_3_buff_0, %c28_i32_84, %c1_i32_85, %c120_i32_86, %c3_i32_87, %c3_i32_88, %c2_i32, %1, %c0_i32_89) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn4_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn4_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_90 = arith.constant 28 : i32
      %c120_i32_91 = arith.constant 120 : i32
      %c40_i32_92 = arith.constant 40 : i32
      func.call @bn4_conv2dk1_skip_ui8_i8_i8(%bn4_act_2_3_buff_0, %view_3, %act_bn4_bn5_buff_0, %act_bn3_bn4_buff_0, %c28_i32_90, %c120_i32_91, %c40_i32_92, %2, %3) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn3_bn4_prod_lock, Release, 1)
      aie.use_lock(%bn4_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, Release, 1)
      aie.use_lock(%bn4_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn5_conv2dk1_relu_i8_ui8(memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32)
    func.func private @bn5_conv2dk3_dw_stride2_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn5_conv2dk1_skip_ui8_i8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, memref<28x1x40xi8>, i32, i32, i32, i32, i32)
    func.func private @bn5_conv2dk1_ui8_i8(memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32)
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn5_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn5_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<10680xi8> to memref<4800xi8>
      %c4800 = arith.constant 4800 : index
      %view_2 = memref.view %bn5_wts_OF_L2L1_cons_buff_0[%c4800][] : memref<10680xi8> to memref<1080xi8>
      %c5880 = arith.constant 5880 : index
      %view_3 = memref.view %bn5_wts_OF_L2L1_cons_buff_0[%c5880][] : memref<10680xi8> to memref<4800xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp14[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp14[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp14[%c2] : memref<16xi32>
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32 = arith.constant 28 : i32
      %c40_i32 = arith.constant 40 : i32
      %c120_i32 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_0, %view, %bn5_act_1_2_buff_0, %c28_i32, %c40_i32, %c120_i32, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_6 = arith.constant 28 : i32
      %c40_i32_7 = arith.constant 40 : i32
      %c120_i32_8 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_1, %view, %bn5_act_1_2_buff_1, %c28_i32_6, %c40_i32_7, %c120_i32_8, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 2)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_9 = arith.constant 28 : i32
      %c1_i32 = arith.constant 1 : i32
      %c120_i32_10 = arith.constant 120 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_0, %bn5_act_1_2_buff_0, %bn5_act_1_2_buff_1, %view_2, %bn5_act_2_3_buff_0, %c28_i32_9, %c1_i32, %c120_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_13 = arith.constant 28 : i32
      %c120_i32_14 = arith.constant 120 : i32
      %c40_i32_15 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_0, %c28_i32_13, %c120_i32_14, %c40_i32_15, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      %c0_16 = arith.constant 0 : index
      %c26 = arith.constant 26 : index
      %c1_17 = arith.constant 1 : index
      %c24 = arith.constant 24 : index
      %c6 = arith.constant 6 : index
      cf.br ^bb1(%c0_16 : index)
    ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
      %4 = arith.cmpi slt, %3, %c24 : index
      cf.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_18 = arith.constant 28 : i32
      %c40_i32_19 = arith.constant 40 : i32
      %c120_i32_20 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_2, %view, %bn5_act_1_2_buff_2, %c28_i32_18, %c40_i32_19, %c120_i32_20, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_21 = arith.constant 28 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c120_i32_23 = arith.constant 120 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c0_i32_27 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_0, %bn5_act_1_2_buff_1, %bn5_act_1_2_buff_2, %view_2, %bn5_act_2_3_buff_0, %c28_i32_21, %c1_i32_22, %c120_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %1, %c0_i32_27) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_28 = arith.constant 28 : i32
      %c120_i32_29 = arith.constant 120 : i32
      %c40_i32_30 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_1, %c28_i32_28, %c120_i32_29, %c40_i32_30, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_31 = arith.constant 28 : i32
      %c40_i32_32 = arith.constant 40 : i32
      %c120_i32_33 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_0, %view, %bn5_act_1_2_buff_0, %c28_i32_31, %c40_i32_32, %c120_i32_33, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_34 = arith.constant 28 : i32
      %c1_i32_35 = arith.constant 1 : i32
      %c120_i32_36 = arith.constant 120 : i32
      %c3_i32_37 = arith.constant 3 : i32
      %c3_i32_38 = arith.constant 3 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c0_i32_40 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_1, %bn5_act_1_2_buff_2, %bn5_act_1_2_buff_0, %view_2, %bn5_act_2_3_buff_0, %c28_i32_34, %c1_i32_35, %c120_i32_36, %c3_i32_37, %c3_i32_38, %c1_i32_39, %1, %c0_i32_40) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_41 = arith.constant 28 : i32
      %c120_i32_42 = arith.constant 120 : i32
      %c40_i32_43 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_0, %c28_i32_41, %c120_i32_42, %c40_i32_43, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_44 = arith.constant 28 : i32
      %c40_i32_45 = arith.constant 40 : i32
      %c120_i32_46 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_1, %view, %bn5_act_1_2_buff_1, %c28_i32_44, %c40_i32_45, %c120_i32_46, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_47 = arith.constant 28 : i32
      %c1_i32_48 = arith.constant 1 : i32
      %c120_i32_49 = arith.constant 120 : i32
      %c3_i32_50 = arith.constant 3 : i32
      %c3_i32_51 = arith.constant 3 : i32
      %c1_i32_52 = arith.constant 1 : i32
      %c0_i32_53 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_2, %bn5_act_1_2_buff_0, %bn5_act_1_2_buff_1, %view_2, %bn5_act_2_3_buff_0, %c28_i32_47, %c1_i32_48, %c120_i32_49, %c3_i32_50, %c3_i32_51, %c1_i32_52, %1, %c0_i32_53) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_54 = arith.constant 28 : i32
      %c120_i32_55 = arith.constant 120 : i32
      %c40_i32_56 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_1, %c28_i32_54, %c120_i32_55, %c40_i32_56, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_57 = arith.constant 28 : i32
      %c40_i32_58 = arith.constant 40 : i32
      %c120_i32_59 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_2, %view, %bn5_act_1_2_buff_2, %c28_i32_57, %c40_i32_58, %c120_i32_59, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_60 = arith.constant 28 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c120_i32_62 = arith.constant 120 : i32
      %c3_i32_63 = arith.constant 3 : i32
      %c3_i32_64 = arith.constant 3 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c0_i32_66 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_0, %bn5_act_1_2_buff_1, %bn5_act_1_2_buff_2, %view_2, %bn5_act_2_3_buff_0, %c28_i32_60, %c1_i32_61, %c120_i32_62, %c3_i32_63, %c3_i32_64, %c1_i32_65, %1, %c0_i32_66) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_67 = arith.constant 28 : i32
      %c120_i32_68 = arith.constant 120 : i32
      %c40_i32_69 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_0, %c28_i32_67, %c120_i32_68, %c40_i32_69, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_70 = arith.constant 28 : i32
      %c40_i32_71 = arith.constant 40 : i32
      %c120_i32_72 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_0, %view, %bn5_act_1_2_buff_0, %c28_i32_70, %c40_i32_71, %c120_i32_72, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_73 = arith.constant 28 : i32
      %c1_i32_74 = arith.constant 1 : i32
      %c120_i32_75 = arith.constant 120 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c3_i32_77 = arith.constant 3 : i32
      %c1_i32_78 = arith.constant 1 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_1, %bn5_act_1_2_buff_2, %bn5_act_1_2_buff_0, %view_2, %bn5_act_2_3_buff_0, %c28_i32_73, %c1_i32_74, %c120_i32_75, %c3_i32_76, %c3_i32_77, %c1_i32_78, %1, %c0_i32_79) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_80 = arith.constant 28 : i32
      %c120_i32_81 = arith.constant 120 : i32
      %c40_i32_82 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_1, %c28_i32_80, %c120_i32_81, %c40_i32_82, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_83 = arith.constant 28 : i32
      %c40_i32_84 = arith.constant 40 : i32
      %c120_i32_85 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_1, %view, %bn5_act_1_2_buff_1, %c28_i32_83, %c40_i32_84, %c120_i32_85, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_86 = arith.constant 28 : i32
      %c1_i32_87 = arith.constant 1 : i32
      %c120_i32_88 = arith.constant 120 : i32
      %c3_i32_89 = arith.constant 3 : i32
      %c3_i32_90 = arith.constant 3 : i32
      %c1_i32_91 = arith.constant 1 : i32
      %c0_i32_92 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_2, %bn5_act_1_2_buff_0, %bn5_act_1_2_buff_1, %view_2, %bn5_act_2_3_buff_0, %c28_i32_86, %c1_i32_87, %c120_i32_88, %c3_i32_89, %c3_i32_90, %c1_i32_91, %1, %c0_i32_92) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_93 = arith.constant 28 : i32
      %c120_i32_94 = arith.constant 120 : i32
      %c40_i32_95 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_0, %c28_i32_93, %c120_i32_94, %c40_i32_95, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      %5 = arith.addi %3, %c6 : index
      cf.br ^bb1(%5 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_96 = arith.constant 28 : i32
      %c40_i32_97 = arith.constant 40 : i32
      %c120_i32_98 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_2, %view, %bn5_act_1_2_buff_2, %c28_i32_96, %c40_i32_97, %c120_i32_98, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_99 = arith.constant 28 : i32
      %c1_i32_100 = arith.constant 1 : i32
      %c120_i32_101 = arith.constant 120 : i32
      %c3_i32_102 = arith.constant 3 : i32
      %c3_i32_103 = arith.constant 3 : i32
      %c1_i32_104 = arith.constant 1 : i32
      %c0_i32_105 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_0, %bn5_act_1_2_buff_1, %bn5_act_1_2_buff_2, %view_2, %bn5_act_2_3_buff_0, %c28_i32_99, %c1_i32_100, %c120_i32_101, %c3_i32_102, %c3_i32_103, %c1_i32_104, %1, %c0_i32_105) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_106 = arith.constant 28 : i32
      %c120_i32_107 = arith.constant 120 : i32
      %c40_i32_108 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_1, %c28_i32_106, %c120_i32_107, %c40_i32_108, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_109 = arith.constant 28 : i32
      %c40_i32_110 = arith.constant 40 : i32
      %c120_i32_111 = arith.constant 120 : i32
      func.call @bn5_conv2dk1_relu_i8_ui8(%act_bn4_bn5_buff_0, %view, %bn5_act_1_2_buff_0, %c28_i32_109, %c40_i32_110, %c120_i32_111, %0) : (memref<28x1x40xi8>, memref<4800xi8>, memref<28x1x120xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn4_bn5_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_112 = arith.constant 28 : i32
      %c1_i32_113 = arith.constant 1 : i32
      %c120_i32_114 = arith.constant 120 : i32
      %c3_i32_115 = arith.constant 3 : i32
      %c3_i32_116 = arith.constant 3 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c0_i32_118 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_1, %bn5_act_1_2_buff_2, %bn5_act_1_2_buff_0, %view_2, %bn5_act_2_3_buff_0, %c28_i32_112, %c1_i32_113, %c120_i32_114, %c3_i32_115, %c3_i32_116, %c1_i32_117, %1, %c0_i32_118) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_119 = arith.constant 28 : i32
      %c120_i32_120 = arith.constant 120 : i32
      %c40_i32_121 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_0, %c28_i32_119, %c120_i32_120, %c40_i32_121, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_122 = arith.constant 28 : i32
      %c1_i32_123 = arith.constant 1 : i32
      %c120_i32_124 = arith.constant 120 : i32
      %c3_i32_125 = arith.constant 3 : i32
      %c3_i32_126 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_127 = arith.constant 0 : i32
      func.call @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(%bn5_act_1_2_buff_2, %bn5_act_1_2_buff_0, %bn5_act_1_2_buff_0, %view_2, %bn5_act_2_3_buff_0, %c28_i32_122, %c1_i32_123, %c120_i32_124, %c3_i32_125, %c3_i32_126, %c2_i32, %1, %c0_i32_127) : (memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>, memref<1080xi8>, memref<28x1x120xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn5_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn5_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn5_bn6_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_128 = arith.constant 28 : i32
      %c120_i32_129 = arith.constant 120 : i32
      %c40_i32_130 = arith.constant 40 : i32
      func.call @bn5_conv2dk1_ui8_i8(%bn5_act_2_3_buff_0, %view_3, %act_bn5_bn6_buff_1, %c28_i32_128, %c120_i32_129, %c40_i32_130, %2) : (memref<28x1x120xui8>, memref<4800xi8>, memref<28x1x40xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn5_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_lock, Release, 1)
      aie.use_lock(%bn5_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a"}
    func.func private @bn6_conv2dk1_relu_i8_ui8(memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32)
    func.func private @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn6_conv2dk3_dw_stride1_relu_ui8_ui8(memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn6_conv2dk1_skip_ui8_i8_i8(memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn6_conv2dk1_ui8_i8(memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn6_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn6_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<30960xi8> to memref<9600xi8>
      %c9600 = arith.constant 9600 : index
      %view_2 = memref.view %bn6_wts_OF_L2L1_cons_buff_0[%c9600][] : memref<30960xi8> to memref<2160xi8>
      %c11760 = arith.constant 11760 : index
      %view_3 = memref.view %bn6_wts_OF_L2L1_cons_buff_0[%c11760][] : memref<30960xi8> to memref<19200xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp12[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp12[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp12[%c2] : memref<16xi32>
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32 = arith.constant 28 : i32
      %c40_i32 = arith.constant 40 : i32
      %c240_i32 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_0, %view, %bn6_act_1_2_buff_0, %c28_i32, %c40_i32, %c240_i32, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_6 = arith.constant 28 : i32
      %c40_i32_7 = arith.constant 40 : i32
      %c240_i32_8 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_1, %view, %bn6_act_1_2_buff_1, %c28_i32_6, %c40_i32_7, %c240_i32_8, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_9 = arith.constant 28 : i32
      %c1_i32 = arith.constant 1 : i32
      %c240_i32_10 = arith.constant 240 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_0, %bn6_act_1_2_buff_0, %bn6_act_1_2_buff_1, %view_2, %bn6_act_2_3_buff_0, %c28_i32_9, %c1_i32, %c240_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c240_i32_13 = arith.constant 240 : i32
      %c80_i32 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_0, %c14_i32, %c240_i32_13, %c80_i32, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      %c0_14 = arith.constant 0 : index
      %c13 = arith.constant 13 : index
      %c1_15 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c6 = arith.constant 6 : index
      cf.br ^bb1(%c0_14 : index)
    ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
      %4 = arith.cmpi slt, %3, %c12 : index
      cf.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_16 = arith.constant 28 : i32
      %c40_i32_17 = arith.constant 40 : i32
      %c240_i32_18 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_2, %view, %bn6_act_1_2_buff_2, %c28_i32_16, %c40_i32_17, %c240_i32_18, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_19 = arith.constant 28 : i32
      %c40_i32_20 = arith.constant 40 : i32
      %c240_i32_21 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_0, %view, %bn6_act_1_2_buff_0, %c28_i32_19, %c40_i32_20, %c240_i32_21, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_22 = arith.constant 28 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c240_i32_24 = arith.constant 240 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c3_i32_26 = arith.constant 3 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_1, %bn6_act_1_2_buff_2, %bn6_act_1_2_buff_0, %view_2, %bn6_act_2_3_buff_0, %c28_i32_22, %c1_i32_23, %c240_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %1, %c0_i32_28) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_29 = arith.constant 14 : i32
      %c240_i32_30 = arith.constant 240 : i32
      %c80_i32_31 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_1, %c14_i32_29, %c240_i32_30, %c80_i32_31, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_32 = arith.constant 28 : i32
      %c40_i32_33 = arith.constant 40 : i32
      %c240_i32_34 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_1, %view, %bn6_act_1_2_buff_1, %c28_i32_32, %c40_i32_33, %c240_i32_34, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_35 = arith.constant 28 : i32
      %c40_i32_36 = arith.constant 40 : i32
      %c240_i32_37 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_2, %view, %bn6_act_1_2_buff_2, %c28_i32_35, %c40_i32_36, %c240_i32_37, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_38 = arith.constant 28 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c240_i32_40 = arith.constant 240 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c3_i32_42 = arith.constant 3 : i32
      %c1_i32_43 = arith.constant 1 : i32
      %c0_i32_44 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_0, %bn6_act_1_2_buff_1, %bn6_act_1_2_buff_2, %view_2, %bn6_act_2_3_buff_0, %c28_i32_38, %c1_i32_39, %c240_i32_40, %c3_i32_41, %c3_i32_42, %c1_i32_43, %1, %c0_i32_44) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_45 = arith.constant 14 : i32
      %c240_i32_46 = arith.constant 240 : i32
      %c80_i32_47 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_0, %c14_i32_45, %c240_i32_46, %c80_i32_47, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_48 = arith.constant 28 : i32
      %c40_i32_49 = arith.constant 40 : i32
      %c240_i32_50 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_0, %view, %bn6_act_1_2_buff_0, %c28_i32_48, %c40_i32_49, %c240_i32_50, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_51 = arith.constant 28 : i32
      %c40_i32_52 = arith.constant 40 : i32
      %c240_i32_53 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_1, %view, %bn6_act_1_2_buff_1, %c28_i32_51, %c40_i32_52, %c240_i32_53, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_54 = arith.constant 28 : i32
      %c1_i32_55 = arith.constant 1 : i32
      %c240_i32_56 = arith.constant 240 : i32
      %c3_i32_57 = arith.constant 3 : i32
      %c3_i32_58 = arith.constant 3 : i32
      %c1_i32_59 = arith.constant 1 : i32
      %c0_i32_60 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_2, %bn6_act_1_2_buff_0, %bn6_act_1_2_buff_1, %view_2, %bn6_act_2_3_buff_0, %c28_i32_54, %c1_i32_55, %c240_i32_56, %c3_i32_57, %c3_i32_58, %c1_i32_59, %1, %c0_i32_60) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_61 = arith.constant 14 : i32
      %c240_i32_62 = arith.constant 240 : i32
      %c80_i32_63 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_1, %c14_i32_61, %c240_i32_62, %c80_i32_63, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_64 = arith.constant 28 : i32
      %c40_i32_65 = arith.constant 40 : i32
      %c240_i32_66 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_2, %view, %bn6_act_1_2_buff_2, %c28_i32_64, %c40_i32_65, %c240_i32_66, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_67 = arith.constant 28 : i32
      %c40_i32_68 = arith.constant 40 : i32
      %c240_i32_69 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_0, %view, %bn6_act_1_2_buff_0, %c28_i32_67, %c40_i32_68, %c240_i32_69, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_70 = arith.constant 28 : i32
      %c1_i32_71 = arith.constant 1 : i32
      %c240_i32_72 = arith.constant 240 : i32
      %c3_i32_73 = arith.constant 3 : i32
      %c3_i32_74 = arith.constant 3 : i32
      %c1_i32_75 = arith.constant 1 : i32
      %c0_i32_76 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_1, %bn6_act_1_2_buff_2, %bn6_act_1_2_buff_0, %view_2, %bn6_act_2_3_buff_0, %c28_i32_70, %c1_i32_71, %c240_i32_72, %c3_i32_73, %c3_i32_74, %c1_i32_75, %1, %c0_i32_76) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_77 = arith.constant 14 : i32
      %c240_i32_78 = arith.constant 240 : i32
      %c80_i32_79 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_0, %c14_i32_77, %c240_i32_78, %c80_i32_79, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_80 = arith.constant 28 : i32
      %c40_i32_81 = arith.constant 40 : i32
      %c240_i32_82 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_1, %view, %bn6_act_1_2_buff_1, %c28_i32_80, %c40_i32_81, %c240_i32_82, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_83 = arith.constant 28 : i32
      %c40_i32_84 = arith.constant 40 : i32
      %c240_i32_85 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_2, %view, %bn6_act_1_2_buff_2, %c28_i32_83, %c40_i32_84, %c240_i32_85, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_86 = arith.constant 28 : i32
      %c1_i32_87 = arith.constant 1 : i32
      %c240_i32_88 = arith.constant 240 : i32
      %c3_i32_89 = arith.constant 3 : i32
      %c3_i32_90 = arith.constant 3 : i32
      %c1_i32_91 = arith.constant 1 : i32
      %c0_i32_92 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_0, %bn6_act_1_2_buff_1, %bn6_act_1_2_buff_2, %view_2, %bn6_act_2_3_buff_0, %c28_i32_86, %c1_i32_87, %c240_i32_88, %c3_i32_89, %c3_i32_90, %c1_i32_91, %1, %c0_i32_92) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_93 = arith.constant 14 : i32
      %c240_i32_94 = arith.constant 240 : i32
      %c80_i32_95 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_1, %c14_i32_93, %c240_i32_94, %c80_i32_95, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_96 = arith.constant 28 : i32
      %c40_i32_97 = arith.constant 40 : i32
      %c240_i32_98 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_0, %view, %bn6_act_1_2_buff_0, %c28_i32_96, %c40_i32_97, %c240_i32_98, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_99 = arith.constant 28 : i32
      %c40_i32_100 = arith.constant 40 : i32
      %c240_i32_101 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_1, %view, %bn6_act_1_2_buff_1, %c28_i32_99, %c40_i32_100, %c240_i32_101, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_102 = arith.constant 28 : i32
      %c1_i32_103 = arith.constant 1 : i32
      %c240_i32_104 = arith.constant 240 : i32
      %c3_i32_105 = arith.constant 3 : i32
      %c3_i32_106 = arith.constant 3 : i32
      %c1_i32_107 = arith.constant 1 : i32
      %c0_i32_108 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_2, %bn6_act_1_2_buff_0, %bn6_act_1_2_buff_1, %view_2, %bn6_act_2_3_buff_0, %c28_i32_102, %c1_i32_103, %c240_i32_104, %c3_i32_105, %c3_i32_106, %c1_i32_107, %1, %c0_i32_108) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_109 = arith.constant 14 : i32
      %c240_i32_110 = arith.constant 240 : i32
      %c80_i32_111 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_0, %c14_i32_109, %c240_i32_110, %c80_i32_111, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      %5 = arith.addi %3, %c6 : index
      cf.br ^bb1(%5 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c28_i32_112 = arith.constant 28 : i32
      %c40_i32_113 = arith.constant 40 : i32
      %c240_i32_114 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_2, %view, %bn6_act_1_2_buff_2, %c28_i32_112, %c40_i32_113, %c240_i32_114, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      %c28_i32_115 = arith.constant 28 : i32
      %c40_i32_116 = arith.constant 40 : i32
      %c240_i32_117 = arith.constant 240 : i32
      func.call @bn6_conv2dk1_relu_i8_ui8(%act_bn5_bn6_cons_buff_0, %view, %bn6_act_1_2_buff_0, %c28_i32_115, %c40_i32_116, %c240_i32_117, %0) : (memref<28x1x40xi8>, memref<9600xi8>, memref<28x1x240xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn6_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c28_i32_118 = arith.constant 28 : i32
      %c1_i32_119 = arith.constant 1 : i32
      %c240_i32_120 = arith.constant 240 : i32
      %c3_i32_121 = arith.constant 3 : i32
      %c3_i32_122 = arith.constant 3 : i32
      %c1_i32_123 = arith.constant 1 : i32
      %c0_i32_124 = arith.constant 0 : i32
      func.call @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(%bn6_act_1_2_buff_1, %bn6_act_1_2_buff_2, %bn6_act_1_2_buff_0, %view_2, %bn6_act_2_3_buff_0, %c28_i32_118, %c1_i32_119, %c240_i32_120, %c3_i32_121, %c3_i32_122, %c1_i32_123, %1, %c0_i32_124) : (memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>, memref<2160xi8>, memref<14x1x240xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn6_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn6_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn6_bn7_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_125 = arith.constant 14 : i32
      %c240_i32_126 = arith.constant 240 : i32
      %c80_i32_127 = arith.constant 80 : i32
      func.call @bn6_conv2dk1_ui8_i8(%bn6_act_2_3_buff_0, %view_3, %act_bn6_bn7_buff_1, %c14_i32_125, %c240_i32_126, %c80_i32_127, %2) : (memref<14x1x240xui8>, memref<19200xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn6_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, Release, 1)
      aie.use_lock(%bn6_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a"}
    func.func private @bn7_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32)
    func.func private @bn7_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn7_conv2dk1_skip_ui8_i8_i8(memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn7_conv2dk1_ui8_i8(memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn7_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn7_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<33800xi8> to memref<16000xi8>
      %c16000 = arith.constant 16000 : index
      %view_2 = memref.view %bn7_wts_OF_L2L1_cons_buff_0[%c16000][] : memref<33800xi8> to memref<1800xi8>
      %c17800 = arith.constant 17800 : index
      %view_3 = memref.view %bn7_wts_OF_L2L1_cons_buff_0[%c17800][] : memref<33800xi8> to memref<16000xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp13[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp13[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp13[%c2] : memref<16xi32>
      %c3 = arith.constant 3 : index
      %3 = memref.load %rtp13[%c3] : memref<16xi32>
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c14_i32 = arith.constant 14 : i32
      %c80_i32 = arith.constant 80 : i32
      %c200_i32 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_0, %view, %bn7_act_1_2_buff_0, %c14_i32, %c80_i32, %c200_i32, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      %c14_i32_6 = arith.constant 14 : i32
      %c80_i32_7 = arith.constant 80 : i32
      %c200_i32_8 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_1, %view, %bn7_act_1_2_buff_1, %c14_i32_6, %c80_i32_7, %c200_i32_8, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_9 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c200_i32_10 = arith.constant 200 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_0, %bn7_act_1_2_buff_0, %bn7_act_1_2_buff_1, %view_2, %bn7_act_2_3_buff_0, %c14_i32_9, %c1_i32, %c200_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c200_i32_14 = arith.constant 200 : i32
      %c80_i32_15 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_0, %act_bn6_bn7_buff_0, %c14_i32_13, %c200_i32_14, %c80_i32_15, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      %c0_16 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_17 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      cf.br ^bb1(%c0_16 : index)
    ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
      %5 = arith.cmpi slt, %4, %c12 : index
      cf.cond_br %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_18 = arith.constant 14 : i32
      %c80_i32_19 = arith.constant 80 : i32
      %c200_i32_20 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_0, %view, %bn7_act_1_2_buff_2, %c14_i32_18, %c80_i32_19, %c200_i32_20, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c200_i32_23 = arith.constant 200 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c0_i32_27 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_0, %bn7_act_1_2_buff_1, %bn7_act_1_2_buff_2, %view_2, %bn7_act_2_3_buff_0, %c14_i32_21, %c1_i32_22, %c200_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %1, %c0_i32_27) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_28 = arith.constant 14 : i32
      %c200_i32_29 = arith.constant 200 : i32
      %c80_i32_30 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_1, %act_bn6_bn7_buff_1, %c14_i32_28, %c200_i32_29, %c80_i32_30, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_31 = arith.constant 14 : i32
      %c80_i32_32 = arith.constant 80 : i32
      %c200_i32_33 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_1, %view, %bn7_act_1_2_buff_0, %c14_i32_31, %c80_i32_32, %c200_i32_33, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_34 = arith.constant 14 : i32
      %c1_i32_35 = arith.constant 1 : i32
      %c200_i32_36 = arith.constant 200 : i32
      %c3_i32_37 = arith.constant 3 : i32
      %c3_i32_38 = arith.constant 3 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c0_i32_40 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_1, %bn7_act_1_2_buff_2, %bn7_act_1_2_buff_0, %view_2, %bn7_act_2_3_buff_0, %c14_i32_34, %c1_i32_35, %c200_i32_36, %c3_i32_37, %c3_i32_38, %c1_i32_39, %1, %c0_i32_40) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_41 = arith.constant 14 : i32
      %c200_i32_42 = arith.constant 200 : i32
      %c80_i32_43 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_0, %act_bn6_bn7_buff_0, %c14_i32_41, %c200_i32_42, %c80_i32_43, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c80_i32_45 = arith.constant 80 : i32
      %c200_i32_46 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_0, %view, %bn7_act_1_2_buff_1, %c14_i32_44, %c80_i32_45, %c200_i32_46, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_47 = arith.constant 14 : i32
      %c1_i32_48 = arith.constant 1 : i32
      %c200_i32_49 = arith.constant 200 : i32
      %c3_i32_50 = arith.constant 3 : i32
      %c3_i32_51 = arith.constant 3 : i32
      %c1_i32_52 = arith.constant 1 : i32
      %c0_i32_53 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_2, %bn7_act_1_2_buff_0, %bn7_act_1_2_buff_1, %view_2, %bn7_act_2_3_buff_0, %c14_i32_47, %c1_i32_48, %c200_i32_49, %c3_i32_50, %c3_i32_51, %c1_i32_52, %1, %c0_i32_53) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_54 = arith.constant 14 : i32
      %c200_i32_55 = arith.constant 200 : i32
      %c80_i32_56 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_1, %act_bn6_bn7_buff_1, %c14_i32_54, %c200_i32_55, %c80_i32_56, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_57 = arith.constant 14 : i32
      %c80_i32_58 = arith.constant 80 : i32
      %c200_i32_59 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_1, %view, %bn7_act_1_2_buff_2, %c14_i32_57, %c80_i32_58, %c200_i32_59, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_60 = arith.constant 14 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c200_i32_62 = arith.constant 200 : i32
      %c3_i32_63 = arith.constant 3 : i32
      %c3_i32_64 = arith.constant 3 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c0_i32_66 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_0, %bn7_act_1_2_buff_1, %bn7_act_1_2_buff_2, %view_2, %bn7_act_2_3_buff_0, %c14_i32_60, %c1_i32_61, %c200_i32_62, %c3_i32_63, %c3_i32_64, %c1_i32_65, %1, %c0_i32_66) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_67 = arith.constant 14 : i32
      %c200_i32_68 = arith.constant 200 : i32
      %c80_i32_69 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_0, %act_bn6_bn7_buff_0, %c14_i32_67, %c200_i32_68, %c80_i32_69, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_70 = arith.constant 14 : i32
      %c80_i32_71 = arith.constant 80 : i32
      %c200_i32_72 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_0, %view, %bn7_act_1_2_buff_0, %c14_i32_70, %c80_i32_71, %c200_i32_72, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_73 = arith.constant 14 : i32
      %c1_i32_74 = arith.constant 1 : i32
      %c200_i32_75 = arith.constant 200 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c3_i32_77 = arith.constant 3 : i32
      %c1_i32_78 = arith.constant 1 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_1, %bn7_act_1_2_buff_2, %bn7_act_1_2_buff_0, %view_2, %bn7_act_2_3_buff_0, %c14_i32_73, %c1_i32_74, %c200_i32_75, %c3_i32_76, %c3_i32_77, %c1_i32_78, %1, %c0_i32_79) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c200_i32_81 = arith.constant 200 : i32
      %c80_i32_82 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_1, %act_bn6_bn7_buff_1, %c14_i32_80, %c200_i32_81, %c80_i32_82, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      aie.use_lock(%act_bn6_bn7_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_83 = arith.constant 14 : i32
      %c80_i32_84 = arith.constant 80 : i32
      %c200_i32_85 = arith.constant 200 : i32
      func.call @bn7_conv2dk1_relu_i8_ui8(%act_bn6_bn7_buff_1, %view, %bn7_act_1_2_buff_1, %c14_i32_83, %c80_i32_84, %c200_i32_85, %0) : (memref<14x1x80xi8>, memref<16000xi8>, memref<14x1x200xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_86 = arith.constant 14 : i32
      %c1_i32_87 = arith.constant 1 : i32
      %c200_i32_88 = arith.constant 200 : i32
      %c3_i32_89 = arith.constant 3 : i32
      %c3_i32_90 = arith.constant 3 : i32
      %c1_i32_91 = arith.constant 1 : i32
      %c0_i32_92 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_2, %bn7_act_1_2_buff_0, %bn7_act_1_2_buff_1, %view_2, %bn7_act_2_3_buff_0, %c14_i32_86, %c1_i32_87, %c200_i32_88, %c3_i32_89, %c3_i32_90, %c1_i32_91, %1, %c0_i32_92) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_93 = arith.constant 14 : i32
      %c200_i32_94 = arith.constant 200 : i32
      %c80_i32_95 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_0, %act_bn6_bn7_buff_0, %c14_i32_93, %c200_i32_94, %c80_i32_95, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      %6 = arith.addi %4, %c6 : index
      cf.br ^bb1(%6 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%bn7_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c200_i32_98 = arith.constant 200 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_101 = arith.constant 0 : i32
      func.call @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(%bn7_act_1_2_buff_0, %bn7_act_1_2_buff_1, %bn7_act_1_2_buff_1, %view_2, %bn7_act_2_3_buff_0, %c14_i32_96, %c1_i32_97, %c200_i32_98, %c3_i32_99, %c3_i32_100, %c2_i32, %1, %c0_i32_101) : (memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>, memref<1800xi8>, memref<14x1x200xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn7_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn7_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn7_bn8_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_102 = arith.constant 14 : i32
      %c200_i32_103 = arith.constant 200 : i32
      %c80_i32_104 = arith.constant 80 : i32
      func.call @bn7_conv2dk1_skip_ui8_i8_i8(%bn7_act_2_3_buff_0, %view_3, %act_bn7_bn8_buff_1, %act_bn6_bn7_buff_1, %c14_i32_102, %c200_i32_103, %c80_i32_104, %2, %3) : (memref<14x1x200xui8>, memref<16000xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn6_bn7_prod_lock, Release, 1)
      aie.use_lock(%bn7_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_lock, Release, 1)
      aie.use_lock(%bn7_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn8_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32)
    func.func private @bn8_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn8_conv2dk1_skip_ui8_i8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn8_conv2dk1_ui8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn8_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn8_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<31096xi8> to memref<14720xi8>
      %c14720 = arith.constant 14720 : index
      %view_2 = memref.view %bn8_wts_OF_L2L1_cons_buff_0[%c14720][] : memref<31096xi8> to memref<1656xi8>
      %c16376 = arith.constant 16376 : index
      %view_3 = memref.view %bn8_wts_OF_L2L1_cons_buff_0[%c16376][] : memref<31096xi8> to memref<14720xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp22[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp22[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp22[%c2] : memref<16xi32>
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c14_i32 = arith.constant 14 : i32
      %c80_i32 = arith.constant 80 : i32
      %c184_i32 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_0, %view, %bn8_act_1_2_buff_0, %c14_i32, %c80_i32, %c184_i32, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      %c14_i32_6 = arith.constant 14 : i32
      %c80_i32_7 = arith.constant 80 : i32
      %c184_i32_8 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_1, %view, %bn8_act_1_2_buff_1, %c14_i32_6, %c80_i32_7, %c184_i32_8, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 2)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_9 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c184_i32_10 = arith.constant 184 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_0, %bn8_act_1_2_buff_0, %bn8_act_1_2_buff_1, %view_2, %bn8_act_2_3_buff_0, %c14_i32_9, %c1_i32, %c184_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c184_i32_14 = arith.constant 184 : i32
      %c80_i32_15 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_0, %c14_i32_13, %c184_i32_14, %c80_i32_15, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      %c0_16 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_17 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      cf.br ^bb1(%c0_16 : index)
    ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
      %4 = arith.cmpi slt, %3, %c12 : index
      cf.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_18 = arith.constant 14 : i32
      %c80_i32_19 = arith.constant 80 : i32
      %c184_i32_20 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_2, %view, %bn8_act_1_2_buff_2, %c14_i32_18, %c80_i32_19, %c184_i32_20, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c184_i32_23 = arith.constant 184 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c0_i32_27 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_0, %bn8_act_1_2_buff_1, %bn8_act_1_2_buff_2, %view_2, %bn8_act_2_3_buff_0, %c14_i32_21, %c1_i32_22, %c184_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %1, %c0_i32_27) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_28 = arith.constant 14 : i32
      %c184_i32_29 = arith.constant 184 : i32
      %c80_i32_30 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_1, %c14_i32_28, %c184_i32_29, %c80_i32_30, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_31 = arith.constant 14 : i32
      %c80_i32_32 = arith.constant 80 : i32
      %c184_i32_33 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_0, %view, %bn8_act_1_2_buff_0, %c14_i32_31, %c80_i32_32, %c184_i32_33, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_34 = arith.constant 14 : i32
      %c1_i32_35 = arith.constant 1 : i32
      %c184_i32_36 = arith.constant 184 : i32
      %c3_i32_37 = arith.constant 3 : i32
      %c3_i32_38 = arith.constant 3 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c0_i32_40 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_1, %bn8_act_1_2_buff_2, %bn8_act_1_2_buff_0, %view_2, %bn8_act_2_3_buff_0, %c14_i32_34, %c1_i32_35, %c184_i32_36, %c3_i32_37, %c3_i32_38, %c1_i32_39, %1, %c0_i32_40) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_41 = arith.constant 14 : i32
      %c184_i32_42 = arith.constant 184 : i32
      %c80_i32_43 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_0, %c14_i32_41, %c184_i32_42, %c80_i32_43, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c80_i32_45 = arith.constant 80 : i32
      %c184_i32_46 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_1, %view, %bn8_act_1_2_buff_1, %c14_i32_44, %c80_i32_45, %c184_i32_46, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_47 = arith.constant 14 : i32
      %c1_i32_48 = arith.constant 1 : i32
      %c184_i32_49 = arith.constant 184 : i32
      %c3_i32_50 = arith.constant 3 : i32
      %c3_i32_51 = arith.constant 3 : i32
      %c1_i32_52 = arith.constant 1 : i32
      %c0_i32_53 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_2, %bn8_act_1_2_buff_0, %bn8_act_1_2_buff_1, %view_2, %bn8_act_2_3_buff_0, %c14_i32_47, %c1_i32_48, %c184_i32_49, %c3_i32_50, %c3_i32_51, %c1_i32_52, %1, %c0_i32_53) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_54 = arith.constant 14 : i32
      %c184_i32_55 = arith.constant 184 : i32
      %c80_i32_56 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_1, %c14_i32_54, %c184_i32_55, %c80_i32_56, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_57 = arith.constant 14 : i32
      %c80_i32_58 = arith.constant 80 : i32
      %c184_i32_59 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_2, %view, %bn8_act_1_2_buff_2, %c14_i32_57, %c80_i32_58, %c184_i32_59, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_60 = arith.constant 14 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c184_i32_62 = arith.constant 184 : i32
      %c3_i32_63 = arith.constant 3 : i32
      %c3_i32_64 = arith.constant 3 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c0_i32_66 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_0, %bn8_act_1_2_buff_1, %bn8_act_1_2_buff_2, %view_2, %bn8_act_2_3_buff_0, %c14_i32_60, %c1_i32_61, %c184_i32_62, %c3_i32_63, %c3_i32_64, %c1_i32_65, %1, %c0_i32_66) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_67 = arith.constant 14 : i32
      %c184_i32_68 = arith.constant 184 : i32
      %c80_i32_69 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_0, %c14_i32_67, %c184_i32_68, %c80_i32_69, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_70 = arith.constant 14 : i32
      %c80_i32_71 = arith.constant 80 : i32
      %c184_i32_72 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_0, %view, %bn8_act_1_2_buff_0, %c14_i32_70, %c80_i32_71, %c184_i32_72, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_73 = arith.constant 14 : i32
      %c1_i32_74 = arith.constant 1 : i32
      %c184_i32_75 = arith.constant 184 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c3_i32_77 = arith.constant 3 : i32
      %c1_i32_78 = arith.constant 1 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_1, %bn8_act_1_2_buff_2, %bn8_act_1_2_buff_0, %view_2, %bn8_act_2_3_buff_0, %c14_i32_73, %c1_i32_74, %c184_i32_75, %c3_i32_76, %c3_i32_77, %c1_i32_78, %1, %c0_i32_79) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c184_i32_81 = arith.constant 184 : i32
      %c80_i32_82 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_1, %c14_i32_80, %c184_i32_81, %c80_i32_82, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_83 = arith.constant 14 : i32
      %c80_i32_84 = arith.constant 80 : i32
      %c184_i32_85 = arith.constant 184 : i32
      func.call @bn8_conv2dk1_relu_i8_ui8(%act_bn7_bn8_cons_buff_1, %view, %bn8_act_1_2_buff_1, %c14_i32_83, %c80_i32_84, %c184_i32_85, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_86 = arith.constant 14 : i32
      %c1_i32_87 = arith.constant 1 : i32
      %c184_i32_88 = arith.constant 184 : i32
      %c3_i32_89 = arith.constant 3 : i32
      %c3_i32_90 = arith.constant 3 : i32
      %c1_i32_91 = arith.constant 1 : i32
      %c0_i32_92 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_2, %bn8_act_1_2_buff_0, %bn8_act_1_2_buff_1, %view_2, %bn8_act_2_3_buff_0, %c14_i32_86, %c1_i32_87, %c184_i32_88, %c3_i32_89, %c3_i32_90, %c1_i32_91, %1, %c0_i32_92) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_93 = arith.constant 14 : i32
      %c184_i32_94 = arith.constant 184 : i32
      %c80_i32_95 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_0, %c14_i32_93, %c184_i32_94, %c80_i32_95, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      %5 = arith.addi %3, %c6 : index
      cf.br ^bb1(%5 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%bn8_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c184_i32_98 = arith.constant 184 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_101 = arith.constant 0 : i32
      func.call @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(%bn8_act_1_2_buff_0, %bn8_act_1_2_buff_1, %bn8_act_1_2_buff_1, %view_2, %bn8_act_2_3_buff_0, %c14_i32_96, %c1_i32_97, %c184_i32_98, %c3_i32_99, %c3_i32_100, %c2_i32, %1, %c0_i32_101) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn8_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn8_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn8_bn9_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_102 = arith.constant 14 : i32
      %c184_i32_103 = arith.constant 184 : i32
      %c80_i32_104 = arith.constant 80 : i32
      func.call @bn8_conv2dk1_ui8_i8(%bn8_act_2_3_buff_0, %view_3, %act_bn8_bn9_buff_1, %c14_i32_102, %c184_i32_103, %c80_i32_104, %2) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn8_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, Release, 1)
      aie.use_lock(%bn8_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn8_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a"}
    func.func private @bn9_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32)
    func.func private @bn9_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn9_conv2dk1_skip_ui8_i8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32)
    func.func private @bn9_conv2dk1_ui8_i8(memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, i32, i32, i32, i32)
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      aie.use_lock(%bn9_wts_OF_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %view = memref.view %bn9_wts_OF_L2L1_cons_buff_0[%c0_1][] : memref<31096xi8> to memref<14720xi8>
      %c14720 = arith.constant 14720 : index
      %view_2 = memref.view %bn9_wts_OF_L2L1_cons_buff_0[%c14720][] : memref<31096xi8> to memref<1656xi8>
      %c16376 = arith.constant 16376 : index
      %view_3 = memref.view %bn9_wts_OF_L2L1_cons_buff_0[%c16376][] : memref<31096xi8> to memref<14720xi8>
      %c0_4 = arith.constant 0 : index
      %0 = memref.load %rtp23[%c0_4] : memref<16xi32>
      %c1_5 = arith.constant 1 : index
      %1 = memref.load %rtp23[%c1_5] : memref<16xi32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %rtp23[%c2] : memref<16xi32>
      %c3 = arith.constant 3 : index
      %3 = memref.load %rtp23[%c3] : memref<16xi32>
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 2)
      %c14_i32 = arith.constant 14 : i32
      %c80_i32 = arith.constant 80 : i32
      %c184_i32 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_0, %view, %bn9_act_1_2_buff_0, %c14_i32, %c80_i32, %c184_i32, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      %c14_i32_6 = arith.constant 14 : i32
      %c80_i32_7 = arith.constant 80 : i32
      %c184_i32_8 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_1, %view, %bn9_act_1_2_buff_1, %c14_i32_6, %c80_i32_7, %c184_i32_8, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 2)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_9 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c184_i32_10 = arith.constant 184 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_0, %bn9_act_1_2_buff_0, %bn9_act_1_2_buff_1, %view_2, %bn9_act_2_3_buff_0, %c14_i32_9, %c1_i32, %c184_i32_10, %c3_i32, %c3_i32_11, %c0_i32, %1, %c0_i32_12) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c184_i32_14 = arith.constant 184 : i32
      %c80_i32_15 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_0, %act_bn8_bn9_buff_0, %c14_i32_13, %c184_i32_14, %c80_i32_15, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      %c0_16 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_17 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      cf.br ^bb1(%c0_16 : index)
    ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
      %5 = arith.cmpi slt, %4, %c12 : index
      cf.cond_br %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_18 = arith.constant 14 : i32
      %c80_i32_19 = arith.constant 80 : i32
      %c184_i32_20 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_0, %view, %bn9_act_1_2_buff_2, %c14_i32_18, %c80_i32_19, %c184_i32_20, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c184_i32_23 = arith.constant 184 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c0_i32_27 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_0, %bn9_act_1_2_buff_1, %bn9_act_1_2_buff_2, %view_2, %bn9_act_2_3_buff_0, %c14_i32_21, %c1_i32_22, %c184_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %1, %c0_i32_27) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_28 = arith.constant 14 : i32
      %c184_i32_29 = arith.constant 184 : i32
      %c80_i32_30 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_1, %act_bn8_bn9_buff_1, %c14_i32_28, %c184_i32_29, %c80_i32_30, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_31 = arith.constant 14 : i32
      %c80_i32_32 = arith.constant 80 : i32
      %c184_i32_33 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_1, %view, %bn9_act_1_2_buff_0, %c14_i32_31, %c80_i32_32, %c184_i32_33, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_34 = arith.constant 14 : i32
      %c1_i32_35 = arith.constant 1 : i32
      %c184_i32_36 = arith.constant 184 : i32
      %c3_i32_37 = arith.constant 3 : i32
      %c3_i32_38 = arith.constant 3 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c0_i32_40 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_1, %bn9_act_1_2_buff_2, %bn9_act_1_2_buff_0, %view_2, %bn9_act_2_3_buff_0, %c14_i32_34, %c1_i32_35, %c184_i32_36, %c3_i32_37, %c3_i32_38, %c1_i32_39, %1, %c0_i32_40) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_41 = arith.constant 14 : i32
      %c184_i32_42 = arith.constant 184 : i32
      %c80_i32_43 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_0, %act_bn8_bn9_buff_0, %c14_i32_41, %c184_i32_42, %c80_i32_43, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c80_i32_45 = arith.constant 80 : i32
      %c184_i32_46 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_0, %view, %bn9_act_1_2_buff_1, %c14_i32_44, %c80_i32_45, %c184_i32_46, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_47 = arith.constant 14 : i32
      %c1_i32_48 = arith.constant 1 : i32
      %c184_i32_49 = arith.constant 184 : i32
      %c3_i32_50 = arith.constant 3 : i32
      %c3_i32_51 = arith.constant 3 : i32
      %c1_i32_52 = arith.constant 1 : i32
      %c0_i32_53 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_2, %bn9_act_1_2_buff_0, %bn9_act_1_2_buff_1, %view_2, %bn9_act_2_3_buff_0, %c14_i32_47, %c1_i32_48, %c184_i32_49, %c3_i32_50, %c3_i32_51, %c1_i32_52, %1, %c0_i32_53) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_54 = arith.constant 14 : i32
      %c184_i32_55 = arith.constant 184 : i32
      %c80_i32_56 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_1, %act_bn8_bn9_buff_1, %c14_i32_54, %c184_i32_55, %c80_i32_56, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_57 = arith.constant 14 : i32
      %c80_i32_58 = arith.constant 80 : i32
      %c184_i32_59 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_1, %view, %bn9_act_1_2_buff_2, %c14_i32_57, %c80_i32_58, %c184_i32_59, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_60 = arith.constant 14 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c184_i32_62 = arith.constant 184 : i32
      %c3_i32_63 = arith.constant 3 : i32
      %c3_i32_64 = arith.constant 3 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c0_i32_66 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_0, %bn9_act_1_2_buff_1, %bn9_act_1_2_buff_2, %view_2, %bn9_act_2_3_buff_0, %c14_i32_60, %c1_i32_61, %c184_i32_62, %c3_i32_63, %c3_i32_64, %c1_i32_65, %1, %c0_i32_66) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_67 = arith.constant 14 : i32
      %c184_i32_68 = arith.constant 184 : i32
      %c80_i32_69 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_0, %act_bn8_bn9_buff_0, %c14_i32_67, %c184_i32_68, %c80_i32_69, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_70 = arith.constant 14 : i32
      %c80_i32_71 = arith.constant 80 : i32
      %c184_i32_72 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_0, %view, %bn9_act_1_2_buff_0, %c14_i32_70, %c80_i32_71, %c184_i32_72, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_73 = arith.constant 14 : i32
      %c1_i32_74 = arith.constant 1 : i32
      %c184_i32_75 = arith.constant 184 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c3_i32_77 = arith.constant 3 : i32
      %c1_i32_78 = arith.constant 1 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_1, %bn9_act_1_2_buff_2, %bn9_act_1_2_buff_0, %view_2, %bn9_act_2_3_buff_0, %c14_i32_73, %c1_i32_74, %c184_i32_75, %c3_i32_76, %c3_i32_77, %c1_i32_78, %1, %c0_i32_79) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c184_i32_81 = arith.constant 184 : i32
      %c80_i32_82 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_1, %act_bn8_bn9_buff_1, %c14_i32_80, %c184_i32_81, %c80_i32_82, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      aie.use_lock(%act_bn8_bn9_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_1_2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_83 = arith.constant 14 : i32
      %c80_i32_84 = arith.constant 80 : i32
      %c184_i32_85 = arith.constant 184 : i32
      func.call @bn9_conv2dk1_relu_i8_ui8(%act_bn8_bn9_buff_1, %view, %bn9_act_1_2_buff_1, %c14_i32_83, %c80_i32_84, %c184_i32_85, %0) : (memref<14x1x80xi8>, memref<14720xi8>, memref<14x1x184xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_1_2_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_86 = arith.constant 14 : i32
      %c1_i32_87 = arith.constant 1 : i32
      %c184_i32_88 = arith.constant 184 : i32
      %c3_i32_89 = arith.constant 3 : i32
      %c3_i32_90 = arith.constant 3 : i32
      %c1_i32_91 = arith.constant 1 : i32
      %c0_i32_92 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_2, %bn9_act_1_2_buff_0, %bn9_act_1_2_buff_1, %view_2, %bn9_act_2_3_buff_0, %c14_i32_86, %c1_i32_87, %c184_i32_88, %c3_i32_89, %c3_i32_90, %c1_i32_91, %1, %c0_i32_92) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_93 = arith.constant 14 : i32
      %c184_i32_94 = arith.constant 184 : i32
      %c80_i32_95 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_0, %act_bn8_bn9_buff_0, %c14_i32_93, %c184_i32_94, %c80_i32_95, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      %6 = arith.addi %4, %c6 : index
      cf.br ^bb1(%6 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%bn9_act_2_3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c184_i32_98 = arith.constant 184 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_101 = arith.constant 0 : i32
      func.call @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(%bn9_act_1_2_buff_0, %bn9_act_1_2_buff_1, %bn9_act_1_2_buff_1, %view_2, %bn9_act_2_3_buff_0, %c14_i32_96, %c1_i32_97, %c184_i32_98, %c3_i32_99, %c3_i32_100, %c2_i32, %1, %c0_i32_101) : (memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>, memref<1656xi8>, memref<14x1x184xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%bn9_act_1_2_prod_lock, Release, 2)
      aie.use_lock(%bn9_act_2_3_cons_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_bn9_bn10_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_102 = arith.constant 14 : i32
      %c184_i32_103 = arith.constant 184 : i32
      %c80_i32_104 = arith.constant 80 : i32
      func.call @bn9_conv2dk1_skip_ui8_i8_i8(%bn9_act_2_3_buff_0, %view_3, %act_bn9_bn10_buff_1, %act_bn8_bn9_buff_1, %c14_i32_102, %c184_i32_103, %c80_i32_104, %2, %3) : (memref<14x1x184xui8>, memref<14720xi8>, memref<14x1x80xi8>, memref<14x1x80xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn8_bn9_prod_lock, Release, 1)
      aie.use_lock(%bn9_act_2_3_prod_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, Release, 1)
      aie.use_lock(%bn9_wts_OF_L2L1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn9_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a"}
    func.func private @bn10_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32)
    func.func private @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn10_conv2dk1_ui8_i8(memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_relu_i8_ui8(memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_skip_ui8_i8_i8(memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32)
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%weightsInBN10_layer1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_1 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_bn9_bn10_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c80_i32 = arith.constant 80 : i32
      %c480_i32 = arith.constant 480 : i32
      %c8_i32 = arith.constant 8 : i32
      func.call @bn10_conv2dk1_relu_i8_ui8(%act_bn9_bn10_buff_0, %weightsInBN10_layer1_cons_buff_0, %B_OF_b10_act_layer1_layer2_buff_0, %c14_i32, %c80_i32, %c480_i32, %c8_i32) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn9_bn10_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_lock, Release, 1)
      aie.use_lock(%act_bn9_bn10_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_3 = arith.constant 14 : i32
      %c80_i32_4 = arith.constant 80 : i32
      %c480_i32_5 = arith.constant 480 : i32
      %c8_i32_6 = arith.constant 8 : i32
      func.call @bn10_conv2dk1_relu_i8_ui8(%act_bn9_bn10_buff_1, %weightsInBN10_layer1_cons_buff_0, %B_OF_b10_act_layer1_layer2_buff_1, %c14_i32_3, %c80_i32_4, %c480_i32_5, %c8_i32_6) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_bn9_bn10_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%weightsInBN10_layer1_cons_prod_lock, Release, 1)
      %5 = arith.addi %0, %c1_0 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "bn10_conv2dk1_fused_relu.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775804 = arith.constant 9223372036854775804 : index
      %c4 = arith.constant 4 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775804 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c480_i32 = arith.constant 480 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_0 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c7_i32 = arith.constant 7 : i32
      %c0_i32_1 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32, %c1_i32, %c480_i32, %c3_i32, %c3_i32_0, %c0_i32, %c7_i32, %c0_i32_1) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_2 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_3 = arith.constant 1 : index
      %c4_4 = arith.constant 4 : index
      cf.br ^bb3(%c0_2 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c12 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_5 = arith.constant 14 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c480_i32_7 = arith.constant 480 : i32
      %c3_i32_8 = arith.constant 3 : i32
      %c3_i32_9 = arith.constant 3 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c7_i32_11 = arith.constant 7 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_5, %c1_i32_6, %c480_i32_7, %c3_i32_8, %c3_i32_9, %c1_i32_10, %c7_i32_11, %c0_i32_12) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %c480_i32_15 = arith.constant 480 : i32
      %c3_i32_16 = arith.constant 3 : i32
      %c3_i32_17 = arith.constant 3 : i32
      %c1_i32_18 = arith.constant 1 : i32
      %c7_i32_19 = arith.constant 7 : i32
      %c0_i32_20 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_13, %c1_i32_14, %c480_i32_15, %c3_i32_16, %c3_i32_17, %c1_i32_18, %c7_i32_19, %c0_i32_20) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c480_i32_23 = arith.constant 480 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c7_i32_27 = arith.constant 7 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_21, %c1_i32_22, %c480_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %c7_i32_27, %c0_i32_28) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_29 = arith.constant 14 : i32
      %c1_i32_30 = arith.constant 1 : i32
      %c480_i32_31 = arith.constant 480 : i32
      %c3_i32_32 = arith.constant 3 : i32
      %c3_i32_33 = arith.constant 3 : i32
      %c1_i32_34 = arith.constant 1 : i32
      %c7_i32_35 = arith.constant 7 : i32
      %c0_i32_36 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_29, %c1_i32_30, %c480_i32_31, %c3_i32_32, %c3_i32_33, %c1_i32_34, %c7_i32_35, %c0_i32_36) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %4 = arith.addi %2, %c4_4 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_37 = arith.constant 14 : i32
      %c1_i32_38 = arith.constant 1 : i32
      %c480_i32_39 = arith.constant 480 : i32
      %c3_i32_40 = arith.constant 3 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c7_i32_42 = arith.constant 7 : i32
      %c0_i32_43 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_37, %c1_i32_38, %c480_i32_39, %c3_i32_40, %c3_i32_41, %c2_i32, %c7_i32_42, %c0_i32_43) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c1_i32_45 = arith.constant 1 : i32
      %c480_i32_46 = arith.constant 480 : i32
      %c3_i32_47 = arith.constant 3 : i32
      %c3_i32_48 = arith.constant 3 : i32
      %c0_i32_49 = arith.constant 0 : i32
      %c7_i32_50 = arith.constant 7 : i32
      %c0_i32_51 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_44, %c1_i32_45, %c480_i32_46, %c3_i32_47, %c3_i32_48, %c0_i32_49, %c7_i32_50, %c0_i32_51) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_52 = arith.constant 0 : index
      %c12_53 = arith.constant 12 : index
      %c1_54 = arith.constant 1 : index
      %c4_55 = arith.constant 4 : index
      cf.br ^bb6(%c0_52 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c12_53 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_56 = arith.constant 14 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c480_i32_58 = arith.constant 480 : i32
      %c3_i32_59 = arith.constant 3 : i32
      %c3_i32_60 = arith.constant 3 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c7_i32_62 = arith.constant 7 : i32
      %c0_i32_63 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_56, %c1_i32_57, %c480_i32_58, %c3_i32_59, %c3_i32_60, %c1_i32_61, %c7_i32_62, %c0_i32_63) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_64 = arith.constant 14 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c480_i32_66 = arith.constant 480 : i32
      %c3_i32_67 = arith.constant 3 : i32
      %c3_i32_68 = arith.constant 3 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c7_i32_70 = arith.constant 7 : i32
      %c0_i32_71 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_64, %c1_i32_65, %c480_i32_66, %c3_i32_67, %c3_i32_68, %c1_i32_69, %c7_i32_70, %c0_i32_71) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_72 = arith.constant 14 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c480_i32_74 = arith.constant 480 : i32
      %c3_i32_75 = arith.constant 3 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c1_i32_77 = arith.constant 1 : i32
      %c7_i32_78 = arith.constant 7 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_72, %c1_i32_73, %c480_i32_74, %c3_i32_75, %c3_i32_76, %c1_i32_77, %c7_i32_78, %c0_i32_79) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c1_i32_81 = arith.constant 1 : i32
      %c480_i32_82 = arith.constant 480 : i32
      %c3_i32_83 = arith.constant 3 : i32
      %c3_i32_84 = arith.constant 3 : i32
      %c1_i32_85 = arith.constant 1 : i32
      %c7_i32_86 = arith.constant 7 : i32
      %c0_i32_87 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_80, %c1_i32_81, %c480_i32_82, %c3_i32_83, %c3_i32_84, %c1_i32_85, %c7_i32_86, %c0_i32_87) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %7 = arith.addi %5, %c4_55 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_88 = arith.constant 14 : i32
      %c1_i32_89 = arith.constant 1 : i32
      %c480_i32_90 = arith.constant 480 : i32
      %c3_i32_91 = arith.constant 3 : i32
      %c3_i32_92 = arith.constant 3 : i32
      %c2_i32_93 = arith.constant 2 : i32
      %c7_i32_94 = arith.constant 7 : i32
      %c0_i32_95 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_88, %c1_i32_89, %c480_i32_90, %c3_i32_91, %c3_i32_92, %c2_i32_93, %c7_i32_94, %c0_i32_95) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c480_i32_98 = arith.constant 480 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c0_i32_101 = arith.constant 0 : i32
      %c7_i32_102 = arith.constant 7 : i32
      %c0_i32_103 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_96, %c1_i32_97, %c480_i32_98, %c3_i32_99, %c3_i32_100, %c0_i32_101, %c7_i32_102, %c0_i32_103) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_104 = arith.constant 0 : index
      %c12_105 = arith.constant 12 : index
      %c1_106 = arith.constant 1 : index
      %c4_107 = arith.constant 4 : index
      cf.br ^bb9(%c0_104 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c12_105 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_108 = arith.constant 14 : i32
      %c1_i32_109 = arith.constant 1 : i32
      %c480_i32_110 = arith.constant 480 : i32
      %c3_i32_111 = arith.constant 3 : i32
      %c3_i32_112 = arith.constant 3 : i32
      %c1_i32_113 = arith.constant 1 : i32
      %c7_i32_114 = arith.constant 7 : i32
      %c0_i32_115 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_108, %c1_i32_109, %c480_i32_110, %c3_i32_111, %c3_i32_112, %c1_i32_113, %c7_i32_114, %c0_i32_115) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_116 = arith.constant 14 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c480_i32_118 = arith.constant 480 : i32
      %c3_i32_119 = arith.constant 3 : i32
      %c3_i32_120 = arith.constant 3 : i32
      %c1_i32_121 = arith.constant 1 : i32
      %c7_i32_122 = arith.constant 7 : i32
      %c0_i32_123 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_116, %c1_i32_117, %c480_i32_118, %c3_i32_119, %c3_i32_120, %c1_i32_121, %c7_i32_122, %c0_i32_123) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_124 = arith.constant 14 : i32
      %c1_i32_125 = arith.constant 1 : i32
      %c480_i32_126 = arith.constant 480 : i32
      %c3_i32_127 = arith.constant 3 : i32
      %c3_i32_128 = arith.constant 3 : i32
      %c1_i32_129 = arith.constant 1 : i32
      %c7_i32_130 = arith.constant 7 : i32
      %c0_i32_131 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_124, %c1_i32_125, %c480_i32_126, %c3_i32_127, %c3_i32_128, %c1_i32_129, %c7_i32_130, %c0_i32_131) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_132 = arith.constant 14 : i32
      %c1_i32_133 = arith.constant 1 : i32
      %c480_i32_134 = arith.constant 480 : i32
      %c3_i32_135 = arith.constant 3 : i32
      %c3_i32_136 = arith.constant 3 : i32
      %c1_i32_137 = arith.constant 1 : i32
      %c7_i32_138 = arith.constant 7 : i32
      %c0_i32_139 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_132, %c1_i32_133, %c480_i32_134, %c3_i32_135, %c3_i32_136, %c1_i32_137, %c7_i32_138, %c0_i32_139) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %10 = arith.addi %8, %c4_107 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_140 = arith.constant 14 : i32
      %c1_i32_141 = arith.constant 1 : i32
      %c480_i32_142 = arith.constant 480 : i32
      %c3_i32_143 = arith.constant 3 : i32
      %c3_i32_144 = arith.constant 3 : i32
      %c2_i32_145 = arith.constant 2 : i32
      %c7_i32_146 = arith.constant 7 : i32
      %c0_i32_147 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_140, %c1_i32_141, %c480_i32_142, %c3_i32_143, %c3_i32_144, %c2_i32_145, %c7_i32_146, %c0_i32_147) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_148 = arith.constant 14 : i32
      %c1_i32_149 = arith.constant 1 : i32
      %c480_i32_150 = arith.constant 480 : i32
      %c3_i32_151 = arith.constant 3 : i32
      %c3_i32_152 = arith.constant 3 : i32
      %c0_i32_153 = arith.constant 0 : i32
      %c7_i32_154 = arith.constant 7 : i32
      %c0_i32_155 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_148, %c1_i32_149, %c480_i32_150, %c3_i32_151, %c3_i32_152, %c0_i32_153, %c7_i32_154, %c0_i32_155) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_156 = arith.constant 0 : index
      %c12_157 = arith.constant 12 : index
      %c1_158 = arith.constant 1 : index
      %c4_159 = arith.constant 4 : index
      cf.br ^bb12(%c0_156 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c12_157 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_160 = arith.constant 14 : i32
      %c1_i32_161 = arith.constant 1 : i32
      %c480_i32_162 = arith.constant 480 : i32
      %c3_i32_163 = arith.constant 3 : i32
      %c3_i32_164 = arith.constant 3 : i32
      %c1_i32_165 = arith.constant 1 : i32
      %c7_i32_166 = arith.constant 7 : i32
      %c0_i32_167 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_160, %c1_i32_161, %c480_i32_162, %c3_i32_163, %c3_i32_164, %c1_i32_165, %c7_i32_166, %c0_i32_167) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_168 = arith.constant 14 : i32
      %c1_i32_169 = arith.constant 1 : i32
      %c480_i32_170 = arith.constant 480 : i32
      %c3_i32_171 = arith.constant 3 : i32
      %c3_i32_172 = arith.constant 3 : i32
      %c1_i32_173 = arith.constant 1 : i32
      %c7_i32_174 = arith.constant 7 : i32
      %c0_i32_175 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_168, %c1_i32_169, %c480_i32_170, %c3_i32_171, %c3_i32_172, %c1_i32_173, %c7_i32_174, %c0_i32_175) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_176 = arith.constant 14 : i32
      %c1_i32_177 = arith.constant 1 : i32
      %c480_i32_178 = arith.constant 480 : i32
      %c3_i32_179 = arith.constant 3 : i32
      %c3_i32_180 = arith.constant 3 : i32
      %c1_i32_181 = arith.constant 1 : i32
      %c7_i32_182 = arith.constant 7 : i32
      %c0_i32_183 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_176, %c1_i32_177, %c480_i32_178, %c3_i32_179, %c3_i32_180, %c1_i32_181, %c7_i32_182, %c0_i32_183) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_184 = arith.constant 14 : i32
      %c1_i32_185 = arith.constant 1 : i32
      %c480_i32_186 = arith.constant 480 : i32
      %c3_i32_187 = arith.constant 3 : i32
      %c3_i32_188 = arith.constant 3 : i32
      %c1_i32_189 = arith.constant 1 : i32
      %c7_i32_190 = arith.constant 7 : i32
      %c0_i32_191 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_184, %c1_i32_185, %c480_i32_186, %c3_i32_187, %c3_i32_188, %c1_i32_189, %c7_i32_190, %c0_i32_191) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %13 = arith.addi %11, %c4_159 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_192 = arith.constant 14 : i32
      %c1_i32_193 = arith.constant 1 : i32
      %c480_i32_194 = arith.constant 480 : i32
      %c3_i32_195 = arith.constant 3 : i32
      %c3_i32_196 = arith.constant 3 : i32
      %c2_i32_197 = arith.constant 2 : i32
      %c7_i32_198 = arith.constant 7 : i32
      %c0_i32_199 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_192, %c1_i32_193, %c480_i32_194, %c3_i32_195, %c3_i32_196, %c2_i32_197, %c7_i32_198, %c0_i32_199) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_200 = arith.constant 14 : i32
      %c1_i32_201 = arith.constant 1 : i32
      %c480_i32_202 = arith.constant 480 : i32
      %c3_i32_203 = arith.constant 3 : i32
      %c3_i32_204 = arith.constant 3 : i32
      %c0_i32_205 = arith.constant 0 : i32
      %c7_i32_206 = arith.constant 7 : i32
      %c0_i32_207 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_200, %c1_i32_201, %c480_i32_202, %c3_i32_203, %c3_i32_204, %c0_i32_205, %c7_i32_206, %c0_i32_207) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_208 = arith.constant 0 : index
      %c12_209 = arith.constant 12 : index
      %c1_210 = arith.constant 1 : index
      %c4_211 = arith.constant 4 : index
      cf.br ^bb16(%c0_208 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb17
      %16 = arith.cmpi slt, %15, %c12_209 : index
      cf.cond_br %16, ^bb17, ^bb18
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_212 = arith.constant 14 : i32
      %c1_i32_213 = arith.constant 1 : i32
      %c480_i32_214 = arith.constant 480 : i32
      %c3_i32_215 = arith.constant 3 : i32
      %c3_i32_216 = arith.constant 3 : i32
      %c1_i32_217 = arith.constant 1 : i32
      %c7_i32_218 = arith.constant 7 : i32
      %c0_i32_219 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_212, %c1_i32_213, %c480_i32_214, %c3_i32_215, %c3_i32_216, %c1_i32_217, %c7_i32_218, %c0_i32_219) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_220 = arith.constant 14 : i32
      %c1_i32_221 = arith.constant 1 : i32
      %c480_i32_222 = arith.constant 480 : i32
      %c3_i32_223 = arith.constant 3 : i32
      %c3_i32_224 = arith.constant 3 : i32
      %c1_i32_225 = arith.constant 1 : i32
      %c7_i32_226 = arith.constant 7 : i32
      %c0_i32_227 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_220, %c1_i32_221, %c480_i32_222, %c3_i32_223, %c3_i32_224, %c1_i32_225, %c7_i32_226, %c0_i32_227) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_228 = arith.constant 14 : i32
      %c1_i32_229 = arith.constant 1 : i32
      %c480_i32_230 = arith.constant 480 : i32
      %c3_i32_231 = arith.constant 3 : i32
      %c3_i32_232 = arith.constant 3 : i32
      %c1_i32_233 = arith.constant 1 : i32
      %c7_i32_234 = arith.constant 7 : i32
      %c0_i32_235 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_228, %c1_i32_229, %c480_i32_230, %c3_i32_231, %c3_i32_232, %c1_i32_233, %c7_i32_234, %c0_i32_235) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_236 = arith.constant 14 : i32
      %c1_i32_237 = arith.constant 1 : i32
      %c480_i32_238 = arith.constant 480 : i32
      %c3_i32_239 = arith.constant 3 : i32
      %c3_i32_240 = arith.constant 3 : i32
      %c1_i32_241 = arith.constant 1 : i32
      %c7_i32_242 = arith.constant 7 : i32
      %c0_i32_243 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_236, %c1_i32_237, %c480_i32_238, %c3_i32_239, %c3_i32_240, %c1_i32_241, %c7_i32_242, %c0_i32_243) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %17 = arith.addi %15, %c4_211 : index
      cf.br ^bb16(%17 : index)
    ^bb18:  // pred: ^bb16
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_244 = arith.constant 14 : i32
      %c1_i32_245 = arith.constant 1 : i32
      %c480_i32_246 = arith.constant 480 : i32
      %c3_i32_247 = arith.constant 3 : i32
      %c3_i32_248 = arith.constant 3 : i32
      %c2_i32_249 = arith.constant 2 : i32
      %c7_i32_250 = arith.constant 7 : i32
      %c0_i32_251 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_244, %c1_i32_245, %c480_i32_246, %c3_i32_247, %c3_i32_248, %c2_i32_249, %c7_i32_250, %c0_i32_251) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_252 = arith.constant 14 : i32
      %c1_i32_253 = arith.constant 1 : i32
      %c480_i32_254 = arith.constant 480 : i32
      %c3_i32_255 = arith.constant 3 : i32
      %c3_i32_256 = arith.constant 3 : i32
      %c0_i32_257 = arith.constant 0 : i32
      %c7_i32_258 = arith.constant 7 : i32
      %c0_i32_259 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_252, %c1_i32_253, %c480_i32_254, %c3_i32_255, %c3_i32_256, %c0_i32_257, %c7_i32_258, %c0_i32_259) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_260 = arith.constant 0 : index
      %c12_261 = arith.constant 12 : index
      %c1_262 = arith.constant 1 : index
      %c4_263 = arith.constant 4 : index
      cf.br ^bb19(%c0_260 : index)
    ^bb19(%18: index):  // 2 preds: ^bb18, ^bb20
      %19 = arith.cmpi slt, %18, %c12_261 : index
      cf.cond_br %19, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_264 = arith.constant 14 : i32
      %c1_i32_265 = arith.constant 1 : i32
      %c480_i32_266 = arith.constant 480 : i32
      %c3_i32_267 = arith.constant 3 : i32
      %c3_i32_268 = arith.constant 3 : i32
      %c1_i32_269 = arith.constant 1 : i32
      %c7_i32_270 = arith.constant 7 : i32
      %c0_i32_271 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_264, %c1_i32_265, %c480_i32_266, %c3_i32_267, %c3_i32_268, %c1_i32_269, %c7_i32_270, %c0_i32_271) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_272 = arith.constant 14 : i32
      %c1_i32_273 = arith.constant 1 : i32
      %c480_i32_274 = arith.constant 480 : i32
      %c3_i32_275 = arith.constant 3 : i32
      %c3_i32_276 = arith.constant 3 : i32
      %c1_i32_277 = arith.constant 1 : i32
      %c7_i32_278 = arith.constant 7 : i32
      %c0_i32_279 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_272, %c1_i32_273, %c480_i32_274, %c3_i32_275, %c3_i32_276, %c1_i32_277, %c7_i32_278, %c0_i32_279) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_280 = arith.constant 14 : i32
      %c1_i32_281 = arith.constant 1 : i32
      %c480_i32_282 = arith.constant 480 : i32
      %c3_i32_283 = arith.constant 3 : i32
      %c3_i32_284 = arith.constant 3 : i32
      %c1_i32_285 = arith.constant 1 : i32
      %c7_i32_286 = arith.constant 7 : i32
      %c0_i32_287 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_280, %c1_i32_281, %c480_i32_282, %c3_i32_283, %c3_i32_284, %c1_i32_285, %c7_i32_286, %c0_i32_287) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_288 = arith.constant 14 : i32
      %c1_i32_289 = arith.constant 1 : i32
      %c480_i32_290 = arith.constant 480 : i32
      %c3_i32_291 = arith.constant 3 : i32
      %c3_i32_292 = arith.constant 3 : i32
      %c1_i32_293 = arith.constant 1 : i32
      %c7_i32_294 = arith.constant 7 : i32
      %c0_i32_295 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_288, %c1_i32_289, %c480_i32_290, %c3_i32_291, %c3_i32_292, %c1_i32_293, %c7_i32_294, %c0_i32_295) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %20 = arith.addi %18, %c4_263 : index
      cf.br ^bb19(%20 : index)
    ^bb21:  // pred: ^bb19
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_296 = arith.constant 14 : i32
      %c1_i32_297 = arith.constant 1 : i32
      %c480_i32_298 = arith.constant 480 : i32
      %c3_i32_299 = arith.constant 3 : i32
      %c3_i32_300 = arith.constant 3 : i32
      %c2_i32_301 = arith.constant 2 : i32
      %c7_i32_302 = arith.constant 7 : i32
      %c0_i32_303 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_296, %c1_i32_297, %c480_i32_298, %c3_i32_299, %c3_i32_300, %c2_i32_301, %c7_i32_302, %c0_i32_303) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_304 = arith.constant 14 : i32
      %c1_i32_305 = arith.constant 1 : i32
      %c480_i32_306 = arith.constant 480 : i32
      %c3_i32_307 = arith.constant 3 : i32
      %c3_i32_308 = arith.constant 3 : i32
      %c0_i32_309 = arith.constant 0 : i32
      %c7_i32_310 = arith.constant 7 : i32
      %c0_i32_311 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_304, %c1_i32_305, %c480_i32_306, %c3_i32_307, %c3_i32_308, %c0_i32_309, %c7_i32_310, %c0_i32_311) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_312 = arith.constant 0 : index
      %c12_313 = arith.constant 12 : index
      %c1_314 = arith.constant 1 : index
      %c4_315 = arith.constant 4 : index
      cf.br ^bb22(%c0_312 : index)
    ^bb22(%21: index):  // 2 preds: ^bb21, ^bb23
      %22 = arith.cmpi slt, %21, %c12_313 : index
      cf.cond_br %22, ^bb23, ^bb24
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_316 = arith.constant 14 : i32
      %c1_i32_317 = arith.constant 1 : i32
      %c480_i32_318 = arith.constant 480 : i32
      %c3_i32_319 = arith.constant 3 : i32
      %c3_i32_320 = arith.constant 3 : i32
      %c1_i32_321 = arith.constant 1 : i32
      %c7_i32_322 = arith.constant 7 : i32
      %c0_i32_323 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_316, %c1_i32_317, %c480_i32_318, %c3_i32_319, %c3_i32_320, %c1_i32_321, %c7_i32_322, %c0_i32_323) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_324 = arith.constant 14 : i32
      %c1_i32_325 = arith.constant 1 : i32
      %c480_i32_326 = arith.constant 480 : i32
      %c3_i32_327 = arith.constant 3 : i32
      %c3_i32_328 = arith.constant 3 : i32
      %c1_i32_329 = arith.constant 1 : i32
      %c7_i32_330 = arith.constant 7 : i32
      %c0_i32_331 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_324, %c1_i32_325, %c480_i32_326, %c3_i32_327, %c3_i32_328, %c1_i32_329, %c7_i32_330, %c0_i32_331) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_332 = arith.constant 14 : i32
      %c1_i32_333 = arith.constant 1 : i32
      %c480_i32_334 = arith.constant 480 : i32
      %c3_i32_335 = arith.constant 3 : i32
      %c3_i32_336 = arith.constant 3 : i32
      %c1_i32_337 = arith.constant 1 : i32
      %c7_i32_338 = arith.constant 7 : i32
      %c0_i32_339 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_2, %B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_332, %c1_i32_333, %c480_i32_334, %c3_i32_335, %c3_i32_336, %c1_i32_337, %c7_i32_338, %c0_i32_339) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_340 = arith.constant 14 : i32
      %c1_i32_341 = arith.constant 1 : i32
      %c480_i32_342 = arith.constant 480 : i32
      %c3_i32_343 = arith.constant 3 : i32
      %c3_i32_344 = arith.constant 3 : i32
      %c1_i32_345 = arith.constant 1 : i32
      %c7_i32_346 = arith.constant 7 : i32
      %c0_i32_347 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_3, %B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_0, %c14_i32_340, %c1_i32_341, %c480_i32_342, %c3_i32_343, %c3_i32_344, %c1_i32_345, %c7_i32_346, %c0_i32_347) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %23 = arith.addi %21, %c4_315 : index
      cf.br ^bb22(%23 : index)
    ^bb24:  // pred: ^bb22
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_348 = arith.constant 14 : i32
      %c1_i32_349 = arith.constant 1 : i32
      %c480_i32_350 = arith.constant 480 : i32
      %c3_i32_351 = arith.constant 3 : i32
      %c3_i32_352 = arith.constant 3 : i32
      %c2_i32_353 = arith.constant 2 : i32
      %c7_i32_354 = arith.constant 7 : i32
      %c0_i32_355 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b10_act_layer1_layer2_cons_buff_0, %B_OF_b10_act_layer1_layer2_cons_buff_1, %B_OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %B_OF_b10_act_layer2_layer3_buff_1, %c14_i32_348, %c1_i32_349, %c480_i32_350, %c3_i32_351, %c3_i32_352, %c2_i32_353, %c7_i32_354, %c0_i32_355) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn10_conv2dk3_dw.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%weightsInBN10_layer3_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_1 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c480_i32 = arith.constant 480 : i32
      %c112_i32 = arith.constant 112 : i32
      %c10_i32 = arith.constant 10 : i32
      func.call @bn10_conv2dk1_ui8_i8(%B_OF_b10_act_layer2_layer3_buff_0, %weightsInBN10_layer3_cons_buff_0, %B_OF_b10_layer3_bn_11_layer1_buff_0, %c14_i32, %c480_i32, %c112_i32, %c10_i32) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_3 = arith.constant 14 : i32
      %c480_i32_4 = arith.constant 480 : i32
      %c112_i32_5 = arith.constant 112 : i32
      %c10_i32_6 = arith.constant 10 : i32
      func.call @bn10_conv2dk1_ui8_i8(%B_OF_b10_act_layer2_layer3_buff_1, %weightsInBN10_layer3_cons_buff_0, %B_OF_b10_layer3_bn_11_layer1_buff_1, %c14_i32_3, %c480_i32_4, %c112_i32_5, %c10_i32_6) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%weightsInBN10_layer3_cons_prod_lock, Release, 1)
      %5 = arith.addi %0, %c1_0 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "bn10_conv2dk1_ui8.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%weightsInBN11_layer1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_1 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c112_i32 = arith.constant 112 : i32
      %c336_i32 = arith.constant 336 : i32
      %c9_i32 = arith.constant 9 : i32
      func.call @bn11_conv2dk1_relu_i8_ui8(%B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0, %weightsInBN11_layer1_cons_buff_0, %B_OF_b11_act_layer1_layer2_buff_0, %c14_i32, %c112_i32, %c336_i32, %c9_i32) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_3 = arith.constant 14 : i32
      %c112_i32_4 = arith.constant 112 : i32
      %c336_i32_5 = arith.constant 336 : i32
      %c9_i32_6 = arith.constant 9 : i32
      func.call @bn11_conv2dk1_relu_i8_ui8(%B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1, %weightsInBN11_layer1_cons_buff_0, %B_OF_b11_act_layer1_layer2_buff_1, %c14_i32_3, %c112_i32_4, %c336_i32_5, %c9_i32_6) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%weightsInBN11_layer1_cons_prod_lock, Release, 1)
      %5 = arith.addi %0, %c1_0 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "bn11_conv2dk1_fused_relu.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775804 = arith.constant 9223372036854775804 : index
      %c4 = arith.constant 4 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775804 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c336_i32 = arith.constant 336 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_0 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_i32_1 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_0, %c0_i32, %c8_i32, %c0_i32_1) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_2 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_3 = arith.constant 1 : index
      %c4_4 = arith.constant 4 : index
      cf.br ^bb3(%c0_2 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c12 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_5 = arith.constant 14 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c336_i32_7 = arith.constant 336 : i32
      %c3_i32_8 = arith.constant 3 : i32
      %c3_i32_9 = arith.constant 3 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c8_i32_11 = arith.constant 8 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_5, %c1_i32_6, %c336_i32_7, %c3_i32_8, %c3_i32_9, %c1_i32_10, %c8_i32_11, %c0_i32_12) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %c336_i32_15 = arith.constant 336 : i32
      %c3_i32_16 = arith.constant 3 : i32
      %c3_i32_17 = arith.constant 3 : i32
      %c1_i32_18 = arith.constant 1 : i32
      %c8_i32_19 = arith.constant 8 : i32
      %c0_i32_20 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_13, %c1_i32_14, %c336_i32_15, %c3_i32_16, %c3_i32_17, %c1_i32_18, %c8_i32_19, %c0_i32_20) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c336_i32_23 = arith.constant 336 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c8_i32_27 = arith.constant 8 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_21, %c1_i32_22, %c336_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %c8_i32_27, %c0_i32_28) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_29 = arith.constant 14 : i32
      %c1_i32_30 = arith.constant 1 : i32
      %c336_i32_31 = arith.constant 336 : i32
      %c3_i32_32 = arith.constant 3 : i32
      %c3_i32_33 = arith.constant 3 : i32
      %c1_i32_34 = arith.constant 1 : i32
      %c8_i32_35 = arith.constant 8 : i32
      %c0_i32_36 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_29, %c1_i32_30, %c336_i32_31, %c3_i32_32, %c3_i32_33, %c1_i32_34, %c8_i32_35, %c0_i32_36) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %4 = arith.addi %2, %c4_4 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_37 = arith.constant 14 : i32
      %c1_i32_38 = arith.constant 1 : i32
      %c336_i32_39 = arith.constant 336 : i32
      %c3_i32_40 = arith.constant 3 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c8_i32_42 = arith.constant 8 : i32
      %c0_i32_43 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_37, %c1_i32_38, %c336_i32_39, %c3_i32_40, %c3_i32_41, %c2_i32, %c8_i32_42, %c0_i32_43) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c1_i32_45 = arith.constant 1 : i32
      %c336_i32_46 = arith.constant 336 : i32
      %c3_i32_47 = arith.constant 3 : i32
      %c3_i32_48 = arith.constant 3 : i32
      %c0_i32_49 = arith.constant 0 : i32
      %c8_i32_50 = arith.constant 8 : i32
      %c0_i32_51 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_44, %c1_i32_45, %c336_i32_46, %c3_i32_47, %c3_i32_48, %c0_i32_49, %c8_i32_50, %c0_i32_51) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_52 = arith.constant 0 : index
      %c12_53 = arith.constant 12 : index
      %c1_54 = arith.constant 1 : index
      %c4_55 = arith.constant 4 : index
      cf.br ^bb6(%c0_52 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c12_53 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_56 = arith.constant 14 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c336_i32_58 = arith.constant 336 : i32
      %c3_i32_59 = arith.constant 3 : i32
      %c3_i32_60 = arith.constant 3 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c8_i32_62 = arith.constant 8 : i32
      %c0_i32_63 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_56, %c1_i32_57, %c336_i32_58, %c3_i32_59, %c3_i32_60, %c1_i32_61, %c8_i32_62, %c0_i32_63) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_64 = arith.constant 14 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c336_i32_66 = arith.constant 336 : i32
      %c3_i32_67 = arith.constant 3 : i32
      %c3_i32_68 = arith.constant 3 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c8_i32_70 = arith.constant 8 : i32
      %c0_i32_71 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_64, %c1_i32_65, %c336_i32_66, %c3_i32_67, %c3_i32_68, %c1_i32_69, %c8_i32_70, %c0_i32_71) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_72 = arith.constant 14 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c336_i32_74 = arith.constant 336 : i32
      %c3_i32_75 = arith.constant 3 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c1_i32_77 = arith.constant 1 : i32
      %c8_i32_78 = arith.constant 8 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_72, %c1_i32_73, %c336_i32_74, %c3_i32_75, %c3_i32_76, %c1_i32_77, %c8_i32_78, %c0_i32_79) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c1_i32_81 = arith.constant 1 : i32
      %c336_i32_82 = arith.constant 336 : i32
      %c3_i32_83 = arith.constant 3 : i32
      %c3_i32_84 = arith.constant 3 : i32
      %c1_i32_85 = arith.constant 1 : i32
      %c8_i32_86 = arith.constant 8 : i32
      %c0_i32_87 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_80, %c1_i32_81, %c336_i32_82, %c3_i32_83, %c3_i32_84, %c1_i32_85, %c8_i32_86, %c0_i32_87) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %7 = arith.addi %5, %c4_55 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_88 = arith.constant 14 : i32
      %c1_i32_89 = arith.constant 1 : i32
      %c336_i32_90 = arith.constant 336 : i32
      %c3_i32_91 = arith.constant 3 : i32
      %c3_i32_92 = arith.constant 3 : i32
      %c2_i32_93 = arith.constant 2 : i32
      %c8_i32_94 = arith.constant 8 : i32
      %c0_i32_95 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_88, %c1_i32_89, %c336_i32_90, %c3_i32_91, %c3_i32_92, %c2_i32_93, %c8_i32_94, %c0_i32_95) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c336_i32_98 = arith.constant 336 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c0_i32_101 = arith.constant 0 : i32
      %c8_i32_102 = arith.constant 8 : i32
      %c0_i32_103 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_96, %c1_i32_97, %c336_i32_98, %c3_i32_99, %c3_i32_100, %c0_i32_101, %c8_i32_102, %c0_i32_103) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_104 = arith.constant 0 : index
      %c12_105 = arith.constant 12 : index
      %c1_106 = arith.constant 1 : index
      %c4_107 = arith.constant 4 : index
      cf.br ^bb9(%c0_104 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c12_105 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_108 = arith.constant 14 : i32
      %c1_i32_109 = arith.constant 1 : i32
      %c336_i32_110 = arith.constant 336 : i32
      %c3_i32_111 = arith.constant 3 : i32
      %c3_i32_112 = arith.constant 3 : i32
      %c1_i32_113 = arith.constant 1 : i32
      %c8_i32_114 = arith.constant 8 : i32
      %c0_i32_115 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_108, %c1_i32_109, %c336_i32_110, %c3_i32_111, %c3_i32_112, %c1_i32_113, %c8_i32_114, %c0_i32_115) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_116 = arith.constant 14 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c336_i32_118 = arith.constant 336 : i32
      %c3_i32_119 = arith.constant 3 : i32
      %c3_i32_120 = arith.constant 3 : i32
      %c1_i32_121 = arith.constant 1 : i32
      %c8_i32_122 = arith.constant 8 : i32
      %c0_i32_123 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_116, %c1_i32_117, %c336_i32_118, %c3_i32_119, %c3_i32_120, %c1_i32_121, %c8_i32_122, %c0_i32_123) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_124 = arith.constant 14 : i32
      %c1_i32_125 = arith.constant 1 : i32
      %c336_i32_126 = arith.constant 336 : i32
      %c3_i32_127 = arith.constant 3 : i32
      %c3_i32_128 = arith.constant 3 : i32
      %c1_i32_129 = arith.constant 1 : i32
      %c8_i32_130 = arith.constant 8 : i32
      %c0_i32_131 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_124, %c1_i32_125, %c336_i32_126, %c3_i32_127, %c3_i32_128, %c1_i32_129, %c8_i32_130, %c0_i32_131) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_132 = arith.constant 14 : i32
      %c1_i32_133 = arith.constant 1 : i32
      %c336_i32_134 = arith.constant 336 : i32
      %c3_i32_135 = arith.constant 3 : i32
      %c3_i32_136 = arith.constant 3 : i32
      %c1_i32_137 = arith.constant 1 : i32
      %c8_i32_138 = arith.constant 8 : i32
      %c0_i32_139 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_132, %c1_i32_133, %c336_i32_134, %c3_i32_135, %c3_i32_136, %c1_i32_137, %c8_i32_138, %c0_i32_139) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %10 = arith.addi %8, %c4_107 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_140 = arith.constant 14 : i32
      %c1_i32_141 = arith.constant 1 : i32
      %c336_i32_142 = arith.constant 336 : i32
      %c3_i32_143 = arith.constant 3 : i32
      %c3_i32_144 = arith.constant 3 : i32
      %c2_i32_145 = arith.constant 2 : i32
      %c8_i32_146 = arith.constant 8 : i32
      %c0_i32_147 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_140, %c1_i32_141, %c336_i32_142, %c3_i32_143, %c3_i32_144, %c2_i32_145, %c8_i32_146, %c0_i32_147) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_148 = arith.constant 14 : i32
      %c1_i32_149 = arith.constant 1 : i32
      %c336_i32_150 = arith.constant 336 : i32
      %c3_i32_151 = arith.constant 3 : i32
      %c3_i32_152 = arith.constant 3 : i32
      %c0_i32_153 = arith.constant 0 : i32
      %c8_i32_154 = arith.constant 8 : i32
      %c0_i32_155 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_148, %c1_i32_149, %c336_i32_150, %c3_i32_151, %c3_i32_152, %c0_i32_153, %c8_i32_154, %c0_i32_155) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_156 = arith.constant 0 : index
      %c12_157 = arith.constant 12 : index
      %c1_158 = arith.constant 1 : index
      %c4_159 = arith.constant 4 : index
      cf.br ^bb12(%c0_156 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c12_157 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_160 = arith.constant 14 : i32
      %c1_i32_161 = arith.constant 1 : i32
      %c336_i32_162 = arith.constant 336 : i32
      %c3_i32_163 = arith.constant 3 : i32
      %c3_i32_164 = arith.constant 3 : i32
      %c1_i32_165 = arith.constant 1 : i32
      %c8_i32_166 = arith.constant 8 : i32
      %c0_i32_167 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_160, %c1_i32_161, %c336_i32_162, %c3_i32_163, %c3_i32_164, %c1_i32_165, %c8_i32_166, %c0_i32_167) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_168 = arith.constant 14 : i32
      %c1_i32_169 = arith.constant 1 : i32
      %c336_i32_170 = arith.constant 336 : i32
      %c3_i32_171 = arith.constant 3 : i32
      %c3_i32_172 = arith.constant 3 : i32
      %c1_i32_173 = arith.constant 1 : i32
      %c8_i32_174 = arith.constant 8 : i32
      %c0_i32_175 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_168, %c1_i32_169, %c336_i32_170, %c3_i32_171, %c3_i32_172, %c1_i32_173, %c8_i32_174, %c0_i32_175) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_176 = arith.constant 14 : i32
      %c1_i32_177 = arith.constant 1 : i32
      %c336_i32_178 = arith.constant 336 : i32
      %c3_i32_179 = arith.constant 3 : i32
      %c3_i32_180 = arith.constant 3 : i32
      %c1_i32_181 = arith.constant 1 : i32
      %c8_i32_182 = arith.constant 8 : i32
      %c0_i32_183 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_176, %c1_i32_177, %c336_i32_178, %c3_i32_179, %c3_i32_180, %c1_i32_181, %c8_i32_182, %c0_i32_183) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_184 = arith.constant 14 : i32
      %c1_i32_185 = arith.constant 1 : i32
      %c336_i32_186 = arith.constant 336 : i32
      %c3_i32_187 = arith.constant 3 : i32
      %c3_i32_188 = arith.constant 3 : i32
      %c1_i32_189 = arith.constant 1 : i32
      %c8_i32_190 = arith.constant 8 : i32
      %c0_i32_191 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_184, %c1_i32_185, %c336_i32_186, %c3_i32_187, %c3_i32_188, %c1_i32_189, %c8_i32_190, %c0_i32_191) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %13 = arith.addi %11, %c4_159 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_192 = arith.constant 14 : i32
      %c1_i32_193 = arith.constant 1 : i32
      %c336_i32_194 = arith.constant 336 : i32
      %c3_i32_195 = arith.constant 3 : i32
      %c3_i32_196 = arith.constant 3 : i32
      %c2_i32_197 = arith.constant 2 : i32
      %c8_i32_198 = arith.constant 8 : i32
      %c0_i32_199 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_192, %c1_i32_193, %c336_i32_194, %c3_i32_195, %c3_i32_196, %c2_i32_197, %c8_i32_198, %c0_i32_199) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_200 = arith.constant 14 : i32
      %c1_i32_201 = arith.constant 1 : i32
      %c336_i32_202 = arith.constant 336 : i32
      %c3_i32_203 = arith.constant 3 : i32
      %c3_i32_204 = arith.constant 3 : i32
      %c0_i32_205 = arith.constant 0 : i32
      %c8_i32_206 = arith.constant 8 : i32
      %c0_i32_207 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_200, %c1_i32_201, %c336_i32_202, %c3_i32_203, %c3_i32_204, %c0_i32_205, %c8_i32_206, %c0_i32_207) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_208 = arith.constant 0 : index
      %c12_209 = arith.constant 12 : index
      %c1_210 = arith.constant 1 : index
      %c4_211 = arith.constant 4 : index
      cf.br ^bb16(%c0_208 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb17
      %16 = arith.cmpi slt, %15, %c12_209 : index
      cf.cond_br %16, ^bb17, ^bb18
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_212 = arith.constant 14 : i32
      %c1_i32_213 = arith.constant 1 : i32
      %c336_i32_214 = arith.constant 336 : i32
      %c3_i32_215 = arith.constant 3 : i32
      %c3_i32_216 = arith.constant 3 : i32
      %c1_i32_217 = arith.constant 1 : i32
      %c8_i32_218 = arith.constant 8 : i32
      %c0_i32_219 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_212, %c1_i32_213, %c336_i32_214, %c3_i32_215, %c3_i32_216, %c1_i32_217, %c8_i32_218, %c0_i32_219) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_220 = arith.constant 14 : i32
      %c1_i32_221 = arith.constant 1 : i32
      %c336_i32_222 = arith.constant 336 : i32
      %c3_i32_223 = arith.constant 3 : i32
      %c3_i32_224 = arith.constant 3 : i32
      %c1_i32_225 = arith.constant 1 : i32
      %c8_i32_226 = arith.constant 8 : i32
      %c0_i32_227 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_220, %c1_i32_221, %c336_i32_222, %c3_i32_223, %c3_i32_224, %c1_i32_225, %c8_i32_226, %c0_i32_227) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_228 = arith.constant 14 : i32
      %c1_i32_229 = arith.constant 1 : i32
      %c336_i32_230 = arith.constant 336 : i32
      %c3_i32_231 = arith.constant 3 : i32
      %c3_i32_232 = arith.constant 3 : i32
      %c1_i32_233 = arith.constant 1 : i32
      %c8_i32_234 = arith.constant 8 : i32
      %c0_i32_235 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_228, %c1_i32_229, %c336_i32_230, %c3_i32_231, %c3_i32_232, %c1_i32_233, %c8_i32_234, %c0_i32_235) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_236 = arith.constant 14 : i32
      %c1_i32_237 = arith.constant 1 : i32
      %c336_i32_238 = arith.constant 336 : i32
      %c3_i32_239 = arith.constant 3 : i32
      %c3_i32_240 = arith.constant 3 : i32
      %c1_i32_241 = arith.constant 1 : i32
      %c8_i32_242 = arith.constant 8 : i32
      %c0_i32_243 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_236, %c1_i32_237, %c336_i32_238, %c3_i32_239, %c3_i32_240, %c1_i32_241, %c8_i32_242, %c0_i32_243) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %17 = arith.addi %15, %c4_211 : index
      cf.br ^bb16(%17 : index)
    ^bb18:  // pred: ^bb16
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_244 = arith.constant 14 : i32
      %c1_i32_245 = arith.constant 1 : i32
      %c336_i32_246 = arith.constant 336 : i32
      %c3_i32_247 = arith.constant 3 : i32
      %c3_i32_248 = arith.constant 3 : i32
      %c2_i32_249 = arith.constant 2 : i32
      %c8_i32_250 = arith.constant 8 : i32
      %c0_i32_251 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_244, %c1_i32_245, %c336_i32_246, %c3_i32_247, %c3_i32_248, %c2_i32_249, %c8_i32_250, %c0_i32_251) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_252 = arith.constant 14 : i32
      %c1_i32_253 = arith.constant 1 : i32
      %c336_i32_254 = arith.constant 336 : i32
      %c3_i32_255 = arith.constant 3 : i32
      %c3_i32_256 = arith.constant 3 : i32
      %c0_i32_257 = arith.constant 0 : i32
      %c8_i32_258 = arith.constant 8 : i32
      %c0_i32_259 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_252, %c1_i32_253, %c336_i32_254, %c3_i32_255, %c3_i32_256, %c0_i32_257, %c8_i32_258, %c0_i32_259) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_260 = arith.constant 0 : index
      %c12_261 = arith.constant 12 : index
      %c1_262 = arith.constant 1 : index
      %c4_263 = arith.constant 4 : index
      cf.br ^bb19(%c0_260 : index)
    ^bb19(%18: index):  // 2 preds: ^bb18, ^bb20
      %19 = arith.cmpi slt, %18, %c12_261 : index
      cf.cond_br %19, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_264 = arith.constant 14 : i32
      %c1_i32_265 = arith.constant 1 : i32
      %c336_i32_266 = arith.constant 336 : i32
      %c3_i32_267 = arith.constant 3 : i32
      %c3_i32_268 = arith.constant 3 : i32
      %c1_i32_269 = arith.constant 1 : i32
      %c8_i32_270 = arith.constant 8 : i32
      %c0_i32_271 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_264, %c1_i32_265, %c336_i32_266, %c3_i32_267, %c3_i32_268, %c1_i32_269, %c8_i32_270, %c0_i32_271) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_272 = arith.constant 14 : i32
      %c1_i32_273 = arith.constant 1 : i32
      %c336_i32_274 = arith.constant 336 : i32
      %c3_i32_275 = arith.constant 3 : i32
      %c3_i32_276 = arith.constant 3 : i32
      %c1_i32_277 = arith.constant 1 : i32
      %c8_i32_278 = arith.constant 8 : i32
      %c0_i32_279 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_272, %c1_i32_273, %c336_i32_274, %c3_i32_275, %c3_i32_276, %c1_i32_277, %c8_i32_278, %c0_i32_279) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_280 = arith.constant 14 : i32
      %c1_i32_281 = arith.constant 1 : i32
      %c336_i32_282 = arith.constant 336 : i32
      %c3_i32_283 = arith.constant 3 : i32
      %c3_i32_284 = arith.constant 3 : i32
      %c1_i32_285 = arith.constant 1 : i32
      %c8_i32_286 = arith.constant 8 : i32
      %c0_i32_287 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_280, %c1_i32_281, %c336_i32_282, %c3_i32_283, %c3_i32_284, %c1_i32_285, %c8_i32_286, %c0_i32_287) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_288 = arith.constant 14 : i32
      %c1_i32_289 = arith.constant 1 : i32
      %c336_i32_290 = arith.constant 336 : i32
      %c3_i32_291 = arith.constant 3 : i32
      %c3_i32_292 = arith.constant 3 : i32
      %c1_i32_293 = arith.constant 1 : i32
      %c8_i32_294 = arith.constant 8 : i32
      %c0_i32_295 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_288, %c1_i32_289, %c336_i32_290, %c3_i32_291, %c3_i32_292, %c1_i32_293, %c8_i32_294, %c0_i32_295) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %20 = arith.addi %18, %c4_263 : index
      cf.br ^bb19(%20 : index)
    ^bb21:  // pred: ^bb19
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_296 = arith.constant 14 : i32
      %c1_i32_297 = arith.constant 1 : i32
      %c336_i32_298 = arith.constant 336 : i32
      %c3_i32_299 = arith.constant 3 : i32
      %c3_i32_300 = arith.constant 3 : i32
      %c2_i32_301 = arith.constant 2 : i32
      %c8_i32_302 = arith.constant 8 : i32
      %c0_i32_303 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_296, %c1_i32_297, %c336_i32_298, %c3_i32_299, %c3_i32_300, %c2_i32_301, %c8_i32_302, %c0_i32_303) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_304 = arith.constant 14 : i32
      %c1_i32_305 = arith.constant 1 : i32
      %c336_i32_306 = arith.constant 336 : i32
      %c3_i32_307 = arith.constant 3 : i32
      %c3_i32_308 = arith.constant 3 : i32
      %c0_i32_309 = arith.constant 0 : i32
      %c8_i32_310 = arith.constant 8 : i32
      %c0_i32_311 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_304, %c1_i32_305, %c336_i32_306, %c3_i32_307, %c3_i32_308, %c0_i32_309, %c8_i32_310, %c0_i32_311) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_312 = arith.constant 0 : index
      %c12_313 = arith.constant 12 : index
      %c1_314 = arith.constant 1 : index
      %c4_315 = arith.constant 4 : index
      cf.br ^bb22(%c0_312 : index)
    ^bb22(%21: index):  // 2 preds: ^bb21, ^bb23
      %22 = arith.cmpi slt, %21, %c12_313 : index
      cf.cond_br %22, ^bb23, ^bb24
    ^bb23:  // pred: ^bb22
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_316 = arith.constant 14 : i32
      %c1_i32_317 = arith.constant 1 : i32
      %c336_i32_318 = arith.constant 336 : i32
      %c3_i32_319 = arith.constant 3 : i32
      %c3_i32_320 = arith.constant 3 : i32
      %c1_i32_321 = arith.constant 1 : i32
      %c8_i32_322 = arith.constant 8 : i32
      %c0_i32_323 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_316, %c1_i32_317, %c336_i32_318, %c3_i32_319, %c3_i32_320, %c1_i32_321, %c8_i32_322, %c0_i32_323) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_324 = arith.constant 14 : i32
      %c1_i32_325 = arith.constant 1 : i32
      %c336_i32_326 = arith.constant 336 : i32
      %c3_i32_327 = arith.constant 3 : i32
      %c3_i32_328 = arith.constant 3 : i32
      %c1_i32_329 = arith.constant 1 : i32
      %c8_i32_330 = arith.constant 8 : i32
      %c0_i32_331 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_324, %c1_i32_325, %c336_i32_326, %c3_i32_327, %c3_i32_328, %c1_i32_329, %c8_i32_330, %c0_i32_331) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_332 = arith.constant 14 : i32
      %c1_i32_333 = arith.constant 1 : i32
      %c336_i32_334 = arith.constant 336 : i32
      %c3_i32_335 = arith.constant 3 : i32
      %c3_i32_336 = arith.constant 3 : i32
      %c1_i32_337 = arith.constant 1 : i32
      %c8_i32_338 = arith.constant 8 : i32
      %c0_i32_339 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_2, %B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_332, %c1_i32_333, %c336_i32_334, %c3_i32_335, %c3_i32_336, %c1_i32_337, %c8_i32_338, %c0_i32_339) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_340 = arith.constant 14 : i32
      %c1_i32_341 = arith.constant 1 : i32
      %c336_i32_342 = arith.constant 336 : i32
      %c3_i32_343 = arith.constant 3 : i32
      %c3_i32_344 = arith.constant 3 : i32
      %c1_i32_345 = arith.constant 1 : i32
      %c8_i32_346 = arith.constant 8 : i32
      %c0_i32_347 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_3, %B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_0, %c14_i32_340, %c1_i32_341, %c336_i32_342, %c3_i32_343, %c3_i32_344, %c1_i32_345, %c8_i32_346, %c0_i32_347) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %23 = arith.addi %21, %c4_315 : index
      cf.br ^bb22(%23 : index)
    ^bb24:  // pred: ^bb22
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_348 = arith.constant 14 : i32
      %c1_i32_349 = arith.constant 1 : i32
      %c336_i32_350 = arith.constant 336 : i32
      %c3_i32_351 = arith.constant 3 : i32
      %c3_i32_352 = arith.constant 3 : i32
      %c2_i32_353 = arith.constant 2 : i32
      %c8_i32_354 = arith.constant 8 : i32
      %c0_i32_355 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%B_OF_b11_act_layer1_layer2_cons_buff_0, %B_OF_b11_act_layer1_layer2_cons_buff_1, %B_OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %B_OF_b11_act_layer2_layer3_buff_1, %c14_i32_348, %c1_i32_349, %c336_i32_350, %c3_i32_351, %c3_i32_352, %c2_i32_353, %c8_i32_354, %c0_i32_355) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn11_conv2dk3_dw.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%weightsInBN11_layer3_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_1 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_skip_cons_cons_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c336_i32 = arith.constant 336 : i32
      %c112_i32 = arith.constant 112 : i32
      %c12_i32 = arith.constant 12 : i32
      %c1_i32 = arith.constant 1 : i32
      func.call @bn11_conv2dk1_skip_ui8_i8_i8(%B_OF_b11_act_layer2_layer3_buff_0, %weightsInBN11_layer3_cons_buff_0, %act_out_buff_0, %OF_b11_skip_cons_buff_0, %c14_i32, %c336_i32, %c112_i32, %c12_i32, %c1_i32) : (memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_skip_cons_prod_lock, Release, 1)
      aie.use_lock(%B_OF_b11_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_skip_cons_cons_lock, AcquireGreaterEqual, 1)
      %c14_i32_3 = arith.constant 14 : i32
      %c336_i32_4 = arith.constant 336 : i32
      %c112_i32_5 = arith.constant 112 : i32
      %c12_i32_6 = arith.constant 12 : i32
      %c1_i32_7 = arith.constant 1 : i32
      func.call @bn11_conv2dk1_skip_ui8_i8_i8(%B_OF_b11_act_layer2_layer3_buff_1, %weightsInBN11_layer3_cons_buff_0, %act_out_buff_1, %OF_b11_skip_cons_buff_1, %c14_i32_3, %c336_i32_4, %c112_i32_5, %c12_i32_6, %c1_i32_7) : (memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%B_OF_b11_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_skip_cons_prod_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%weightsInBN11_layer3_cons_prod_lock, Release, 1)
      %5 = arith.addi %0, %c1_0 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "bn11_conv2dk1_skip.o"}
    aie.shim_dma_allocation @wts_b10_L3L2(MM2S, 0, 2)
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
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_b10_L3L2_cons_prod_lock, AcquireGreaterEqual, 3)
      aie.dma_bd(%wts_b10_L3L2_cons_buff_0 : memref<96480xi8>, 0, 96480) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%wts_b10_L3L2_cons_cons_lock, Release, 3)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%wts_b10_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b10_L3L2_cons_buff_0 : memref<96480xi8>, 0, 38400) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%wts_b10_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%wts_b10_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b10_L3L2_cons_buff_0 : memref<96480xi8>, 38400, 4320) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%wts_b10_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 2, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%wts_b10_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b10_L3L2_cons_buff_0 : memref<96480xi8>, 42720, 53760) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%wts_b10_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 1, ^bb9, ^bb15)
    ^bb9:  // 2 preds: ^bb8, ^bb14
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_2 : memref<14x1x112xi8>, 0, 1568) {bd_id = 27 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_3 : memref<14x1x112xi8>, 0, 1568) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_4 : memref<14x1x112xi8>, 0, 1568) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_5 : memref<14x1x112xi8>, 0, 1568) {bd_id = 30 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb9
    ^bb15:  // pred: ^bb8
      %5 = aie.dma_start(MM2S, 3, ^bb16, ^bb22)
    ^bb16:  // 2 preds: ^bb15, ^bb21
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_2 : memref<14x1x112xi8>, 0, 1568) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_3 : memref<14x1x112xi8>, 0, 1568) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_4 : memref<14x1x112xi8>, 0, 1568) {bd_id = 35 : i32, next_bd_id = 36 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_1_cons_buff_5 : memref<14x1x112xi8>, 0, 1568) {bd_id = 36 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb16
    ^bb22:  // pred: ^bb15
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN10_layer1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN10_layer1_cons_buff_0 : memref<38400xi8>, 0, 38400) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN10_layer1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_act_layer1_layer2_buff_0 : memref<14x1x480xui8>, 0, 6720) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_OF_b10_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_act_layer1_layer2_buff_1 : memref<14x1x480xui8>, 0, 6720) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_OF_b10_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN10_layer2_cons_buff_0 : memref<4320xi8>, 0, 4320) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_act_layer1_layer2_cons_buff_0 : memref<14x1x480xui8>, 0, 6720) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_act_layer1_layer2_cons_buff_1 : memref<14x1x480xui8>, 0, 6720) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_act_layer1_layer2_cons_buff_2 : memref<14x1x480xui8>, 0, 6720) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_act_layer1_layer2_cons_buff_3 : memref<14x1x480xui8>, 0, 6720) {bd_id = 4 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @wts_b11_L3L2(MM2S, 0, 3)
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN10_layer3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN10_layer3_cons_buff_0 : memref<53760xi8>, 0, 53760) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN10_layer3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_3_1 = aie.memtile_dma(%tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_b11_L3L2_cons_prod_lock, AcquireGreaterEqual, 3)
      aie.dma_bd(%wts_b11_L3L2_cons_buff_0 : memref<78288xi8>, 0, 78288) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%wts_b11_L3L2_cons_cons_lock, Release, 3)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%wts_b11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b11_L3L2_cons_buff_0 : memref<78288xi8>, 0, 37632) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%wts_b11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%wts_b11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b11_L3L2_cons_buff_0 : memref<78288xi8>, 37632, 3024) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%wts_b11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 2, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%wts_b11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b11_L3L2_cons_buff_0 : memref<78288xi8>, 40656, 37632) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%wts_b11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN11_layer1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN11_layer1_cons_buff_0 : memref<37632xi8>, 0, 37632) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN11_layer1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b11_act_layer1_layer2_buff_0 : memref<14x1x336xui8>, 0, 4704) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_OF_b11_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b11_act_layer1_layer2_buff_1 : memref<14x1x336xui8>, 0, 4704) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_OF_b11_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN11_layer2_cons_buff_0 : memref<3024xi8>, 0, 3024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b11_act_layer1_layer2_cons_buff_0 : memref<14x1x336xui8>, 0, 4704) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b11_act_layer1_layer2_cons_buff_1 : memref<14x1x336xui8>, 0, 4704) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b11_act_layer1_layer2_cons_buff_2 : memref<14x1x336xui8>, 0, 4704) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_OF_b11_act_layer1_layer2_cons_buff_3 : memref<14x1x336xui8>, 0, 4704) {bd_id = 4 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @act_in(MM2S, 0, 0)
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN11_layer3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN11_layer3_cons_buff_0 : memref<37632xi8>, 0, 37632) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN11_layer3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%act_out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_out_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%act_out_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_out_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_out_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%OF_b11_skip_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_skip_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_b11_skip_cons_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b11_skip_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_skip_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_b11_skip_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @wts_OF_01_L3L2(MM2S, 1, 0)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%act_in_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_in_cons_buff_0 : memref<112x1x16xui8>, 0, 1792) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_in_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_in_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_in_cons_buff_1 : memref<112x1x16xui8>, 0, 1792) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%act_in_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%act_in_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_in_cons_buff_2 : memref<112x1x16xui8>, 0, 1792) {bd_id = 2 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%act_in_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%bn0_1_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn0_1_wts_OF_L2L1_cons_buff_0 : memref<3536xi8>, 0, 3536) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%bn0_1_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_OF_01_L3L2_cons_prod_lock, AcquireGreaterEqual, 5)
      aie.dma_bd(%wts_OF_01_L3L2_cons_buff_0 : memref<34256xi8>, 0, 34256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%wts_OF_01_L3L2_cons_cons_lock, Release, 5)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%wts_OF_01_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_01_L3L2_cons_buff_0 : memref<34256xi8>, 0, 3536) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%wts_OF_01_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%wts_OF_01_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_01_L3L2_cons_buff_0 : memref<34256xi8>, 3536, 4104) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%wts_OF_01_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 2, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%wts_OF_01_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_01_L3L2_cons_buff_0 : memref<34256xi8>, 7640, 5256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%wts_OF_01_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 3, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%wts_OF_01_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_01_L3L2_cons_buff_0 : memref<34256xi8>, 12896, 10680) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%wts_OF_01_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(MM2S, 4, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%wts_OF_01_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_01_L3L2_cons_buff_0 : memref<34256xi8>, 23576, 10680) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%wts_OF_01_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn2_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn2_wts_OF_L2L1_cons_buff_0 : memref<4104xi8>, 0, 4104) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn2_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn3_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn3_wts_OF_L2L1_cons_buff_0 : memref<5256xi8>, 0, 5256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn3_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn4_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn4_wts_OF_L2L1_cons_buff_0 : memref<10680xi8>, 0, 10680) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn4_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.shim_dma_allocation @wts_OF_11_L3L2(MM2S, 0, 1)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn5_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn5_wts_OF_L2L1_cons_buff_0 : memref<10680xi8>, 0, 10680) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn5_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%act_bn5_bn6_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn5_bn6_buff_0 : memref<28x1x40xi8>, 0, 1120) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%act_bn5_bn6_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_bn5_bn6_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn5_bn6_buff_1 : memref<28x1x40xi8>, 0, 1120) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_bn5_bn6_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_OF_11_L3L2_cons_prod_lock, AcquireGreaterEqual, 4)
      aie.dma_bd(%wts_OF_11_L3L2_cons_buff_0 : memref<126952xi8>, 0, 126952) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%wts_OF_11_L3L2_cons_cons_lock, Release, 4)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%wts_OF_11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_11_L3L2_cons_buff_0 : memref<126952xi8>, 0, 30960) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%wts_OF_11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%wts_OF_11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_11_L3L2_cons_buff_0 : memref<126952xi8>, 30960, 33800) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%wts_OF_11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 2, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%wts_OF_11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_11_L3L2_cons_buff_0 : memref<126952xi8>, 64760, 31096) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%wts_OF_11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 3, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%wts_OF_11_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_OF_11_L3L2_cons_buff_0 : memref<126952xi8>, 95856, 31096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%wts_OF_11_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn6_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn6_wts_OF_L2L1_cons_buff_0 : memref<30960xi8>, 0, 30960) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn6_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb5
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn5_bn6_cons_buff_0 : memref<28x1x40xi8>, 0, 1120) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn5_bn6_cons_buff_1 : memref<28x1x40xi8>, 0, 1120) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%act_bn5_bn6_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn5_bn6_cons_buff_2 : memref<28x1x40xi8>, 0, 1120) {bd_id = 3 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_bn5_bn6_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn7_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn7_wts_OF_L2L1_cons_buff_0 : memref<33800xi8>, 0, 33800) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn7_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%act_bn7_bn8_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn7_bn8_buff_0 : memref<14x1x80xi8>, 0, 1120) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%act_bn7_bn8_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_bn7_bn8_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn7_bn8_buff_1 : memref<14x1x80xi8>, 0, 1120) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_bn7_bn8_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn8_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn8_wts_OF_L2L1_cons_buff_0 : memref<31096xi8>, 0, 31096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn8_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb5
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn7_bn8_cons_buff_0 : memref<14x1x80xi8>, 0, 1120) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn7_bn8_cons_buff_1 : memref<14x1x80xi8>, 0, 1120) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%act_bn7_bn8_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_bn7_bn8_cons_buff_2 : memref<14x1x80xi8>, 0, 1120) {bd_id = 3 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_bn7_bn8_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb6:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @act_out(S2MM, 0, 3)
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bn9_wts_OF_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%bn9_wts_OF_L2L1_cons_buff_0 : memref<31096xi8>, 0, 31096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bn9_wts_OF_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_2_5 : East, %switchbox_3_5 : West)
    aie.wire(%tile_3_5 : Core, %switchbox_3_5 : Core)
    aie.wire(%tile_3_5 : DMA, %switchbox_3_5 : DMA)
    aie.wire(%switchbox_3_4 : North, %switchbox_3_5 : South)
  }
}

