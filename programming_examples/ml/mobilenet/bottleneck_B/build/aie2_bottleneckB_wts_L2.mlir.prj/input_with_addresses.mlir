module {
  aie.device(npu1_3col) {
    memref.global "public" @OF_b12_act_layer2_layer3 : memref<7x1x336xui8>
    memref.global "public" @OF_b12_act_layer1_layer2_cons : memref<14x1x336xui8>
    memref.global "public" @OF_b12_act_layer1_layer2 : memref<14x1x336xui8>
    memref.global "public" @OF_b11_layer3_bn_12_layer1 : memref<14x1x112xi8>
    memref.global "public" @OF_b11_act_layer2_layer3 : memref<14x1x336xui8>
    memref.global "public" @OF_b11_act_layer1_layer2_cons : memref<14x1x336xui8>
    memref.global "public" @OF_b11_act_layer1_layer2 : memref<14x1x336xui8>
    memref.global "public" @OF_b11_skip_cons : memref<14x1x112xi8>
    memref.global "public" @OF_b11_skip : memref<14x1x112xi8>
    memref.global "public" @OF_b10_layer3_bn_11_layer1_0_cons : memref<14x1x112xi8>
    memref.global "public" @OF_b10_layer3_bn_11_layer1_1_cons : memref<14x1x112xi8>
    memref.global "public" @OF_b10_layer3_bn_11_layer1 : memref<14x1x112xi8>
    memref.global "public" @OF_b10_act_layer2_layer3 : memref<14x1x480xui8>
    memref.global "public" @OF_b10_act_layer1_layer2_cons : memref<14x1x480xui8>
    memref.global "public" @OF_b10_act_layer1_layer2 : memref<14x1x480xui8>
    memref.global "public" @weightsInBN12_layer3_cons : memref<26880xi8>
    memref.global "public" @weightsInBN12_layer3 : memref<26880xi8>
    memref.global "public" @weightsInBN12_layer2_cons : memref<3024xi8>
    memref.global "public" @weightsInBN12_layer2 : memref<3024xi8>
    memref.global "public" @weightsInBN12_layer1_cons : memref<37632xi8>
    memref.global "public" @weightsInBN12_layer1 : memref<37632xi8>
    memref.global "public" @wts_b12_L3L2_cons : memref<67536xi8>
    memref.global "public" @wts_b12_L3L2 : memref<67536xi8>
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
    memref.global "public" @act_out_cons : memref<7x1x80xi8>
    memref.global "public" @act_out : memref<7x1x80xi8>
    memref.global "public" @act_in_cons : memref<14x1x80xi8>
    memref.global "public" @act_in : memref<14x1x80xi8>
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
    %OF_b12_act_layer2_layer3_buff_0 = aie.buffer(%tile_1_2) {address = 22864 : i32, sym_name = "OF_b12_act_layer2_layer3_buff_0"} : memref<7x1x336xui8> 
    %OF_b12_act_layer2_layer3_buff_1 = aie.buffer(%tile_1_2) {address = 25216 : i32, sym_name = "OF_b12_act_layer2_layer3_buff_1"} : memref<7x1x336xui8> 
    %OF_b12_act_layer2_layer3_prod_lock = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "OF_b12_act_layer2_layer3_prod_lock"}
    %OF_b12_act_layer2_layer3_cons_lock = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "OF_b12_act_layer2_layer3_cons_lock"}
    %OF_b12_act_layer1_layer2_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_buff_0"} : memref<14x1x336xui8> 
    %OF_b12_act_layer1_layer2_cons_buff_1 = aie.buffer(%tile_1_2) {address = 5728 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_buff_1"} : memref<14x1x336xui8> 
    %OF_b12_act_layer1_layer2_cons_buff_2 = aie.buffer(%tile_1_2) {address = 10432 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_buff_2"} : memref<14x1x336xui8> 
    %OF_b12_act_layer1_layer2_cons_buff_3 = aie.buffer(%tile_1_2) {address = 15136 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_buff_3"} : memref<14x1x336xui8> 
    %OF_b12_act_layer1_layer2_cons_prod_lock = aie.lock(%tile_1_2, 2) {init = 4 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_prod_lock"}
    %OF_b12_act_layer1_layer2_cons_cons_lock = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_cons_lock"}
    %OF_b12_act_layer1_layer2_buff_0 = aie.buffer(%tile_1_3) {address = 38656 : i32, sym_name = "OF_b12_act_layer1_layer2_buff_0"} : memref<14x1x336xui8> 
    %OF_b12_act_layer1_layer2_buff_1 = aie.buffer(%tile_1_3) {address = 43360 : i32, sym_name = "OF_b12_act_layer1_layer2_buff_1"} : memref<14x1x336xui8> 
    %OF_b12_act_layer1_layer2_prod_lock = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "OF_b12_act_layer1_layer2_prod_lock"}
    %OF_b12_act_layer1_layer2_cons_lock = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "OF_b12_act_layer1_layer2_cons_lock"}
    %OF_b11_layer3_bn_12_layer1_buff_0 = aie.buffer(%tile_1_4) {address = 38656 : i32, sym_name = "OF_b11_layer3_bn_12_layer1_buff_0"} : memref<14x1x112xi8> 
    %OF_b11_layer3_bn_12_layer1_buff_1 = aie.buffer(%tile_1_4) {address = 40224 : i32, sym_name = "OF_b11_layer3_bn_12_layer1_buff_1"} : memref<14x1x112xi8> 
    %OF_b11_layer3_bn_12_layer1_prod_lock = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "OF_b11_layer3_bn_12_layer1_prod_lock"}
    %OF_b11_layer3_bn_12_layer1_cons_lock = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "OF_b11_layer3_bn_12_layer1_cons_lock"}
    %OF_b11_act_layer2_layer3_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, sym_name = "OF_b11_act_layer2_layer3_buff_0"} : memref<14x1x336xui8> 
    %OF_b11_act_layer2_layer3_buff_1 = aie.buffer(%tile_1_5) {address = 5728 : i32, sym_name = "OF_b11_act_layer2_layer3_buff_1"} : memref<14x1x336xui8> 
    %OF_b11_act_layer2_layer3_prod_lock = aie.lock(%tile_1_5, 4) {init = 2 : i32, sym_name = "OF_b11_act_layer2_layer3_prod_lock"}
    %OF_b11_act_layer2_layer3_cons_lock = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "OF_b11_act_layer2_layer3_cons_lock"}
    %OF_b11_act_layer1_layer2_cons_buff_0 = aie.buffer(%tile_1_5) {address = 10432 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_buff_0"} : memref<14x1x336xui8> 
    %OF_b11_act_layer1_layer2_cons_buff_1 = aie.buffer(%tile_1_5) {address = 15136 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_buff_1"} : memref<14x1x336xui8> 
    %OF_b11_act_layer1_layer2_cons_buff_2 = aie.buffer(%tile_1_5) {address = 19840 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_buff_2"} : memref<14x1x336xui8> 
    %OF_b11_act_layer1_layer2_cons_buff_3 = aie.buffer(%tile_1_5) {address = 24544 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_buff_3"} : memref<14x1x336xui8> 
    %OF_b11_act_layer1_layer2_cons_prod_lock = aie.lock(%tile_1_5, 2) {init = 4 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_prod_lock"}
    %OF_b11_act_layer1_layer2_cons_cons_lock = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_cons_lock"}
    %OF_b11_act_layer1_layer2_buff_0 = aie.buffer(%tile_0_5) {address = 38656 : i32, sym_name = "OF_b11_act_layer1_layer2_buff_0"} : memref<14x1x336xui8> 
    %OF_b11_act_layer1_layer2_buff_1 = aie.buffer(%tile_0_5) {address = 43360 : i32, sym_name = "OF_b11_act_layer1_layer2_buff_1"} : memref<14x1x336xui8> 
    %OF_b11_act_layer1_layer2_prod_lock = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "OF_b11_act_layer1_layer2_prod_lock"}
    %OF_b11_act_layer1_layer2_cons_lock = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "OF_b11_act_layer1_layer2_cons_lock"}
    %OF_b11_skip_cons_buff_0 = aie.buffer(%tile_1_4) {address = 41792 : i32, sym_name = "OF_b11_skip_cons_buff_0"} : memref<14x1x112xi8> 
    %OF_b11_skip_cons_buff_1 = aie.buffer(%tile_1_4) {address = 43360 : i32, sym_name = "OF_b11_skip_cons_buff_1"} : memref<14x1x112xi8> 
    %OF_b11_skip_cons_prod_lock = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "OF_b11_skip_cons_prod_lock"}
    %OF_b11_skip_cons_cons_lock = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "OF_b11_skip_cons_cons_lock"}
    %OF_b10_layer3_bn_11_layer1_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 48064 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_0_cons_buff_0"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49632 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_0_cons_buff_1"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_0_cons_prod_lock = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_0_cons_prod_lock"}
    %OF_b10_layer3_bn_11_layer1_0_cons_cons_lock = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_0_cons_cons_lock"}
    %OF_b10_layer3_bn_11_layer1_1_cons_buff_0 = aie.buffer(%tile_0_1) {address = 96480 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_buff_0"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_1_cons_buff_1 = aie.buffer(%tile_0_1) {address = 98048 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_buff_1"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_1_cons_buff_2 = aie.buffer(%tile_0_1) {address = 99616 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_buff_2"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_1_cons_buff_3 = aie.buffer(%tile_0_1) {address = 101184 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_buff_3"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_1_cons_buff_4 = aie.buffer(%tile_0_1) {address = 102752 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_buff_4"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_1_cons_buff_5 = aie.buffer(%tile_0_1) {address = 104320 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_buff_5"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_1_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 6 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_prod_lock"}
    %OF_b10_layer3_bn_11_layer1_1_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_1_cons_cons_lock"}
    %OF_b10_layer3_bn_11_layer1_buff_0 = aie.buffer(%tile_0_4) {address = 54784 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_buff_0"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_buff_1 = aie.buffer(%tile_0_4) {address = 56352 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_buff_1"} : memref<14x1x112xi8> 
    %OF_b10_layer3_bn_11_layer1_prod_lock = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_prod_lock"}
    %OF_b10_layer3_bn_11_layer1_cons_lock = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "OF_b10_layer3_bn_11_layer1_cons_lock"}
    %OF_b10_act_layer2_layer3_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "OF_b10_act_layer2_layer3_buff_0"} : memref<14x1x480xui8> 
    %OF_b10_act_layer2_layer3_buff_1 = aie.buffer(%tile_0_3) {address = 7744 : i32, sym_name = "OF_b10_act_layer2_layer3_buff_1"} : memref<14x1x480xui8> 
    %OF_b10_act_layer2_layer3_prod_lock = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "OF_b10_act_layer2_layer3_prod_lock"}
    %OF_b10_act_layer2_layer3_cons_lock = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "OF_b10_act_layer2_layer3_cons_lock"}
    %OF_b10_act_layer1_layer2_cons_buff_0 = aie.buffer(%tile_0_3) {address = 14464 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_buff_0"} : memref<14x1x480xui8> 
    %OF_b10_act_layer1_layer2_cons_buff_1 = aie.buffer(%tile_0_3) {address = 21184 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_buff_1"} : memref<14x1x480xui8> 
    %OF_b10_act_layer1_layer2_cons_buff_2 = aie.buffer(%tile_0_3) {address = 27904 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_buff_2"} : memref<14x1x480xui8> 
    %OF_b10_act_layer1_layer2_cons_buff_3 = aie.buffer(%tile_0_3) {address = 34624 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_buff_3"} : memref<14x1x480xui8> 
    %OF_b10_act_layer1_layer2_cons_prod_lock = aie.lock(%tile_0_3, 2) {init = 4 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_prod_lock"}
    %OF_b10_act_layer1_layer2_cons_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_cons_lock"}
    %OF_b10_act_layer1_layer2_buff_0 = aie.buffer(%tile_0_2) {address = 39424 : i32, sym_name = "OF_b10_act_layer1_layer2_buff_0"} : memref<14x1x480xui8> 
    %OF_b10_act_layer1_layer2_buff_1 = aie.buffer(%tile_0_2) {address = 46144 : i32, sym_name = "OF_b10_act_layer1_layer2_buff_1"} : memref<14x1x480xui8> 
    %OF_b10_act_layer1_layer2_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "OF_b10_act_layer1_layer2_prod_lock"}
    %OF_b10_act_layer1_layer2_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "OF_b10_act_layer1_layer2_cons_lock"}
    %weightsInBN12_layer3_cons_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, sym_name = "weightsInBN12_layer3_cons_buff_0"} : memref<26880xi8> 
    %weightsInBN12_layer3_cons_prod_lock = aie.lock(%tile_2_2, 2) {init = 1 : i32, sym_name = "weightsInBN12_layer3_cons_prod_lock"}
    %weightsInBN12_layer3_cons_cons_lock = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "weightsInBN12_layer3_cons_cons_lock"}
    %weightsInBN12_layer2_cons_buff_0 = aie.buffer(%tile_1_2) {address = 19840 : i32, sym_name = "weightsInBN12_layer2_cons_buff_0"} : memref<3024xi8> 
    %weightsInBN12_layer2_cons_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "weightsInBN12_layer2_cons_prod_lock"}
    %weightsInBN12_layer2_cons_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "weightsInBN12_layer2_cons_cons_lock"}
    %weightsInBN12_layer1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, sym_name = "weightsInBN12_layer1_cons_buff_0"} : memref<37632xi8> 
    %weightsInBN12_layer1_cons_prod_lock = aie.lock(%tile_1_3, 0) {init = 1 : i32, sym_name = "weightsInBN12_layer1_cons_prod_lock"}
    %weightsInBN12_layer1_cons_cons_lock = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "weightsInBN12_layer1_cons_cons_lock"}
    %wts_b12_L3L2_cons_buff_0 = aie.buffer(%tile_2_1) {address = 0 : i32, sym_name = "wts_b12_L3L2_cons_buff_0"} : memref<67536xi8> 
    %wts_b12_L3L2_cons_prod_lock = aie.lock(%tile_2_1, 0) {init = 3 : i32, sym_name = "wts_b12_L3L2_cons_prod_lock"}
    %wts_b12_L3L2_cons_cons_lock = aie.lock(%tile_2_1, 1) {init = 0 : i32, sym_name = "wts_b12_L3L2_cons_cons_lock"}
    %wts_b12_L3L2_prod_lock = aie.lock(%tile_1_0, 4) {init = 1 : i32, sym_name = "wts_b12_L3L2_prod_lock"}
    %wts_b12_L3L2_cons_lock = aie.lock(%tile_1_0, 5) {init = 0 : i32, sym_name = "wts_b12_L3L2_cons_lock"}
    %weightsInBN11_layer3_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, sym_name = "weightsInBN11_layer3_cons_buff_0"} : memref<37632xi8> 
    %weightsInBN11_layer3_cons_prod_lock = aie.lock(%tile_1_4, 0) {init = 1 : i32, sym_name = "weightsInBN11_layer3_cons_prod_lock"}
    %weightsInBN11_layer3_cons_cons_lock = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "weightsInBN11_layer3_cons_cons_lock"}
    %weightsInBN11_layer2_cons_buff_0 = aie.buffer(%tile_1_5) {address = 29248 : i32, sym_name = "weightsInBN11_layer2_cons_buff_0"} : memref<3024xi8> 
    %weightsInBN11_layer2_cons_prod_lock = aie.lock(%tile_1_5, 0) {init = 1 : i32, sym_name = "weightsInBN11_layer2_cons_prod_lock"}
    %weightsInBN11_layer2_cons_cons_lock = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "weightsInBN11_layer2_cons_cons_lock"}
    %weightsInBN11_layer1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "weightsInBN11_layer1_cons_buff_0"} : memref<37632xi8> 
    %weightsInBN11_layer1_cons_prod_lock = aie.lock(%tile_0_5, 0) {init = 1 : i32, sym_name = "weightsInBN11_layer1_cons_prod_lock"}
    %weightsInBN11_layer1_cons_cons_lock = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "weightsInBN11_layer1_cons_cons_lock"}
    %wts_b11_L3L2_cons_buff_0 = aie.buffer(%tile_1_1) {address = 0 : i32, sym_name = "wts_b11_L3L2_cons_buff_0"} : memref<78288xi8> 
    %wts_b11_L3L2_cons_prod_lock = aie.lock(%tile_1_1, 0) {init = 3 : i32, sym_name = "wts_b11_L3L2_cons_prod_lock"}
    %wts_b11_L3L2_cons_cons_lock = aie.lock(%tile_1_1, 1) {init = 0 : i32, sym_name = "wts_b11_L3L2_cons_cons_lock"}
    %wts_b11_L3L2_prod_lock = aie.lock(%tile_1_0, 2) {init = 1 : i32, sym_name = "wts_b11_L3L2_prod_lock"}
    %wts_b11_L3L2_cons_lock = aie.lock(%tile_1_0, 3) {init = 0 : i32, sym_name = "wts_b11_L3L2_cons_lock"}
    %weightsInBN10_layer3_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "weightsInBN10_layer3_cons_buff_0"} : memref<53760xi8> 
    %weightsInBN10_layer3_cons_prod_lock = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "weightsInBN10_layer3_cons_prod_lock"}
    %weightsInBN10_layer3_cons_cons_lock = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "weightsInBN10_layer3_cons_cons_lock"}
    %weightsInBN10_layer2_cons_buff_0 = aie.buffer(%tile_0_3) {address = 41344 : i32, sym_name = "weightsInBN10_layer2_cons_buff_0"} : memref<4320xi8> 
    %weightsInBN10_layer2_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "weightsInBN10_layer2_cons_prod_lock"}
    %weightsInBN10_layer2_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "weightsInBN10_layer2_cons_cons_lock"}
    %weightsInBN10_layer1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "weightsInBN10_layer1_cons_buff_0"} : memref<38400xi8> 
    %weightsInBN10_layer1_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "weightsInBN10_layer1_cons_prod_lock"}
    %weightsInBN10_layer1_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "weightsInBN10_layer1_cons_cons_lock"}
    %wts_b10_L3L2_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "wts_b10_L3L2_cons_buff_0"} : memref<96480xi8> 
    %wts_b10_L3L2_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 3 : i32, sym_name = "wts_b10_L3L2_cons_prod_lock"}
    %wts_b10_L3L2_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "wts_b10_L3L2_cons_cons_lock"}
    %wts_b10_L3L2_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "wts_b10_L3L2_prod_lock"}
    %wts_b10_L3L2_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "wts_b10_L3L2_cons_lock"}
    %act_out_cons_prod_lock = aie.lock(%tile_1_0, 0) {init = 1 : i32, sym_name = "act_out_cons_prod_lock"}
    %act_out_cons_cons_lock = aie.lock(%tile_1_0, 1) {init = 0 : i32, sym_name = "act_out_cons_cons_lock"}
    %act_out_buff_0 = aie.buffer(%tile_2_2) {address = 27904 : i32, sym_name = "act_out_buff_0"} : memref<7x1x80xi8> 
    %act_out_buff_1 = aie.buffer(%tile_2_2) {address = 28464 : i32, sym_name = "act_out_buff_1"} : memref<7x1x80xi8> 
    %act_out_prod_lock = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "act_out_prod_lock"}
    %act_out_cons_lock = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "act_out_cons_lock"}
    %act_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 52864 : i32, sym_name = "act_in_cons_buff_0"} : memref<14x1x80xi8> 
    %act_in_cons_buff_1 = aie.buffer(%tile_0_2) {address = 53984 : i32, sym_name = "act_in_cons_buff_1"} : memref<14x1x80xi8> 
    %act_in_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "act_in_cons_prod_lock"}
    %act_in_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "act_in_cons_cons_lock"}
    %act_in_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "act_in_prod_lock"}
    %act_in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "act_in_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%tile_1_1, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%tile_1_0, DMA : 1, %tile_2_1, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%tile_2_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%tile_2_1, DMA : 2, %tile_2_2, DMA : 0)
    %bn10_1_rtp = aie.buffer(%tile_0_2) {address = 55104 : i32, sym_name = "bn10_1_rtp"} : memref<16xi32> 
    %bn10_2_rtp = aie.buffer(%tile_0_3) {address = 45664 : i32, sym_name = "bn10_2_rtp"} : memref<16xi32> 
    %bn10_3_rtp = aie.buffer(%tile_0_4) {address = 57920 : i32, sym_name = "bn10_3_rtp"} : memref<16xi32> 
    %bn11_1_rtp = aie.buffer(%tile_0_5) {address = 51200 : i32, sym_name = "bn11_1_rtp"} : memref<16xi32> 
    %bn11_2_rtp = aie.buffer(%tile_1_5) {address = 32272 : i32, sym_name = "bn11_2_rtp"} : memref<16xi32> 
    %bn11_3_rtp = aie.buffer(%tile_1_4) {address = 44928 : i32, sym_name = "bn11_3_rtp"} : memref<16xi32> 
    %bn12_1_rtp = aie.buffer(%tile_1_3) {address = 48064 : i32, sym_name = "bn12_1_rtp"} : memref<16xi32> 
    %bn12_2_rtp = aie.buffer(%tile_1_2) {address = 27568 : i32, sym_name = "bn12_2_rtp"} : memref<16xi32> 
    %bn12_3_rtp = aie.buffer(%tile_2_2) {address = 29024 : i32, sym_name = "bn12_3_rtp"} : memref<16xi32> 
    func.func private @bn10_conv2dk1_relu_i8_ui8(memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32)
    func.func private @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn10_conv2dk1_ui8_i8(memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_relu_i8_ui8(memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32)
    func.func private @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn11_conv2dk1_skip_ui8_i8_i8(memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32)
    func.func private @bn12_conv2dk1_relu_i8_ui8(memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32)
    func.func private @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @bn12_conv2dk1_ui8_i8(memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%tile_0_1, DMA : 3, %tile_1_4, DMA : 1)
    aie.flow(%tile_0_5, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %tile_1_2, DMA : 1)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.use_lock(%weightsInBN10_layer1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c80_i32 = arith.constant 80 : i32
      %c480_i32 = arith.constant 480 : i32
      %c9_i32 = arith.constant 9 : i32
      func.call @bn10_conv2dk1_relu_i8_ui8(%act_in_cons_buff_0, %weightsInBN10_layer1_cons_buff_0, %OF_b10_act_layer1_layer2_buff_0, %c14_i32, %c80_i32, %c480_i32, %c9_i32) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_lock, Release, 1)
      aie.use_lock(%act_in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_2 = arith.constant 14 : i32
      %c80_i32_3 = arith.constant 80 : i32
      %c480_i32_4 = arith.constant 480 : i32
      %c9_i32_5 = arith.constant 9 : i32
      func.call @bn10_conv2dk1_relu_i8_ui8(%act_in_cons_buff_1, %weightsInBN10_layer1_cons_buff_0, %OF_b10_act_layer1_layer2_buff_1, %c14_i32_2, %c80_i32_3, %c480_i32_4, %c9_i32_5) : (memref<14x1x80xi8>, memref<38400xi8>, memref<14x1x480xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_in_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.use_lock(%weightsInBN10_layer1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn10_conv2dk1_fused_relu.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
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
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c480_i32 = arith.constant 480 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_0 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_i32_1 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32, %c1_i32, %c480_i32, %c3_i32, %c3_i32_0, %c0_i32, %c8_i32, %c0_i32_1) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_2 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_3 = arith.constant 1 : index
      %c4_4 = arith.constant 4 : index
      cf.br ^bb3(%c0_2 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c12 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_5 = arith.constant 14 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c480_i32_7 = arith.constant 480 : i32
      %c3_i32_8 = arith.constant 3 : i32
      %c3_i32_9 = arith.constant 3 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c8_i32_11 = arith.constant 8 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_5, %c1_i32_6, %c480_i32_7, %c3_i32_8, %c3_i32_9, %c1_i32_10, %c8_i32_11, %c0_i32_12) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %c480_i32_15 = arith.constant 480 : i32
      %c3_i32_16 = arith.constant 3 : i32
      %c3_i32_17 = arith.constant 3 : i32
      %c1_i32_18 = arith.constant 1 : i32
      %c8_i32_19 = arith.constant 8 : i32
      %c0_i32_20 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_13, %c1_i32_14, %c480_i32_15, %c3_i32_16, %c3_i32_17, %c1_i32_18, %c8_i32_19, %c0_i32_20) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c480_i32_23 = arith.constant 480 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c8_i32_27 = arith.constant 8 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_21, %c1_i32_22, %c480_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %c8_i32_27, %c0_i32_28) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_29 = arith.constant 14 : i32
      %c1_i32_30 = arith.constant 1 : i32
      %c480_i32_31 = arith.constant 480 : i32
      %c3_i32_32 = arith.constant 3 : i32
      %c3_i32_33 = arith.constant 3 : i32
      %c1_i32_34 = arith.constant 1 : i32
      %c8_i32_35 = arith.constant 8 : i32
      %c0_i32_36 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_29, %c1_i32_30, %c480_i32_31, %c3_i32_32, %c3_i32_33, %c1_i32_34, %c8_i32_35, %c0_i32_36) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %4 = arith.addi %2, %c4_4 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_37 = arith.constant 14 : i32
      %c1_i32_38 = arith.constant 1 : i32
      %c480_i32_39 = arith.constant 480 : i32
      %c3_i32_40 = arith.constant 3 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c8_i32_42 = arith.constant 8 : i32
      %c0_i32_43 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_37, %c1_i32_38, %c480_i32_39, %c3_i32_40, %c3_i32_41, %c2_i32, %c8_i32_42, %c0_i32_43) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c1_i32_45 = arith.constant 1 : i32
      %c480_i32_46 = arith.constant 480 : i32
      %c3_i32_47 = arith.constant 3 : i32
      %c3_i32_48 = arith.constant 3 : i32
      %c0_i32_49 = arith.constant 0 : i32
      %c8_i32_50 = arith.constant 8 : i32
      %c0_i32_51 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_44, %c1_i32_45, %c480_i32_46, %c3_i32_47, %c3_i32_48, %c0_i32_49, %c8_i32_50, %c0_i32_51) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_52 = arith.constant 0 : index
      %c12_53 = arith.constant 12 : index
      %c1_54 = arith.constant 1 : index
      %c4_55 = arith.constant 4 : index
      cf.br ^bb6(%c0_52 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c12_53 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_56 = arith.constant 14 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c480_i32_58 = arith.constant 480 : i32
      %c3_i32_59 = arith.constant 3 : i32
      %c3_i32_60 = arith.constant 3 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c8_i32_62 = arith.constant 8 : i32
      %c0_i32_63 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_56, %c1_i32_57, %c480_i32_58, %c3_i32_59, %c3_i32_60, %c1_i32_61, %c8_i32_62, %c0_i32_63) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_64 = arith.constant 14 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c480_i32_66 = arith.constant 480 : i32
      %c3_i32_67 = arith.constant 3 : i32
      %c3_i32_68 = arith.constant 3 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c8_i32_70 = arith.constant 8 : i32
      %c0_i32_71 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_64, %c1_i32_65, %c480_i32_66, %c3_i32_67, %c3_i32_68, %c1_i32_69, %c8_i32_70, %c0_i32_71) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_72 = arith.constant 14 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c480_i32_74 = arith.constant 480 : i32
      %c3_i32_75 = arith.constant 3 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c1_i32_77 = arith.constant 1 : i32
      %c8_i32_78 = arith.constant 8 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_72, %c1_i32_73, %c480_i32_74, %c3_i32_75, %c3_i32_76, %c1_i32_77, %c8_i32_78, %c0_i32_79) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c1_i32_81 = arith.constant 1 : i32
      %c480_i32_82 = arith.constant 480 : i32
      %c3_i32_83 = arith.constant 3 : i32
      %c3_i32_84 = arith.constant 3 : i32
      %c1_i32_85 = arith.constant 1 : i32
      %c8_i32_86 = arith.constant 8 : i32
      %c0_i32_87 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_80, %c1_i32_81, %c480_i32_82, %c3_i32_83, %c3_i32_84, %c1_i32_85, %c8_i32_86, %c0_i32_87) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %7 = arith.addi %5, %c4_55 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_88 = arith.constant 14 : i32
      %c1_i32_89 = arith.constant 1 : i32
      %c480_i32_90 = arith.constant 480 : i32
      %c3_i32_91 = arith.constant 3 : i32
      %c3_i32_92 = arith.constant 3 : i32
      %c2_i32_93 = arith.constant 2 : i32
      %c8_i32_94 = arith.constant 8 : i32
      %c0_i32_95 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_88, %c1_i32_89, %c480_i32_90, %c3_i32_91, %c3_i32_92, %c2_i32_93, %c8_i32_94, %c0_i32_95) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c480_i32_98 = arith.constant 480 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c0_i32_101 = arith.constant 0 : i32
      %c8_i32_102 = arith.constant 8 : i32
      %c0_i32_103 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_96, %c1_i32_97, %c480_i32_98, %c3_i32_99, %c3_i32_100, %c0_i32_101, %c8_i32_102, %c0_i32_103) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_104 = arith.constant 0 : index
      %c12_105 = arith.constant 12 : index
      %c1_106 = arith.constant 1 : index
      %c4_107 = arith.constant 4 : index
      cf.br ^bb9(%c0_104 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c12_105 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_108 = arith.constant 14 : i32
      %c1_i32_109 = arith.constant 1 : i32
      %c480_i32_110 = arith.constant 480 : i32
      %c3_i32_111 = arith.constant 3 : i32
      %c3_i32_112 = arith.constant 3 : i32
      %c1_i32_113 = arith.constant 1 : i32
      %c8_i32_114 = arith.constant 8 : i32
      %c0_i32_115 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_108, %c1_i32_109, %c480_i32_110, %c3_i32_111, %c3_i32_112, %c1_i32_113, %c8_i32_114, %c0_i32_115) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_116 = arith.constant 14 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c480_i32_118 = arith.constant 480 : i32
      %c3_i32_119 = arith.constant 3 : i32
      %c3_i32_120 = arith.constant 3 : i32
      %c1_i32_121 = arith.constant 1 : i32
      %c8_i32_122 = arith.constant 8 : i32
      %c0_i32_123 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_116, %c1_i32_117, %c480_i32_118, %c3_i32_119, %c3_i32_120, %c1_i32_121, %c8_i32_122, %c0_i32_123) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_124 = arith.constant 14 : i32
      %c1_i32_125 = arith.constant 1 : i32
      %c480_i32_126 = arith.constant 480 : i32
      %c3_i32_127 = arith.constant 3 : i32
      %c3_i32_128 = arith.constant 3 : i32
      %c1_i32_129 = arith.constant 1 : i32
      %c8_i32_130 = arith.constant 8 : i32
      %c0_i32_131 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_124, %c1_i32_125, %c480_i32_126, %c3_i32_127, %c3_i32_128, %c1_i32_129, %c8_i32_130, %c0_i32_131) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_132 = arith.constant 14 : i32
      %c1_i32_133 = arith.constant 1 : i32
      %c480_i32_134 = arith.constant 480 : i32
      %c3_i32_135 = arith.constant 3 : i32
      %c3_i32_136 = arith.constant 3 : i32
      %c1_i32_137 = arith.constant 1 : i32
      %c8_i32_138 = arith.constant 8 : i32
      %c0_i32_139 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_132, %c1_i32_133, %c480_i32_134, %c3_i32_135, %c3_i32_136, %c1_i32_137, %c8_i32_138, %c0_i32_139) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %10 = arith.addi %8, %c4_107 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_140 = arith.constant 14 : i32
      %c1_i32_141 = arith.constant 1 : i32
      %c480_i32_142 = arith.constant 480 : i32
      %c3_i32_143 = arith.constant 3 : i32
      %c3_i32_144 = arith.constant 3 : i32
      %c2_i32_145 = arith.constant 2 : i32
      %c8_i32_146 = arith.constant 8 : i32
      %c0_i32_147 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_140, %c1_i32_141, %c480_i32_142, %c3_i32_143, %c3_i32_144, %c2_i32_145, %c8_i32_146, %c0_i32_147) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_148 = arith.constant 14 : i32
      %c1_i32_149 = arith.constant 1 : i32
      %c480_i32_150 = arith.constant 480 : i32
      %c3_i32_151 = arith.constant 3 : i32
      %c3_i32_152 = arith.constant 3 : i32
      %c0_i32_153 = arith.constant 0 : i32
      %c8_i32_154 = arith.constant 8 : i32
      %c0_i32_155 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_148, %c1_i32_149, %c480_i32_150, %c3_i32_151, %c3_i32_152, %c0_i32_153, %c8_i32_154, %c0_i32_155) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_156 = arith.constant 0 : index
      %c12_157 = arith.constant 12 : index
      %c1_158 = arith.constant 1 : index
      %c4_159 = arith.constant 4 : index
      cf.br ^bb12(%c0_156 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c12_157 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_160 = arith.constant 14 : i32
      %c1_i32_161 = arith.constant 1 : i32
      %c480_i32_162 = arith.constant 480 : i32
      %c3_i32_163 = arith.constant 3 : i32
      %c3_i32_164 = arith.constant 3 : i32
      %c1_i32_165 = arith.constant 1 : i32
      %c8_i32_166 = arith.constant 8 : i32
      %c0_i32_167 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_160, %c1_i32_161, %c480_i32_162, %c3_i32_163, %c3_i32_164, %c1_i32_165, %c8_i32_166, %c0_i32_167) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_168 = arith.constant 14 : i32
      %c1_i32_169 = arith.constant 1 : i32
      %c480_i32_170 = arith.constant 480 : i32
      %c3_i32_171 = arith.constant 3 : i32
      %c3_i32_172 = arith.constant 3 : i32
      %c1_i32_173 = arith.constant 1 : i32
      %c8_i32_174 = arith.constant 8 : i32
      %c0_i32_175 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_168, %c1_i32_169, %c480_i32_170, %c3_i32_171, %c3_i32_172, %c1_i32_173, %c8_i32_174, %c0_i32_175) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_176 = arith.constant 14 : i32
      %c1_i32_177 = arith.constant 1 : i32
      %c480_i32_178 = arith.constant 480 : i32
      %c3_i32_179 = arith.constant 3 : i32
      %c3_i32_180 = arith.constant 3 : i32
      %c1_i32_181 = arith.constant 1 : i32
      %c8_i32_182 = arith.constant 8 : i32
      %c0_i32_183 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_176, %c1_i32_177, %c480_i32_178, %c3_i32_179, %c3_i32_180, %c1_i32_181, %c8_i32_182, %c0_i32_183) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_184 = arith.constant 14 : i32
      %c1_i32_185 = arith.constant 1 : i32
      %c480_i32_186 = arith.constant 480 : i32
      %c3_i32_187 = arith.constant 3 : i32
      %c3_i32_188 = arith.constant 3 : i32
      %c1_i32_189 = arith.constant 1 : i32
      %c8_i32_190 = arith.constant 8 : i32
      %c0_i32_191 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_184, %c1_i32_185, %c480_i32_186, %c3_i32_187, %c3_i32_188, %c1_i32_189, %c8_i32_190, %c0_i32_191) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %13 = arith.addi %11, %c4_159 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_192 = arith.constant 14 : i32
      %c1_i32_193 = arith.constant 1 : i32
      %c480_i32_194 = arith.constant 480 : i32
      %c3_i32_195 = arith.constant 3 : i32
      %c3_i32_196 = arith.constant 3 : i32
      %c2_i32_197 = arith.constant 2 : i32
      %c8_i32_198 = arith.constant 8 : i32
      %c0_i32_199 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_192, %c1_i32_193, %c480_i32_194, %c3_i32_195, %c3_i32_196, %c2_i32_197, %c8_i32_198, %c0_i32_199) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      cf.br ^bb16(%c9223372036854775804 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb20
      %16 = arith.cmpi slt, %15, %c9223372036854775807 : index
      cf.cond_br %16, ^bb17, ^bb21
    ^bb17:  // pred: ^bb16
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_200 = arith.constant 14 : i32
      %c1_i32_201 = arith.constant 1 : i32
      %c480_i32_202 = arith.constant 480 : i32
      %c3_i32_203 = arith.constant 3 : i32
      %c3_i32_204 = arith.constant 3 : i32
      %c0_i32_205 = arith.constant 0 : i32
      %c8_i32_206 = arith.constant 8 : i32
      %c0_i32_207 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_200, %c1_i32_201, %c480_i32_202, %c3_i32_203, %c3_i32_204, %c0_i32_205, %c8_i32_206, %c0_i32_207) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %c0_208 = arith.constant 0 : index
      %c12_209 = arith.constant 12 : index
      %c1_210 = arith.constant 1 : index
      %c4_211 = arith.constant 4 : index
      cf.br ^bb18(%c0_208 : index)
    ^bb18(%17: index):  // 2 preds: ^bb17, ^bb19
      %18 = arith.cmpi slt, %17, %c12_209 : index
      cf.cond_br %18, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_212 = arith.constant 14 : i32
      %c1_i32_213 = arith.constant 1 : i32
      %c480_i32_214 = arith.constant 480 : i32
      %c3_i32_215 = arith.constant 3 : i32
      %c3_i32_216 = arith.constant 3 : i32
      %c1_i32_217 = arith.constant 1 : i32
      %c8_i32_218 = arith.constant 8 : i32
      %c0_i32_219 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_212, %c1_i32_213, %c480_i32_214, %c3_i32_215, %c3_i32_216, %c1_i32_217, %c8_i32_218, %c0_i32_219) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_220 = arith.constant 14 : i32
      %c1_i32_221 = arith.constant 1 : i32
      %c480_i32_222 = arith.constant 480 : i32
      %c3_i32_223 = arith.constant 3 : i32
      %c3_i32_224 = arith.constant 3 : i32
      %c1_i32_225 = arith.constant 1 : i32
      %c8_i32_226 = arith.constant 8 : i32
      %c0_i32_227 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_220, %c1_i32_221, %c480_i32_222, %c3_i32_223, %c3_i32_224, %c1_i32_225, %c8_i32_226, %c0_i32_227) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_228 = arith.constant 14 : i32
      %c1_i32_229 = arith.constant 1 : i32
      %c480_i32_230 = arith.constant 480 : i32
      %c3_i32_231 = arith.constant 3 : i32
      %c3_i32_232 = arith.constant 3 : i32
      %c1_i32_233 = arith.constant 1 : i32
      %c8_i32_234 = arith.constant 8 : i32
      %c0_i32_235 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_2, %OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_228, %c1_i32_229, %c480_i32_230, %c3_i32_231, %c3_i32_232, %c1_i32_233, %c8_i32_234, %c0_i32_235) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_236 = arith.constant 14 : i32
      %c1_i32_237 = arith.constant 1 : i32
      %c480_i32_238 = arith.constant 480 : i32
      %c3_i32_239 = arith.constant 3 : i32
      %c3_i32_240 = arith.constant 3 : i32
      %c1_i32_241 = arith.constant 1 : i32
      %c8_i32_242 = arith.constant 8 : i32
      %c0_i32_243 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_3, %OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_0, %c14_i32_236, %c1_i32_237, %c480_i32_238, %c3_i32_239, %c3_i32_240, %c1_i32_241, %c8_i32_242, %c0_i32_243) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %19 = arith.addi %17, %c4_211 : index
      cf.br ^bb18(%19 : index)
    ^bb20:  // pred: ^bb18
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_244 = arith.constant 14 : i32
      %c1_i32_245 = arith.constant 1 : i32
      %c480_i32_246 = arith.constant 480 : i32
      %c3_i32_247 = arith.constant 3 : i32
      %c3_i32_248 = arith.constant 3 : i32
      %c2_i32_249 = arith.constant 2 : i32
      %c8_i32_250 = arith.constant 8 : i32
      %c0_i32_251 = arith.constant 0 : i32
      func.call @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b10_act_layer1_layer2_cons_buff_0, %OF_b10_act_layer1_layer2_cons_buff_1, %OF_b10_act_layer1_layer2_cons_buff_1, %weightsInBN10_layer2_cons_buff_0, %OF_b10_act_layer2_layer3_buff_1, %c14_i32_244, %c1_i32_245, %c480_i32_246, %c3_i32_247, %c3_i32_248, %c2_i32_249, %c8_i32_250, %c0_i32_251) : (memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>, memref<4320xi8>, memref<14x1x480xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, Release, 1)
      %20 = arith.addi %15, %c1 : index
      cf.br ^bb16(%20 : index)
    ^bb21:  // pred: ^bb16
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn10_conv2dk3_dw.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      aie.use_lock(%weightsInBN10_layer3_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c480_i32 = arith.constant 480 : i32
      %c112_i32 = arith.constant 112 : i32
      %c9_i32 = arith.constant 9 : i32
      func.call @bn10_conv2dk1_ui8_i8(%OF_b10_act_layer2_layer3_buff_0, %weightsInBN10_layer3_cons_buff_0, %OF_b10_layer3_bn_11_layer1_buff_0, %c14_i32, %c480_i32, %c112_i32, %c9_i32) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_2 = arith.constant 14 : i32
      %c480_i32_3 = arith.constant 480 : i32
      %c112_i32_4 = arith.constant 112 : i32
      %c9_i32_5 = arith.constant 9 : i32
      func.call @bn10_conv2dk1_ui8_i8(%OF_b10_act_layer2_layer3_buff_1, %weightsInBN10_layer3_cons_buff_0, %OF_b10_layer3_bn_11_layer1_buff_1, %c14_i32_2, %c480_i32_3, %c112_i32_4, %c9_i32_5) : (memref<14x1x480xui8>, memref<53760xi8>, memref<14x1x112xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.use_lock(%weightsInBN10_layer3_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn10_conv2dk1_ui8.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      aie.use_lock(%weightsInBN11_layer1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c112_i32 = arith.constant 112 : i32
      %c336_i32 = arith.constant 336 : i32
      %c9_i32 = arith.constant 9 : i32
      func.call @bn11_conv2dk1_relu_i8_ui8(%OF_b10_layer3_bn_11_layer1_0_cons_buff_0, %weightsInBN11_layer1_cons_buff_0, %OF_b11_act_layer1_layer2_buff_0, %c14_i32, %c112_i32, %c336_i32, %c9_i32) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_lock, Release, 1)
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_2 = arith.constant 14 : i32
      %c112_i32_3 = arith.constant 112 : i32
      %c336_i32_4 = arith.constant 336 : i32
      %c9_i32_5 = arith.constant 9 : i32
      func.call @bn11_conv2dk1_relu_i8_ui8(%OF_b10_layer3_bn_11_layer1_0_cons_buff_1, %weightsInBN11_layer1_cons_buff_0, %OF_b11_act_layer1_layer2_buff_1, %c14_i32_2, %c112_i32_3, %c336_i32_4, %c9_i32_5) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.use_lock(%weightsInBN11_layer1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn11_conv2dk1_fused_relu.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
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
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c336_i32 = arith.constant 336 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_0 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_i32_1 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_0, %c0_i32, %c8_i32, %c0_i32_1) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_2 = arith.constant 0 : index
      %c12 = arith.constant 12 : index
      %c1_3 = arith.constant 1 : index
      %c4_4 = arith.constant 4 : index
      cf.br ^bb3(%c0_2 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c12 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_5 = arith.constant 14 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c336_i32_7 = arith.constant 336 : i32
      %c3_i32_8 = arith.constant 3 : i32
      %c3_i32_9 = arith.constant 3 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c8_i32_11 = arith.constant 8 : i32
      %c0_i32_12 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_5, %c1_i32_6, %c336_i32_7, %c3_i32_8, %c3_i32_9, %c1_i32_10, %c8_i32_11, %c0_i32_12) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_13 = arith.constant 14 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %c336_i32_15 = arith.constant 336 : i32
      %c3_i32_16 = arith.constant 3 : i32
      %c3_i32_17 = arith.constant 3 : i32
      %c1_i32_18 = arith.constant 1 : i32
      %c8_i32_19 = arith.constant 8 : i32
      %c0_i32_20 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_13, %c1_i32_14, %c336_i32_15, %c3_i32_16, %c3_i32_17, %c1_i32_18, %c8_i32_19, %c0_i32_20) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_21 = arith.constant 14 : i32
      %c1_i32_22 = arith.constant 1 : i32
      %c336_i32_23 = arith.constant 336 : i32
      %c3_i32_24 = arith.constant 3 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c1_i32_26 = arith.constant 1 : i32
      %c8_i32_27 = arith.constant 8 : i32
      %c0_i32_28 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_21, %c1_i32_22, %c336_i32_23, %c3_i32_24, %c3_i32_25, %c1_i32_26, %c8_i32_27, %c0_i32_28) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_29 = arith.constant 14 : i32
      %c1_i32_30 = arith.constant 1 : i32
      %c336_i32_31 = arith.constant 336 : i32
      %c3_i32_32 = arith.constant 3 : i32
      %c3_i32_33 = arith.constant 3 : i32
      %c1_i32_34 = arith.constant 1 : i32
      %c8_i32_35 = arith.constant 8 : i32
      %c0_i32_36 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_29, %c1_i32_30, %c336_i32_31, %c3_i32_32, %c3_i32_33, %c1_i32_34, %c8_i32_35, %c0_i32_36) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %4 = arith.addi %2, %c4_4 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_37 = arith.constant 14 : i32
      %c1_i32_38 = arith.constant 1 : i32
      %c336_i32_39 = arith.constant 336 : i32
      %c3_i32_40 = arith.constant 3 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c8_i32_42 = arith.constant 8 : i32
      %c0_i32_43 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_37, %c1_i32_38, %c336_i32_39, %c3_i32_40, %c3_i32_41, %c2_i32, %c8_i32_42, %c0_i32_43) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_44 = arith.constant 14 : i32
      %c1_i32_45 = arith.constant 1 : i32
      %c336_i32_46 = arith.constant 336 : i32
      %c3_i32_47 = arith.constant 3 : i32
      %c3_i32_48 = arith.constant 3 : i32
      %c0_i32_49 = arith.constant 0 : i32
      %c8_i32_50 = arith.constant 8 : i32
      %c0_i32_51 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_44, %c1_i32_45, %c336_i32_46, %c3_i32_47, %c3_i32_48, %c0_i32_49, %c8_i32_50, %c0_i32_51) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_52 = arith.constant 0 : index
      %c12_53 = arith.constant 12 : index
      %c1_54 = arith.constant 1 : index
      %c4_55 = arith.constant 4 : index
      cf.br ^bb6(%c0_52 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c12_53 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_56 = arith.constant 14 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c336_i32_58 = arith.constant 336 : i32
      %c3_i32_59 = arith.constant 3 : i32
      %c3_i32_60 = arith.constant 3 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c8_i32_62 = arith.constant 8 : i32
      %c0_i32_63 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_56, %c1_i32_57, %c336_i32_58, %c3_i32_59, %c3_i32_60, %c1_i32_61, %c8_i32_62, %c0_i32_63) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_64 = arith.constant 14 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c336_i32_66 = arith.constant 336 : i32
      %c3_i32_67 = arith.constant 3 : i32
      %c3_i32_68 = arith.constant 3 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c8_i32_70 = arith.constant 8 : i32
      %c0_i32_71 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_64, %c1_i32_65, %c336_i32_66, %c3_i32_67, %c3_i32_68, %c1_i32_69, %c8_i32_70, %c0_i32_71) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_72 = arith.constant 14 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c336_i32_74 = arith.constant 336 : i32
      %c3_i32_75 = arith.constant 3 : i32
      %c3_i32_76 = arith.constant 3 : i32
      %c1_i32_77 = arith.constant 1 : i32
      %c8_i32_78 = arith.constant 8 : i32
      %c0_i32_79 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_72, %c1_i32_73, %c336_i32_74, %c3_i32_75, %c3_i32_76, %c1_i32_77, %c8_i32_78, %c0_i32_79) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_80 = arith.constant 14 : i32
      %c1_i32_81 = arith.constant 1 : i32
      %c336_i32_82 = arith.constant 336 : i32
      %c3_i32_83 = arith.constant 3 : i32
      %c3_i32_84 = arith.constant 3 : i32
      %c1_i32_85 = arith.constant 1 : i32
      %c8_i32_86 = arith.constant 8 : i32
      %c0_i32_87 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_80, %c1_i32_81, %c336_i32_82, %c3_i32_83, %c3_i32_84, %c1_i32_85, %c8_i32_86, %c0_i32_87) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %7 = arith.addi %5, %c4_55 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_88 = arith.constant 14 : i32
      %c1_i32_89 = arith.constant 1 : i32
      %c336_i32_90 = arith.constant 336 : i32
      %c3_i32_91 = arith.constant 3 : i32
      %c3_i32_92 = arith.constant 3 : i32
      %c2_i32_93 = arith.constant 2 : i32
      %c8_i32_94 = arith.constant 8 : i32
      %c0_i32_95 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_88, %c1_i32_89, %c336_i32_90, %c3_i32_91, %c3_i32_92, %c2_i32_93, %c8_i32_94, %c0_i32_95) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_96 = arith.constant 14 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %c336_i32_98 = arith.constant 336 : i32
      %c3_i32_99 = arith.constant 3 : i32
      %c3_i32_100 = arith.constant 3 : i32
      %c0_i32_101 = arith.constant 0 : i32
      %c8_i32_102 = arith.constant 8 : i32
      %c0_i32_103 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_96, %c1_i32_97, %c336_i32_98, %c3_i32_99, %c3_i32_100, %c0_i32_101, %c8_i32_102, %c0_i32_103) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_104 = arith.constant 0 : index
      %c12_105 = arith.constant 12 : index
      %c1_106 = arith.constant 1 : index
      %c4_107 = arith.constant 4 : index
      cf.br ^bb9(%c0_104 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c12_105 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_108 = arith.constant 14 : i32
      %c1_i32_109 = arith.constant 1 : i32
      %c336_i32_110 = arith.constant 336 : i32
      %c3_i32_111 = arith.constant 3 : i32
      %c3_i32_112 = arith.constant 3 : i32
      %c1_i32_113 = arith.constant 1 : i32
      %c8_i32_114 = arith.constant 8 : i32
      %c0_i32_115 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_108, %c1_i32_109, %c336_i32_110, %c3_i32_111, %c3_i32_112, %c1_i32_113, %c8_i32_114, %c0_i32_115) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_116 = arith.constant 14 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c336_i32_118 = arith.constant 336 : i32
      %c3_i32_119 = arith.constant 3 : i32
      %c3_i32_120 = arith.constant 3 : i32
      %c1_i32_121 = arith.constant 1 : i32
      %c8_i32_122 = arith.constant 8 : i32
      %c0_i32_123 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_116, %c1_i32_117, %c336_i32_118, %c3_i32_119, %c3_i32_120, %c1_i32_121, %c8_i32_122, %c0_i32_123) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_124 = arith.constant 14 : i32
      %c1_i32_125 = arith.constant 1 : i32
      %c336_i32_126 = arith.constant 336 : i32
      %c3_i32_127 = arith.constant 3 : i32
      %c3_i32_128 = arith.constant 3 : i32
      %c1_i32_129 = arith.constant 1 : i32
      %c8_i32_130 = arith.constant 8 : i32
      %c0_i32_131 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_124, %c1_i32_125, %c336_i32_126, %c3_i32_127, %c3_i32_128, %c1_i32_129, %c8_i32_130, %c0_i32_131) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_132 = arith.constant 14 : i32
      %c1_i32_133 = arith.constant 1 : i32
      %c336_i32_134 = arith.constant 336 : i32
      %c3_i32_135 = arith.constant 3 : i32
      %c3_i32_136 = arith.constant 3 : i32
      %c1_i32_137 = arith.constant 1 : i32
      %c8_i32_138 = arith.constant 8 : i32
      %c0_i32_139 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_132, %c1_i32_133, %c336_i32_134, %c3_i32_135, %c3_i32_136, %c1_i32_137, %c8_i32_138, %c0_i32_139) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %10 = arith.addi %8, %c4_107 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_140 = arith.constant 14 : i32
      %c1_i32_141 = arith.constant 1 : i32
      %c336_i32_142 = arith.constant 336 : i32
      %c3_i32_143 = arith.constant 3 : i32
      %c3_i32_144 = arith.constant 3 : i32
      %c2_i32_145 = arith.constant 2 : i32
      %c8_i32_146 = arith.constant 8 : i32
      %c0_i32_147 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_140, %c1_i32_141, %c336_i32_142, %c3_i32_143, %c3_i32_144, %c2_i32_145, %c8_i32_146, %c0_i32_147) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_148 = arith.constant 14 : i32
      %c1_i32_149 = arith.constant 1 : i32
      %c336_i32_150 = arith.constant 336 : i32
      %c3_i32_151 = arith.constant 3 : i32
      %c3_i32_152 = arith.constant 3 : i32
      %c0_i32_153 = arith.constant 0 : i32
      %c8_i32_154 = arith.constant 8 : i32
      %c0_i32_155 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_148, %c1_i32_149, %c336_i32_150, %c3_i32_151, %c3_i32_152, %c0_i32_153, %c8_i32_154, %c0_i32_155) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_156 = arith.constant 0 : index
      %c12_157 = arith.constant 12 : index
      %c1_158 = arith.constant 1 : index
      %c4_159 = arith.constant 4 : index
      cf.br ^bb12(%c0_156 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c12_157 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_160 = arith.constant 14 : i32
      %c1_i32_161 = arith.constant 1 : i32
      %c336_i32_162 = arith.constant 336 : i32
      %c3_i32_163 = arith.constant 3 : i32
      %c3_i32_164 = arith.constant 3 : i32
      %c1_i32_165 = arith.constant 1 : i32
      %c8_i32_166 = arith.constant 8 : i32
      %c0_i32_167 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_160, %c1_i32_161, %c336_i32_162, %c3_i32_163, %c3_i32_164, %c1_i32_165, %c8_i32_166, %c0_i32_167) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_168 = arith.constant 14 : i32
      %c1_i32_169 = arith.constant 1 : i32
      %c336_i32_170 = arith.constant 336 : i32
      %c3_i32_171 = arith.constant 3 : i32
      %c3_i32_172 = arith.constant 3 : i32
      %c1_i32_173 = arith.constant 1 : i32
      %c8_i32_174 = arith.constant 8 : i32
      %c0_i32_175 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_168, %c1_i32_169, %c336_i32_170, %c3_i32_171, %c3_i32_172, %c1_i32_173, %c8_i32_174, %c0_i32_175) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_176 = arith.constant 14 : i32
      %c1_i32_177 = arith.constant 1 : i32
      %c336_i32_178 = arith.constant 336 : i32
      %c3_i32_179 = arith.constant 3 : i32
      %c3_i32_180 = arith.constant 3 : i32
      %c1_i32_181 = arith.constant 1 : i32
      %c8_i32_182 = arith.constant 8 : i32
      %c0_i32_183 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_176, %c1_i32_177, %c336_i32_178, %c3_i32_179, %c3_i32_180, %c1_i32_181, %c8_i32_182, %c0_i32_183) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_184 = arith.constant 14 : i32
      %c1_i32_185 = arith.constant 1 : i32
      %c336_i32_186 = arith.constant 336 : i32
      %c3_i32_187 = arith.constant 3 : i32
      %c3_i32_188 = arith.constant 3 : i32
      %c1_i32_189 = arith.constant 1 : i32
      %c8_i32_190 = arith.constant 8 : i32
      %c0_i32_191 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_184, %c1_i32_185, %c336_i32_186, %c3_i32_187, %c3_i32_188, %c1_i32_189, %c8_i32_190, %c0_i32_191) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %13 = arith.addi %11, %c4_159 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_192 = arith.constant 14 : i32
      %c1_i32_193 = arith.constant 1 : i32
      %c336_i32_194 = arith.constant 336 : i32
      %c3_i32_195 = arith.constant 3 : i32
      %c3_i32_196 = arith.constant 3 : i32
      %c2_i32_197 = arith.constant 2 : i32
      %c8_i32_198 = arith.constant 8 : i32
      %c0_i32_199 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_192, %c1_i32_193, %c336_i32_194, %c3_i32_195, %c3_i32_196, %c2_i32_197, %c8_i32_198, %c0_i32_199) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      cf.br ^bb16(%c9223372036854775804 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb20
      %16 = arith.cmpi slt, %15, %c9223372036854775807 : index
      cf.cond_br %16, ^bb17, ^bb21
    ^bb17:  // pred: ^bb16
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_200 = arith.constant 14 : i32
      %c1_i32_201 = arith.constant 1 : i32
      %c336_i32_202 = arith.constant 336 : i32
      %c3_i32_203 = arith.constant 3 : i32
      %c3_i32_204 = arith.constant 3 : i32
      %c0_i32_205 = arith.constant 0 : i32
      %c8_i32_206 = arith.constant 8 : i32
      %c0_i32_207 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_200, %c1_i32_201, %c336_i32_202, %c3_i32_203, %c3_i32_204, %c0_i32_205, %c8_i32_206, %c0_i32_207) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %c0_208 = arith.constant 0 : index
      %c12_209 = arith.constant 12 : index
      %c1_210 = arith.constant 1 : index
      %c4_211 = arith.constant 4 : index
      cf.br ^bb18(%c0_208 : index)
    ^bb18(%17: index):  // 2 preds: ^bb17, ^bb19
      %18 = arith.cmpi slt, %17, %c12_209 : index
      cf.cond_br %18, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_212 = arith.constant 14 : i32
      %c1_i32_213 = arith.constant 1 : i32
      %c336_i32_214 = arith.constant 336 : i32
      %c3_i32_215 = arith.constant 3 : i32
      %c3_i32_216 = arith.constant 3 : i32
      %c1_i32_217 = arith.constant 1 : i32
      %c8_i32_218 = arith.constant 8 : i32
      %c0_i32_219 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_212, %c1_i32_213, %c336_i32_214, %c3_i32_215, %c3_i32_216, %c1_i32_217, %c8_i32_218, %c0_i32_219) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_220 = arith.constant 14 : i32
      %c1_i32_221 = arith.constant 1 : i32
      %c336_i32_222 = arith.constant 336 : i32
      %c3_i32_223 = arith.constant 3 : i32
      %c3_i32_224 = arith.constant 3 : i32
      %c1_i32_225 = arith.constant 1 : i32
      %c8_i32_226 = arith.constant 8 : i32
      %c0_i32_227 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_220, %c1_i32_221, %c336_i32_222, %c3_i32_223, %c3_i32_224, %c1_i32_225, %c8_i32_226, %c0_i32_227) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_228 = arith.constant 14 : i32
      %c1_i32_229 = arith.constant 1 : i32
      %c336_i32_230 = arith.constant 336 : i32
      %c3_i32_231 = arith.constant 3 : i32
      %c3_i32_232 = arith.constant 3 : i32
      %c1_i32_233 = arith.constant 1 : i32
      %c8_i32_234 = arith.constant 8 : i32
      %c0_i32_235 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_2, %OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_228, %c1_i32_229, %c336_i32_230, %c3_i32_231, %c3_i32_232, %c1_i32_233, %c8_i32_234, %c0_i32_235) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_236 = arith.constant 14 : i32
      %c1_i32_237 = arith.constant 1 : i32
      %c336_i32_238 = arith.constant 336 : i32
      %c3_i32_239 = arith.constant 3 : i32
      %c3_i32_240 = arith.constant 3 : i32
      %c1_i32_241 = arith.constant 1 : i32
      %c8_i32_242 = arith.constant 8 : i32
      %c0_i32_243 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_3, %OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_0, %c14_i32_236, %c1_i32_237, %c336_i32_238, %c3_i32_239, %c3_i32_240, %c1_i32_241, %c8_i32_242, %c0_i32_243) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %19 = arith.addi %17, %c4_211 : index
      cf.br ^bb18(%19 : index)
    ^bb20:  // pred: ^bb18
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_244 = arith.constant 14 : i32
      %c1_i32_245 = arith.constant 1 : i32
      %c336_i32_246 = arith.constant 336 : i32
      %c3_i32_247 = arith.constant 3 : i32
      %c3_i32_248 = arith.constant 3 : i32
      %c2_i32_249 = arith.constant 2 : i32
      %c8_i32_250 = arith.constant 8 : i32
      %c0_i32_251 = arith.constant 0 : i32
      func.call @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(%OF_b11_act_layer1_layer2_cons_buff_0, %OF_b11_act_layer1_layer2_cons_buff_1, %OF_b11_act_layer1_layer2_cons_buff_1, %weightsInBN11_layer2_cons_buff_0, %OF_b11_act_layer2_layer3_buff_1, %c14_i32_244, %c1_i32_245, %c336_i32_246, %c3_i32_247, %c3_i32_248, %c2_i32_249, %c8_i32_250, %c0_i32_251) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<14x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, Release, 1)
      %20 = arith.addi %15, %c1 : index
      cf.br ^bb16(%20 : index)
    ^bb21:  // pred: ^bb16
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn11_conv2dk3_dw.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      aie.use_lock(%weightsInBN11_layer3_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_skip_cons_cons_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c336_i32 = arith.constant 336 : i32
      %c112_i32 = arith.constant 112 : i32
      %c12_i32 = arith.constant 12 : i32
      %c1_i32 = arith.constant 1 : i32
      func.call @bn11_conv2dk1_skip_ui8_i8_i8(%OF_b11_act_layer2_layer3_buff_0, %weightsInBN11_layer3_cons_buff_0, %OF_b11_layer3_bn_12_layer1_buff_0, %OF_b11_skip_cons_buff_0, %c14_i32, %c336_i32, %c112_i32, %c12_i32, %c1_i32) : (memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_skip_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b11_skip_cons_cons_lock, AcquireGreaterEqual, 1)
      %c14_i32_2 = arith.constant 14 : i32
      %c336_i32_3 = arith.constant 336 : i32
      %c112_i32_4 = arith.constant 112 : i32
      %c12_i32_5 = arith.constant 12 : i32
      %c1_i32_6 = arith.constant 1 : i32
      func.call @bn11_conv2dk1_skip_ui8_i8_i8(%OF_b11_act_layer2_layer3_buff_1, %weightsInBN11_layer3_cons_buff_0, %OF_b11_layer3_bn_12_layer1_buff_1, %OF_b11_skip_cons_buff_1, %c14_i32_2, %c336_i32_3, %c112_i32_4, %c12_i32_5, %c1_i32_6) : (memref<14x1x336xui8>, memref<37632xi8>, memref<14x1x112xi8>, memref<14x1x112xi8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_skip_cons_prod_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.use_lock(%weightsInBN11_layer3_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn11_conv2dk1_skip.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      aie.use_lock(%weightsInBN12_layer1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c14 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c112_i32 = arith.constant 112 : i32
      %c336_i32 = arith.constant 336 : i32
      %c8_i32 = arith.constant 8 : i32
      func.call @bn12_conv2dk1_relu_i8_ui8(%OF_b11_layer3_bn_12_layer1_buff_0, %weightsInBN12_layer1_cons_buff_0, %OF_b12_act_layer1_layer2_buff_0, %c14_i32, %c112_i32, %c336_i32, %c8_i32) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_prod_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_lock, Release, 1)
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_2 = arith.constant 14 : i32
      %c112_i32_3 = arith.constant 112 : i32
      %c336_i32_4 = arith.constant 336 : i32
      %c8_i32_5 = arith.constant 8 : i32
      func.call @bn12_conv2dk1_relu_i8_ui8(%OF_b11_layer3_bn_12_layer1_buff_1, %weightsInBN12_layer1_cons_buff_0, %OF_b12_act_layer1_layer2_buff_1, %c14_i32_2, %c112_i32_3, %c336_i32_4, %c8_i32_5) : (memref<14x1x112xi8>, memref<37632xi8>, memref<14x1x336xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b11_layer3_bn_12_layer1_prod_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.use_lock(%weightsInBN12_layer1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn12_conv2dk1_fused_relu.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      aie.use_lock(%weightsInBN12_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
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
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32 = arith.constant 14 : i32
      %c1_i32 = arith.constant 1 : i32
      %c336_i32 = arith.constant 336 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_0 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c7_i32 = arith.constant 7 : i32
      %c0_i32_1 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_0, %c0_i32, %c7_i32, %c0_i32_1) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %c0_2 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1_3 = arith.constant 1 : index
      %c4_4 = arith.constant 4 : index
      %c4_5 = arith.constant 4 : index
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_6 = arith.constant 14 : i32
      %c1_i32_7 = arith.constant 1 : i32
      %c336_i32_8 = arith.constant 336 : i32
      %c3_i32_9 = arith.constant 3 : i32
      %c3_i32_10 = arith.constant 3 : i32
      %c1_i32_11 = arith.constant 1 : i32
      %c7_i32_12 = arith.constant 7 : i32
      %c0_i32_13 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_6, %c1_i32_7, %c336_i32_8, %c3_i32_9, %c3_i32_10, %c1_i32_11, %c7_i32_12, %c0_i32_13) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_14 = arith.constant 14 : i32
      %c1_i32_15 = arith.constant 1 : i32
      %c336_i32_16 = arith.constant 336 : i32
      %c3_i32_17 = arith.constant 3 : i32
      %c3_i32_18 = arith.constant 3 : i32
      %c1_i32_19 = arith.constant 1 : i32
      %c7_i32_20 = arith.constant 7 : i32
      %c0_i32_21 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_14, %c1_i32_15, %c336_i32_16, %c3_i32_17, %c3_i32_18, %c1_i32_19, %c7_i32_20, %c0_i32_21) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_22 = arith.constant 14 : i32
      %c1_i32_23 = arith.constant 1 : i32
      %c336_i32_24 = arith.constant 336 : i32
      %c3_i32_25 = arith.constant 3 : i32
      %c3_i32_26 = arith.constant 3 : i32
      %c1_i32_27 = arith.constant 1 : i32
      %c7_i32_28 = arith.constant 7 : i32
      %c0_i32_29 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_22, %c1_i32_23, %c336_i32_24, %c3_i32_25, %c3_i32_26, %c1_i32_27, %c7_i32_28, %c0_i32_29) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_30 = arith.constant 14 : i32
      %c1_i32_31 = arith.constant 1 : i32
      %c336_i32_32 = arith.constant 336 : i32
      %c3_i32_33 = arith.constant 3 : i32
      %c3_i32_34 = arith.constant 3 : i32
      %c1_i32_35 = arith.constant 1 : i32
      %c7_i32_36 = arith.constant 7 : i32
      %c0_i32_37 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_30, %c1_i32_31, %c336_i32_32, %c3_i32_33, %c3_i32_34, %c1_i32_35, %c7_i32_36, %c0_i32_37) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      cf.br ^bb3(%c4_4 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c6 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_38 = arith.constant 14 : i32
      %c1_i32_39 = arith.constant 1 : i32
      %c336_i32_40 = arith.constant 336 : i32
      %c3_i32_41 = arith.constant 3 : i32
      %c3_i32_42 = arith.constant 3 : i32
      %c1_i32_43 = arith.constant 1 : i32
      %c7_i32_44 = arith.constant 7 : i32
      %c0_i32_45 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_38, %c1_i32_39, %c336_i32_40, %c3_i32_41, %c3_i32_42, %c1_i32_43, %c7_i32_44, %c0_i32_45) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      %4 = arith.addi %2, %c1_3 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_46 = arith.constant 14 : i32
      %c1_i32_47 = arith.constant 1 : i32
      %c336_i32_48 = arith.constant 336 : i32
      %c3_i32_49 = arith.constant 3 : i32
      %c3_i32_50 = arith.constant 3 : i32
      %c0_i32_51 = arith.constant 0 : i32
      %c7_i32_52 = arith.constant 7 : i32
      %c0_i32_53 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_46, %c1_i32_47, %c336_i32_48, %c3_i32_49, %c3_i32_50, %c0_i32_51, %c7_i32_52, %c0_i32_53) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %c0_54 = arith.constant 0 : index
      %c6_55 = arith.constant 6 : index
      %c1_56 = arith.constant 1 : index
      %c4_57 = arith.constant 4 : index
      %c4_58 = arith.constant 4 : index
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_59 = arith.constant 14 : i32
      %c1_i32_60 = arith.constant 1 : i32
      %c336_i32_61 = arith.constant 336 : i32
      %c3_i32_62 = arith.constant 3 : i32
      %c3_i32_63 = arith.constant 3 : i32
      %c1_i32_64 = arith.constant 1 : i32
      %c7_i32_65 = arith.constant 7 : i32
      %c0_i32_66 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_59, %c1_i32_60, %c336_i32_61, %c3_i32_62, %c3_i32_63, %c1_i32_64, %c7_i32_65, %c0_i32_66) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_67 = arith.constant 14 : i32
      %c1_i32_68 = arith.constant 1 : i32
      %c336_i32_69 = arith.constant 336 : i32
      %c3_i32_70 = arith.constant 3 : i32
      %c3_i32_71 = arith.constant 3 : i32
      %c1_i32_72 = arith.constant 1 : i32
      %c7_i32_73 = arith.constant 7 : i32
      %c0_i32_74 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_67, %c1_i32_68, %c336_i32_69, %c3_i32_70, %c3_i32_71, %c1_i32_72, %c7_i32_73, %c0_i32_74) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_75 = arith.constant 14 : i32
      %c1_i32_76 = arith.constant 1 : i32
      %c336_i32_77 = arith.constant 336 : i32
      %c3_i32_78 = arith.constant 3 : i32
      %c3_i32_79 = arith.constant 3 : i32
      %c1_i32_80 = arith.constant 1 : i32
      %c7_i32_81 = arith.constant 7 : i32
      %c0_i32_82 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_75, %c1_i32_76, %c336_i32_77, %c3_i32_78, %c3_i32_79, %c1_i32_80, %c7_i32_81, %c0_i32_82) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_83 = arith.constant 14 : i32
      %c1_i32_84 = arith.constant 1 : i32
      %c336_i32_85 = arith.constant 336 : i32
      %c3_i32_86 = arith.constant 3 : i32
      %c3_i32_87 = arith.constant 3 : i32
      %c1_i32_88 = arith.constant 1 : i32
      %c7_i32_89 = arith.constant 7 : i32
      %c0_i32_90 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_83, %c1_i32_84, %c336_i32_85, %c3_i32_86, %c3_i32_87, %c1_i32_88, %c7_i32_89, %c0_i32_90) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      cf.br ^bb6(%c4_57 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c6_55 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_91 = arith.constant 14 : i32
      %c1_i32_92 = arith.constant 1 : i32
      %c336_i32_93 = arith.constant 336 : i32
      %c3_i32_94 = arith.constant 3 : i32
      %c3_i32_95 = arith.constant 3 : i32
      %c1_i32_96 = arith.constant 1 : i32
      %c7_i32_97 = arith.constant 7 : i32
      %c0_i32_98 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_91, %c1_i32_92, %c336_i32_93, %c3_i32_94, %c3_i32_95, %c1_i32_96, %c7_i32_97, %c0_i32_98) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      %7 = arith.addi %5, %c1_56 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_99 = arith.constant 14 : i32
      %c1_i32_100 = arith.constant 1 : i32
      %c336_i32_101 = arith.constant 336 : i32
      %c3_i32_102 = arith.constant 3 : i32
      %c3_i32_103 = arith.constant 3 : i32
      %c0_i32_104 = arith.constant 0 : i32
      %c7_i32_105 = arith.constant 7 : i32
      %c0_i32_106 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_99, %c1_i32_100, %c336_i32_101, %c3_i32_102, %c3_i32_103, %c0_i32_104, %c7_i32_105, %c0_i32_106) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %c0_107 = arith.constant 0 : index
      %c6_108 = arith.constant 6 : index
      %c1_109 = arith.constant 1 : index
      %c4_110 = arith.constant 4 : index
      %c4_111 = arith.constant 4 : index
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_112 = arith.constant 14 : i32
      %c1_i32_113 = arith.constant 1 : i32
      %c336_i32_114 = arith.constant 336 : i32
      %c3_i32_115 = arith.constant 3 : i32
      %c3_i32_116 = arith.constant 3 : i32
      %c1_i32_117 = arith.constant 1 : i32
      %c7_i32_118 = arith.constant 7 : i32
      %c0_i32_119 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_112, %c1_i32_113, %c336_i32_114, %c3_i32_115, %c3_i32_116, %c1_i32_117, %c7_i32_118, %c0_i32_119) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_120 = arith.constant 14 : i32
      %c1_i32_121 = arith.constant 1 : i32
      %c336_i32_122 = arith.constant 336 : i32
      %c3_i32_123 = arith.constant 3 : i32
      %c3_i32_124 = arith.constant 3 : i32
      %c1_i32_125 = arith.constant 1 : i32
      %c7_i32_126 = arith.constant 7 : i32
      %c0_i32_127 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_120, %c1_i32_121, %c336_i32_122, %c3_i32_123, %c3_i32_124, %c1_i32_125, %c7_i32_126, %c0_i32_127) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_128 = arith.constant 14 : i32
      %c1_i32_129 = arith.constant 1 : i32
      %c336_i32_130 = arith.constant 336 : i32
      %c3_i32_131 = arith.constant 3 : i32
      %c3_i32_132 = arith.constant 3 : i32
      %c1_i32_133 = arith.constant 1 : i32
      %c7_i32_134 = arith.constant 7 : i32
      %c0_i32_135 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_128, %c1_i32_129, %c336_i32_130, %c3_i32_131, %c3_i32_132, %c1_i32_133, %c7_i32_134, %c0_i32_135) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_136 = arith.constant 14 : i32
      %c1_i32_137 = arith.constant 1 : i32
      %c336_i32_138 = arith.constant 336 : i32
      %c3_i32_139 = arith.constant 3 : i32
      %c3_i32_140 = arith.constant 3 : i32
      %c1_i32_141 = arith.constant 1 : i32
      %c7_i32_142 = arith.constant 7 : i32
      %c0_i32_143 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_136, %c1_i32_137, %c336_i32_138, %c3_i32_139, %c3_i32_140, %c1_i32_141, %c7_i32_142, %c0_i32_143) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      cf.br ^bb9(%c4_110 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c6_108 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_144 = arith.constant 14 : i32
      %c1_i32_145 = arith.constant 1 : i32
      %c336_i32_146 = arith.constant 336 : i32
      %c3_i32_147 = arith.constant 3 : i32
      %c3_i32_148 = arith.constant 3 : i32
      %c1_i32_149 = arith.constant 1 : i32
      %c7_i32_150 = arith.constant 7 : i32
      %c0_i32_151 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_144, %c1_i32_145, %c336_i32_146, %c3_i32_147, %c3_i32_148, %c1_i32_149, %c7_i32_150, %c0_i32_151) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      %10 = arith.addi %8, %c1_109 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_152 = arith.constant 14 : i32
      %c1_i32_153 = arith.constant 1 : i32
      %c336_i32_154 = arith.constant 336 : i32
      %c3_i32_155 = arith.constant 3 : i32
      %c3_i32_156 = arith.constant 3 : i32
      %c0_i32_157 = arith.constant 0 : i32
      %c7_i32_158 = arith.constant 7 : i32
      %c0_i32_159 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_152, %c1_i32_153, %c336_i32_154, %c3_i32_155, %c3_i32_156, %c0_i32_157, %c7_i32_158, %c0_i32_159) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %c0_160 = arith.constant 0 : index
      %c6_161 = arith.constant 6 : index
      %c1_162 = arith.constant 1 : index
      %c4_163 = arith.constant 4 : index
      %c4_164 = arith.constant 4 : index
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_165 = arith.constant 14 : i32
      %c1_i32_166 = arith.constant 1 : i32
      %c336_i32_167 = arith.constant 336 : i32
      %c3_i32_168 = arith.constant 3 : i32
      %c3_i32_169 = arith.constant 3 : i32
      %c1_i32_170 = arith.constant 1 : i32
      %c7_i32_171 = arith.constant 7 : i32
      %c0_i32_172 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_165, %c1_i32_166, %c336_i32_167, %c3_i32_168, %c3_i32_169, %c1_i32_170, %c7_i32_171, %c0_i32_172) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_173 = arith.constant 14 : i32
      %c1_i32_174 = arith.constant 1 : i32
      %c336_i32_175 = arith.constant 336 : i32
      %c3_i32_176 = arith.constant 3 : i32
      %c3_i32_177 = arith.constant 3 : i32
      %c1_i32_178 = arith.constant 1 : i32
      %c7_i32_179 = arith.constant 7 : i32
      %c0_i32_180 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_173, %c1_i32_174, %c336_i32_175, %c3_i32_176, %c3_i32_177, %c1_i32_178, %c7_i32_179, %c0_i32_180) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_181 = arith.constant 14 : i32
      %c1_i32_182 = arith.constant 1 : i32
      %c336_i32_183 = arith.constant 336 : i32
      %c3_i32_184 = arith.constant 3 : i32
      %c3_i32_185 = arith.constant 3 : i32
      %c1_i32_186 = arith.constant 1 : i32
      %c7_i32_187 = arith.constant 7 : i32
      %c0_i32_188 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_181, %c1_i32_182, %c336_i32_183, %c3_i32_184, %c3_i32_185, %c1_i32_186, %c7_i32_187, %c0_i32_188) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_189 = arith.constant 14 : i32
      %c1_i32_190 = arith.constant 1 : i32
      %c336_i32_191 = arith.constant 336 : i32
      %c3_i32_192 = arith.constant 3 : i32
      %c3_i32_193 = arith.constant 3 : i32
      %c1_i32_194 = arith.constant 1 : i32
      %c7_i32_195 = arith.constant 7 : i32
      %c0_i32_196 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_189, %c1_i32_190, %c336_i32_191, %c3_i32_192, %c3_i32_193, %c1_i32_194, %c7_i32_195, %c0_i32_196) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      cf.br ^bb12(%c4_163 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c6_161 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_197 = arith.constant 14 : i32
      %c1_i32_198 = arith.constant 1 : i32
      %c336_i32_199 = arith.constant 336 : i32
      %c3_i32_200 = arith.constant 3 : i32
      %c3_i32_201 = arith.constant 3 : i32
      %c1_i32_202 = arith.constant 1 : i32
      %c7_i32_203 = arith.constant 7 : i32
      %c0_i32_204 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_197, %c1_i32_198, %c336_i32_199, %c3_i32_200, %c3_i32_201, %c1_i32_202, %c7_i32_203, %c0_i32_204) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      %13 = arith.addi %11, %c1_162 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      cf.br ^bb16(%c9223372036854775804 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb20
      %16 = arith.cmpi slt, %15, %c9223372036854775807 : index
      cf.cond_br %16, ^bb17, ^bb21
    ^bb17:  // pred: ^bb16
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      %c14_i32_205 = arith.constant 14 : i32
      %c1_i32_206 = arith.constant 1 : i32
      %c336_i32_207 = arith.constant 336 : i32
      %c3_i32_208 = arith.constant 3 : i32
      %c3_i32_209 = arith.constant 3 : i32
      %c0_i32_210 = arith.constant 0 : i32
      %c7_i32_211 = arith.constant 7 : i32
      %c0_i32_212 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_205, %c1_i32_206, %c336_i32_207, %c3_i32_208, %c3_i32_209, %c0_i32_210, %c7_i32_211, %c0_i32_212) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %c0_213 = arith.constant 0 : index
      %c6_214 = arith.constant 6 : index
      %c1_215 = arith.constant 1 : index
      %c4_216 = arith.constant 4 : index
      %c4_217 = arith.constant 4 : index
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_218 = arith.constant 14 : i32
      %c1_i32_219 = arith.constant 1 : i32
      %c336_i32_220 = arith.constant 336 : i32
      %c3_i32_221 = arith.constant 3 : i32
      %c3_i32_222 = arith.constant 3 : i32
      %c1_i32_223 = arith.constant 1 : i32
      %c7_i32_224 = arith.constant 7 : i32
      %c0_i32_225 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_218, %c1_i32_219, %c336_i32_220, %c3_i32_221, %c3_i32_222, %c1_i32_223, %c7_i32_224, %c0_i32_225) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_226 = arith.constant 14 : i32
      %c1_i32_227 = arith.constant 1 : i32
      %c336_i32_228 = arith.constant 336 : i32
      %c3_i32_229 = arith.constant 3 : i32
      %c3_i32_230 = arith.constant 3 : i32
      %c1_i32_231 = arith.constant 1 : i32
      %c7_i32_232 = arith.constant 7 : i32
      %c0_i32_233 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_226, %c1_i32_227, %c336_i32_228, %c3_i32_229, %c3_i32_230, %c1_i32_231, %c7_i32_232, %c0_i32_233) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_234 = arith.constant 14 : i32
      %c1_i32_235 = arith.constant 1 : i32
      %c336_i32_236 = arith.constant 336 : i32
      %c3_i32_237 = arith.constant 3 : i32
      %c3_i32_238 = arith.constant 3 : i32
      %c1_i32_239 = arith.constant 1 : i32
      %c7_i32_240 = arith.constant 7 : i32
      %c0_i32_241 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_234, %c1_i32_235, %c336_i32_236, %c3_i32_237, %c3_i32_238, %c1_i32_239, %c7_i32_240, %c0_i32_241) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_242 = arith.constant 14 : i32
      %c1_i32_243 = arith.constant 1 : i32
      %c336_i32_244 = arith.constant 336 : i32
      %c3_i32_245 = arith.constant 3 : i32
      %c3_i32_246 = arith.constant 3 : i32
      %c1_i32_247 = arith.constant 1 : i32
      %c7_i32_248 = arith.constant 7 : i32
      %c0_i32_249 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_2, %OF_b12_act_layer1_layer2_cons_buff_3, %OF_b12_act_layer1_layer2_cons_buff_0, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_1, %c14_i32_242, %c1_i32_243, %c336_i32_244, %c3_i32_245, %c3_i32_246, %c1_i32_247, %c7_i32_248, %c0_i32_249) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      cf.br ^bb18(%c4_216 : index)
    ^bb18(%17: index):  // 2 preds: ^bb17, ^bb19
      %18 = arith.cmpi slt, %17, %c6_214 : index
      cf.cond_br %18, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, AcquireGreaterEqual, 1)
      %c14_i32_250 = arith.constant 14 : i32
      %c1_i32_251 = arith.constant 1 : i32
      %c336_i32_252 = arith.constant 336 : i32
      %c3_i32_253 = arith.constant 3 : i32
      %c3_i32_254 = arith.constant 3 : i32
      %c1_i32_255 = arith.constant 1 : i32
      %c7_i32_256 = arith.constant 7 : i32
      %c0_i32_257 = arith.constant 0 : i32
      func.call @bn12_conv2dk3_dw_stride2_relu_ui8_ui8(%OF_b12_act_layer1_layer2_cons_buff_0, %OF_b12_act_layer1_layer2_cons_buff_1, %OF_b12_act_layer1_layer2_cons_buff_2, %weightsInBN12_layer2_cons_buff_0, %OF_b12_act_layer2_layer3_buff_0, %c14_i32_250, %c1_i32_251, %c336_i32_252, %c3_i32_253, %c3_i32_254, %c1_i32_255, %c7_i32_256, %c0_i32_257) : (memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>, memref<3024xi8>, memref<7x1x336xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 2)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, Release, 1)
      %19 = arith.addi %17, %c1_215 : index
      cf.br ^bb18(%19 : index)
    ^bb20:  // pred: ^bb18
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, Release, 1)
      %20 = arith.addi %15, %c1 : index
      cf.br ^bb16(%20 : index)
    ^bb21:  // pred: ^bb16
      aie.use_lock(%weightsInBN12_layer2_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn12_conv2dk3_dw_stride2.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      aie.use_lock(%weightsInBN12_layer3_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1_1 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      %c2_2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c6 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32 = arith.constant 7 : i32
      %c336_i32 = arith.constant 336 : i32
      %c80_i32 = arith.constant 80 : i32
      %c10_i32 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_0, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_0, %c7_i32, %c336_i32, %c80_i32, %c10_i32) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_3 = arith.constant 7 : i32
      %c336_i32_4 = arith.constant 336 : i32
      %c80_i32_5 = arith.constant 80 : i32
      %c10_i32_6 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_1, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_1, %c7_i32_3, %c336_i32_4, %c80_i32_5, %c10_i32_6) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      %4 = arith.addi %2, %c2_2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_7 = arith.constant 7 : i32
      %c336_i32_8 = arith.constant 336 : i32
      %c80_i32_9 = arith.constant 80 : i32
      %c10_i32_10 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_0, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_0, %c7_i32_7, %c336_i32_8, %c80_i32_9, %c10_i32_10) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      %c0_11 = arith.constant 0 : index
      %c7_12 = arith.constant 7 : index
      %c1_13 = arith.constant 1 : index
      %c6_14 = arith.constant 6 : index
      %c2_15 = arith.constant 2 : index
      cf.br ^bb6(%c0_11 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c6_14 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_16 = arith.constant 7 : i32
      %c336_i32_17 = arith.constant 336 : i32
      %c80_i32_18 = arith.constant 80 : i32
      %c10_i32_19 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_1, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_1, %c7_i32_16, %c336_i32_17, %c80_i32_18, %c10_i32_19) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_20 = arith.constant 7 : i32
      %c336_i32_21 = arith.constant 336 : i32
      %c80_i32_22 = arith.constant 80 : i32
      %c10_i32_23 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_0, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_0, %c7_i32_20, %c336_i32_21, %c80_i32_22, %c10_i32_23) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      %7 = arith.addi %5, %c2_15 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_24 = arith.constant 7 : i32
      %c336_i32_25 = arith.constant 336 : i32
      %c80_i32_26 = arith.constant 80 : i32
      %c10_i32_27 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_1, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_1, %c7_i32_24, %c336_i32_25, %c80_i32_26, %c10_i32_27) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      %8 = arith.addi %0, %c2 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      %c0_28 = arith.constant 0 : index
      %c7_29 = arith.constant 7 : index
      %c1_30 = arith.constant 1 : index
      %c6_31 = arith.constant 6 : index
      %c2_32 = arith.constant 2 : index
      cf.br ^bb10(%c0_28 : index)
    ^bb10(%9: index):  // 2 preds: ^bb9, ^bb11
      %10 = arith.cmpi slt, %9, %c6_31 : index
      cf.cond_br %10, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_33 = arith.constant 7 : i32
      %c336_i32_34 = arith.constant 336 : i32
      %c80_i32_35 = arith.constant 80 : i32
      %c10_i32_36 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_0, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_0, %c7_i32_33, %c336_i32_34, %c80_i32_35, %c10_i32_36) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_37 = arith.constant 7 : i32
      %c336_i32_38 = arith.constant 336 : i32
      %c80_i32_39 = arith.constant 80 : i32
      %c10_i32_40 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_1, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_1, %c7_i32_37, %c336_i32_38, %c80_i32_39, %c10_i32_40) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      %11 = arith.addi %9, %c2_32 : index
      cf.br ^bb10(%11 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%OF_b12_act_layer2_layer3_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_out_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_41 = arith.constant 7 : i32
      %c336_i32_42 = arith.constant 336 : i32
      %c80_i32_43 = arith.constant 80 : i32
      %c10_i32_44 = arith.constant 10 : i32
      func.call @bn12_conv2dk1_ui8_i8(%OF_b12_act_layer2_layer3_buff_0, %weightsInBN12_layer3_cons_buff_0, %act_out_buff_0, %c7_i32_41, %c336_i32_42, %c80_i32_43, %c10_i32_44) : (memref<7x1x336xui8>, memref<26880xi8>, memref<7x1x80xi8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%OF_b12_act_layer2_layer3_prod_lock, Release, 1)
      aie.use_lock(%act_out_cons_lock, Release, 1)
      aie.use_lock(%weightsInBN12_layer3_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "bn12_conv2dk1_ui8.o"}
    aie.shim_dma_allocation @act_in(MM2S, 0, 0)
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
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%act_in_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_in_cons_buff_0 : memref<14x1x80xi8>, 0, 1120) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_in_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_in_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_in_cons_buff_1 : memref<14x1x80xi8>, 0, 1120) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%act_in_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%weightsInBN10_layer1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN10_layer1_cons_buff_0 : memref<38400xi8>, 0, 38400) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%weightsInBN10_layer1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_act_layer1_layer2_buff_0 : memref<14x1x480xui8>, 0, 6720) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_b10_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_act_layer1_layer2_buff_1 : memref<14x1x480xui8>, 0, 6720) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_b10_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @act_out(S2MM, 0, 1)
    aie.shim_dma_allocation @wts_b10_L3L2(MM2S, 1, 0)
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%act_out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_out_buff_0 : memref<7x1x80xi8>, 0, 560) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_out_prod_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_out_buff_1 : memref<7x1x80xi8>, 0, 560) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%act_out_prod_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%weightsInBN12_layer3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN12_layer3_cons_buff_0 : memref<26880xi8>, 0, 26880) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%weightsInBN12_layer3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
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
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_2 : memref<14x1x112xi8>, 0, 1568) {bd_id = 27 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_3 : memref<14x1x112xi8>, 0, 1568) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_4 : memref<14x1x112xi8>, 0, 1568) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_5 : memref<14x1x112xi8>, 0, 1568) {bd_id = 30 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb9
    ^bb15:  // pred: ^bb8
      %5 = aie.dma_start(MM2S, 3, ^bb16, ^bb22)
    ^bb16:  // 2 preds: ^bb15, ^bb21
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_2 : memref<14x1x112xi8>, 0, 1568) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_3 : memref<14x1x112xi8>, 0, 1568) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_4 : memref<14x1x112xi8>, 0, 1568) {bd_id = 35 : i32, next_bd_id = 36 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_1_cons_buff_5 : memref<14x1x112xi8>, 0, 1568) {bd_id = 36 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb16
    ^bb22:  // pred: ^bb15
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN10_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN10_layer2_cons_buff_0 : memref<4320xi8>, 0, 4320) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN10_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_act_layer1_layer2_cons_buff_0 : memref<14x1x480xui8>, 0, 6720) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_act_layer1_layer2_cons_buff_1 : memref<14x1x480xui8>, 0, 6720) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_act_layer1_layer2_cons_buff_2 : memref<14x1x480xui8>, 0, 6720) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_act_layer1_layer2_cons_buff_3 : memref<14x1x480xui8>, 0, 6720) {bd_id = 4 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b10_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @wts_b11_L3L2(MM2S, 0, 1)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN10_layer3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN10_layer3_cons_buff_0 : memref<53760xi8>, 0, 53760) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN10_layer3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
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
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN11_layer1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN11_layer1_cons_buff_0 : memref<37632xi8>, 0, 37632) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN11_layer1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_0_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b10_layer3_bn_11_layer1_0_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b10_layer3_bn_11_layer1_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_act_layer1_layer2_buff_0 : memref<14x1x336xui8>, 0, 4704) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_b11_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_act_layer1_layer2_buff_1 : memref<14x1x336xui8>, 0, 4704) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_b11_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN11_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN11_layer2_cons_buff_0 : memref<3024xi8>, 0, 3024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN11_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_act_layer1_layer2_cons_buff_0 : memref<14x1x336xui8>, 0, 4704) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_act_layer1_layer2_cons_buff_1 : memref<14x1x336xui8>, 0, 4704) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_act_layer1_layer2_cons_buff_2 : memref<14x1x336xui8>, 0, 4704) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_act_layer1_layer2_cons_buff_3 : memref<14x1x336xui8>, 0, 4704) {bd_id = 4 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b11_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @wts_b12_L3L2(MM2S, 1, 1)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN11_layer3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN11_layer3_cons_buff_0 : memref<37632xi8>, 0, 37632) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN11_layer3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%OF_b11_skip_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_skip_cons_buff_0 : memref<14x1x112xi8>, 0, 1568) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b11_skip_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b11_skip_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b11_skip_cons_buff_1 : memref<14x1x112xi8>, 0, 1568) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b11_skip_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_b12_L3L2_cons_prod_lock, AcquireGreaterEqual, 3)
      aie.dma_bd(%wts_b12_L3L2_cons_buff_0 : memref<67536xi8>, 0, 67536) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%wts_b12_L3L2_cons_cons_lock, Release, 3)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%wts_b12_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b12_L3L2_cons_buff_0 : memref<67536xi8>, 0, 37632) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%wts_b12_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%wts_b12_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b12_L3L2_cons_buff_0 : memref<67536xi8>, 37632, 3024) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%wts_b12_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 2, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%wts_b12_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_b12_L3L2_cons_buff_0 : memref<67536xi8>, 40656, 26880) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%wts_b12_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN12_layer1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN12_layer1_cons_buff_0 : memref<37632xi8>, 0, 37632) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN12_layer1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b12_act_layer1_layer2_buff_0 : memref<14x1x336xui8>, 0, 4704) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b12_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b12_act_layer1_layer2_buff_1 : memref<14x1x336xui8>, 0, 4704) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b12_act_layer1_layer2_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%weightsInBN12_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%weightsInBN12_layer2_cons_buff_0 : memref<3024xi8>, 0, 3024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%weightsInBN12_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b12_act_layer1_layer2_cons_buff_0 : memref<14x1x336xui8>, 0, 4704) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b12_act_layer1_layer2_cons_buff_1 : memref<14x1x336xui8>, 0, 4704) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b12_act_layer1_layer2_cons_buff_2 : memref<14x1x336xui8>, 0, 4704) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_b12_act_layer1_layer2_cons_buff_3 : memref<14x1x336xui8>, 0, 4704) {bd_id = 4 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_b12_act_layer1_layer2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
  }
}
