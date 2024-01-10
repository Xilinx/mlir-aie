module {
  aie.device(ipu) {
    memref.global "public" @outOFL2L3_cons : memref<32x1x256xui8>
    memref.global "public" @outOFL2L3 : memref<32x1x256xui8>
    memref.global "public" @act_4_5 : memref<32x1x32xui8>
    memref.global "public" @act_3_5 : memref<32x1x32xui8>
    memref.global "public" @act_2_3_4_0_cons : memref<32x1x64xui8>
    memref.global "public" @act_2_3_4_1_cons : memref<32x1x64xui8>
    memref.global "public" @act_2_3_4 : memref<32x1x64xui8>
    memref.global "public" @wts_buf_02_cons : memref<32768xi8>
    memref.global "public" @wts_buf_02 : memref<32768xi8>
    memref.global "public" @wts_buf_01_0_cons : memref<36864xi8>
    memref.global "public" @wts_buf_01_1_cons : memref<36864xi8>
    memref.global "public" @wts_buf_01 : memref<36864xi8>
    memref.global "public" @wts_buf_00_cons : memref<4096xi8>
    memref.global "public" @wts_buf_00 : memref<4096xi8>
    memref.global "public" @inOF_wts_0_L3L2_cons : memref<73728xi8>
    memref.global "public" @inOF_wts_0_L3L2 : memref<73728xi8>
    memref.global "public" @skip_buf_cons : memref<32x1x64xi8>
    memref.global "public" @skip_buf : memref<32x1x64xi8>
    memref.global "public" @inOF_act_L3L2_0_cons : memref<32x1x64xi8>
    memref.global "public" @inOF_act_L3L2_1_cons : memref<32x1x64xi8>
    memref.global "public" @inOF_act_L3L2 : memref<32x1x64xi8>
    %tile_0_0 = aie.tile(0, 0)
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
    }
    %tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
    }
    %tile_0_5 = aie.tile(0, 5)
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
    }
    %outOFL2L3_cons_prod_lock = aie.lock(%tile_0_0, 4) {init = 0 : i32, sym_name = "outOFL2L3_cons_prod_lock"}
    %outOFL2L3_cons_cons_lock = aie.lock(%tile_0_0, 5) {init = 0 : i32, sym_name = "outOFL2L3_cons_cons_lock"}
    %outOFL2L3_buff_0 = aie.buffer(%tile_0_4) {address = 33792 : i32, sym_name = "outOFL2L3_buff_0"} : memref<32x1x256xui8>
    %outOFL2L3_buff_1 = aie.buffer(%tile_0_4) {address = 41984 : i32, sym_name = "outOFL2L3_buff_1"} : memref<32x1x256xui8>
    %outOFL2L3_prod_lock = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "outOFL2L3_prod_lock"}
    %outOFL2L3_cons_lock = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "outOFL2L3_cons_lock"}
    %act_4_5_buff_0 = aie.buffer(%tile_0_5) {address = 46080 : i32, sym_name = "act_4_5_buff_0"} : memref<32x1x32xui8>
    %act_4_5_buff_1 = aie.buffer(%tile_0_5) {address = 47104 : i32, sym_name = "act_4_5_buff_1"} : memref<32x1x32xui8>
    %act_4_5_prod_lock = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "act_4_5_prod_lock"}
    %act_4_5_cons_lock = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "act_4_5_cons_lock"}
    %act_3_5_buff_0 = aie.buffer(%tile_0_3) {address = 46080 : i32, sym_name = "act_3_5_buff_0"} : memref<32x1x32xui8>
    %act_3_5_buff_1 = aie.buffer(%tile_0_3) {address = 47104 : i32, sym_name = "act_3_5_buff_1"} : memref<32x1x32xui8>
    %act_3_5_prod_lock = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "act_3_5_prod_lock"}
    %act_3_5_cons_lock = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "act_3_5_cons_lock"}
    %act_2_3_4_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 37888 : i32, sym_name = "act_2_3_4_0_cons_buff_0"} : memref<32x1x64xui8>
    %act_2_3_4_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 39936 : i32, sym_name = "act_2_3_4_0_cons_buff_1"} : memref<32x1x64xui8>
    %act_2_3_4_0_cons_buff_2 = aie.buffer(%tile_0_3) {address = 41984 : i32, sym_name = "act_2_3_4_0_cons_buff_2"} : memref<32x1x64xui8>
    %act_2_3_4_0_cons_buff_3 = aie.buffer(%tile_0_3) {address = 44032 : i32, sym_name = "act_2_3_4_0_cons_buff_3"} : memref<32x1x64xui8>
    %act_2_3_4_0_cons_prod_lock = aie.lock(%tile_0_3, 2) {init = 4 : i32, sym_name = "act_2_3_4_0_cons_prod_lock"}
    %act_2_3_4_0_cons_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "act_2_3_4_0_cons_cons_lock"}
    %act_2_3_4_1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 37888 : i32, sym_name = "act_2_3_4_1_cons_buff_0"} : memref<32x1x64xui8>
    %act_2_3_4_1_cons_buff_1 = aie.buffer(%tile_0_5) {address = 39936 : i32, sym_name = "act_2_3_4_1_cons_buff_1"} : memref<32x1x64xui8>
    %act_2_3_4_1_cons_buff_2 = aie.buffer(%tile_0_5) {address = 41984 : i32, sym_name = "act_2_3_4_1_cons_buff_2"} : memref<32x1x64xui8>
    %act_2_3_4_1_cons_buff_3 = aie.buffer(%tile_0_5) {address = 44032 : i32, sym_name = "act_2_3_4_1_cons_buff_3"} : memref<32x1x64xui8>
    %act_2_3_4_1_cons_prod_lock = aie.lock(%tile_0_5, 2) {init = 4 : i32, sym_name = "act_2_3_4_1_cons_prod_lock"}
    %act_2_3_4_1_cons_cons_lock = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "act_2_3_4_1_cons_cons_lock"}
    %act_2_3_4_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, sym_name = "act_2_3_4_buff_0"} : memref<32x1x64xui8>
    %act_2_3_4_buff_1 = aie.buffer(%tile_0_2) {address = 7168 : i32, sym_name = "act_2_3_4_buff_1"} : memref<32x1x64xui8>
    %act_2_3_4_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "act_2_3_4_prod_lock"}
    %act_2_3_4_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "act_2_3_4_cons_lock"}
    %wts_buf_02_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "wts_buf_02_cons_buff_0"} : memref<32768xi8>
    %wts_buf_02_cons_prod_lock = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "wts_buf_02_cons_prod_lock"}
    %wts_buf_02_cons_cons_lock = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "wts_buf_02_cons_cons_lock"}
    %wts_buf_01_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "wts_buf_01_0_cons_buff_0"} : memref<36864xi8>
    %wts_buf_01_0_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "wts_buf_01_0_cons_prod_lock"}
    %wts_buf_01_0_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "wts_buf_01_0_cons_cons_lock"}
    %wts_buf_01_1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "wts_buf_01_1_cons_buff_0"} : memref<36864xi8>
    %wts_buf_01_1_cons_prod_lock = aie.lock(%tile_0_5, 0) {init = 1 : i32, sym_name = "wts_buf_01_1_cons_prod_lock"}
    %wts_buf_01_1_cons_cons_lock = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "wts_buf_01_1_cons_cons_lock"}
    %wts_buf_00_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "wts_buf_00_cons_buff_0"} : memref<4096xi8>
    %wts_buf_00_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "wts_buf_00_cons_prod_lock"}
    %wts_buf_00_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "wts_buf_00_cons_cons_lock"}
    %inOF_wts_0_L3L2_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "inOF_wts_0_L3L2_cons_buff_0"} : memref<73728xi8>
    %inOF_wts_0_L3L2_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 3 : i32, sym_name = "inOF_wts_0_L3L2_cons_prod_lock"}
    %inOF_wts_0_L3L2_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "inOF_wts_0_L3L2_cons_cons_lock"}
    %inOF_wts_0_L3L2_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "inOF_wts_0_L3L2_prod_lock"}
    %inOF_wts_0_L3L2_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "inOF_wts_0_L3L2_cons_lock"}
    %skip_buf_cons_buff_0 = aie.buffer(%tile_0_4) {address = 50176 : i32, sym_name = "skip_buf_cons_buff_0"} : memref<32x1x64xi8>
    %skip_buf_cons_buff_1 = aie.buffer(%tile_0_4) {address = 52224 : i32, sym_name = "skip_buf_cons_buff_1"} : memref<32x1x64xi8>
    %skip_buf_cons_prod_lock = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "skip_buf_cons_prod_lock"}
    %skip_buf_cons_cons_lock = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "skip_buf_cons_cons_lock"}
    %inOF_act_L3L2_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "inOF_act_L3L2_0_cons_buff_0"} : memref<32x1x64xi8>
    %inOF_act_L3L2_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 11264 : i32, sym_name = "inOF_act_L3L2_0_cons_buff_1"} : memref<32x1x64xi8>
    %inOF_act_L3L2_0_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "inOF_act_L3L2_0_cons_prod_lock"}
    %inOF_act_L3L2_0_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_0_cons_cons_lock"}
    %inOF_act_L3L2_1_cons_buff_0 = aie.buffer(%tile_0_1) {address = 73728 : i32, sym_name = "inOF_act_L3L2_1_cons_buff_0"} : memref<32x1x64xi8>
    %inOF_act_L3L2_1_cons_buff_1 = aie.buffer(%tile_0_1) {address = 75776 : i32, sym_name = "inOF_act_L3L2_1_cons_buff_1"} : memref<32x1x64xi8>
    %inOF_act_L3L2_1_cons_buff_2 = aie.buffer(%tile_0_1) {address = 77824 : i32, sym_name = "inOF_act_L3L2_1_cons_buff_2"} : memref<32x1x64xi8>
    %inOF_act_L3L2_1_cons_buff_3 = aie.buffer(%tile_0_1) {address = 79872 : i32, sym_name = "inOF_act_L3L2_1_cons_buff_3"} : memref<32x1x64xi8>
    %inOF_act_L3L2_1_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 4 : i32, sym_name = "inOF_act_L3L2_1_cons_prod_lock"}
    %inOF_act_L3L2_1_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_1_cons_cons_lock"}
    %inOF_act_L3L2_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "inOF_act_L3L2_prod_lock"}
    %inOF_act_L3L2_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_cons_lock"}
    %rtp2 = aie.buffer(%tile_0_2) {address = 13312 : i32, sym_name = "rtp2"} : memref<16xi32>
    %rtp3 = aie.buffer(%tile_0_3) {address = 48128 : i32, sym_name = "rtp3"} : memref<16xi32>
    %rtp4 = aie.buffer(%tile_0_5) {address = 48128 : i32, sym_name = "rtp4"} : memref<16xi32>
    %rtp5 = aie.buffer(%tile_0_4) {address = 54272 : i32, sym_name = "rtp5"} : memref<16xi32>
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 3, %tile_0_4, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_0, DMA : 0)
    func.func private @conv2dk1(memref<32x1x64xi8>, memref<4096xi8>, memref<32x1x64xui8>, i32, i32, i32, i32)
    func.func private @conv2dk3(memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @conv2dk1_skip(memref<32x1x32xui8>, memref<32x1x32xui8>, memref<32768xi8>, memref<32x1x256xui8>, memref<32x1x64xi8>, i32, i32, i32, i32, i32, i32, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32_i32 = arith.constant 32 : i32
      %c32 = arith.constant 32 : index
      %c64_i32 = arith.constant 64 : i32
      %c64_i32_0 = arith.constant 64 : i32
      %c4294967295 = arith.constant 4294967295 : index
      %c1_1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%wts_buf_00_cons_cons_lock, AcquireGreaterEqual, 1)
      %2 = memref.load %rtp2[%c0] : memref<16xi32>
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0 : index)
    ^bb3(%3: index):  // 2 preds: ^bb2, ^bb4
      %4 = arith.cmpi slt, %3, %c32 : index
      cf.cond_br %4, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk1(%inOF_act_L3L2_0_cons_buff_0, %wts_buf_00_cons_buff_0, %act_2_3_4_buff_0, %c32_i32, %c64_i32, %c64_i32_0, %2) : (memref<32x1x64xi8>, memref<4096xi8>, memref<32x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_cons_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk1(%inOF_act_L3L2_0_cons_buff_1, %wts_buf_00_cons_buff_0, %act_2_3_4_buff_1, %c32_i32, %c64_i32, %c64_i32_0, %2) : (memref<32x1x64xi8>, memref<4096xi8>, memref<32x1x64xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_cons_lock, Release, 1)
      %5 = arith.addi %3, %c2 : index
      cf.br ^bb3(%5 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%wts_buf_00_cons_prod_lock, Release, 1)
      %6 = arith.addi %0, %c1_1 : index
      cf.br ^bb1(%6 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "conv2dk1.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32_i32 = arith.constant 32 : i32
      %c30 = arith.constant 30 : index
      %c64_i32 = arith.constant 64 : i32
      %c32_i32_0 = arith.constant 32 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_1 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_2 = arith.constant 0 : i32
      %c11_i32 = arith.constant 11 : i32
      %c4294967295 = arith.constant 4294967295 : index
      %c4294967292 = arith.constant 4294967292 : index
      %c4 = arith.constant 4 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c4294967292 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28 = arith.constant 28 : index
      %c4_3 = arith.constant 4 : index
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c28 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %4 = arith.addi %2, %c4_3 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28_4 = arith.constant 28 : index
      %c4_5 = arith.constant 4 : index
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c28_4 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %7 = arith.addi %5, %c4_5 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28_6 = arith.constant 28 : index
      %c4_7 = arith.constant 4 : index
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c28_6 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %10 = arith.addi %8, %c4_7 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28_8 = arith.constant 28 : index
      %c4_9 = arith.constant 4 : index
      cf.br ^bb12(%c0 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c28_8 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %13 = arith.addi %11, %c4_9 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28_10 = arith.constant 28 : index
      %c4_11 = arith.constant 4 : index
      cf.br ^bb16(%c0 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb17
      %16 = arith.cmpi slt, %15, %c28_10 : index
      cf.cond_br %16, ^bb17, ^bb18
    ^bb17:  // pred: ^bb16
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %17 = arith.addi %15, %c4_11 : index
      cf.br ^bb16(%17 : index)
    ^bb18:  // pred: ^bb16
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28_12 = arith.constant 28 : index
      %c4_13 = arith.constant 4 : index
      cf.br ^bb19(%c0 : index)
    ^bb19(%18: index):  // 2 preds: ^bb18, ^bb20
      %19 = arith.cmpi slt, %18, %c28_12 : index
      cf.cond_br %19, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %20 = arith.addi %18, %c4_13 : index
      cf.br ^bb19(%20 : index)
    ^bb21:  // pred: ^bb19
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      %c28_14 = arith.constant 28 : index
      %c4_15 = arith.constant 4 : index
      cf.br ^bb22(%c0 : index)
    ^bb22(%21: index):  // 2 preds: ^bb21, ^bb23
      %22 = arith.cmpi slt, %21, %c28_14 : index
      cf.cond_br %22, ^bb23, ^bb24
    ^bb23:  // pred: ^bb22
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      %23 = arith.addi %21, %c4_15 : index
      cf.br ^bb22(%23 : index)
    ^bb24:  // pred: ^bb22
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_0, %act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_1, %act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_0_cons_buff_2, %act_2_3_4_0_cons_buff_3, %act_2_3_4_0_cons_buff_3, %wts_buf_01_0_cons_buff_0, %act_3_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_3_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32_i32 = arith.constant 32 : i32
      %c30 = arith.constant 30 : index
      %c64_i32 = arith.constant 64 : i32
      %c32_i32_0 = arith.constant 32 : i32
      %c3_i32 = arith.constant 3 : i32
      %c3_i32_1 = arith.constant 3 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c32_i32_2 = arith.constant 32 : i32
      %c4294967295 = arith.constant 4294967295 : index
      %c11_i32 = arith.constant 11 : i32
      %c4294967292 = arith.constant 4294967292 : index
      %c4 = arith.constant 4 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c4294967292 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28 = arith.constant 28 : index
      %c4_3 = arith.constant 4 : index
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c28 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %4 = arith.addi %2, %c4_3 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28_4 = arith.constant 28 : index
      %c4_5 = arith.constant 4 : index
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c28_4 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %7 = arith.addi %5, %c4_5 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28_6 = arith.constant 28 : index
      %c4_7 = arith.constant 4 : index
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c28_6 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %10 = arith.addi %8, %c4_7 : index
      cf.br ^bb9(%10 : index)
    ^bb11:  // pred: ^bb9
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28_8 = arith.constant 28 : index
      %c4_9 = arith.constant 4 : index
      cf.br ^bb12(%c0 : index)
    ^bb12(%11: index):  // 2 preds: ^bb11, ^bb13
      %12 = arith.cmpi slt, %11, %c28_8 : index
      cf.cond_br %12, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %13 = arith.addi %11, %c4_9 : index
      cf.br ^bb12(%13 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      %14 = arith.addi %0, %c4 : index
      cf.br ^bb1(%14 : index)
    ^bb15:  // pred: ^bb1
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28_10 = arith.constant 28 : index
      %c4_11 = arith.constant 4 : index
      cf.br ^bb16(%c0 : index)
    ^bb16(%15: index):  // 2 preds: ^bb15, ^bb17
      %16 = arith.cmpi slt, %15, %c28_10 : index
      cf.cond_br %16, ^bb17, ^bb18
    ^bb17:  // pred: ^bb16
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %17 = arith.addi %15, %c4_11 : index
      cf.br ^bb16(%17 : index)
    ^bb18:  // pred: ^bb16
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28_12 = arith.constant 28 : index
      %c4_13 = arith.constant 4 : index
      cf.br ^bb19(%c0 : index)
    ^bb19(%18: index):  // 2 preds: ^bb18, ^bb20
      %19 = arith.cmpi slt, %18, %c28_12 : index
      cf.cond_br %19, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %20 = arith.addi %18, %c4_13 : index
      cf.br ^bb19(%20 : index)
    ^bb21:  // pred: ^bb19
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 2)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      %c28_14 = arith.constant 28 : index
      %c4_15 = arith.constant 4 : index
      cf.br ^bb22(%c0 : index)
    ^bb22(%21: index):  // 2 preds: ^bb21, ^bb23
      %22 = arith.cmpi slt, %21, %c28_14 : index
      cf.cond_br %22, ^bb23, ^bb24
    ^bb23:  // pred: ^bb22
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      %23 = arith.addi %21, %c4_15 : index
      cf.br ^bb22(%23 : index)
    ^bb24:  // pred: ^bb22
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_0, %act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_1, %act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_0, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c1_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk3(%act_2_3_4_1_cons_buff_2, %act_2_3_4_1_cons_buff_3, %act_2_3_4_1_cons_buff_3, %wts_buf_01_1_cons_buff_0, %act_4_5_buff_1, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c2_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_4_5_cons_lock, Release, 1)
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, Release, 2)
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32_i32 = arith.constant 32 : i32
      %c32 = arith.constant 32 : index
      %c64_i32 = arith.constant 64 : i32
      %c256_i32 = arith.constant 256 : i32
      %c64_i32_0 = arith.constant 64 : i32
      %c4294967295 = arith.constant 4294967295 : index
      %c1_1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%wts_buf_02_cons_cons_lock, AcquireGreaterEqual, 1)
      %2 = memref.load %rtp5[%c0] : memref<16xi32>
      %3 = memref.load %rtp5[%c1] : memref<16xi32>
      %4 = memref.load %rtp5[%c2] : memref<16xi32>
      %c2_2 = arith.constant 2 : index
      cf.br ^bb3(%c0 : index)
    ^bb3(%5: index):  // 2 preds: ^bb2, ^bb4
      %6 = arith.cmpi slt, %5, %c32 : index
      cf.cond_br %6, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_3_5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%outOFL2L3_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%skip_buf_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk1_skip(%act_3_5_buff_0, %act_4_5_buff_0, %wts_buf_02_cons_buff_0, %outOFL2L3_buff_0, %skip_buf_cons_buff_0, %c32_i32, %c64_i32, %c256_i32, %c64_i32_0, %2, %3, %4) : (memref<32x1x32xui8>, memref<32x1x32xui8>, memref<32768xi8>, memref<32x1x256xui8>, memref<32x1x64xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%outOFL2L3_cons_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, Release, 1)
      aie.use_lock(%skip_buf_cons_prod_lock, Release, 1)
      aie.use_lock(%act_3_5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%act_4_5_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%outOFL2L3_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%skip_buf_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @conv2dk1_skip(%act_3_5_buff_1, %act_4_5_buff_1, %wts_buf_02_cons_buff_0, %outOFL2L3_buff_1, %skip_buf_cons_buff_1, %c32_i32, %c64_i32, %c256_i32, %c64_i32_0, %2, %3, %4) : (memref<32x1x32xui8>, memref<32x1x32xui8>, memref<32768xi8>, memref<32x1x256xui8>, memref<32x1x64xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%outOFL2L3_cons_lock, Release, 1)
      aie.use_lock(%act_3_5_prod_lock, Release, 1)
      aie.use_lock(%act_4_5_prod_lock, Release, 1)
      aie.use_lock(%skip_buf_cons_prod_lock, Release, 1)
      %7 = arith.addi %5, %c2_2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%wts_buf_02_cons_prod_lock, Release, 1)
      %8 = arith.addi %0, %c1_1 : index
      cf.br ^bb1(%8 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_with = "conv2dk1_skip.o"}
    aie.shim_dma_allocation @inOF_act_L3L2(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<16384xi32>, %arg1: memref<18432xi32>, %arg2: memref<65536xi32>) {
      aiex.ipu.write32 {address = 13312 : ui32, column = 0 : i32, row = 2 : i32, value = 9 : ui32}
      aiex.ipu.write32 {address = 48128 : ui32, column = 0 : i32, row = 3 : i32, value = 11 : ui32}
      aiex.ipu.write32 {address = 48128 : ui32, column = 0 : i32, row = 5 : i32, value = 11 : ui32}
      aiex.ipu.write32 {address = 54272 : ui32, column = 0 : i32, row = 4 : i32, value = 11 : ui32}
      aiex.ipu.write32 {address = 54276 : ui32, column = 0 : i32, row = 4 : i32, value = 4294967295 : ui32}
      aiex.ipu.write32 {address = 54280 : ui32, column = 0 : i32, row = 4 : i32, value = 10 : ui32}
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c32_i32 = arith.constant 32 : i32
      %c32_i32_0 = arith.constant 32 : i32
      %c32_i32_1 = arith.constant 32 : i32
      %c64_i32 = arith.constant 64 : i32
      %c64_i32_2 = arith.constant 64 : i32
      %c1_i32_3 = arith.constant 1 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %c64_i32_5 = arith.constant 64 : i32
      %c16_i32 = arith.constant 16 : i32
      %c16384_i32 = arith.constant 16384 : i32
      %c512_i32 = arith.constant 512 : i32
      %c0_i32_6 = arith.constant 0 : i32
      %c16384_i64 = arith.constant 16384 : i64
      %c65536_i64 = arith.constant 65536 : i64
      %c18432_i64 = arith.constant 18432 : i64
      aiex.ipu.writebd_shimtile {bd_id = 0 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.ipu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}
      aiex.ipu.writebd_shimtile {bd_id = 2 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, ddr_id = 2 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.ipu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483650 : ui32}
      aiex.ipu.writebd_shimtile {bd_id = 2 : i32, buffer_length = 18432 : i32, buffer_offset = 0 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, ddr_id = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.ipu.write32 {address = 119324 : ui32, column = 0 : i32, row = 0 : i32, value = 2 : ui32}
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_0_cons_buff_0 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_0_cons_buff_1 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%wts_buf_00_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_buf_00_cons_buff_0 : memref<4096xi8>, 0, 4096)
      aie.use_lock(%wts_buf_00_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%act_2_3_4_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_buff_0 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%act_2_3_4_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_buff_1 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_0 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_1 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_2 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_3 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb10)
    ^bb6:  // 2 preds: ^bb5, ^bb9
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_0 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_1 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_2 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_3 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb10:  // pred: ^bb5
      %2 = aie.dma_start(S2MM, 1, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%inOF_wts_0_L3L2_cons_prod_lock, AcquireGreaterEqual, 3)
      aie.dma_bd(%inOF_wts_0_L3L2_cons_buff_0 : memref<73728xi8>, 0, 73728)
      aie.use_lock(%inOF_wts_0_L3L2_cons_cons_lock, Release, 3)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      %3 = aie.dma_start(MM2S, 1, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%inOF_wts_0_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_wts_0_L3L2_cons_buff_0 : memref<73728xi8>, 0, 4096)
      aie.use_lock(%inOF_wts_0_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %4 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%inOF_wts_0_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_wts_0_L3L2_cons_buff_0 : memref<73728xi8>, 4096, 36864)
      aie.use_lock(%inOF_wts_0_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      %5 = aie.dma_start(MM2S, 3, ^bb17, ^bb18)
    ^bb17:  // 2 preds: ^bb16, ^bb17
      aie.use_lock(%inOF_wts_0_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_wts_0_L3L2_cons_buff_0 : memref<73728xi8>, 40960, 32768)
      aie.use_lock(%inOF_wts_0_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb17
    ^bb18:  // pred: ^bb16
      aie.end
    }
    aie.shim_dma_allocation @inOF_wts_0_L3L2(MM2S, 1, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%skip_buf_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%skip_buf_cons_buff_0 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%skip_buf_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%skip_buf_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%skip_buf_cons_buff_1 : memref<32x1x64xi8>, 0, 2048)
      aie.use_lock(%skip_buf_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%wts_buf_02_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_buf_02_cons_buff_0 : memref<32768xi8>, 0, 32768)
      aie.use_lock(%wts_buf_02_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%outOFL2L3_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOFL2L3_buff_0 : memref<32x1x256xui8>, 0, 8192)
      aie.use_lock(%outOFL2L3_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%outOFL2L3_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outOFL2L3_buff_1 : memref<32x1x256xui8>, 0, 8192)
      aie.use_lock(%outOFL2L3_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_buf_01_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_buf_01_0_cons_buff_0 : memref<36864xi8>, 0, 36864)
      aie.use_lock(%wts_buf_01_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_0_cons_buff_0 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_0_cons_buff_1 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_0_cons_buff_2 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%act_2_3_4_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_0_cons_buff_3 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @outOFL2L3(S2MM, 0, 0)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%wts_buf_01_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_buf_01_1_cons_buff_0 : memref<36864xi8>, 0, 36864)
      aie.use_lock(%wts_buf_01_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb7)
    ^bb3:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_1_cons_buff_0 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_1_cons_buff_1 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_1_cons_buff_2 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%act_2_3_4_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_2_3_4_1_cons_buff_3 : memref<32x1x64xui8>, 0, 2048)
      aie.use_lock(%act_2_3_4_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb7:  // pred: ^bb2
      aie.end
    }
  }
}

