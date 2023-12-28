module {
  aie.device(ipu) {
    memref.global "public" @outC_cons : memref<16384xi16>
    memref.global "public" @outC : memref<16384xi16>
    memref.global "public" @memC3_cons : memref<64x64xi16>
    memref.global "public" @memC3 : memref<64x64xi16>
    memref.global "public" @memC2_cons : memref<64x64xi16>
    memref.global "public" @memC2 : memref<64x64xi16>
    memref.global "public" @memC1_cons : memref<64x64xi16>
    memref.global "public" @memC1 : memref<64x64xi16>
    memref.global "public" @memC0_cons : memref<64x64xi16>
    memref.global "public" @memC0 : memref<64x64xi16>
    memref.global "public" @memB_0_cons : memref<32x64xi16>
    memref.global "public" @memB_1_cons : memref<32x64xi16>
    memref.global "public" @memB_2_cons : memref<32x64xi16>
    memref.global "public" @memB_3_cons : memref<32x64xi16>
    memref.global "public" @memB : memref<32x64xi16>
    memref.global "public" @inB_cons : memref<2048xi16>
    memref.global "public" @inB : memref<2048xi16>
    memref.global "public" @memA3_cons : memref<64x32xi16>
    memref.global "public" @memA3 : memref<64x32xi16>
    memref.global "public" @memA2_cons : memref<64x32xi16>
    memref.global "public" @memA2 : memref<64x32xi16>
    memref.global "public" @memA1_cons : memref<64x32xi16>
    memref.global "public" @memA1 : memref<64x32xi16>
    memref.global "public" @memA0_cons : memref<64x32xi16>
    memref.global "public" @memA0 : memref<64x32xi16>
    memref.global "public" @inA_cons : memref<8192xi16>
    memref.global "public" @inA : memref<8192xi16>
    func.func private @zero_scalar_i16(memref<64x64xi16>)
    func.func private @zero_i16(memref<64x64xi16>)
    func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    %outC_cons_prod_lock = aie.lock(%tile_0_0, 4) {init = 0 : i32, sym_name = "outC_cons_prod_lock"}
    %outC_cons_cons_lock = aie.lock(%tile_0_0, 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock"}
    %outC_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "outC_buff_0"} : memref<16384xi16>
    %outC_buff_1 = aie.buffer(%tile_0_1) {address = 32768 : i32, sym_name = "outC_buff_1"} : memref<16384xi16>
    %outC_prod_lock = aie.lock(%tile_0_1, 4) {init = 8 : i32, sym_name = "outC_prod_lock"}
    %outC_cons_lock = aie.lock(%tile_0_1, 5) {init = 0 : i32, sym_name = "outC_cons_lock"}
    %memC3_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "memC3_buff_0"} : memref<64x64xi16>
    %memC3_buff_1 = aie.buffer(%tile_0_5) {address = 9216 : i32, sym_name = "memC3_buff_1"} : memref<64x64xi16>
    %memC3_prod_lock = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "memC3_prod_lock"}
    %memC3_cons_lock = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "memC3_cons_lock"}
    %memC2_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "memC2_buff_0"} : memref<64x64xi16>
    %memC2_buff_1 = aie.buffer(%tile_0_4) {address = 9216 : i32, sym_name = "memC2_buff_1"} : memref<64x64xi16>
    %memC2_prod_lock = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "memC2_prod_lock"}
    %memC2_cons_lock = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "memC2_cons_lock"}
    %memC1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "memC1_buff_0"} : memref<64x64xi16>
    %memC1_buff_1 = aie.buffer(%tile_0_3) {address = 9216 : i32, sym_name = "memC1_buff_1"} : memref<64x64xi16>
    %memC1_prod_lock = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "memC1_prod_lock"}
    %memC1_cons_lock = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "memC1_cons_lock"}
    %memC0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "memC0_buff_0"} : memref<64x64xi16>
    %memC0_buff_1 = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "memC0_buff_1"} : memref<64x64xi16>
    %memC0_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "memC0_prod_lock"}
    %memC0_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "memC0_cons_lock"}
    %memB_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "memB_0_cons_buff_0"} : memref<32x64xi16>
    %memB_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 21504 : i32, sym_name = "memB_0_cons_buff_1"} : memref<32x64xi16>
    %memB_0_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memB_0_cons_prod_lock"}
    %memB_0_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memB_0_cons_cons_lock"}
    %memB_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 17408 : i32, sym_name = "memB_1_cons_buff_0"} : memref<32x64xi16>
    %memB_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 21504 : i32, sym_name = "memB_1_cons_buff_1"} : memref<32x64xi16>
    %memB_1_cons_prod_lock = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "memB_1_cons_prod_lock"}
    %memB_1_cons_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "memB_1_cons_cons_lock"}
    %memB_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 17408 : i32, sym_name = "memB_2_cons_buff_0"} : memref<32x64xi16>
    %memB_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 21504 : i32, sym_name = "memB_2_cons_buff_1"} : memref<32x64xi16>
    %memB_2_cons_prod_lock = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "memB_2_cons_prod_lock"}
    %memB_2_cons_cons_lock = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "memB_2_cons_cons_lock"}
    %memB_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 17408 : i32, sym_name = "memB_3_cons_buff_0"} : memref<32x64xi16>
    %memB_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 21504 : i32, sym_name = "memB_3_cons_buff_1"} : memref<32x64xi16>
    %memB_3_cons_prod_lock = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "memB_3_cons_prod_lock"}
    %memB_3_cons_cons_lock = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "memB_3_cons_cons_lock"}
    %inB_cons_buff_0 = aie.buffer(%tile_0_1) {address = 98304 : i32, sym_name = "inB_cons_buff_0"} : memref<2048xi16>
    %inB_cons_buff_1 = aie.buffer(%tile_0_1) {address = 102400 : i32, sym_name = "inB_cons_buff_1"} : memref<2048xi16>
    %inB_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
    %inB_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
    %inB_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "inB_prod_lock"}
    %inB_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "inB_cons_lock"}
    %memA3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 25600 : i32, sym_name = "memA3_cons_buff_0"} : memref<64x32xi16>
    %memA3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 29696 : i32, sym_name = "memA3_cons_buff_1"} : memref<64x32xi16>
    %memA3_cons_prod_lock = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "memA3_cons_prod_lock"}
    %memA3_cons_cons_lock = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "memA3_cons_cons_lock"}
    %memA2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 25600 : i32, sym_name = "memA2_cons_buff_0"} : memref<64x32xi16>
    %memA2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 29696 : i32, sym_name = "memA2_cons_buff_1"} : memref<64x32xi16>
    %memA2_cons_prod_lock = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "memA2_cons_prod_lock"}
    %memA2_cons_cons_lock = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "memA2_cons_cons_lock"}
    %memA1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 25600 : i32, sym_name = "memA1_cons_buff_0"} : memref<64x32xi16>
    %memA1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 29696 : i32, sym_name = "memA1_cons_buff_1"} : memref<64x32xi16>
    %memA1_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "memA1_cons_prod_lock"}
    %memA1_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "memA1_cons_cons_lock"}
    %memA0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "memA0_cons_buff_0"} : memref<64x32xi16>
    %memA0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 29696 : i32, sym_name = "memA0_cons_buff_1"} : memref<64x32xi16>
    %memA0_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memA0_cons_prod_lock"}
    %memA0_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memA0_cons_cons_lock"}
    %inA_cons_buff_0 = aie.buffer(%tile_0_1) {address = 65536 : i32, sym_name = "inA_cons_buff_0"} : memref<8192xi16>
    %inA_cons_buff_1 = aie.buffer(%tile_0_1) {address = 81920 : i32, sym_name = "inA_cons_buff_1"} : memref<8192xi16>
    %inA_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 8 : i32, sym_name = "inA_cons_prod_lock"}
    %inA_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
    %inA_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "inA_prod_lock"}
    %inA_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "inA_cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_1, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_5, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_4, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_3, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 3)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 4)
    aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 5)
    aie.flow(%tile_0_1, DMA : 5, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967294 step %c2 {
        %c0_10 = arith.constant 0 : index
        %c2_11 = arith.constant 2 : index
        %c1_12 = arith.constant 1 : index
        aie.use_lock(%memC0_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC0_buff_0) : (memref<64x64xi16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c4_14 = arith.constant 4 : index
        %c1_15 = arith.constant 1 : index
        %c2_16 = arith.constant 2 : index
        scf.for %arg1 = %c0_13 to %c4_14 step %c2_16 {
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_0, %memB_0_cons_buff_0, %memC0_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_1, %memB_0_cons_buff_1, %memC0_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC0_cons_lock, Release, 1)
        aie.use_lock(%memC0_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC0_buff_1) : (memref<64x64xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c4_18 = arith.constant 4 : index
        %c1_19 = arith.constant 1 : index
        %c2_20 = arith.constant 2 : index
        scf.for %arg1 = %c0_17 to %c4_18 step %c2_20 {
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_0, %memB_0_cons_buff_0, %memC0_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_1, %memB_0_cons_buff_1, %memC0_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC0_cons_lock, Release, 1)
        %c0_21 = arith.constant 0 : index
        %c2_22 = arith.constant 2 : index
        %c1_23 = arith.constant 1 : index
        aie.use_lock(%memC0_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC0_buff_0) : (memref<64x64xi16>) -> ()
        %c0_24 = arith.constant 0 : index
        %c4_25 = arith.constant 4 : index
        %c1_26 = arith.constant 1 : index
        %c2_27 = arith.constant 2 : index
        scf.for %arg1 = %c0_24 to %c4_25 step %c2_27 {
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_0, %memB_0_cons_buff_0, %memC0_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_1, %memB_0_cons_buff_1, %memC0_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC0_cons_lock, Release, 1)
        aie.use_lock(%memC0_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC0_buff_1) : (memref<64x64xi16>) -> ()
        %c0_28 = arith.constant 0 : index
        %c4_29 = arith.constant 4 : index
        %c1_30 = arith.constant 1 : index
        %c2_31 = arith.constant 2 : index
        scf.for %arg1 = %c0_28 to %c4_29 step %c2_31 {
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_0, %memB_0_cons_buff_0, %memC0_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
          aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA0_cons_buff_1, %memB_0_cons_buff_1, %memC0_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA0_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC0_cons_lock, Release, 1)
      }
      %c0_0 = arith.constant 0 : index
      %c2_1 = arith.constant 2 : index
      %c1_2 = arith.constant 1 : index
      aie.use_lock(%memC0_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC0_buff_0) : (memref<64x64xi16>) -> ()
      %c0_3 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_4 = arith.constant 1 : index
      %c2_5 = arith.constant 2 : index
      scf.for %arg0 = %c0_3 to %c4 step %c2_5 {
        func.call @matmul_i16_i16(%memA0_cons_buff_1, %memB_0_cons_buff_1, %memC0_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA0_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
        aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA0_cons_buff_0, %memB_0_cons_buff_0, %memC0_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA0_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC0_cons_lock, Release, 1)
      aie.use_lock(%memC0_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC0_buff_1) : (memref<64x64xi16>) -> ()
      %c0_6 = arith.constant 0 : index
      %c4_7 = arith.constant 4 : index
      %c1_8 = arith.constant 1 : index
      %c2_9 = arith.constant 2 : index
      scf.for %arg0 = %c0_6 to %c4_7 step %c2_9 {
        aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA0_cons_buff_1, %memB_0_cons_buff_1, %memC0_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA0_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
        aie.use_lock(%memA0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_0_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA0_cons_buff_0, %memB_0_cons_buff_0, %memC0_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA0_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_0_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC0_cons_lock, Release, 1)
      aie.end
    } {link_with = "mm.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967294 step %c2 {
        %c0_10 = arith.constant 0 : index
        %c2_11 = arith.constant 2 : index
        %c1_12 = arith.constant 1 : index
        aie.use_lock(%memC1_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC1_buff_0) : (memref<64x64xi16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c4_14 = arith.constant 4 : index
        %c1_15 = arith.constant 1 : index
        %c2_16 = arith.constant 2 : index
        scf.for %arg1 = %c0_13 to %c4_14 step %c2_16 {
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_0, %memB_1_cons_buff_0, %memC1_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_1, %memB_1_cons_buff_1, %memC1_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC1_cons_lock, Release, 1)
        aie.use_lock(%memC1_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC1_buff_1) : (memref<64x64xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c4_18 = arith.constant 4 : index
        %c1_19 = arith.constant 1 : index
        %c2_20 = arith.constant 2 : index
        scf.for %arg1 = %c0_17 to %c4_18 step %c2_20 {
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_0, %memB_1_cons_buff_0, %memC1_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_1, %memB_1_cons_buff_1, %memC1_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC1_cons_lock, Release, 1)
        %c0_21 = arith.constant 0 : index
        %c2_22 = arith.constant 2 : index
        %c1_23 = arith.constant 1 : index
        aie.use_lock(%memC1_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC1_buff_0) : (memref<64x64xi16>) -> ()
        %c0_24 = arith.constant 0 : index
        %c4_25 = arith.constant 4 : index
        %c1_26 = arith.constant 1 : index
        %c2_27 = arith.constant 2 : index
        scf.for %arg1 = %c0_24 to %c4_25 step %c2_27 {
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_0, %memB_1_cons_buff_0, %memC1_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_1, %memB_1_cons_buff_1, %memC1_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC1_cons_lock, Release, 1)
        aie.use_lock(%memC1_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC1_buff_1) : (memref<64x64xi16>) -> ()
        %c0_28 = arith.constant 0 : index
        %c4_29 = arith.constant 4 : index
        %c1_30 = arith.constant 1 : index
        %c2_31 = arith.constant 2 : index
        scf.for %arg1 = %c0_28 to %c4_29 step %c2_31 {
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_0, %memB_1_cons_buff_0, %memC1_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
          aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA1_cons_buff_1, %memB_1_cons_buff_1, %memC1_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA1_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC1_cons_lock, Release, 1)
      }
      %c0_0 = arith.constant 0 : index
      %c2_1 = arith.constant 2 : index
      %c1_2 = arith.constant 1 : index
      aie.use_lock(%memC1_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC1_buff_0) : (memref<64x64xi16>) -> ()
      %c0_3 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_4 = arith.constant 1 : index
      %c2_5 = arith.constant 2 : index
      scf.for %arg0 = %c0_3 to %c4 step %c2_5 {
        func.call @matmul_i16_i16(%memA1_cons_buff_1, %memB_1_cons_buff_1, %memC1_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA1_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
        aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA1_cons_buff_0, %memB_1_cons_buff_0, %memC1_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA1_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC1_cons_lock, Release, 1)
      aie.use_lock(%memC1_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC1_buff_1) : (memref<64x64xi16>) -> ()
      %c0_6 = arith.constant 0 : index
      %c4_7 = arith.constant 4 : index
      %c1_8 = arith.constant 1 : index
      %c2_9 = arith.constant 2 : index
      scf.for %arg0 = %c0_6 to %c4_7 step %c2_9 {
        aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA1_cons_buff_1, %memB_1_cons_buff_1, %memC1_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA1_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
        aie.use_lock(%memA1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_1_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA1_cons_buff_0, %memB_1_cons_buff_0, %memC1_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA1_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_1_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC1_cons_lock, Release, 1)
      aie.end
    } {link_with = "mm.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967294 step %c2 {
        %c0_10 = arith.constant 0 : index
        %c2_11 = arith.constant 2 : index
        %c1_12 = arith.constant 1 : index
        aie.use_lock(%memC2_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC2_buff_0) : (memref<64x64xi16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c4_14 = arith.constant 4 : index
        %c1_15 = arith.constant 1 : index
        %c2_16 = arith.constant 2 : index
        scf.for %arg1 = %c0_13 to %c4_14 step %c2_16 {
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_0, %memB_2_cons_buff_0, %memC2_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_1, %memB_2_cons_buff_1, %memC2_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC2_cons_lock, Release, 1)
        aie.use_lock(%memC2_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC2_buff_1) : (memref<64x64xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c4_18 = arith.constant 4 : index
        %c1_19 = arith.constant 1 : index
        %c2_20 = arith.constant 2 : index
        scf.for %arg1 = %c0_17 to %c4_18 step %c2_20 {
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_0, %memB_2_cons_buff_0, %memC2_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_1, %memB_2_cons_buff_1, %memC2_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC2_cons_lock, Release, 1)
        %c0_21 = arith.constant 0 : index
        %c2_22 = arith.constant 2 : index
        %c1_23 = arith.constant 1 : index
        aie.use_lock(%memC2_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC2_buff_0) : (memref<64x64xi16>) -> ()
        %c0_24 = arith.constant 0 : index
        %c4_25 = arith.constant 4 : index
        %c1_26 = arith.constant 1 : index
        %c2_27 = arith.constant 2 : index
        scf.for %arg1 = %c0_24 to %c4_25 step %c2_27 {
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_0, %memB_2_cons_buff_0, %memC2_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_1, %memB_2_cons_buff_1, %memC2_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC2_cons_lock, Release, 1)
        aie.use_lock(%memC2_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC2_buff_1) : (memref<64x64xi16>) -> ()
        %c0_28 = arith.constant 0 : index
        %c4_29 = arith.constant 4 : index
        %c1_30 = arith.constant 1 : index
        %c2_31 = arith.constant 2 : index
        scf.for %arg1 = %c0_28 to %c4_29 step %c2_31 {
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_0, %memB_2_cons_buff_0, %memC2_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
          aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA2_cons_buff_1, %memB_2_cons_buff_1, %memC2_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA2_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC2_cons_lock, Release, 1)
      }
      %c0_0 = arith.constant 0 : index
      %c2_1 = arith.constant 2 : index
      %c1_2 = arith.constant 1 : index
      aie.use_lock(%memC2_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC2_buff_0) : (memref<64x64xi16>) -> ()
      %c0_3 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_4 = arith.constant 1 : index
      %c2_5 = arith.constant 2 : index
      scf.for %arg0 = %c0_3 to %c4 step %c2_5 {
        func.call @matmul_i16_i16(%memA2_cons_buff_1, %memB_2_cons_buff_1, %memC2_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA2_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
        aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA2_cons_buff_0, %memB_2_cons_buff_0, %memC2_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA2_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC2_cons_lock, Release, 1)
      aie.use_lock(%memC2_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC2_buff_1) : (memref<64x64xi16>) -> ()
      %c0_6 = arith.constant 0 : index
      %c4_7 = arith.constant 4 : index
      %c1_8 = arith.constant 1 : index
      %c2_9 = arith.constant 2 : index
      scf.for %arg0 = %c0_6 to %c4_7 step %c2_9 {
        aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA2_cons_buff_1, %memB_2_cons_buff_1, %memC2_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA2_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
        aie.use_lock(%memA2_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_2_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA2_cons_buff_0, %memB_2_cons_buff_0, %memC2_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA2_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_2_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC2_cons_lock, Release, 1)
      aie.end
    } {link_with = "mm.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967294 step %c2 {
        %c0_10 = arith.constant 0 : index
        %c2_11 = arith.constant 2 : index
        %c1_12 = arith.constant 1 : index
        aie.use_lock(%memC3_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC3_buff_0) : (memref<64x64xi16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c4_14 = arith.constant 4 : index
        %c1_15 = arith.constant 1 : index
        %c2_16 = arith.constant 2 : index
        scf.for %arg1 = %c0_13 to %c4_14 step %c2_16 {
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_0, %memB_3_cons_buff_0, %memC3_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_1, %memB_3_cons_buff_1, %memC3_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC3_cons_lock, Release, 1)
        aie.use_lock(%memC3_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC3_buff_1) : (memref<64x64xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c4_18 = arith.constant 4 : index
        %c1_19 = arith.constant 1 : index
        %c2_20 = arith.constant 2 : index
        scf.for %arg1 = %c0_17 to %c4_18 step %c2_20 {
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_0, %memB_3_cons_buff_0, %memC3_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_1, %memB_3_cons_buff_1, %memC3_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC3_cons_lock, Release, 1)
        %c0_21 = arith.constant 0 : index
        %c2_22 = arith.constant 2 : index
        %c1_23 = arith.constant 1 : index
        aie.use_lock(%memC3_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC3_buff_0) : (memref<64x64xi16>) -> ()
        %c0_24 = arith.constant 0 : index
        %c4_25 = arith.constant 4 : index
        %c1_26 = arith.constant 1 : index
        %c2_27 = arith.constant 2 : index
        scf.for %arg1 = %c0_24 to %c4_25 step %c2_27 {
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_0, %memB_3_cons_buff_0, %memC3_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_1, %memB_3_cons_buff_1, %memC3_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC3_cons_lock, Release, 1)
        aie.use_lock(%memC3_prod_lock, AcquireGreaterEqual, 1)
        func.call @zero_i16(%memC3_buff_1) : (memref<64x64xi16>) -> ()
        %c0_28 = arith.constant 0 : index
        %c4_29 = arith.constant 4 : index
        %c1_30 = arith.constant 1 : index
        %c2_31 = arith.constant 2 : index
        scf.for %arg1 = %c0_28 to %c4_29 step %c2_31 {
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_0, %memB_3_cons_buff_0, %memC3_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
          aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
          aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
          func.call @matmul_i16_i16(%memA3_cons_buff_1, %memB_3_cons_buff_1, %memC3_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
          aie.use_lock(%memA3_cons_prod_lock, Release, 1)
          aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
        }
        aie.use_lock(%memC3_cons_lock, Release, 1)
      }
      %c0_0 = arith.constant 0 : index
      %c2_1 = arith.constant 2 : index
      %c1_2 = arith.constant 1 : index
      aie.use_lock(%memC3_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC3_buff_0) : (memref<64x64xi16>) -> ()
      %c0_3 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_4 = arith.constant 1 : index
      %c2_5 = arith.constant 2 : index
      scf.for %arg0 = %c0_3 to %c4 step %c2_5 {
        func.call @matmul_i16_i16(%memA3_cons_buff_1, %memB_3_cons_buff_1, %memC3_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA3_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
        aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA3_cons_buff_0, %memB_3_cons_buff_0, %memC3_buff_0) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA3_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC3_cons_lock, Release, 1)
      aie.use_lock(%memC3_prod_lock, AcquireGreaterEqual, 1)
      func.call @zero_i16(%memC3_buff_1) : (memref<64x64xi16>) -> ()
      %c0_6 = arith.constant 0 : index
      %c4_7 = arith.constant 4 : index
      %c1_8 = arith.constant 1 : index
      %c2_9 = arith.constant 2 : index
      scf.for %arg0 = %c0_6 to %c4_7 step %c2_9 {
        aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA3_cons_buff_1, %memB_3_cons_buff_1, %memC3_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA3_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
        aie.use_lock(%memA3_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%memB_3_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @matmul_i16_i16(%memA3_cons_buff_0, %memB_3_cons_buff_0, %memC3_buff_1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
        aie.use_lock(%memA3_cons_prod_lock, Release, 1)
        aie.use_lock(%memB_3_cons_prod_lock, Release, 1)
      }
      aie.use_lock(%memC3_cons_lock, Release, 1)
      aie.end
    } {link_with = "mm.o"}
    aie.shim_dma_allocation @inA(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<16384xi32>, %arg1: memref<8192xi32>, %arg2: memref<16384xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 2, 256, 32][16384, 32, 64]) {id = 0 : i64, metadata = @outC} : memref<16384xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][2, 4, 256, 16][0, 16, 64]) {id = 1 : i64, metadata = @inA} : memref<16384xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 2, 128, 32][0, 32, 64]) {id = 2 : i64, metadata = @inB} : memref<8192xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual, 4)
      aie.dma_bd(%inA_cons_buff_0 : memref<8192xi16>, 0, 8192)
      aie.use_lock(%inA_cons_cons_lock, Release, 4)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inA_cons_prod_lock, AcquireGreaterEqual, 4)
      aie.dma_bd(%inA_cons_buff_1 : memref<8192xi16>, 0, 8192)
      aie.use_lock(%inA_cons_cons_lock, Release, 4)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<8192xi16>, 0, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<8192xi16>, 0, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<8192xi16>, 4096, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<8192xi16>, 4096, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<8192xi16>, 8192, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<8192xi16>, 8192, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_0 : memref<8192xi16>, 12288, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%inA_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inA_cons_buff_1 : memref<8192xi16>, 12288, 2048, [<size = 16, stride = 64>, <size = 8, stride = 2>, <size = 4, stride = 16>, <size = 2, stride = 1>])
      aie.use_lock(%inA_cons_prod_lock, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<2048xi16>, 0, 2048)
      aie.use_lock(%inB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inB_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<2048xi16>, 0, 2048)
      aie.use_lock(%inB_cons_cons_lock, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<2048xi16>, 0, 2048, [<size = 8, stride = 128>, <size = 16, stride = 2>, <size = 4, stride = 32>, <size = 2, stride = 1>])
      aie.use_lock(%inB_cons_prod_lock, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%inB_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<2048xi16>, 0, 2048, [<size = 8, stride = 128>, <size = 16, stride = 2>, <size = 4, stride = 32>, <size = 2, stride = 1>])
      aie.use_lock(%inB_cons_prod_lock, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 2, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<16384xi16>, 0, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<16384xi16>, 0, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(S2MM, 3, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<16384xi16>, 8192, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<16384xi16>, 8192, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %9 = aie.dma_start(S2MM, 4, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<16384xi16>, 16384, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<16384xi16>, 16384, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      %10 = aie.dma_start(S2MM, 5, ^bb31, ^bb33)
    ^bb31:  // 2 preds: ^bb30, ^bb32
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_0 : memref<16384xi16>, 24576, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%outC_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%outC_buff_1 : memref<16384xi16>, 24576, 4096)
      aie.use_lock(%outC_cons_lock, Release, 1)
      aie.next_bd ^bb31
    ^bb33:  // pred: ^bb30
      %11 = aie.dma_start(MM2S, 5, ^bb34, ^bb36)
    ^bb34:  // 2 preds: ^bb33, ^bb35
      aie.use_lock(%outC_cons_lock, AcquireGreaterEqual, 4)
      aie.dma_bd(%outC_buff_0 : memref<16384xi16>, 0, 16384, [<size = 16, stride = 128>, <size = 4, stride = 2>, <size = 16, stride = 8>, <size = 2, stride = 1>])
      aie.use_lock(%outC_prod_lock, Release, 4)
      aie.next_bd ^bb35
    ^bb35:  // pred: ^bb34
      aie.use_lock(%outC_cons_lock, AcquireGreaterEqual, 4)
      aie.dma_bd(%outC_buff_1 : memref<16384xi16>, 0, 16384, [<size = 16, stride = 128>, <size = 4, stride = 2>, <size = 16, stride = 8>, <size = 2, stride = 1>])
      aie.use_lock(%outC_prod_lock, Release, 4)
      aie.next_bd ^bb34
    ^bb36:  // pred: ^bb33
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_cons_buff_0 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA0_cons_buff_1 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_0_cons_buff_0 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_0_cons_buff_1 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC0_buff_0 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC0_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC0_buff_1 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC0_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_cons_buff_0 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA1_cons_buff_1 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_1_cons_buff_0 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_1_cons_buff_1 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC1_buff_0 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC1_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC1_buff_1 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC1_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_cons_buff_0 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA2_cons_buff_1 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_2_cons_buff_0 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_2_cons_buff_1 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC2_buff_0 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC2_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC2_buff_1 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC2_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @inB(MM2S, 1, 0)
    aie.shim_dma_allocation @outC(S2MM, 0, 0)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memA3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_cons_buff_0 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memA3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memA3_cons_buff_1 : memref<64x32xi16>, 0, 2048)
      aie.use_lock(%memA3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memB_3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_3_cons_buff_0 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memB_3_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_3_cons_buff_1 : memref<32x64xi16>, 0, 2048)
      aie.use_lock(%memB_3_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%memC3_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC3_buff_0 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC3_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%memC3_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%memC3_buff_1 : memref<64x64xi16>, 0, 4096)
      aie.use_lock(%memC3_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
  }
}
