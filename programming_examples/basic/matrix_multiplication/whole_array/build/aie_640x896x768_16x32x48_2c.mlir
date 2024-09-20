module {
  aie.device(npu1_2col) {
    func.func private @zero_scalar_i32(memref<16x48xi32>)
    func.func private @zero_i32(memref<16x48xi32>)
    func.func private @matmul_scalar_i16_i32(memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>)
    func.func private @matmul_i16_i32(memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    aie.objectfifo @A_L2L1_0(%tile_0_1 toStream [<size = 4, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile_0_2, %tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x32xi16>>
    aie.objectfifo @A_L2L1_1(%tile_0_1 toStream [<size = 4, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile_0_3, %tile_1_3}, 2 : i32) : !aie.objectfifo<memref<16x32xi16>>
    aie.objectfifo @A_L2L1_2(%tile_1_1 toStream [<size = 4, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile_0_4, %tile_1_4}, 2 : i32) : !aie.objectfifo<memref<16x32xi16>>
    aie.objectfifo @A_L2L1_3(%tile_1_1 toStream [<size = 4, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%tile_0_5, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<16x32xi16>>
    aie.objectfifo @A_L3L2_0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xi16>>
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0, @A_L2L1_1]([] [0, 512])
    aie.objectfifo @A_L3L2_1(%tile_1_0, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1024xi16>>
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_2, @A_L2L1_3]([] [0, 512])
    aie.objectfifo @B_L3L2_0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1536xi16>>
    aie.objectfifo @B_L2L1_0(%tile_0_1 toStream [<size = 8, stride = 192>, <size = 12, stride = 4>, <size = 4, stride = 48>, <size = 4, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x48xi16>>
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [])
    aie.objectfifo @B_L3L2_1(%tile_1_0, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1536xi16>>
    aie.objectfifo @B_L2L1_1(%tile_1_1 toStream [<size = 8, stride = 192>, <size = 12, stride = 4>, <size = 4, stride = 48>, <size = 4, stride = 1>], {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x48xi16>>
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L2L3_0(%tile_0_1 toStream [<size = 4, stride = 192>, <size = 4, stride = 4>, <size = 12, stride = 16>, <size = 4, stride = 1>], {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<3072xi32>>
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 768, 1536, 2304] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x48xi32>>
    aie.objectfifo @C_L2L3_1(%tile_1_1 toStream [<size = 4, stride = 192>, <size = 4, stride = 4>, <size = 12, stride = 16>, <size = 4, stride = 1>], {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<3072xi32>>
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 768, 1536, 2304] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c80 = arith.constant 80 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c80 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<16x48xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x48xi32>> -> memref<16x48xi32>
          func.call @zero_i32(%1) : (memref<16x48xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c28 = arith.constant 28 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c28 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xi16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xi16>> -> memref<16x32xi16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x48xi16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x48xi16>> -> memref<32x48xi16>
            func.call @matmul_i16_i32(%3, %5, %1) : (memref<16x32xi16>, memref<32x48xi16>, memref<16x48xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_16x32x48.o"}
    aiex.runtime_sequence(%arg0: memref<573440xi16>, %arg1: memref<688128xi16>, %arg2: memref<491520xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][2, 8, 64, 48][49152, 96, 768, 1]) {id = 0 : i64, metadata = @C_L2L3_0} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][8, 28, 32, 32][0, 32, 896, 1]) {id = 1 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 2 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 57344][8, 28, 32, 32][0, 32, 896, 1]) {id = 3 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 4 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 48][2, 8, 64, 48][49152, 96, 768, 1]) {id = 0 : i64, metadata = @C_L2L3_1} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 28672][8, 28, 32, 32][0, 32, 896, 1]) {id = 1 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 2 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 86016][8, 28, 32, 32][0, 32, 896, 1]) {id = 3 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 4 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 98304][2, 8, 64, 48][49152, 96, 768, 1]) {id = 8 : i64, metadata = @C_L2L3_0} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 114688][8, 28, 32, 32][0, 32, 896, 1]) {id = 9 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 10 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 172032][8, 28, 32, 32][0, 32, 896, 1]) {id = 11 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 12 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 98352][2, 8, 64, 48][49152, 96, 768, 1]) {id = 8 : i64, metadata = @C_L2L3_1} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 143360][8, 28, 32, 32][0, 32, 896, 1]) {id = 9 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 10 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 200704][8, 28, 32, 32][0, 32, 896, 1]) {id = 11 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 12 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 196608][2, 8, 64, 48][49152, 96, 768, 1]) {id = 0 : i64, metadata = @C_L2L3_0} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 229376][8, 28, 32, 32][0, 32, 896, 1]) {id = 1 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 2 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 286720][8, 28, 32, 32][0, 32, 896, 1]) {id = 3 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 4 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 196656][2, 8, 64, 48][49152, 96, 768, 1]) {id = 0 : i64, metadata = @C_L2L3_1} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 258048][8, 28, 32, 32][0, 32, 896, 1]) {id = 1 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 2 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 315392][8, 28, 32, 32][0, 32, 896, 1]) {id = 3 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 4 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 294912][2, 8, 64, 48][49152, 96, 768, 1]) {id = 8 : i64, metadata = @C_L2L3_0} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 344064][8, 28, 32, 32][0, 32, 896, 1]) {id = 9 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 10 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 401408][8, 28, 32, 32][0, 32, 896, 1]) {id = 11 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 12 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 294960][2, 8, 64, 48][49152, 96, 768, 1]) {id = 8 : i64, metadata = @C_L2L3_1} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 372736][8, 28, 32, 32][0, 32, 896, 1]) {id = 9 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 10 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 430080][8, 28, 32, 32][0, 32, 896, 1]) {id = 11 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 12 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 393216][2, 8, 64, 48][49152, 96, 768, 1]) {id = 0 : i64, metadata = @C_L2L3_0} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 458752][8, 28, 32, 32][0, 32, 896, 1]) {id = 1 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 2 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 516096][8, 28, 32, 32][0, 32, 896, 1]) {id = 3 : i64, metadata = @A_L3L2_0} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 28, 32, 48][96, 24576, 768, 1]) {id = 4 : i64, metadata = @B_L3L2_0} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 393264][2, 8, 64, 48][49152, 96, 768, 1]) {id = 0 : i64, metadata = @C_L2L3_1} : memref<491520xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 487424][8, 28, 32, 32][0, 32, 896, 1]) {id = 1 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 2 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 544768][8, 28, 32, 32][0, 32, 896, 1]) {id = 3 : i64, metadata = @A_L3L2_1} : memref<573440xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 48][8, 28, 32, 48][96, 24576, 768, 1]) {id = 4 : i64, metadata = @B_L3L2_1} : memref<688128xi16>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}

