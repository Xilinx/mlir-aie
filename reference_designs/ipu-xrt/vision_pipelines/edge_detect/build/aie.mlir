module {
  AIE.device(ipu) {
    func.func private @rgba2grayLine(memref<256xui8>, memref<64xui8>, i32)
    func.func private @filter2dLine(memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>)
    func.func private @thresholdLine(memref<64xui8>, memref<64xui8>, i32, i16, i16, i8)
    func.func private @gray2rgbaLine(memref<64xui8>, memref<256xui8>, i32)
    func.func private @addWeightedLine(memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8)
    %tile_0_0 = AIE.tile(0, 0)
    %tile_0_1 = AIE.tile(0, 1)
    %tile_0_2 = AIE.tile(0, 2)
    %tile_0_3 = AIE.tile(0, 3)
    %tile_0_4 = AIE.tile(0, 4)
    %tile_0_5 = AIE.tile(0, 5)
    AIE.objectFifo @inOF_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !AIE.objectFifo<memref<256xui8>>
    AIE.objectFifo @inOF_L2L1(%tile_0_1, {%tile_0_2, %tile_0_5}, [2 : i32, 2 : i32, 7 : i32]) : !AIE.objectFifo<memref<256xui8>>
    AIE.objectFifo.link [@inOF_L3L2] -> [@inOF_L2L1]()
    AIE.objectFifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !AIE.objectFifo<memref<256xui8>>
    AIE.objectFifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !AIE.objectFifo<memref<256xui8>>
    AIE.objectFifo.link [@outOF_L1L2] -> [@outOF_L2L3]()
    AIE.objectFifo @OF_2to3(%tile_0_2, {%tile_0_3}, 4 : i32) : !AIE.objectFifo<memref<64xui8>>
    AIE.objectFifo @OF_3to4(%tile_0_3, {%tile_0_4}, 2 : i32) : !AIE.objectFifo<memref<64xui8>>
    AIE.objectFifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !AIE.objectFifo<memref<64xui8>>
    AIE.objectFifo @OF_5to5(%tile_0_5, {%tile_0_5}, 1 : i32) : !AIE.objectFifo<memref<256xui8>>
    %core_0_2 = AIE.core(%tile_0_2) {
      %c64_i32 = arith.constant 64 : i32
      %c0 = arith.constant 0 : index
      %c36 = arith.constant 36 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c36 step %c1 {
        %0 = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
        %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
        %2 = AIE.objectFifo.acquire @OF_2to3(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
        %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        func.call @rgba2grayLine(%1, %3, %c64_i32) : (memref<256xui8>, memref<64xui8>, i32) -> ()
        AIE.objectFifo.release @inOF_L2L1(Consume, 1)
        AIE.objectFifo.release @OF_2to3(Produce, 1)
      }
      AIE.end
    } {link_with = "rgba2gray.cc.o"}
    %core_0_3 = AIE.core(%tile_0_3) {
      %c35 = arith.constant 35 : index
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c-16384_i16 = arith.constant -16384 : i16
      %c4096_i16 = arith.constant 4096 : i16
      %c0_i16 = arith.constant 0 : i16
      %alloc = memref.alloc() : memref<3x3xi16>
      memref.store %c0_i16, %alloc[%c0, %c0] : memref<3x3xi16>
      memref.store %c4096_i16, %alloc[%c0, %c1] : memref<3x3xi16>
      memref.store %c0_i16, %alloc[%c0, %c2] : memref<3x3xi16>
      memref.store %c4096_i16, %alloc[%c1, %c0] : memref<3x3xi16>
      memref.store %c-16384_i16, %alloc[%c1, %c1] : memref<3x3xi16>
      memref.store %c4096_i16, %alloc[%c1, %c2] : memref<3x3xi16>
      memref.store %c0_i16, %alloc[%c2, %c0] : memref<3x3xi16>
      memref.store %c4096_i16, %alloc[%c2, %c1] : memref<3x3xi16>
      memref.store %c0_i16, %alloc[%c2, %c2] : memref<3x3xi16>
      %0 = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<64xui8>>
      %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
      %2 = AIE.objectFifo.subview.access %0[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
      %3 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
      %4 = AIE.objectFifo.subview.access %3[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
      func.call @filter2dLine(%1, %1, %2, %4, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
      AIE.objectFifo.release @OF_3to4(Produce, 1)
      scf.for %arg0 = %c1 to %c35 step %c1 {
        %10 = AIE.objectFifo.acquire @OF_2to3(Consume, 3) : !AIE.objectFifoSubview<memref<64xui8>>
        %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        %12 = AIE.objectFifo.subview.access %10[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        %13 = AIE.objectFifo.subview.access %10[2] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        %14 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
        %15 = AIE.objectFifo.subview.access %14[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        func.call @filter2dLine(%11, %12, %13, %15, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
        AIE.objectFifo.release @OF_2to3(Consume, 1)
        AIE.objectFifo.release @OF_3to4(Produce, 1)
      }
      %5 = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<64xui8>>
      %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
      %7 = AIE.objectFifo.subview.access %5[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
      %8 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
      %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
      func.call @filter2dLine(%6, %7, %7, %9, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
      AIE.objectFifo.release @OF_2to3(Consume, 2)
      AIE.objectFifo.release @OF_3to4(Produce, 1)
      AIE.end
    } {link_with = "filter2d.cc.o"}
    %core_0_4 = AIE.core(%tile_0_4) {
      %c64_i32 = arith.constant 64 : i32
      %c10_i16 = arith.constant 10 : i16
      %c255_i16 = arith.constant 255 : i16
      %c0_i8 = arith.constant 0 : i8
      %c0 = arith.constant 0 : index
      %c36 = arith.constant 36 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c36 step %c1 {
        %0 = AIE.objectFifo.acquire @OF_3to4(Consume, 1) : !AIE.objectFifoSubview<memref<64xui8>>
        %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        %2 = AIE.objectFifo.acquire @OF_4to5(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
        %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        func.call @thresholdLine(%1, %3, %c64_i32, %c10_i16, %c255_i16, %c0_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
        AIE.objectFifo.release @OF_3to4(Consume, 1)
        AIE.objectFifo.release @OF_4to5(Produce, 1)
      }
      AIE.end
    } {link_with = "threshold.cc.o"}
    %core_0_5 = AIE.core(%tile_0_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0_i8 = arith.constant 0 : i8
      %c16384_i16 = arith.constant 16384 : i16
      %c64_i32 = arith.constant 64 : i32
      %c0 = arith.constant 0 : index
      %c36 = arith.constant 36 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c36 step %c1 {
        %0 = AIE.objectFifo.acquire @OF_4to5(Consume, 1) : !AIE.objectFifoSubview<memref<64xui8>>
        %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
        %2 = AIE.objectFifo.acquire @OF_5to5(Produce, 1) : !AIE.objectFifoSubview<memref<256xui8>>
        %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
        func.call @gray2rgbaLine(%1, %3, %c64_i32) : (memref<64xui8>, memref<256xui8>, i32) -> ()
        AIE.objectFifo.release @OF_4to5(Consume, 1)
        AIE.objectFifo.release @OF_5to5(Produce, 1)
        %4 = AIE.objectFifo.acquire @OF_5to5(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
        %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
        %6 = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
        %7 = AIE.objectFifo.subview.access %6[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
        %8 = AIE.objectFifo.acquire @outOF_L1L2(Produce, 1) : !AIE.objectFifoSubview<memref<256xui8>>
        %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
        func.call @addWeightedLine(%5, %7, %9, %c256_i32, %c16384_i16, %c16384_i16, %c0_i8) : (memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8) -> ()
        AIE.objectFifo.release @OF_5to5(Consume, 1)
        AIE.objectFifo.release @inOF_L2L1(Consume, 1)
        AIE.objectFifo.release @outOF_L1L2(Produce, 1)
      }
      AIE.end
    } {link_with = "combined_gray2rgba_addWeighted.a"}
    func.func @sequence(%arg0: memref<2304xi32>, %arg1: memref<2304xi32>, %arg2: memref<2304xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c36_i32 = arith.constant 36 : i32
      %c64_i32 = arith.constant 64 : i32
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c36_i32, %c64_i32] [%c0_i32, %c0_i32, %c64_i32]) {id = 0 : i32, metadata = @outOF_L2L3} : (i32, i32, memref<2304xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c36_i32, %c64_i32] [%c0_i32, %c0_i32, %c64_i32]) {id = 1 : i32, metadata = @inOF_L3L2} : (i32, i32, memref<2304xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

