module {
  AIE.device(ipu) {
    func.func private @zero_scalar_i16(memref<64x64xi16>)
    func.func private @zero_i16(memref<64x64xi16>)
    func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
    %tile_0_0 = AIE.tile(0, 0)
    %tile_0_1 = AIE.tile(0, 1)
    %tile_0_2 = AIE.tile(0, 2)
    AIE.objectFifo @inA(%tile_0_0, {%tile_0_2}, 2 : i32) : !AIE.objectFifo<memref<64x32xi16>>
    AIE.objectFifo @inB(%tile_0_0, {%tile_0_2}, 2 : i32) : !AIE.objectFifo<memref<32x64xi16>>
    AIE.objectFifo @outC(%tile_0_2, {%tile_0_0}, 2 : i32) : !AIE.objectFifo<memref<64x64xi16>>
    %core_0_2 = AIE.core(%tile_0_2) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          %0 = AIE.objectFifo.acquire @outC(Produce, 1) : !AIE.objectFifoSubview<memref<64x64xi16>>
          %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64x64xi16>> -> memref<64x64xi16>
          func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
          scf.for %arg2 = %c0 to %c4 step %c1 {
            %2 = AIE.objectFifo.acquire @inA(Consume, 1) : !AIE.objectFifoSubview<memref<64x32xi16>>
            %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<64x32xi16>> -> memref<64x32xi16>
            %4 = AIE.objectFifo.acquire @inB(Consume, 1) : !AIE.objectFifoSubview<memref<32x64xi16>>
            %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<32x64xi16>> -> memref<32x64xi16>
            func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
            AIE.objectFifo.release @inA(Consume, 1)
            AIE.objectFifo.release @inB(Consume, 1)
          }
          AIE.objectFifo.release @outC(Produce, 1)
        }
      }
      AIE.end
    } {link_with = "mm.o"}
    func.func @sequence(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>, %arg2: memref<8192xi32>) {
      %c2048_i32 = arith.constant 2048 : i32
      %c16_i32 = arith.constant 16 : i32
      %c4_i32 = arith.constant 4 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      %c64_i32 = arith.constant 64 : i32
      %c32_i32 = arith.constant 32 : i32
      %c4096_i32 = arith.constant 4096 : i32
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c2_i32, %c64_i32, %c32_i32] [%c4096_i32, %c32_i32, %c64_i32]) {id = 0 : i32, metadata = @outC} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c4_i32, %c64_i32, %c16_i32] [%c0_i32, %c16_i32, %c64_i32]) {id = 1 : i32, metadata = @inA} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c4_i32, %c32_i32, %c32_i32] [%c32_i32, %c2048_i32, %c64_i32]) {id = 2 : i32, metadata = @inB} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c4096_i32] [%c2_i32, %c4_i32, %c64_i32, %c16_i32] [%c0_i32, %c16_i32, %c64_i32]) {id = 3 : i32, metadata = @inA} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c4_i32, %c32_i32, %c32_i32] [%c32_i32, %c2048_i32, %c64_i32]) {id = 4 : i32, metadata = @inB} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

