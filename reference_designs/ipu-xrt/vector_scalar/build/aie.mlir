module {
  AIE.device(ipu) {
    func.func private @scale_int32(memref<1024xi32>, memref<1024xi32>)
    %tile_0_0 = AIE.tile(0, 0)
    %tile_0_2 = AIE.tile(0, 2)
    AIE.objectFifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
    AIE.objectFifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
    %core_0_2 = AIE.core(%tile_0_2) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          %0 = AIE.objectFifo.acquire @out(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
          %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
          %2 = AIE.objectFifo.acquire @in(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
          %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
          func.call @scale_int32(%3, %1) : (memref<1024xi32>, memref<1024xi32>) -> ()
          AIE.objectFifo.release @in(Consume, 1)
          AIE.objectFifo.release @out(Produce, 1)
        }
      }
      AIE.end
    } {link_with = "scale.o"}
    func.func @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4096_i32 = arith.constant 4096 : i32
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c4096_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 0 : i32, metadata = @out} : (i32, i32, memref<4096xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c4096_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 1 : i32, metadata = @in} : (i32, i32, memref<4096xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

