module {
  aie.device(npu1_1col) {
    func.func private @passthrough_64_i32(memref<64xi32>, memref<64xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        %2 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        func.call @passthrough_64_i32(%1, %3) : (memref<64xi32>, memref<64xi32>) -> ()
        aie.objectfifo.release @output_fifo(Produce, 1)
        aie.objectfifo.release @input_fifo(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aiex.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
