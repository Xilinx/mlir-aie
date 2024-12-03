module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>> 
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>> 
    func.func private @passthrough_10_i32(memref<10xi32>, memref<10xi32>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c5 step %c1 {
        %0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %c0_0 = arith.constant 0 : index
        %c5_1 = arith.constant 5 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c5_1 step %c1_2 {
          %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
          func.call @passthrough_10_i32(%1, %3) : (memref<10xi32>, memref<10xi32>) -> ()
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @in(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aiex.runtime_sequence(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 50][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in} : memref<10xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 250][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<10xi32>
      aiex.npu.dma_wait {symbol = @in}
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

