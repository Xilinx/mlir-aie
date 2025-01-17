module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @shim_to_mem(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x20xi32>> 
    aie.objectfifo @mem_to_comp(%tile_0_1 dimensionsToStream [<size = 1, stride = 160>, <size = 4, stride = 5>, <size = 8, stride = 20>, <size = 5, stride = 1>], {%tile_0_2 dimensionsFromStream [<size = 4, stride = 20>, <size = 2, stride = 80>, <size = 20, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<8x20xi32>> 
    aie.objectfifo.link [@shim_to_mem] -> [@mem_to_comp]([] [])
    aie.objectfifo @comp_to_mem(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x20xi32>> 
    aie.objectfifo @mem_to_shim(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8x20xi32>> 
    aie.objectfifo.link [@comp_to_mem] -> [@mem_to_shim]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
          %0 = aie.objectfifo.acquire @mem_to_comp(Consume, 1) : !aie.objectfifosubview<memref<8x20xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x20xi32>> -> memref<8x20xi32>
          %2 = aie.objectfifo.acquire @comp_to_mem(Produce, 1) : !aie.objectfifosubview<memref<8x20xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x20xi32>> -> memref<8x20xi32>
          %c0_3 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_4 = arith.constant 1 : index
          scf.for %arg2 = %c0_3 to %c8 step %c1_4 {
            %c0_5 = arith.constant 0 : index
            %c20 = arith.constant 20 : index
            %c1_6 = arith.constant 1 : index
            scf.for %arg3 = %c0_5 to %c20 step %c1_6 {
              %4 = memref.load %1[%arg2, %arg3] : memref<8x20xi32>
              memref.store %4, %3[%arg2, %arg3] : memref<8x20xi32>
            }
          }
          aie.objectfifo.release @mem_to_comp(Consume, 1)
          aie.objectfifo.release @comp_to_mem(Produce, 1)
        }
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<160xi32>, %arg1: memref<160xi32>, %arg2: memref<160xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 160][0, 0, 0, 1]) {id = 1 : i64, metadata = @shim_to_mem} : memref<160xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 160][0, 0, 0, 1]) {id = 0 : i64, metadata = @mem_to_shim} : memref<160xi32>
      aiex.npu.dma_wait {symbol = @mem_to_shim}
    }
  }
}

