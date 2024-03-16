module {
  aie.device(xcvc1902) {
    %tile_6_0 = aie.tile(6, 0)
    %tile_6_2 = aie.tile(6, 2)
    aie.objectfifo @in(%tile_6_0, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%tile_6_2, {%tile_6_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
          %0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %4 = memref.load %1[%arg2] : memref<16xi32>
            %c3_i32 = arith.constant 3 : i32
            %5 = arith.muli %4, %c3_i32 : i32
            memref.store %5, %3[%arg2] : memref<16xi32>
          }
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
      }
      aie.end
    }
    func.func @sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<64xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<64xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

