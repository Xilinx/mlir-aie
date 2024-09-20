module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in1(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c768 = arith.constant 768 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c768 step %c1_1 {
          %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %2 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %4 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %6 = memref.load %1[%arg2] : memref<16xi32>
            %7 = memref.load %3[%arg2] : memref<16xi32>
            %8 = arith.muli %6, %7 : i32
            memref.store %8, %5[%arg2] : memref<16xi32>
          }
          aie.objectfifo.release @in1(Consume, 1)
          aie.objectfifo.release @in2(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<12288xi32>, %arg1: memref<12288xi32>, %arg2: memref<12288xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 12288][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<12288xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 12288][0, 0, 0, 1]) {id = 1 : i64, metadata = @in1} : memref<12288xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 12288][0, 0, 0, 1]) {id = 2 : i64, metadata = @in2} : memref<12288xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}

