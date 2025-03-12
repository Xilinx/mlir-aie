module {
  aie.device(npu1_1col) {
    func.func private @zero_i32(memref<32x32xi32>)
    func.func private @matmul_i8_i32(memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inA(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xi8>> 
    aie.objectfifo @memA(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi8>> 
    aie.objectfifo.link [@inA] -> [@memA]([] [])
    aie.objectfifo @inB(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xi8>> 
    aie.objectfifo @memB(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi8>> 
    aie.objectfifo.link [@inB] -> [@memB]([] [])
    aie.objectfifo @memC(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @outC(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo.link [@memC] -> [@outC]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<32x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xi32>> -> memref<32x32xi32>
          func.call @zero_i32(%1) : (memref<32x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<32x32xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xi8>> -> memref<32x32xi8>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xi8>> -> memref<32x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<32x32xi8>, memref<32x32xi8>, memref<32x32xi32>) -> ()
            aie.objectfifo.release @memA(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_32x32x32.o"}
    aiex.runtime_sequence(%arg0: memref<65536xi8>, %arg1: memref<65536xi8>, %arg2: memref<65536xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][2, 8, 32, 32][8192, 32, 256, 1]) {id = 0 : i64, metadata = @outC} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][8, 8, 32, 32][0, 32, 256, 1]) {id = 1 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 2 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][8, 8, 32, 32][0, 32, 256, 1]) {id = 3 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 4 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 16384][2, 8, 32, 32][8192, 32, 256, 1]) {id = 8 : i64, metadata = @outC} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][8, 8, 32, 32][0, 32, 256, 1]) {id = 9 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 10 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][8, 8, 32, 32][0, 32, 256, 1]) {id = 11 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 12 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 32768][2, 8, 32, 32][8192, 32, 256, 1]) {id = 0 : i64, metadata = @outC} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 32768][8, 8, 32, 32][0, 32, 256, 1]) {id = 1 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 2 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 40960][8, 8, 32, 32][0, 32, 256, 1]) {id = 3 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 4 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 49152][2, 8, 32, 32][8192, 32, 256, 1]) {id = 8 : i64, metadata = @outC} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 49152][8, 8, 32, 32][0, 32, 256, 1]) {id = 9 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 10 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 57344][8, 8, 32, 32][0, 32, 256, 1]) {id = 11 : i64, metadata = @inA} : memref<65536xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][8, 8, 32, 32][32, 8192, 256, 1]) {id = 12 : i64, metadata = @inB} : memref<65536xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}

