module {
  aie.device(npu1_1col) {
    func.func private @zero_i16(memref<64x96xi16>)
    func.func private @matmul_i8_i16(memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inA(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x120xi8>> 
    aie.objectfifo @memA(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 480>, <size = 15, stride = 8>, <size = 4, stride = 120>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x120xi8>> 
    aie.objectfifo.link [@inA] -> [@memA]([] [])
    aie.objectfifo @inB(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<120x96xi8>> 
    aie.objectfifo @memB(%mem_tile_0_1 dimensionsToStream [<size = 15, stride = 768>, <size = 12, stride = 8>, <size = 8, stride = 96>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<120x96xi8>> 
    aie.objectfifo.link [@inB] -> [@memB]([] [])
    aie.objectfifo @memC(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x96xi16>> 
    aie.objectfifo @outC(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 384>, <size = 4, stride = 8>, <size = 12, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x96xi16>> 
    aie.objectfifo.link [@memC] -> [@outC]([] [])
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c40 = arith.constant 40 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c40 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<64x96xi16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x96xi16>> -> memref<64x96xi16>
          func.call @zero_i16(%1) : (memref<64x96xi16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<64x120xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x120xi8>> -> memref<64x120xi8>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<120x96xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<120x96xi8>> -> memref<120x96xi8>
            func.call @matmul_i8_i16(%3, %5, %1) : (memref<64x120xi8>, memref<120x96xi8>, memref<64x96xi16>) -> ()
            aie.objectfifo.release @memA(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm_64x120x96.o"}
    aiex.runtime_sequence(%arg0: memref<245760xi8>, %arg1: memref<230400xi8>, %arg2: memref<245760xi16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 7995392 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 491520 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 2 : i32, arg_plus = 491520 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 15 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][2, 5, 64, 96][30720, 96, 480, 1]) {id = 0 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][5, 4, 64, 120][0, 120, 480, 1]) {id = 1 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 2 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 30720][5, 4, 64, 120][0, 120, 480, 1]) {id = 3 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 4 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 61440][2, 5, 64, 96][30720, 96, 480, 1]) {id = 8 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 61440][5, 4, 64, 120][0, 120, 480, 1]) {id = 9 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 10 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 92160][5, 4, 64, 120][0, 120, 480, 1]) {id = 11 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 12 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 122880][2, 5, 64, 96][30720, 96, 480, 1]) {id = 0 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 122880][5, 4, 64, 120][0, 120, 480, 1]) {id = 1 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 2 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 153600][5, 4, 64, 120][0, 120, 480, 1]) {id = 3 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 4 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 184320][2, 5, 64, 96][30720, 96, 480, 1]) {id = 8 : i64, metadata = @outC} : memref<245760xi16>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 184320][5, 4, 64, 120][0, 120, 480, 1]) {id = 9 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 10 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 215040][5, 4, 64, 120][0, 120, 480, 1]) {id = 11 : i64, metadata = @inA} : memref<245760xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][5, 4, 120, 96][96, 57600, 480, 1]) {id = 12 : i64, metadata = @inB} : memref<230400xi8>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}

