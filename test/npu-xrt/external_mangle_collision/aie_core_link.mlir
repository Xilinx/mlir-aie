module {
  func.func private @kernel(memref<8xi32>, i32)

  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
    
    aie.objectfifo @objFifo_in2(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in2] ([] [])
    
    aie.objectfifo @objFifo_out2(%t03, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_out2] -> [@objFifo_out0] ([] [])
  
    aie.core(%t02) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : i32
  
      scf.for %steps = %c0 to %c8 step %c1 {
        %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        
        memref.copy %elem0, %elem1 : memref<8xi32> to memref<8xi32>
        
        func.call @kernel(%elem1, %c10) : (memref<8xi32>, i32) -> ()
        
        aie.objectfifo.release @objFifo_in1(Consume, 1)
        aie.objectfifo.release @objFifo_out1(Produce, 1)
      }
      aie.end
    } { link_with="kernel1.o" }

    aie.core(%t03) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : i32
  
      scf.for %steps = %c0 to %c8 step %c1 {
        %subview0 = aie.objectfifo.acquire @objFifo_in2(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = aie.objectfifo.acquire @objFifo_out2(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        
        memref.copy %elem0, %elem1 : memref<8xi32> to memref<8xi32>
        
        func.call @kernel(%elem1, %c5) : (memref<8xi32>, i32) -> ()
        
        aie.objectfifo.release @objFifo_in2(Consume, 1)
        aie.objectfifo.release @objFifo_out2(Produce, 1)
      }
      aie.end
    } { link_with="kernel2.o" }

    aie.runtime_sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_out0, id = 1 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_in0, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
