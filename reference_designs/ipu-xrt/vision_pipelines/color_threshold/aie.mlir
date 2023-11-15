module {
  AIE.device(ipu) {
    %0 = AIE.tile(0, 0)
    %1 = AIE.tile(0, 1)
    %2 = AIE.tile(0, 2)
    %3 = AIE.tile(0, 3)
    %4 = AIE.tile(0, 4)
    %5 = AIE.tile(0, 5)
    %rtp0 = AIE.buffer(%2) {sym_name = "rtp0"} : memref<16xi32>
    %rtp1 = AIE.buffer(%3) {sym_name = "rtp1"} : memref<16xi32>
    %rtp2 = AIE.buffer(%4) {sym_name = "rtp2"} : memref<16xi32>
    %rtp3 = AIE.buffer(%5) {sym_name = "rtp3"} : memref<16xi32>
    AIE.objectFifo @objFifo_in0(%0, {%1}, 2 : i32) : !AIE.objectFifo<memref<512xui8>>
    AIE.objectFifo @objFifo_in1(%1, {%2}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo @objFifo_in2(%1, {%3}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo @objFifo_in3(%1, {%4}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo @objFifo_in4(%1, {%5}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo.link [@objFifo_in0] -> [@objFifo_in1, @objFifo_in2, @objFifo_in3, @objFifo_in4] ()
    AIE.objectFifo @objFifo_out0(%1, {%0}, 2 : i32) : !AIE.objectFifo<memref<512xui8>>
    AIE.objectFifo @objFifo_out1(%2, {%1}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo @objFifo_out2(%3, {%1}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo @objFifo_out3(%4, {%1}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo @objFifo_out4(%5, {%1}, 2 : i32) : !AIE.objectFifo<memref<128xui8>>
    AIE.objectFifo.link [@objFifo_out1, @objFifo_out2, @objFifo_out3, @objFifo_out4] -> [@objFifo_out0] ()
    func.func private @thresholdLine(%in: memref<128xui8>, %out: memref<128xui8>, %lineWidth: i32,  %thresholdValue: i32, %maxValue: i32, %thresholdType: i8) -> ()
    %24 = AIE.core(%2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %lineWidth = arith.constant 128 : i32
      %maxValue = arith.constant 255 : i32
      %th = arith.constant 100 : i32
      %v0 = arith.constant 0 : i32
      memref.store %th, %rtp0[%c0] : memref<16xi32>
      memref.store %v0, %rtp0[%c1] : memref<16xi32>
      scf.for %arg0 = %c0 to %c4096 step %c1 {
        %subview0 = AIE.objectFifo.acquire @objFifo_in1(Consume, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %subview1 = AIE.objectFifo.acquire @objFifo_out1(Produce, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %thresh = memref.load %rtp0[%c0] : memref<16xi32>
        %tt = memref.load %rtp0[%c1] : memref<16xi32>
        %threshType = arith.trunci %tt : i32 to i8
        func.call @thresholdLine(%elem0,%elem1,%lineWidth,%thresh,%maxValue,%threshType) : (memref<128xui8>, memref<128xui8>, i32, i32, i32, i8) -> ()
        AIE.objectFifo.release @objFifo_in1(Consume, 1)
        AIE.objectFifo.release @objFifo_out1(Produce, 1)
      }
      AIE.end
    } {link_with = "threshold.o"}
    %34 = AIE.core(%3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %lineWidth = arith.constant 128 : i32
      %maxValue = arith.constant 255 : i32
      %th = arith.constant 100 : i32
      %v0 = arith.constant 0 : i32
      memref.store %th, %rtp1[%c0] : memref<16xi32>
      memref.store %v0, %rtp1[%c1] : memref<16xi32>
      scf.for %arg0 = %c0 to %c4096 step %c1 {
        %subview0 = AIE.objectFifo.acquire @objFifo_in2(Consume, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %subview1 = AIE.objectFifo.acquire @objFifo_out2(Produce, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %thresh = memref.load %rtp1[%c0] : memref<16xi32>
        %tt = memref.load %rtp1[%c1] : memref<16xi32>
        %threshType = arith.trunci %tt : i32 to i8
        func.call @thresholdLine(%elem0,%elem1,%lineWidth,%thresh,%maxValue,%threshType) : (memref<128xui8>, memref<128xui8>, i32, i32, i32, i8) -> ()
        AIE.objectFifo.release @objFifo_in2(Consume, 1)
        AIE.objectFifo.release @objFifo_out2(Produce, 1)
      }
      AIE.end
    } {link_with = "threshold.o"}
    %44 = AIE.core(%4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %lineWidth = arith.constant 128 : i32
      %maxValue = arith.constant 255 : i32
      %th = arith.constant 100 : i32
      %v0 = arith.constant 0 : i32
      memref.store %th, %rtp2[%c0] : memref<16xi32>
      memref.store %v0, %rtp2[%c1] : memref<16xi32>
      scf.for %arg0 = %c0 to %c4096 step %c1 {
        %subview0 = AIE.objectFifo.acquire @objFifo_in3(Consume, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %subview1 = AIE.objectFifo.acquire @objFifo_out3(Produce, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %thresh = memref.load %rtp2[%c0] : memref<16xi32>
        %tt = memref.load %rtp2[%c1] : memref<16xi32>
        %threshType = arith.trunci %tt : i32 to i8
        func.call @thresholdLine(%elem0,%elem1,%lineWidth,%thresh,%maxValue,%threshType) : (memref<128xui8>, memref<128xui8>, i32, i32, i32, i8) -> ()
        AIE.objectFifo.release @objFifo_in3(Consume, 1)
        AIE.objectFifo.release @objFifo_out3(Produce, 1)
      }
      AIE.end
    } {link_with = "threshold.o"}
    %54 = AIE.core(%5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %lineWidth = arith.constant 128 : i32
      %maxValue = arith.constant 255 : i32
      %th = arith.constant 100 : i32
      %v0 = arith.constant 0 : i32
      memref.store %th, %rtp3[%c0] : memref<16xi32>
      memref.store %v0, %rtp3[%c1] : memref<16xi32>
      scf.for %arg0 = %c0 to %c4096 step %c1 {
        %subview0 = AIE.objectFifo.acquire @objFifo_in4(Consume, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %subview1 = AIE.objectFifo.acquire @objFifo_out4(Produce, 1) : !AIE.objectFifoSubview<memref<128xui8>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<128xui8>> -> memref<128xui8>
        %thresh = memref.load %rtp3[%c0] : memref<16xi32>
        %tt = memref.load %rtp3[%c1] : memref<16xi32>
        %threshType = arith.trunci %tt : i32 to i8
        func.call @thresholdLine(%elem0,%elem1,%lineWidth,%thresh,%maxValue,%threshType) : (memref<128xui8>, memref<128xui8>, i32, i32, i32, i8) -> ()
        AIE.objectFifo.release @objFifo_in4(Consume, 1)
        AIE.objectFifo.release @objFifo_out4(Produce, 1)
      }
      AIE.end
    } {link_with = "threshold.o"}
    func.func @sequence(%in : memref<2048xi32>, %buf : memref<32xi32>, %out : memref<2048xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      %c128 = arith.constant 128 : i32
      %c512 = arith.constant 512 : i32
      AIEX.ipu.rtp_write(0, 2, 0, 50) { buffer_sym_name = "rtp0" }
      AIEX.ipu.rtp_write(0, 3, 0, 50) { buffer_sym_name = "rtp1" }
      AIEX.ipu.rtp_write(0, 4, 0, 50) { buffer_sym_name = "rtp2" }
      AIEX.ipu.rtp_write(0, 5, 0, 50) { buffer_sym_name = "rtp3" }
      AIEX.ipu.rtp_write(0, 2, 1, 0) { buffer_sym_name = "rtp0" }
      AIEX.ipu.rtp_write(0, 3, 1, 0) { buffer_sym_name = "rtp1" }
      AIEX.ipu.rtp_write(0, 4, 1, 0) { buffer_sym_name = "rtp2" }
      AIEX.ipu.rtp_write(0, 5, 1, 0) { buffer_sym_name = "rtp3" }
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0,%c0,%c0,%c0][%c1,%c4,%c4,%c128][%c0,%c128,%c512]) { metadata = @objFifo_out0, id = 1 : i32 } : (i32, i32, memref<2048xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0,%c0,%c0,%c0][%c1,%c4,%c4,%c128][%c0,%c128,%c512]) { metadata = @objFifo_in0, id = 0 : i32 } : (i32, i32, memref<2048xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      AIEX.ipu.sync { column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
      return
    }
  }
}

