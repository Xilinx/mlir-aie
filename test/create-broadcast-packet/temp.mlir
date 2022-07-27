module @test_broadcast_packet {
  %0 = AIE.tile(6, 3)
  %1 = AIE.tile(6, 4)
  %2 = AIE.tile(7, 2)
  %3 = AIE.tile(7, 3)
  %4 = AIE.tile(7, 4)
  AIE.packet_flow(1) {
    AIE.packet_source<%2, DMA : 0>
    AIE.packet_dest<%4, DMA : 0>
    AIE.packet_dest<%1, DMA : 0>
  }
  AIE.packet_flow(0) {
    AIE.packet_source<%2, DMA : 0>
    AIE.packet_dest<%3, DMA : 0>
    AIE.packet_dest<%0, DMA : 0>
  }
}

