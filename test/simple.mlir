module {
  %00 = aie.core(0, 0)
  %11 = aie.core(1, 1)
  %01 = aie.core(0, 1)
  aie.flow(%00, "DMA" : 0, %11, "ME" : 1)
  aie.packet_flow(0x10) {
    aie.packet_source < %00, "ME" : 0>
	 aie.packet_dest < %11, "ME" : 0>
	 aie.packet_dest < %01, "DMA" : 1>
  }
}
