module {
  %00 = aie.core(0, 0)
  %11 = aie.core(1, 1)
  aie.flow(%00, "DMA" : 0, %11, "ME" : 1)
}
