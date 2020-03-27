

module {
  %2 = aie.core(2, 3)
  %3 = aie.core(2, 2)
  aie.flow(%2, "ME" : 0, %3, "ME" : 1)
  aie.flow(%3, "ME" : 0, %3, "ME" : 0)
  aie.flow(%3, "ME" : 1, %2, "ME" : 1)
  }
