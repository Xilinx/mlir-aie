// RUN: aiecc.py %s

module @test00_itsalive {
  %tile12 = AIE.tile(1, 2)

  %buf12_0 = AIE.buffer(%tile12) { sym_name = "a", address = 0 } : memref<256xi32>

  %core12 = AIE.core(%tile12) {
    %val1 = constant 1 : i32
    %idx1 = constant 3 : index
    %2 = addi %val1, %val1 : i32
    AIE.end
  }
}
