
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module @test1_core_llvm1 {
  %tile12 = AIE.tile(1, 2)

  %buf12_0 = AIE.buffer(%tile12) { sym_name = "a" } : memref<256xi32>
//  %buf12_1 = AIE.buffer(%tile12) { sym_name = "b" } : memref<256xi32>

//  %lock12_0 = AIE.lock(%tile12, 0)
//  %lock12_1 = AIE.lock(%tile12, 1)

  %core12 = AIE.core(%tile12) {
//    AIE.useLock(%lock12_0, "Acquire", 0, 0)
    %val1 = constant 1 : i32
    %idx1 = constant 3 : index
    store %val1, %buf12_0[%idx1] : memref<256xi32>
    %val2 = constant 2 : i32
    %idx2 = constant 5 : index
    store %val2, %buf12_0[%idx2] : memref<256xi32>
//    AIE.useLock(%lock12_1, "Release", 1, 0)
    AIE.end
  }

}
