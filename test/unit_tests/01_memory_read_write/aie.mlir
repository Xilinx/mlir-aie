
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module @test1_core_llvm1 {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

//  %lock13_0 = AIE.lock(%tile13, 0)
//  %lock13_1 = AIE.lock(%tile13, 1)

  %core13 = AIE.core(%tile13) {
//    AIE.useLock(%lock13_0, "Acquire", 0, 0)
    %val1 = constant 7 : i32
    %idx1 = constant 3 : index
    %2 = addi %val1, %val1 : i32
    store %2, %buf13_0[%idx1] : memref<256xi32>
    %val2 = constant 8 : i32
    %idx2 = constant 5 : index
    store %val2, %buf13_0[%idx2] : memref<256xi32>
    %val3 = load %buf13_0[%idx1] : memref<256xi32>
    %idx3 = constant 9 : index
    store %val3,%buf13_0[%idx3] : memref<256xi32>
//    AIE.useLock(%lock13_1, "Release", 1, 0)
    AIE.end
  }

}
