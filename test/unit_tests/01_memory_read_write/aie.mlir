// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test01_memory_read_write {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

  %core13 = AIE.core(%tile13) {
    %val1 = constant 7 : i32
    %idx1 = constant 3 : index
    %2 = addi %val1, %val1 : i32
    memref.store %2, %buf13_0[%idx1] : memref<256xi32>
    %val2 = constant 8 : i32
    %idx2 = constant 5 : index
    memref.store %val2, %buf13_0[%idx2] : memref<256xi32>
    %val3 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %idx3 = constant 9 : index
    memref.store %val3,%buf13_0[%idx3] : memref<256xi32>
    AIE.end
  }
}
