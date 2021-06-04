// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%S/../../../runtime_lib %S/../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf

module @test {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xf32>

  %core13 = AIE.core(%tile13) {
    %val1 = constant 7.0 : f32
    %idx1 = constant 3 : index
    %2 = addf %val1, %val1 : f32
    memref.store %2, %buf13_0[%idx1] : memref<256xf32>
    %val2 = constant 8.0 : f32
    %idx2 = constant 5 : index
    memref.store %val2, %buf13_0[%idx2] : memref<256xf32>
    %val3 = memref.load %buf13_0[%idx1] : memref<256xf32>
    %idx3 = constant 9 : index
    memref.store %val3,%buf13_0[%idx3] : memref<256xf32>
    AIE.end
  }
}
