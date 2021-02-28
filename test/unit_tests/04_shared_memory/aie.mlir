// RUN: aiecc.py --sysroot=/group/xrlabs/platforms/pynq_on_versal_vck190_hacked/vck190-sysroot %s %S/test.cpp ../../../runtime_lib/test_library.cpp -o test.elf

module @test3_core_llvm1 {
  %tile13 = AIE.tile(1, 3)
  %tile14 = AIE.tile(1, 4)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>
  %buf14_0 = AIE.buffer(%tile14) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) // input buffer lock
  %lock13_5 = AIE.lock(%tile13, 5) // interbuffer lock
  %lock14_7 = AIE.lock(%tile14, 7) // output buffer lock

  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 1, 0) // acquire for read(e.g. input ping)
    AIE.useLock(%lock13_5, "Acquire", 0, 0) // acquire for write
    %idx1 = constant 3 : index
    %val1 = load %buf13_0[%idx1] : memref<256xi32>
    %2    = addi %val1, %val1 : i32
    %3 = addi %2, %val1 : i32
    %4 = addi %3, %val1 : i32
    %5 = addi %4, %val1 : i32
    %idx2 = constant 5 : index
    store %5, %buf13_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock13_3, "Release", 0, 0) // release for write
    AIE.useLock(%lock13_5, "Release", 1, 0) // release for read
    AIE.end
  }

  %core14 = AIE.core(%tile14) {
    AIE.useLock(%lock13_5, "Acquire", 1, 0) // acquire for read(e.g. input ping)
    AIE.useLock(%lock14_7, "Acquire", 0, 0) // acquire for write
    %idx1 = constant 5 : index
    %val1 = load %buf13_1[%idx1] : memref<256xi32>
    %2    = addi %val1, %val1 : i32
    %3 = addi %2, %val1 : i32
    %4 = addi %3, %val1 : i32
    %5 = addi %4, %val1 : i32
    %idx2 = constant 5 : index
    store %5, %buf14_0[%idx2] : memref<256xi32>
    AIE.useLock(%lock13_5, "Release", 0, 0) // release for write
    AIE.useLock(%lock14_7, "Release", 1, 0) // release for read
    AIE.end
  }
}
