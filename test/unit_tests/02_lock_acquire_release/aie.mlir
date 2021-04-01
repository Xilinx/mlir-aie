// RUN: aiecc.py --sysroot=${VITIS_SYSROOT} %s -I%S/../../../runtime_lib %S/../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf

module @test02_lock_acquire_release {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3)
  %lock13_5 = AIE.lock(%tile13, 5)

  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 0, 0) // acquire for write (e.g. input ping)
    AIE.useLock(%lock13_5, "Acquire", 0, 0) // acquire for write (e.g. input ping)
    AIE.useLock(%lock13_5, "Release", 1, 0) // release for read 
    AIE.end
  }

}
