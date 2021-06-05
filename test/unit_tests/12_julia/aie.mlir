// RUN: clang++ --target=aie -c -I/usr/include/aie %S/kernel.cpp
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%S/../../../runtime_lib/ %S/../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf


module @test {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3)

  func private @func(%A: memref<256xi32>, %B: memref<256xi32>) -> ()

  %core13 = AIE.core(%tile13) {
//    AIE.useLock(%lock13_3, "Acquire", 1, 0) // acquire
    call @func(%buf13_0, %buf13_1) : (memref<256xi32>, memref<256xi32>) -> ()
//    AIE.useLock(%lock13_3, "Release", 0, 0) // release for write
    AIE.end
  } { link_with="kernel.o" }
}
