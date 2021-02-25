// RUN: aiecc.py --sysroot=/group/xrlabs/platforms/pynq_on_versal_vck190_hacked/vck190-sysroot %s %S/test.cpp -o test.elf

module @test3_core_llvm1 {
  %tile13 = AIE.tile(1, 3)
  %tile23 = AIE.tile(2, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
//  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>
  %buf23_0 = AIE.buffer(%tile23) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) // input buffer lock
  %lock23_7 = AIE.lock(%tile23, 7) // output buffer lock

  %core13 = AIE.core(%tile13) {
//    AIE.useLock(%lock13_3, "Acquire", 1, 0) // acquire for read(e.g. input ping)
//    %idx1 = constant 3 : index
//    %val1 = load %buf13_0[%idx1] : memref<256xi32>
//    %2    = addi %val1, %val1 : i32
//    %3 = addi %2, %val1 : i32
//    %4 = addi %3, %val1 : i32
//    %5 = addi %4, %val1 : i32
//    %idx2 = constant 5 : index
//    puc_mcd(upd_elem(v16int16,%5,0))
//    AIE.useLock(%lock13_3, "Release", 0, 0) // release for write
    AIE.end
  }


  %core23 = AIE.core(%tile23) {
//    AIE.useLock(%lock23_7, "Acquire", 0, 0) // acquire for write
//    %idx1 = constant 3 : index
//    %val1 = ext_elem(srs(read_cascade,0),0)
//    %2    = addi %val1, %val1 : i32
//    %3 = addi %2, %val1 : i32
//    %4 = addi %3, %val1 : i32
//    %5 = addi %4, %val1 : i32
//    %idx2 = constant 5 : index
//    store %5, %buf23_0[%idx2] : memref<256xi32>
//    AIE.useLock(%lock23_7, "Release", 1, 0) // release for read
    AIE.end
  }

}
