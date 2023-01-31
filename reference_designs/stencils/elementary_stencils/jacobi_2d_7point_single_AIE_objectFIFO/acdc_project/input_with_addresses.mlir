module @stencil_2d_7point {
  %0 = AIE.tile(7, 1)
  %1 = AIE.switchbox(%0) {
  }
  %2 = AIE.tile(7, 0)
  %3 = AIE.switchbox(%2) {
  }
  %4 = AIE.lock(%0, 14) {sym_name = "lock71_14"}
  AIE.flow(%2, DMA : 0, %0, DMA : 0)
  %5 = AIE.lock(%2, 0) {sym_name = "of_0_lock_0"}
  %6 = AIE.buffer(%0) {address = 4096 : i32, sym_name = "of_1_buff_0"} : memref<256xf32>
  %7 = AIE.lock(%0, 0) {sym_name = "of_1_lock_0"}
  %8 = AIE.buffer(%0) {address = 5120 : i32, sym_name = "of_1_buff_1"} : memref<256xf32>
  %9 = AIE.lock(%0, 1) {sym_name = "of_1_lock_1"}
  %10 = AIE.buffer(%0) {address = 6144 : i32, sym_name = "of_1_buff_2"} : memref<256xf32>
  %11 = AIE.lock(%0, 2) {sym_name = "of_1_lock_2"}
  %12 = AIE.buffer(%0) {address = 7168 : i32, sym_name = "of_1_buff_3"} : memref<256xf32>
  %13 = AIE.lock(%0, 3) {sym_name = "of_1_lock_3"}
  AIE.flow(%0, DMA : 0, %2, DMA : 0)
  %14 = AIE.buffer(%0) {address = 8192 : i32, sym_name = "of_2_buff_0"} : memref<256xf32>
  %15 = AIE.lock(%0, 4) {sym_name = "of_2_lock_0"}
  %16 = AIE.buffer(%0) {address = 9216 : i32, sym_name = "of_2_buff_1"} : memref<256xf32>
  %17 = AIE.lock(%0, 5) {sym_name = "of_2_lock_1"}
  %18 = AIE.lock(%2, 1) {sym_name = "of_3_lock_0"}
  %19 = AIE.external_buffer {sym_name = "ddr_test_buffer_in0"} : memref<768xf32>
  %20 = AIE.external_buffer {sym_name = "ddr_test_buffer_out"} : memref<256xf32>
  func.func private @stencil_2d_7point_fp32(memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>)
  %21 = AIE.core(%0) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    AIE.useLock(%4, Acquire, 0)
    AIE.useLock(%7, Acquire, 1)
    AIE.useLock(%9, Acquire, 1)
    AIE.useLock(%11, Acquire, 1)
    AIE.useLock(%15, Acquire, 0)
    func.call @stencil_2d_7point_fp32(%6, %8, %10, %14) : (memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) -> ()
    AIE.useLock(%7, Release, 0)
    AIE.useLock(%15, Release, 1)
    AIE.useLock(%4, Release, 0)
    AIE.end
  } {link_with = "stencil_2d_7point_fp32.o"}
  %22 = AIE.mem(%0) {
    %24 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    AIE.useLock(%15, Acquire, 1)
    AIE.dmaBd(<%14 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%15, Release, 0)
    AIE.nextBd ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%17, Acquire, 1)
    AIE.dmaBd(<%16 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%17, Release, 0)
    AIE.nextBd ^bb1
  ^bb3:  // pred: ^bb0
    %25 = AIE.dmaStart(S2MM, 0, ^bb4, ^bb8)
  ^bb4:  // 2 preds: ^bb3, ^bb7
    AIE.useLock(%7, Acquire, 0)
    AIE.dmaBd(<%6 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%7, Release, 1)
    AIE.nextBd ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%9, Acquire, 0)
    AIE.dmaBd(<%8 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%9, Release, 1)
    AIE.nextBd ^bb6
  ^bb6:  // pred: ^bb5
    AIE.useLock(%11, Acquire, 0)
    AIE.dmaBd(<%10 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%11, Release, 1)
    AIE.nextBd ^bb7
  ^bb7:  // pred: ^bb6
    AIE.useLock(%13, Acquire, 0)
    AIE.dmaBd(<%12 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%13, Release, 1)
    AIE.nextBd ^bb4
  ^bb8:  // pred: ^bb3
    AIE.end
  }
  %23 = AIE.shimDMA(%2) {
    %24 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%18, Acquire, 0)
    AIE.dmaBd(<%20 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%18, Release, 1)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb0
    %25 = AIE.dmaStart(MM2S, 0, ^bb3, ^bb4)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%5, Acquire, 1)
    AIE.dmaBd(<%19 : memref<768xf32>, 0, 768>, 0)
    AIE.useLock(%5, Release, 0)
    AIE.nextBd ^bb3
  ^bb4:  // pred: ^bb2
    AIE.end
  }
}

