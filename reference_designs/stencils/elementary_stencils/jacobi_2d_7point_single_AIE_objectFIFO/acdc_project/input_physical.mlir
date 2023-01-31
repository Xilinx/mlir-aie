module @stencil_2d_7point {
  %0 = AIE.tile(7, 1)
  %1 = AIE.switchbox(%0) {
    AIE.connect<South : 0, DMA : 0>
    AIE.connect<DMA : 0, South : 0>
  }
  %2 = AIE.tile(7, 0)
  %3 = AIE.switchbox(%2) {
    AIE.connect<South : 3, North : 0>
    AIE.connect<North : 0, South : 2>
  }
  %4 = AIE.lock(%0, 14) {sym_name = "lock71_14"}
  %5 = AIE.shimmux(%2) {
    AIE.connect<DMA : 0, North : 3>
    AIE.connect<North : 2, DMA : 0>
  }
  %6 = AIE.lock(%2, 0) {sym_name = "of_0_lock_0"}
  %7 = AIE.buffer(%0) {address = 4096 : i32, sym_name = "of_1_buff_0"} : memref<256xf32>
  %8 = AIE.lock(%0, 0) {sym_name = "of_1_lock_0"}
  %9 = AIE.buffer(%0) {address = 5120 : i32, sym_name = "of_1_buff_1"} : memref<256xf32>
  %10 = AIE.lock(%0, 1) {sym_name = "of_1_lock_1"}
  %11 = AIE.buffer(%0) {address = 6144 : i32, sym_name = "of_1_buff_2"} : memref<256xf32>
  %12 = AIE.lock(%0, 2) {sym_name = "of_1_lock_2"}
  %13 = AIE.buffer(%0) {address = 7168 : i32, sym_name = "of_1_buff_3"} : memref<256xf32>
  %14 = AIE.lock(%0, 3) {sym_name = "of_1_lock_3"}
  %15 = AIE.buffer(%0) {address = 8192 : i32, sym_name = "of_2_buff_0"} : memref<256xf32>
  %16 = AIE.lock(%0, 4) {sym_name = "of_2_lock_0"}
  %17 = AIE.buffer(%0) {address = 9216 : i32, sym_name = "of_2_buff_1"} : memref<256xf32>
  %18 = AIE.lock(%0, 5) {sym_name = "of_2_lock_1"}
  %19 = AIE.lock(%2, 1) {sym_name = "of_3_lock_0"}
  %20 = AIE.external_buffer {sym_name = "ddr_test_buffer_in0"} : memref<768xf32>
  %21 = AIE.external_buffer {sym_name = "ddr_test_buffer_out"} : memref<256xf32>
  func.func private @stencil_2d_7point_fp32(memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>)
  %22 = AIE.core(%0) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    AIE.useLock(%4, Acquire, 0)
    AIE.useLock(%8, Acquire, 1)
    AIE.useLock(%10, Acquire, 1)
    AIE.useLock(%12, Acquire, 1)
    AIE.useLock(%16, Acquire, 0)
    func.call @stencil_2d_7point_fp32(%7, %9, %11, %15) : (memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>) -> ()
    AIE.useLock(%8, Release, 0)
    AIE.useLock(%16, Release, 1)
    AIE.useLock(%4, Release, 0)
    AIE.end
  } {link_with = "stencil_2d_7point_fp32.o"}
  %23 = AIE.mem(%0) {
    %25 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    AIE.useLock(%16, Acquire, 1)
    AIE.dmaBd(<%15 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%16, Release, 0)
    AIE.nextBd ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%18, Acquire, 1)
    AIE.dmaBd(<%17 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%18, Release, 0)
    AIE.nextBd ^bb1
  ^bb3:  // pred: ^bb0
    %26 = AIE.dmaStart(S2MM, 0, ^bb4, ^bb8)
  ^bb4:  // 2 preds: ^bb3, ^bb7
    AIE.useLock(%8, Acquire, 0)
    AIE.dmaBd(<%7 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%8, Release, 1)
    AIE.nextBd ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBd(<%9 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%10, Release, 1)
    AIE.nextBd ^bb6
  ^bb6:  // pred: ^bb5
    AIE.useLock(%12, Acquire, 0)
    AIE.dmaBd(<%11 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%12, Release, 1)
    AIE.nextBd ^bb7
  ^bb7:  // pred: ^bb6
    AIE.useLock(%14, Acquire, 0)
    AIE.dmaBd(<%13 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%14, Release, 1)
    AIE.nextBd ^bb4
  ^bb8:  // pred: ^bb3
    AIE.end
  }
  %24 = AIE.shimDMA(%2) {
    %25 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%19, Acquire, 0)
    AIE.dmaBd(<%21 : memref<256xf32>, 0, 256>, 0)
    AIE.useLock(%19, Release, 1)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb0
    %26 = AIE.dmaStart(MM2S, 0, ^bb3, ^bb4)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%6, Acquire, 1)
    AIE.dmaBd(<%20 : memref<768xf32>, 0, 768>, 0)
    AIE.useLock(%6, Release, 0)
    AIE.nextBd ^bb3
  ^bb4:  // pred: ^bb2
    AIE.end
  }
  AIE.wire(%5 : North, %3 : South)
  AIE.wire(%2 : DMA, %5 : DMA)
  AIE.wire(%0 : Core, %1 : Core)
  AIE.wire(%0 : DMA, %1 : DMA)
  AIE.wire(%3 : North, %1 : South)
}

