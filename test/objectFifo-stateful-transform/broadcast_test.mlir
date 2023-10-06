//===- broadcast_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: September 5th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: module @broadcast {
//CHECK:   AIE.device(xcvc1902) {
//CHECK:     memref.global "public" @broadcast_of_0_cons : memref<16xi32>
//CHECK:     memref.global "public" @broadcast_of_1_cons : memref<16xi32>
//CHECK:     memref.global "public" @broadcast_of_2_cons : memref<16xi32>
//CHECK:     memref.global "public" @broadcast_of_3_cons : memref<16xi32>
//CHECK:     memref.global "public" @broadcast_of : memref<16xi32>
//CHECK:     %0 = AIE.tile(1, 2)
//CHECK:     %1 = AIE.tile(1, 3)
//CHECK:     %2 = AIE.tile(1, 4)
//CHECK:     %3 = AIE.tile(3, 2)
//CHECK:     %4 = AIE.tile(3, 3)
//CHECK:     %5 = AIE.buffer(%0) {sym_name = "broadcast_of_0_cons_buff_0"} : memref<16xi32>
//CHECK:     %6 = AIE.buffer(%0) {sym_name = "broadcast_of_0_cons_buff_1"} : memref<16xi32>
//CHECK:     %7 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "broadcast_of_0_cons_lock_0"}
//CHECK:     %8 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "broadcast_of_0_cons_lock_1"}
//CHECK:     %9 = AIE.buffer(%2) {sym_name = "broadcast_of_1_cons_buff_0"} : memref<16xi32>
//CHECK:     %10 = AIE.buffer(%2) {sym_name = "broadcast_of_1_cons_buff_1"} : memref<16xi32>
//CHECK:     %11 = AIE.buffer(%2) {sym_name = "broadcast_of_1_cons_buff_2"} : memref<16xi32>
//CHECK:     %12 = AIE.lock(%2, 0) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_0"}
//CHECK:     %13 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_1"}
//CHECK:     %14 = AIE.lock(%2, 2) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_2"}
//CHECK:     %15 = AIE.buffer(%3) {sym_name = "broadcast_of_2_cons_buff_0"} : memref<16xi32>
//CHECK:     %16 = AIE.buffer(%3) {sym_name = "broadcast_of_2_cons_buff_1"} : memref<16xi32>
//CHECK:     %17 = AIE.buffer(%3) {sym_name = "broadcast_of_2_cons_buff_2"} : memref<16xi32>
//CHECK:     %18 = AIE.buffer(%3) {sym_name = "broadcast_of_2_cons_buff_3"} : memref<16xi32>
//CHECK:     %19 = AIE.lock(%3, 0) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_0"}
//CHECK:     %20 = AIE.lock(%3, 1) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_1"}
//CHECK:     %21 = AIE.lock(%3, 2) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_2"}
//CHECK:     %22 = AIE.lock(%3, 3) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_3"}
//CHECK:     %23 = AIE.buffer(%4) {sym_name = "broadcast_of_3_cons_buff_0"} : memref<16xi32>
//CHECK:     %24 = AIE.buffer(%4) {sym_name = "broadcast_of_3_cons_buff_1"} : memref<16xi32>
//CHECK:     %25 = AIE.buffer(%4) {sym_name = "broadcast_of_3_cons_buff_2"} : memref<16xi32>
//CHECK:     %26 = AIE.lock(%4, 0) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_0"}
//CHECK:     %27 = AIE.lock(%4, 1) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_1"}
//CHECK:     %28 = AIE.lock(%4, 2) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_2"}
//CHECK:     %29 = AIE.buffer(%1) {sym_name = "broadcast_of_buff_0"} : memref<16xi32>
//CHECK:     %30 = AIE.buffer(%1) {sym_name = "broadcast_of_buff_1"} : memref<16xi32>
//CHECK:     %31 = AIE.lock(%1, 0) {init = 0 : i32, sym_name = "broadcast_of_lock_0"}
//CHECK:     %32 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "broadcast_of_lock_1"}
//CHECK:     AIE.flow(%1, DMA : 0, %4, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %3, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %2, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %0, DMA : 0)
//CHECK:     func.func @some_work(%arg0: memref<16xi32>) {
//CHECK:       return
//CHECK:     }
//CHECK:     %33 = AIE.core(%1) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c2 = arith.constant 2 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c2 {
//CHECK:         AIE.useLock(%31, Acquire, 0)
//CHECK:         func.call @some_work(%29) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%31, Release, 1)
//CHECK:         AIE.useLock(%32, Acquire, 0)
//CHECK:         func.call @some_work(%30) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%32, Release, 1)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %34 = AIE.core(%0) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c2 = arith.constant 2 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c2 {
//CHECK:         AIE.useLock(%7, Acquire, 1)
//CHECK:         func.call @some_work(%5) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%7, Release, 0)
//CHECK:         AIE.useLock(%8, Acquire, 1)
//CHECK:         func.call @some_work(%6) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%8, Release, 0)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %35 = AIE.core(%2) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c3 = arith.constant 3 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c3 {
//CHECK:         AIE.useLock(%12, Acquire, 1)
//CHECK:         AIE.useLock(%13, Acquire, 1)
//CHECK:         func.call @some_work(%9) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%10) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%12, Release, 0)
//CHECK:         AIE.useLock(%13, Release, 0)
//CHECK:         AIE.useLock(%14, Acquire, 1)
//CHECK:         AIE.useLock(%12, Acquire, 1)
//CHECK:         func.call @some_work(%11) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%9) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%14, Release, 0)
//CHECK:         AIE.useLock(%12, Release, 0)
//CHECK:         AIE.useLock(%13, Acquire, 1)
//CHECK:         AIE.useLock(%14, Acquire, 1)
//CHECK:         func.call @some_work(%10) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%11) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%13, Release, 0)
//CHECK:         AIE.useLock(%14, Release, 0)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %36 = AIE.core(%3) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c4 = arith.constant 4 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c4 {
//CHECK:         AIE.useLock(%19, Acquire, 1)
//CHECK:         AIE.useLock(%20, Acquire, 1)
//CHECK:         AIE.useLock(%21, Acquire, 1)
//CHECK:         func.call @some_work(%15) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%16) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%17) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%19, Release, 0)
//CHECK:         AIE.useLock(%22, Acquire, 1)
//CHECK:         func.call @some_work(%16) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%17) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%18) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%20, Release, 0)
//CHECK:         AIE.useLock(%19, Acquire, 1)
//CHECK:         func.call @some_work(%17) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%18) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%15) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%21, Release, 0)
//CHECK:         AIE.useLock(%20, Acquire, 1)
//CHECK:         func.call @some_work(%18) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%15) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%16) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%22, Release, 0)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %37 = AIE.core(%4) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c3 = arith.constant 3 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c3 {
//CHECK:         AIE.useLock(%26, Acquire, 1)
//CHECK:         AIE.useLock(%27, Acquire, 1)
//CHECK:         func.call @some_work(%23) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%24) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%26, Release, 0)
//CHECK:         AIE.useLock(%28, Acquire, 1)
//CHECK:         func.call @some_work(%24) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%25) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%27, Release, 0)
//CHECK:         AIE.useLock(%26, Acquire, 1)
//CHECK:         func.call @some_work(%25) : (memref<16xi32>) -> ()
//CHECK:         func.call @some_work(%23) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%28, Release, 0)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %38 = AIE.mem(%1) {
//CHECK:       %43 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%31, Acquire, 1)
//CHECK:       AIE.dmaBd(<%29 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%31, Release, 0)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%32, Acquire, 1)
//CHECK:       AIE.dmaBd(<%30 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%32, Release, 0)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %39 = AIE.mem(%0) {
//CHECK:       %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%7, Acquire, 0)
//CHECK:       AIE.dmaBd(<%5 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%8, Acquire, 0)
//CHECK:       AIE.dmaBd(<%6 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%8, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %40 = AIE.mem(%2) {
//CHECK:       %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
//CHECK:       AIE.useLock(%12, Acquire, 0)
//CHECK:       AIE.dmaBd(<%9 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%12, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%13, Acquire, 0)
//CHECK:       AIE.dmaBd(<%10 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%13, Release, 1)
//CHECK:       AIE.nextBd ^bb3
//CHECK:     ^bb3:  // pred: ^bb2
//CHECK:       AIE.useLock(%14, Acquire, 0)
//CHECK:       AIE.dmaBd(<%11 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%14, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb4:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %41 = AIE.mem(%3) {
//CHECK:       %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
//CHECK:       AIE.useLock(%19, Acquire, 0)
//CHECK:       AIE.dmaBd(<%15 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%20, Acquire, 0)
//CHECK:       AIE.dmaBd(<%16 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%20, Release, 1)
//CHECK:       AIE.nextBd ^bb3
//CHECK:     ^bb3:  // pred: ^bb2
//CHECK:       AIE.useLock(%21, Acquire, 0)
//CHECK:       AIE.dmaBd(<%17 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%21, Release, 1)
//CHECK:       AIE.nextBd ^bb4
//CHECK:     ^bb4:  // pred: ^bb3
//CHECK:       AIE.useLock(%22, Acquire, 0)
//CHECK:       AIE.dmaBd(<%18 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%22, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb5:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %42 = AIE.mem(%4) {
//CHECK:       %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
//CHECK:       AIE.useLock(%26, Acquire, 0)
//CHECK:       AIE.dmaBd(<%23 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%26, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%27, Acquire, 0)
//CHECK:       AIE.dmaBd(<%24 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%27, Release, 1)
//CHECK:       AIE.nextBd ^bb3
//CHECK:     ^bb3:  // pred: ^bb2
//CHECK:       AIE.useLock(%28, Acquire, 0)
//CHECK:       AIE.dmaBd(<%25 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%28, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb4:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }

module @broadcast {
 AIE.device(xcvc1902) {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)
    %tile14 = AIE.tile(1, 4)
    %tile32 = AIE.tile(3, 2)
    %tile33 = AIE.tile(3, 3)

    AIE.objectFifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3, 4, 3]) : !AIE.objectFifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core13 = AIE.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = AIE.objectFifo.acquire @broadcast_of (Produce, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @broadcast_of (Produce, 1)
        }
        
        AIE.end
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = AIE.objectFifo.acquire @broadcast_of (Consume, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @broadcast_of (Consume, 1)
        }
        
        AIE.end
    }

    %core14 = AIE.core(%tile14) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = AIE.objectFifo.acquire @broadcast_of (Consume, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @broadcast_of (Consume, 2)
        }
        
        AIE.end
    }

    %core32 = AIE.core(%tile32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 { 
            %subview = AIE.objectFifo.acquire @broadcast_of (Consume, 3) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem2 = AIE.objectFifo.subview.access %subview[2] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @broadcast_of (Consume, 1)
        }
        
        AIE.end
    }

    %core33 = AIE.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 { 
            %subview = AIE.objectFifo.acquire @broadcast_of (Consume, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @broadcast_of (Consume, 1)
        }
        
        AIE.end
    }
 }
}
