//===- bd_chain_on_memtile/bd_chain_with_repeat_count.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s


// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb5, repeat_count = 4)
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       aie.use_lock(%of1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%of1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%of1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%of1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%of1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.end
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%tile_1_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 4)
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       aie.use_lock(%of1_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%of1_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.end
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module {
  aie.device(npu1) {
    %tile11 = aie.tile(1, 1)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of1 (%tile11, {%tile13}, 1 : i32) {repeat_count = 3 : i32, iter_count = 5 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
