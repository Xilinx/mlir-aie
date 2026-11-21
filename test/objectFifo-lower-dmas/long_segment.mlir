// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An object past one buffer descriptor's length (131071 words on npu2) moves
// as a chain of descriptors, none past the length. Each BD has one acquire
// and one release field: the first takes the object's lock and releases by
// zero, the last acquires disabled (acq_en = false) and gives the lock back.

module @long_segment {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)

    %b0 = aie.buffer(%mem) {sym_name = "b0"} : memref<262144xbf16>
    %free = aie.lock(%mem) {init = 1 : i32, sym_name = "free"}
    %full = aie.lock(%mem) {init = 0 : i32, sym_name = "full"}

    aie.objectfifo.pool @pool(%mem) {
      depth = 1 : i32, buffers = [@b0]
    } : memref<262144xbf16> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 262144 : i32}
    }
    aie.objectfifo.dma_endpoint @fill(%mem) fills @pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @drain(%mem) drains @pool {channelIndex = 0 : i32}
  }
}

// CHECK-LABEL: @long_segment
// CHECK:       aie.memtile_dma
// CHECK:         aie.dma_start(S2MM, 0, ^[[F0:bb[0-9]+]], ^{{bb[0-9]+}})
// CHECK:       ^[[F0]]:
// CHECK:         aie.use_lock(%free, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%b0 : memref<262144xbf16> offset = 0 len = 131072)
// CHECK:         aie.use_lock(%full, Release, %c0{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[F1:bb[0-9]+]]
// CHECK:       ^[[F1]]:
// CHECK:         aie.use_lock(%free, AcquireGreaterEqual, %{{.*}}) {acq_en = false}
// CHECK-NEXT:    aie.dma_bd(%b0 : memref<262144xbf16> offset = 131072 len = 131072)
// CHECK:         aie.use_lock(%full, Release, %c1{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[F0]]
// CHECK:         aie.dma_start(MM2S, 0, ^[[D0:bb[0-9]+]], ^{{bb[0-9]+}})
// CHECK:       ^[[D0]]:
// CHECK:         aie.use_lock(%full, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%b0 : memref<262144xbf16> offset = 0 len = 131072)
// CHECK:         aie.use_lock(%free, Release, %c0{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[D1:bb[0-9]+]]
// CHECK:       ^[[D1]]:
// CHECK:         aie.use_lock(%full, AcquireGreaterEqual, %{{.*}}) {acq_en = false}
// CHECK-NEXT:    aie.dma_bd(%b0 : memref<262144xbf16> offset = 131072 len = 131072)
// CHECK:         aie.use_lock(%free, Release, %c1{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[D0]]
