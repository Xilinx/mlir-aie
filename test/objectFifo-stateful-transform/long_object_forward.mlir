// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A memtile forward of one 262144-element bf16 object (131072 words, past
// npu2's 131071-word BD length): each side of the memtile moves the object
// as two chained BDs, the lock taken by the first and given back by the last.

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.objectfifo @in(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<262144xbf16>>
    aie.objectfifo @in_fwd(%mem, {%shim}, 1 : i32) : !aie.objectfifo<memref<262144xbf16>>
    aie.objectfifo.link [@in] -> [@in_fwd]([] [])
  }
}

// CHECK:       aie.memtile_dma
// CHECK:         aie.dma_start(S2MM, 0, ^[[F0:bb[0-9]+]], ^{{bb[0-9]+}})
// CHECK:       ^[[F0]]:
// CHECK:         aie.use_lock(%[[PROD:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF:.*]] : memref<262144xbf16> offset = 0 len = 131072)
// CHECK-NEXT:    aie.next_bd ^[[F1:bb[0-9]+]]
// CHECK:       ^[[F1]]:
// CHECK-NEXT:    aie.dma_bd(%[[BUF]] : memref<262144xbf16> offset = 131072 len = 131072)
// CHECK:         aie.use_lock(%[[CONS:.*]], Release, %{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[F0]]
// CHECK:         aie.dma_start(MM2S, 0, ^[[D0:bb[0-9]+]], ^{{bb[0-9]+}})
// CHECK:       ^[[D0]]:
// CHECK:         aie.use_lock(%[[CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF]] : memref<262144xbf16> offset = 0 len = 131072)
// CHECK-NEXT:    aie.next_bd ^[[D1:bb[0-9]+]]
// CHECK:       ^[[D1]]:
// CHECK-NEXT:    aie.dma_bd(%[[BUF]] : memref<262144xbf16> offset = 131072 len = 131072)
// CHECK:         aie.use_lock(%[[PROD]], Release, %{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[D0]]
