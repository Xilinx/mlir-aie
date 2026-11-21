// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-lock-ids --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-cdo --cdo-debug=true -o %t.cdo 2>&1 | FileCheck %s --check-prefix=CDO

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A memtile forward of one 262144-element bf16 object (131072 words, past
// npu2's 131071-word BD length): each side of the memtile moves the object as
// two chained BDs. The first acquires and releases by zero, the second
// acquires disabled and releases, so each BD fills its one acquire and one
// release field. The CDO's lock words (VALID_BD, LOCK_REL_VALUE[30:24],
// LOCK_REL_ID[23:16], LOCK_ACQ_ENABLE[15], LOCK_ACQ_VALUE[14:8],
// LOCK_ACQ_ID[7:0]) bear it out.

// S2MM: acquire 64 (>= 1), release 65 by 0; then acquire disabled, release 65 by 1.
// CDO-DAG: Address: 0x00000000001A001C {{.*}} is: 0x8041FF40
// CDO-DAG: Address: 0x00000000001A003C {{.*}} is: 0x81417F40
// MM2S: acquire 65 (>= 1), release 64 by 0; then acquire disabled, release 64 by 1.
// CDO-DAG: Address: 0x00000000001A005C {{.*}} is: 0x8040FF41
// CDO-DAG: Address: 0x00000000001A007C {{.*}} is: 0x81407F41

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
// CHECK:         aie.use_lock(%[[CONS:.*]], Release, %c0{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[F1:bb[0-9]+]]
// CHECK:       ^[[F1]]:
// CHECK:         aie.use_lock(%[[PROD]], AcquireGreaterEqual, %{{.*}}) {acq_en = false}
// CHECK-NEXT:    aie.dma_bd(%[[BUF]] : memref<262144xbf16> offset = 131072 len = 131072)
// CHECK:         aie.use_lock(%[[CONS]], Release, %c1{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[F0]]
// CHECK:         aie.dma_start(MM2S, 0, ^[[D0:bb[0-9]+]], ^{{bb[0-9]+}})
// CHECK:       ^[[D0]]:
// CHECK:         aie.use_lock(%[[CONS]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF]] : memref<262144xbf16> offset = 0 len = 131072)
// CHECK:         aie.use_lock(%[[PROD]], Release, %c0{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[D1:bb[0-9]+]]
// CHECK:       ^[[D1]]:
// CHECK:         aie.use_lock(%[[CONS]], AcquireGreaterEqual, %{{.*}}) {acq_en = false}
// CHECK-NEXT:    aie.dma_bd(%[[BUF]] : memref<262144xbf16> offset = 131072 len = 131072)
// CHECK:         aie.use_lock(%[[PROD]], Release, %c1{{.*}})
// CHECK-NEXT:    aie.next_bd ^[[D0]]
