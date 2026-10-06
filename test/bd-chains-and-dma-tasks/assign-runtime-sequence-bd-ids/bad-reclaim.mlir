//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='reclaim-bds=true' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: not aie-opt --aie-assign-runtime-sequence-bd-ids \
// RUN:   --split-input-file %s 2>&1 | FileCheck %s --check-prefix=OFF

// Reclaim only takes ids from a task that is not started again later: its BDs
// must still hold their descriptors when the restart pushes them. Here every
// task holding an id is restarted, so the pool is genuinely exhausted.
// OFF: 'aiex.dma_configure_task' op Too many simultaneously active buffer descriptors
// OFF-SAME: DMATasks.md). Or let the compiler poll for finished tasks and take their ids (reclaim-bds, aiecc --reclaim-runtime-bds), provided no task depends on a push issued after it.{{ ?$}}

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @all_restarted(%arg0: memref<64xi32>) {
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%shim, S2MM, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t3)
    // expected-error@+1 {{could not take any back either}}
    %new = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
    aiex.dma_start_task(%t0)
    aiex.dma_start_task(%t1)
    aiex.dma_start_task(%t2)
    aiex.dma_start_task(%t3)
  }
}

// -----

// The same pool with no restarts: reclaim takes t0 behind an idle poll (see
// reclaim.mlir). It is off by default, since that poll hangs if t0 can only
// finish after a later push, so by default the allocation fails as it always
// has and the error names the option.
// OFF: 'aiex.dma_configure_task' op Too many simultaneously active buffer descriptors
// OFF-SAME: DMATasks.md). Or let the compiler poll for finished tasks and take their ids (reclaim-bds, aiecc --reclaim-runtime-bds), provided no task depends on a push issued after it.{{ ?$}}

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @reclaim_off(%arg0: memref<64xi32>) {
    // expected-note@+1 {{the task it waits for}}
    %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%shim, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%shim, S2MM, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%shim, S2MM, 1) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t3)
    // expected-remark@+1 {{channel of an earlier task is idle and takes back its ids 0, 1, 2, 3}}
    %new = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%new)
  }
}
