//===- bd_verify_transfer_in_bounds.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// A BD's constant transfer must end inside its buffer, counting the offset:
// the length bounds a BD without dims, the furthest index one with them. Task
// BDs are checked by their dma_configure_task.

// An omitted len is the whole buffer, so with an offset it runs past the end.
aie.device(npu2) {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) : memref<1024xi32>
  aie.mem(%t) {
    aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    // expected-error@+2 {{transfer of 1024 elements at offset 512 runs past the end of the 1024-element buffer}}
    // expected-note@+1 {{an omitted `len` is the whole buffer; give the length to transfer from this offset}}
    aie.dma_bd(%buf : memref<1024xi32> offset = 512)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) : memref<1024xi32>
  aie.mem(%t) {
    aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    // expected-error@+1 {{transfer of 1024 elements at offset 512 runs past the end of the 1024-element buffer}}
    aie.dma_bd(%buf : memref<1024xi32> offset = 512 len = 1024)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// Control: the rest of the buffer from the offset fits.
aie.device(npu2) {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) : memref<1024xi32>
  aie.mem(%t) {
    aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<1024xi32> offset = 512 len = 512)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// The dims alone reach index 1023; the offset pushes them past the end.
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<1024xi32>
  aie.memtile_dma(%t) {
    aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    // expected-error@+1 {{Specified stride(s) and size(s) result in out of bounds access in buffer, for index 1535 in memref of length 1024.}}
    aie.dma_bd(%buf : memref<1024xi32> offset = 512 len = 1024 sizes = [4, 256] strides = [256, 1])
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 0 : i32} : memref<1024xi32>
  aie.runtime_sequence @task_no_len() {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error@+2 {{transfer of 1024 elements at offset 512 runs past the end of the 1024-element buffer}}
      // expected-note@+1 {{an omitted `len` is the whole buffer; give the length to transfer from this offset}}
      aie.dma_bd(%buf : memref<1024xi32> offset = 512) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// Control: a stride-0 repeat transfers more elements than the buffer holds,
// but every index it reads is inside it.
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 0 : i32} : memref<256xi32>
  aie.runtime_sequence @task_repeat() {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 1024 sizes = [4, 256] strides = [0, 1]) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// Control: a runtime-sequence argument's type does not bound the host buffer.
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @host(%arg0: memref<1024xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 1024) {bd_id = 0 : i32}
      aie.end
    }
  }
}
