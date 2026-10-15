//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// 300 f32 = 1200 bytes into 256-byte objects: the last one never fills.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<300xf32>) {
      // expected-error@+1 {{moves 1200 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<300xf32> offset = 0 len = 300 sizes = [1, 1, 1, 300] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// 256 f32 = 1024 bytes is exactly 4 objects.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<256xf32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<256xf32> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Only the extent is compared, not the element type: 256 i8 is one object of
// 64 i32.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<256xi8>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<256xi8> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The repeat count multiplies the transfer: 4 issues of 64 bytes is one object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 16 sizes = [1, 1, 4, 4] strides = [0, 0, 4, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A sub-byte element counts as the whole byte it occupies on both sides:
// 3 i4 is 3 bytes against 2-byte objects.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<2xi4>>
    aie.runtime_sequence(%arg0: memref<3xi4>) {
      // expected-error@+1 {{moves 3 bytes through objectFIFO @of, which is not a whole number of its 2-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<3xi4> offset = 0 len = 3 sizes = [1, 1, 1, 3] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A fill is measured against the consumer's object: 32 i32 is one 128-byte
// consumer object, though only half a producer object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>> -> !aie.objectfifo<memref<32xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A drain is measured against the producer's object, the unit the producer
// sends in.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t02, {%t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>> -> !aie.objectfifo<memref<32xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A padding producer sends more than its object holds, so it is not checked.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    aie.objectfifo @of(%t01 dimensionsToStream [<size = 8, stride = 8>, <size = 8, stride = 1>], {%t00}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>]>} : !aie.objectfifo<memref<64xi8>>
    aie.runtime_sequence(%arg0: memref<72xi8>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<72xi8> offset = 0 len = 72 sizes = [1, 1, 1, 72] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A runtime repeat count leaves nothing constant to compare.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<300xf32>, %n: i32) {
      %t = aiex.dma_configure_task_for @of repeat %n : i32 {
        aie.dma_bd(%arg0 : memref<300xf32> offset = 0 len = 300 sizes = [1, 1, 1, 300] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A repeat on dma_start_task re-runs the task: two starts of half an object
// are one whole object, so the task alone is not checked.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t) {repeat_count = 1 : i32}
    }
  }
}
