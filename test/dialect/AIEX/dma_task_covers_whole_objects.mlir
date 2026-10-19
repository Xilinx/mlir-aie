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

// -----

// Two fills of half an object, from two host buffers, concatenate into one.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

// Two drains of half an object split it across two host buffers.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t02, {%t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }
}

// -----

// Half an object then a whole one leaves the second object half full. The
// first partial transfer reports, once.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<64xi32>) {
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of, which, with this runtime sequence's other transfers through it (384 bytes in all), is not a whole number of its 256-byte objects}}
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

// A transfer in a loop has no static count to sum, so it is not checked.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %t = aiex.dma_configure_task_for @of {
          aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
          aie.end
        }
        aiex.dma_start_task(%t)
      }
    }
  }
}

// -----

// An omitted len implies the whole buffer, as DMABDOp::getLenInBytes()
// assumes: 300 f32 (1200 bytes) into 256-byte objects ends mid-object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<300xf32>) {
      // expected-error@+1 {{moves 1200 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<300xf32>)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A linked fifo's far-end BD is sized by the link's shared pool rather than
// this fifo's own type: @of_out's larger 96-i32 objects own the pool, so 96
// i32 is whole, though not whole 64-i32 (256-byte) @of_in objects.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of_in(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<96xi32>) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<96xi32> offset = 0 len = 96 sizes = [1, 1, 1, 96] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The link's owner side (@of_in, the larger of the two and so the pool's
// real size) is still checked even though the sizes differ: a transfer that
// doesn't cover its own 384-byte objects is rejected, not waved through just
// because @of_out is a different (smaller) size.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of_in(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo @of_out(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{moves 256 bytes through objectFIFO @of_in, which is not a whole number of its 384-byte objects}}
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A distribute link's single combined side owns the pool at its own declared
// size, unlike the several small sides whose segments are offset-derived: a
// transfer through @of_in that doesn't cover its own 512-byte objects is
// still rejected, even though @of_in and its smaller outputs are unequal
// sizes (which would otherwise defer the check entirely).
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_in(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of_out0(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out1(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out0, @of_out1]([] [0, 64])
    aie.runtime_sequence(%arg0: memref<96xi32>) {
      // expected-error@+1 {{moves 384 bytes through objectFIFO @of_in, which is not a whole number of its 512-byte objects}}
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<96xi32> offset = 0 len = 96 sizes = [1, 1, 1, 96] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A start inside a loop may run more than once, so it is not summed as one
// execution: two issues of half an object (started lexically once, inside a
// two-iteration loop) are not flagged, though the configuring task itself
// stays in the sequence's entry block.
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
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        aiex.dma_start_task(%t)
      }
    }
  }
}

// -----

// A cyclic BD chain runs indefinitely, so walking its body once isn't its
// true extent: this self-looping 32-i32 BD is not flagged, though 128 bytes
// alone is not a whole 256-byte object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.next_bd ^loop
      ^loop:
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.next_bd ^loop
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Only the chain reachable from the entry is queued at a start: the
// disconnected self-looping block never streams, so 128 bytes is all that
// moves, not a whole 256-byte object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      ^orphan:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 32 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.next_bd ^orphan
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The same disconnected block doesn't count against a whole entry chain
// either: 256 bytes stream, not 384.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<96xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<96xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      ^orphan:
        aie.dma_bd(%arg0 : memref<96xi32> offset = 64 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.next_bd ^orphan
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A pinned next_bd_id chains outside the block successors, so the chain's
// extent can't be followed and is not checked.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A scratchpad length_parameter lengthens the BD at run time, so its static
// len alone is not the transfer's extent and is not checked: 32 i32 plus
// n * 32-i32 units is unknown, though 32 i32 alone is not a whole object.
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
            {length_parameter = @n, length_unit = 32 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A negative repeat_count is rejected at its own op, rather than folding to a
// bogus negative byte count that happens to clear the modulus check below.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      // expected-error@+1 {{attribute 'repeat_count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
      } {repeat_count = -2 : i32}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// An omitted len's implicit whole-buffer element count stays a true int64:
// past INT32_MAX elements it must not wrap into a garbled (and possibly
// falsely-whole) byte count.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<3000000001xi8>) {
      // expected-error@+1 {{moves 3000000001 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<3000000001xi8>)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// An explicit len is unsigned, as DMABDOp::getLenInBytes() reads it: 3e9 i8
// is 2,500,000 whole 1200-byte objects, not a negative byte count.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<1200xi8>>
    aie.runtime_sequence(%arg0: memref<3000000000xi8>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<3000000000xi8> offset = 0 len = 3000000000)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A byte count past INT64_MAX is not checked, rather than wrapping: 2.4e10
// bytes repeated 1e9 + 1 times is whole 300-byte objects, which a wrapped
// count is not.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<300xi8>>
    aie.runtime_sequence(%arg0: memref<3000000000xi64>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<3000000000xi64> offset = 0 len = 3000000000)
        aie.end
      } {repeat_count = 1000000000 : i32}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// With runtime sizes, an omitted len is their inner extent, not the whole
// 300-i32 buffer, so it is not checked: %n = 64 fills one whole object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<300xi32>, %n: i64) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<300xi32> offset = 0 sizes = [1, 1, 1, %n] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A plain link's pool owner is picked by element count, as createLinkPools
// does: @of_in's 128 i8 outnumber @of_out's 64 i32, so the pool (and the
// memtile BD @of_out drains from) holds 128 bytes, and a 128-byte drain is
// whole though @of_out declares 256-byte objects.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of_in(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo @of_out(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A linked fill lands in the link's pool, sized by its owner's declared type
// rather than the fill's consumerElemType: @of_in owns a 256-byte pool, so a
// 128-byte fill (one 32-i32 consumer object) leaves its BD incomplete.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of_in(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<64xi32>> -> !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of_out(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<32xi32>) {
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of_in, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A dma_memcpy_nd through the same fifo shares its objects too, so the
// sequence's total is unknown: 128 bytes from the task plus 128 from the
// memcpy fill one 256-byte object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32][1, 1, 1, 32][0, 0, 0, 1]) {id = 1 : i64, metadata = @of} : memref<64xi32>
    }
  }
}

// -----

// A task that is never started streams nothing, so it doesn't hide the other
// transfer's partial object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
    }
  }
}

// -----

// A task value forwarded through scf.if is started again under another name,
// so its starts can't be counted: the two starts here move one whole object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %n: i32) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
      %c0 = arith.constant 0 : i32
      %again = arith.cmpi ne, %n, %c0 : i32
      %u = scf.if %again -> (index) {
        scf.yield %t : index
      } else {
        scf.yield %t : index
      }
      aiex.dma_start_task(%u)
    }
  }
}

// -----

// A broadcast fill reaches every consumer's BD: the memtile's lands in the
// 384-byte link pool @of_out owns, but the core's in its own pool of 256-byte
// @of_in objects, so 384 bytes leaves the core's second object incomplete.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_in(%t00, {%t01, %t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<96xi32>) {
      // expected-error@+1 {{moves 384 bytes through objectFIFO @of_in, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<96xi32> offset = 0 len = 96 sizes = [1, 1, 1, 96] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The same broadcast with 768 bytes completes both consumers' BDs: two
// 384-byte link-pool objects and three 256-byte core objects.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_in(%t00, {%t01, %t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<192xi32>) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<192xi32> offset = 0 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A broadcast drain's producer off the link's shared tile keeps its own pool of
// 256-byte @of_in objects, whatever the 384-byte link pool @of_out owns, so 384
// bytes leaves the core's second object incomplete.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_in(%t02, {%t01, %t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<96xi32>) {
      // expected-error@+1 {{moves 384 bytes through objectFIFO @of_in, which is not a whole number of its 256-byte objects}}
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<96xi32> offset = 0 len = 96 sizes = [1, 1, 1, 96] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A broadcast stalls on its slowest consumer, so the memtile receives only
// what the shim drains: 512 bytes completes two of the core's objects but
// leaves the second 384-byte link-pool object incomplete.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_in(%t02, {%t01, %t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<128xi32>) {
      // expected-error@+1 {{moves 512 bytes through objectFIFO @of_in, which is not a whole number of its 384-byte objects}}
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<128xi32> offset = 0 len = 128 sizes = [1, 1, 1, 128] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The same broadcast drain with 768 bytes completes three of the core's
// objects and two of the memtile's.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_in(%t02, {%t01, %t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%t01, {%t03}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<192xi32>) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<192xi32> offset = 0 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The fifo verifier rejects a dynamically-shaped object type, which only the
// generic form can spell, before the link pool's owner is sized from it.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of_in(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{type should be static shaped memref, but got 'memref<?xi32>'}}
    "aie.objectfifo"(%t01, %t02) <{dimensionsFromStreamPerConsumer = #aie<bd_dim_layout_array_array[[]]>, dimensionsToStream = #aie<bd_dim_layout_array[]>, disable_synchronization = false, elemNumber = 2 : i32, elemType = !aie.objectfifo<memref<?xi32>>, plio = false, sym_name = "of_out", via_DMA = false}> : (index, index) -> ()
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.runtime_sequence(%arg0: memref<96xi32>) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%arg0 : memref<96xi32> offset = 0 len = 96 sizes = [1, 1, 1, 96] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// An object whose byte size passes INT64_MAX is not checked, rather than
// wrapping: 2^61 + 1 i64 would wrap to an 8-byte object that 12 bytes misses.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<2305843009213693953xi64>>
    aie.runtime_sequence(%arg0: memref<3xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<3xi32> offset = 0 len = 3)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A BD chain started for the fifo shares its objects too, so the sequence's
// total is unknown: 128 bytes from the task plus 128 from the chain fill one
// 256-byte object.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.bd_chain @half(%buf: memref<32xi32>) {
      aie.dma_bd(%buf : memref<32xi32> offset = 0 len = 32)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
      %c = aiex.dma_start_bd_chain_for @half(%arg1) : (memref<32xi32>) for @of
    }
  }
}

// -----

// A dma_channel_reset_for re-arms the fifo's far ends and drops a partly
// filled object, so transfers on either side of it don't share one: the half
// object before the reset is partial in its own segment.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_channel_reset_for(@of)
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

// After the reset, the half object is summed only with its own segment: the
// whole object before the reset doesn't count toward it.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_channel_reset_for(@of)
      // expected-error@+1 {{moves 128 bytes through objectFIFO @of, which is not a whole number of its 256-byte objects}}
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

// Two halves that complete an object before the reset, and a whole object
// after it, are whole in each segment.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %t0 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg1 : memref<32xi32> offset = 0 len = 32 sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_channel_reset_for(@of)
      %t2 = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg2 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t2)
    }
  }
}
