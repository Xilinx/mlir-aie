//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two-run xrt::runlist test where each kernel's runtime sequence has 6 host
// buffer arguments (3 in / 3 out). This is the only test that exercises the
// firmware command-chain walker (multi-run runlist) together with host buffer
// arguments beyond the first 5 -- the DDR address patches for arg_idx >= 5 fold
// in the AIE-space translation offset (see AIETargetNPU.cpp), while the chain
// walker requires each per-command slot to be correctly sized. Three add-by-
// constant lanes, one per column. The runlist cascades run0 (ADDONE, +1) into
// run1 (ADDTWO, +2), so a correct result also proves in-order chain execution.

module {
  aie.device(NPUDEVICE) @aie_add_1 {
    %s0 = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %s1 = aie.tile(1, 0)
    %m1 = aie.tile(1, 1)
    %c1 = aie.tile(1, 2)
    %s2 = aie.tile(2, 0)
    %m2 = aie.tile(2, 1)
    %c2 = aie.tile(2, 2)

    aie.objectfifo @in0 (%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @inc0(%m0, {%c0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in0] -> [@inc0] ([] [])
    aie.objectfifo @outc0(%c0, {%m0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out0 (%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@outc0] -> [@out0] ([] [])

    aie.objectfifo @in1 (%s1, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @inc1(%m1, {%c1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in1] -> [@inc1] ([] [])
    aie.objectfifo @outc1(%c1, {%m1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out1 (%m1, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@outc1] -> [@out1] ([] [])

    aie.objectfifo @in2 (%s2, {%m2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @inc2(%m2, {%c2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in2] -> [@inc2] ([] [])
    aie.objectfifo @outc2(%c2, {%m2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out2 (%m2, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@outc2] -> [@out2] ([] [])

    aie.core(%c0) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %st = arith.constant 1 : index
      %val = arith.constant 1 : i32
      scf.for %s = %lb to %ub step %st {
        %sv0 = aie.objectfifo.acquire @inc0(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sv1 = aie.objectfifo.acquire @outc0(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %i = %lb to %ub step %st {
          %0 = memref.load %e0[%i] : memref<8xi32>
          %1 = arith.addi %0, %val : i32
          memref.store %1, %e1[%i] : memref<8xi32>
        }
        aie.objectfifo.release @inc0(Consume, 1)
        aie.objectfifo.release @outc0(Produce, 1)
      }
      aie.end
    }
    aie.core(%c1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %st = arith.constant 1 : index
      %val = arith.constant 1 : i32
      scf.for %s = %lb to %ub step %st {
        %sv0 = aie.objectfifo.acquire @inc1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sv1 = aie.objectfifo.acquire @outc1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %i = %lb to %ub step %st {
          %0 = memref.load %e0[%i] : memref<8xi32>
          %1 = arith.addi %0, %val : i32
          memref.store %1, %e1[%i] : memref<8xi32>
        }
        aie.objectfifo.release @inc1(Consume, 1)
        aie.objectfifo.release @outc1(Produce, 1)
      }
      aie.end
    }
    aie.core(%c2) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %st = arith.constant 1 : index
      %val = arith.constant 1 : i32
      scf.for %s = %lb to %ub step %st {
        %sv0 = aie.objectfifo.acquire @inc2(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sv1 = aie.objectfifo.acquire @outc2(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %i = %lb to %ub step %st {
          %0 = memref.load %e0[%i] : memref<8xi32>
          %1 = arith.addi %0, %val : i32
          memref.store %1, %e1[%i] : memref<8xi32>
        }
        aie.objectfifo.release @inc2(Consume, 1)
        aie.objectfifo.release @outc2(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%i0: memref<64xi32>, %o0: memref<64xi32>, %i1: memref<64xi32>, %o1: memref<64xi32>, %i2: memref<64xi32>, %o2: memref<64xi32>) {
      %c0i = arith.constant 0 : i64
      %c1i = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%o0[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @out0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%i0[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @in0, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%o1[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @out1, id = 3 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%i1[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @in1, id = 2 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%o2[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @out2, id = 5 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%i2[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @in2, id = 4 : i64 } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @out0 }
      aiex.npu.dma_wait { symbol = @out1 }
      aiex.npu.dma_wait { symbol = @out2 }
    }
  }
  aie.device(NPUDEVICE) @aie_add_2 {
    %s0 = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %s1 = aie.tile(1, 0)
    %m1 = aie.tile(1, 1)
    %c1 = aie.tile(1, 2)
    %s2 = aie.tile(2, 0)
    %m2 = aie.tile(2, 1)
    %c2 = aie.tile(2, 2)

    aie.objectfifo @in0 (%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @inc0(%m0, {%c0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in0] -> [@inc0] ([] [])
    aie.objectfifo @outc0(%c0, {%m0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out0 (%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@outc0] -> [@out0] ([] [])

    aie.objectfifo @in1 (%s1, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @inc1(%m1, {%c1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in1] -> [@inc1] ([] [])
    aie.objectfifo @outc1(%c1, {%m1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out1 (%m1, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@outc1] -> [@out1] ([] [])

    aie.objectfifo @in2 (%s2, {%m2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @inc2(%m2, {%c2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@in2] -> [@inc2] ([] [])
    aie.objectfifo @outc2(%c2, {%m2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out2 (%m2, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@outc2] -> [@out2] ([] [])

    aie.core(%c0) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %st = arith.constant 1 : index
      %val = arith.constant 2 : i32
      scf.for %s = %lb to %ub step %st {
        %sv0 = aie.objectfifo.acquire @inc0(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sv1 = aie.objectfifo.acquire @outc0(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %i = %lb to %ub step %st {
          %0 = memref.load %e0[%i] : memref<8xi32>
          %1 = arith.addi %0, %val : i32
          memref.store %1, %e1[%i] : memref<8xi32>
        }
        aie.objectfifo.release @inc0(Consume, 1)
        aie.objectfifo.release @outc0(Produce, 1)
      }
      aie.end
    }
    aie.core(%c1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %st = arith.constant 1 : index
      %val = arith.constant 2 : i32
      scf.for %s = %lb to %ub step %st {
        %sv0 = aie.objectfifo.acquire @inc1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sv1 = aie.objectfifo.acquire @outc1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %i = %lb to %ub step %st {
          %0 = memref.load %e0[%i] : memref<8xi32>
          %1 = arith.addi %0, %val : i32
          memref.store %1, %e1[%i] : memref<8xi32>
        }
        aie.objectfifo.release @inc1(Consume, 1)
        aie.objectfifo.release @outc1(Produce, 1)
      }
      aie.end
    }
    aie.core(%c2) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %st = arith.constant 1 : index
      %val = arith.constant 2 : i32
      scf.for %s = %lb to %ub step %st {
        %sv0 = aie.objectfifo.acquire @inc2(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sv1 = aie.objectfifo.acquire @outc2(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %i = %lb to %ub step %st {
          %0 = memref.load %e0[%i] : memref<8xi32>
          %1 = arith.addi %0, %val : i32
          memref.store %1, %e1[%i] : memref<8xi32>
        }
        aie.objectfifo.release @inc2(Consume, 1)
        aie.objectfifo.release @outc2(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%i0: memref<64xi32>, %o0: memref<64xi32>, %i1: memref<64xi32>, %o1: memref<64xi32>, %i2: memref<64xi32>, %o2: memref<64xi32>) {
      %c0i = arith.constant 0 : i64
      %c1i = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%o0[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @out0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%i0[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @in0, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%o1[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @out1, id = 3 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%i1[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @in1, id = 2 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%o2[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @out2, id = 5 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%i2[%c0i,%c0i,%c0i,%c0i][%c1i,%c1i,%c1i,%c64][%c0i,%c0i,%c0i,%c1i]) { metadata = @in2, id = 4 : i64 } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @out0 }
      aiex.npu.dma_wait { symbol = @out1 }
      aiex.npu.dma_wait { symbol = @out2 }
    }
  }
}
