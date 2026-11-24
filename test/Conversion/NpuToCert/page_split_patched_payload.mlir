//===- page_split_patched_payload.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Each page carries its own copy of every payload it pushes, so a page split
// must keep each cert.apply_offset_57 on the same page as every later uC-DMA
// push of the payload it patches.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// Interior cut: reject the cut before the second write32, inside the @pB
// patch/push span, and split before the patch instead.
//
// CHECK-LABEL: aie.device(xcve3858) @interior
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@pA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainA)
// CHECK-NOT: aiex.cert.apply_offset_57(@pB
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@pB
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainB)
aie.device(xcve3858) @interior {
  memref.global "private" constant @pA : memref<980xi32> = dense<0>
  memref.global "private" constant @pB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainA {
    aiex.cert.uc_dma_bd @pA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainB {
    aiex.cert.uc_dma_bd @pB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@pA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainA)
      aiex.cert.apply_offset_57(@pB, 1, 0)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.write32(0x1004, 2)
      aiex.cert.write32(0x1008, 3)
      aiex.cert.uc_dma_write_des_sync(@chainB)
    }
  }
}

// -----

// Job-boundary cut: reject the boundary before job 3, which separates @qB's
// patch in job 2 from its push in job 3, and split before job 2 instead.
//
// CHECK-LABEL: aie.device(xcve3858) @job_boundary
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@qA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainQA)
// CHECK-NOT: aiex.cert.apply_offset_57(@qB
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@qB
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainQB)
aie.device(xcve3858) @job_boundary {
  memref.global "private" constant @qA : memref<975xi32> = dense<0>
  memref.global "private" constant @qB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainQA {
    aiex.cert.uc_dma_bd @qA, 0, 975, false
  }
  aiex.cert.uc_dma_chain @chainQB {
    aiex.cert.uc_dma_bd @qB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@qA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainQA)
    }
    aiex.cert.job(2) {
      aiex.cert.apply_offset_57(@qB, 1, 0)
    }
    aiex.cert.job(3) {
      aiex.cert.uc_dma_write_des_sync(@chainQB)
    }
  }
}

// -----

// Control: the interior case without the @rB patch. An unpatched payload
// imposes no constraint, so the cut before the second write32 is kept.
//
// CHECK-LABEL: aie.device(xcve3858) @unpatched
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@rA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainRA)
// CHECK: aiex.cert.write32(4096, 1)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.write32(4100, 2)
// CHECK: aiex.cert.write32(4104, 3)
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainRB)
aie.device(xcve3858) @unpatched {
  memref.global "private" constant @rA : memref<980xi32> = dense<0>
  memref.global "private" constant @rB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainRA {
    aiex.cert.uc_dma_bd @rA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainRB {
    aiex.cert.uc_dma_bd @rB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@rA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainRA)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.write32(0x1004, 2)
      aiex.cert.write32(0x1008, 3)
      aiex.cert.uc_dma_write_des_sync(@chainRB)
    }
  }
}

// -----

// Repeated pushes need the same patched copy. The greedy cut before the
// second @sA push is rejected; split after it instead.
//
// CHECK-LABEL: aie.device(xcve3858) @repeated_push
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@sA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSA)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSW)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSA)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSV)
aie.device(xcve3858) @repeated_push {
  memref.global "private" constant @sA : memref<700xi32> = dense<0>
  memref.global "private" constant @sW : memref<400xi32> = dense<0>
  memref.global "private" constant @sV : memref<900xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainSA {
    aiex.cert.uc_dma_bd @sA, 0, 700, false
  }
  aiex.cert.uc_dma_chain @chainSW {
    aiex.cert.uc_dma_bd @sW, 0, 400, false
  }
  aiex.cert.uc_dma_chain @chainSV {
    aiex.cert.uc_dma_bd @sV, 0, 900, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@sA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainSA)
      aiex.cert.uc_dma_write_des_sync(@chainSW)
      aiex.cert.uc_dma_write_des_sync(@chainSA)
      aiex.cert.uc_dma_write_des_sync(@chainSV)
    }
  }
}

// -----

// Each page carries one copy of each chain and payload, however many pushes
// name them. This page repeats @chainTA and shares @tA with @chainTA2; counted
// once each, it fits the 8192-byte limit, so the indivisible span is kept
// whole instead of being rejected.
//
// CHECK-LABEL: aie.device(xcve3858) @shared_payload_fits
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@tA, 1, 0)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainTA2)
// CHECK-NOT: aiex.cert.page
aie.device(xcve3858) @shared_payload_fits {
  memref.global "private" constant @tA : memref<980xi32> = dense<0>
  memref.global "private" constant @tB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainTA {
    aiex.cert.uc_dma_bd @tA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainTB {
    aiex.cert.uc_dma_bd @tB, 0, 1000, false
  }
  aiex.cert.uc_dma_chain @chainTA2 {
    aiex.cert.uc_dma_bd @tA, 4096, 980, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@tA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainTA)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.uc_dma_write_des_sync(@chainTB)
      aiex.cert.uc_dma_write_des_sync(@chainTA)
      aiex.cert.apply_offset_57(@tA, 1, 2)
      aiex.cert.uc_dma_write_des_sync(@chainTA2)
    }
  }
}
