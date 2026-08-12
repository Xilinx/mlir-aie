//===- page_split_dma_chain_overhead.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 48 independent 40-element (160-byte) blockwrites: 48*160 = 7680 bytes of
// payload, under cert_page_size (8000), but 48*(160+16) = 8448 bytes once the
// splitter's per-BD descriptor overhead is counted -- over the hard 8192-byte
// uC page limit. Merging all 48 into one atomic chain before any page exists
// would make the resulting oversized page unsplittable (the splitter can only
// cut between sibling ops, never inside a chain's BD list). Expect two legal
// pages instead, each holding one fully-merged chain.

// RUN: aie-opt --aie-npu-to-cert %s | aie-opt --cert-legalize-pages | FileCheck %s

// CHECK:      aiex.cert.uc_dma_chain @[[C0:[a-zA-Z0-9_]+]] {
// CHECK-COUNT-22: aiex.cert.uc_dma_bd
// CHECK-NEXT: }
// CHECK:      aiex.cert.uc_dma_chain @[[C1:[a-zA-Z0-9_]+]] {
// CHECK-COUNT-26: aiex.cert.uc_dma_bd
// CHECK-NEXT: }
// CHECK:      aiex.cert.page {
// CHECK-NEXT:   aiex.cert.job
// CHECK-NEXT:     aiex.cert.uc_dma_write_des_sync(@[[C0]])
// CHECK:      aiex.cert.page {
// CHECK-NEXT:   aiex.cert.job
// CHECK-NEXT:     aiex.cert.uc_dma_write_des_sync(@[[C1]])

aie.device(npu2) {
  memref.global "private" constant @data_0 : memref<40xi32> = dense<1>
  memref.global "private" constant @data_1 : memref<40xi32> = dense<2>
  memref.global "private" constant @data_2 : memref<40xi32> = dense<3>
  memref.global "private" constant @data_3 : memref<40xi32> = dense<4>
  memref.global "private" constant @data_4 : memref<40xi32> = dense<5>
  memref.global "private" constant @data_5 : memref<40xi32> = dense<6>
  memref.global "private" constant @data_6 : memref<40xi32> = dense<7>
  memref.global "private" constant @data_7 : memref<40xi32> = dense<8>
  memref.global "private" constant @data_8 : memref<40xi32> = dense<9>
  memref.global "private" constant @data_9 : memref<40xi32> = dense<10>
  memref.global "private" constant @data_10 : memref<40xi32> = dense<11>
  memref.global "private" constant @data_11 : memref<40xi32> = dense<12>
  memref.global "private" constant @data_12 : memref<40xi32> = dense<13>
  memref.global "private" constant @data_13 : memref<40xi32> = dense<14>
  memref.global "private" constant @data_14 : memref<40xi32> = dense<15>
  memref.global "private" constant @data_15 : memref<40xi32> = dense<16>
  memref.global "private" constant @data_16 : memref<40xi32> = dense<17>
  memref.global "private" constant @data_17 : memref<40xi32> = dense<18>
  memref.global "private" constant @data_18 : memref<40xi32> = dense<19>
  memref.global "private" constant @data_19 : memref<40xi32> = dense<20>
  memref.global "private" constant @data_20 : memref<40xi32> = dense<21>
  memref.global "private" constant @data_21 : memref<40xi32> = dense<22>
  memref.global "private" constant @data_22 : memref<40xi32> = dense<23>
  memref.global "private" constant @data_23 : memref<40xi32> = dense<24>
  memref.global "private" constant @data_24 : memref<40xi32> = dense<25>
  memref.global "private" constant @data_25 : memref<40xi32> = dense<26>
  memref.global "private" constant @data_26 : memref<40xi32> = dense<27>
  memref.global "private" constant @data_27 : memref<40xi32> = dense<28>
  memref.global "private" constant @data_28 : memref<40xi32> = dense<29>
  memref.global "private" constant @data_29 : memref<40xi32> = dense<30>
  memref.global "private" constant @data_30 : memref<40xi32> = dense<31>
  memref.global "private" constant @data_31 : memref<40xi32> = dense<32>
  memref.global "private" constant @data_32 : memref<40xi32> = dense<33>
  memref.global "private" constant @data_33 : memref<40xi32> = dense<34>
  memref.global "private" constant @data_34 : memref<40xi32> = dense<35>
  memref.global "private" constant @data_35 : memref<40xi32> = dense<36>
  memref.global "private" constant @data_36 : memref<40xi32> = dense<37>
  memref.global "private" constant @data_37 : memref<40xi32> = dense<38>
  memref.global "private" constant @data_38 : memref<40xi32> = dense<39>
  memref.global "private" constant @data_39 : memref<40xi32> = dense<40>
  memref.global "private" constant @data_40 : memref<40xi32> = dense<41>
  memref.global "private" constant @data_41 : memref<40xi32> = dense<42>
  memref.global "private" constant @data_42 : memref<40xi32> = dense<43>
  memref.global "private" constant @data_43 : memref<40xi32> = dense<44>
  memref.global "private" constant @data_44 : memref<40xi32> = dense<45>
  memref.global "private" constant @data_45 : memref<40xi32> = dense<46>
  memref.global "private" constant @data_46 : memref<40xi32> = dense<47>
  memref.global "private" constant @data_47 : memref<40xi32> = dense<48>
  aie.runtime_sequence @seq() {
    %g0 = memref.get_global @data_0 : memref<40xi32>
    aiex.npu.blockwrite(%g0) {address = 0 : ui32} : memref<40xi32>
    %g1 = memref.get_global @data_1 : memref<40xi32>
    aiex.npu.blockwrite(%g1) {address = 160 : ui32} : memref<40xi32>
    %g2 = memref.get_global @data_2 : memref<40xi32>
    aiex.npu.blockwrite(%g2) {address = 320 : ui32} : memref<40xi32>
    %g3 = memref.get_global @data_3 : memref<40xi32>
    aiex.npu.blockwrite(%g3) {address = 480 : ui32} : memref<40xi32>
    %g4 = memref.get_global @data_4 : memref<40xi32>
    aiex.npu.blockwrite(%g4) {address = 640 : ui32} : memref<40xi32>
    %g5 = memref.get_global @data_5 : memref<40xi32>
    aiex.npu.blockwrite(%g5) {address = 800 : ui32} : memref<40xi32>
    %g6 = memref.get_global @data_6 : memref<40xi32>
    aiex.npu.blockwrite(%g6) {address = 960 : ui32} : memref<40xi32>
    %g7 = memref.get_global @data_7 : memref<40xi32>
    aiex.npu.blockwrite(%g7) {address = 1120 : ui32} : memref<40xi32>
    %g8 = memref.get_global @data_8 : memref<40xi32>
    aiex.npu.blockwrite(%g8) {address = 1280 : ui32} : memref<40xi32>
    %g9 = memref.get_global @data_9 : memref<40xi32>
    aiex.npu.blockwrite(%g9) {address = 1440 : ui32} : memref<40xi32>
    %g10 = memref.get_global @data_10 : memref<40xi32>
    aiex.npu.blockwrite(%g10) {address = 1600 : ui32} : memref<40xi32>
    %g11 = memref.get_global @data_11 : memref<40xi32>
    aiex.npu.blockwrite(%g11) {address = 1760 : ui32} : memref<40xi32>
    %g12 = memref.get_global @data_12 : memref<40xi32>
    aiex.npu.blockwrite(%g12) {address = 1920 : ui32} : memref<40xi32>
    %g13 = memref.get_global @data_13 : memref<40xi32>
    aiex.npu.blockwrite(%g13) {address = 2080 : ui32} : memref<40xi32>
    %g14 = memref.get_global @data_14 : memref<40xi32>
    aiex.npu.blockwrite(%g14) {address = 2240 : ui32} : memref<40xi32>
    %g15 = memref.get_global @data_15 : memref<40xi32>
    aiex.npu.blockwrite(%g15) {address = 2400 : ui32} : memref<40xi32>
    %g16 = memref.get_global @data_16 : memref<40xi32>
    aiex.npu.blockwrite(%g16) {address = 2560 : ui32} : memref<40xi32>
    %g17 = memref.get_global @data_17 : memref<40xi32>
    aiex.npu.blockwrite(%g17) {address = 2720 : ui32} : memref<40xi32>
    %g18 = memref.get_global @data_18 : memref<40xi32>
    aiex.npu.blockwrite(%g18) {address = 2880 : ui32} : memref<40xi32>
    %g19 = memref.get_global @data_19 : memref<40xi32>
    aiex.npu.blockwrite(%g19) {address = 3040 : ui32} : memref<40xi32>
    %g20 = memref.get_global @data_20 : memref<40xi32>
    aiex.npu.blockwrite(%g20) {address = 3200 : ui32} : memref<40xi32>
    %g21 = memref.get_global @data_21 : memref<40xi32>
    aiex.npu.blockwrite(%g21) {address = 3360 : ui32} : memref<40xi32>
    %g22 = memref.get_global @data_22 : memref<40xi32>
    aiex.npu.blockwrite(%g22) {address = 3520 : ui32} : memref<40xi32>
    %g23 = memref.get_global @data_23 : memref<40xi32>
    aiex.npu.blockwrite(%g23) {address = 3680 : ui32} : memref<40xi32>
    %g24 = memref.get_global @data_24 : memref<40xi32>
    aiex.npu.blockwrite(%g24) {address = 3840 : ui32} : memref<40xi32>
    %g25 = memref.get_global @data_25 : memref<40xi32>
    aiex.npu.blockwrite(%g25) {address = 4000 : ui32} : memref<40xi32>
    %g26 = memref.get_global @data_26 : memref<40xi32>
    aiex.npu.blockwrite(%g26) {address = 4160 : ui32} : memref<40xi32>
    %g27 = memref.get_global @data_27 : memref<40xi32>
    aiex.npu.blockwrite(%g27) {address = 4320 : ui32} : memref<40xi32>
    %g28 = memref.get_global @data_28 : memref<40xi32>
    aiex.npu.blockwrite(%g28) {address = 4480 : ui32} : memref<40xi32>
    %g29 = memref.get_global @data_29 : memref<40xi32>
    aiex.npu.blockwrite(%g29) {address = 4640 : ui32} : memref<40xi32>
    %g30 = memref.get_global @data_30 : memref<40xi32>
    aiex.npu.blockwrite(%g30) {address = 4800 : ui32} : memref<40xi32>
    %g31 = memref.get_global @data_31 : memref<40xi32>
    aiex.npu.blockwrite(%g31) {address = 4960 : ui32} : memref<40xi32>
    %g32 = memref.get_global @data_32 : memref<40xi32>
    aiex.npu.blockwrite(%g32) {address = 5120 : ui32} : memref<40xi32>
    %g33 = memref.get_global @data_33 : memref<40xi32>
    aiex.npu.blockwrite(%g33) {address = 5280 : ui32} : memref<40xi32>
    %g34 = memref.get_global @data_34 : memref<40xi32>
    aiex.npu.blockwrite(%g34) {address = 5440 : ui32} : memref<40xi32>
    %g35 = memref.get_global @data_35 : memref<40xi32>
    aiex.npu.blockwrite(%g35) {address = 5600 : ui32} : memref<40xi32>
    %g36 = memref.get_global @data_36 : memref<40xi32>
    aiex.npu.blockwrite(%g36) {address = 5760 : ui32} : memref<40xi32>
    %g37 = memref.get_global @data_37 : memref<40xi32>
    aiex.npu.blockwrite(%g37) {address = 5920 : ui32} : memref<40xi32>
    %g38 = memref.get_global @data_38 : memref<40xi32>
    aiex.npu.blockwrite(%g38) {address = 6080 : ui32} : memref<40xi32>
    %g39 = memref.get_global @data_39 : memref<40xi32>
    aiex.npu.blockwrite(%g39) {address = 6240 : ui32} : memref<40xi32>
    %g40 = memref.get_global @data_40 : memref<40xi32>
    aiex.npu.blockwrite(%g40) {address = 6400 : ui32} : memref<40xi32>
    %g41 = memref.get_global @data_41 : memref<40xi32>
    aiex.npu.blockwrite(%g41) {address = 6560 : ui32} : memref<40xi32>
    %g42 = memref.get_global @data_42 : memref<40xi32>
    aiex.npu.blockwrite(%g42) {address = 6720 : ui32} : memref<40xi32>
    %g43 = memref.get_global @data_43 : memref<40xi32>
    aiex.npu.blockwrite(%g43) {address = 6880 : ui32} : memref<40xi32>
    %g44 = memref.get_global @data_44 : memref<40xi32>
    aiex.npu.blockwrite(%g44) {address = 7040 : ui32} : memref<40xi32>
    %g45 = memref.get_global @data_45 : memref<40xi32>
    aiex.npu.blockwrite(%g45) {address = 7200 : ui32} : memref<40xi32>
    %g46 = memref.get_global @data_46 : memref<40xi32>
    aiex.npu.blockwrite(%g46) {address = 7360 : ui32} : memref<40xi32>
    %g47 = memref.get_global @data_47 : memref<40xi32>
    aiex.npu.blockwrite(%g47) {address = 7520 : ui32} : memref<40xi32>
  }
}
