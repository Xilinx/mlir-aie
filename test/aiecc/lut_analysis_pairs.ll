; Copyright (C) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

@table_a = external global [32 x i8]
@table_b = external global [32 x i8]
@unrelated_a = external global [32 x i8]
@unrelated_b = external global [32 x i8]

declare <16 x i32> @llvm.aie2p.vsel32(<16 x i32>, <16 x i32>, i32)
declare <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32>)
declare <16 x i32> @llvm.aie2.vbroadcast32.I512(i32)
declare <16 x i32> @llvm.aie2.vsel32(<16 x i32>, <16 x i32>, i32)
declare <8 x i32> @llvm.aie2.ext.I256.I512(<16 x i32>, i32)
declare <8 x i32> @llvm.aie2.load.4x16.lo(<8 x i32>)
declare <8 x i32> @llvm.aie2.load.4x32.lo(<8 x i32>)
declare <8 x i32> @llvm.aie2.load.4x32.hi(<8 x i32>)
declare <8 x i32> @llvm.aie2p.load.4x64.lo(<8 x i32>)
declare <8 x i32> @llvm.aie2p.load.4x64.hi(<8 x i32>)

define <8 x i32> @gather_with_unrelated_blend(<16 x i32> %offset, <16 x i32> %alternative) {
  %a = insertelement <16 x i32> poison, i32 ptrtoint (ptr @table_a to i32), i32 0
  %aa = shufflevector <16 x i32> %a, <16 x i32> poison, <16 x i32> zeroinitializer
  %b = insertelement <16 x i32> poison, i32 ptrtoint (ptr @table_b to i32), i32 0
  %bb = shufflevector <16 x i32> %b, <16 x i32> poison, <16 x i32> zeroinitializer
  %c = insertelement <16 x i32> poison, i32 ptrtoint (ptr @unrelated_a to i32), i32 0
  %cc = shufflevector <16 x i32> %c, <16 x i32> poison, <16 x i32> zeroinitializer
  %d = insertelement <16 x i32> poison, i32 ptrtoint (ptr @unrelated_b to i32), i32 0
  %dd = shufflevector <16 x i32> %d, <16 x i32> poison, <16 x i32> zeroinitializer
  %unrelated = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %cc, <16 x i32> %dd, i32 52428)
  %pair = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %aa, <16 x i32> %bb, i32 52428)
  %indices = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %offset, <16 x i32> %alternative, i32 52428)
  %addr = add <16 x i32> %pair, %indices
  %half = shufflevector <16 x i32> %addr, <16 x i32> %unrelated, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %result = call <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}

define <8 x i32> @unresolved_gather(<8 x i32> %addr) {
  %result = call <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32> %addr)
  ret <8 x i32> %result
}

define <8 x i32> @offset_gather() {
  %a = insertelement <16 x i32> poison, i32 ptrtoint (ptr getelementptr ([32 x i8], ptr @table_a, i32 0, i32 16) to i32), i32 0
  %aa = shufflevector <16 x i32> %a, <16 x i32> poison, <16 x i32> zeroinitializer
  %b = insertelement <16 x i32> poison, i32 ptrtoint (ptr @table_b to i32), i32 0
  %bb = shufflevector <16 x i32> %b, <16 x i32> poison, <16 x i32> zeroinitializer
  %pair = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %aa, <16 x i32> %bb, i32 52428)
  %half = shufflevector <16 x i32> %pair, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %result = call <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}

define <8 x i32> @same_table() {
  %a = insertelement <16 x i32> poison, i32 ptrtoint (ptr @table_a to i32), i32 0
  %aa = shufflevector <16 x i32> %a, <16 x i32> poison, <16 x i32> zeroinitializer
  %pair = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %aa, <16 x i32> %aa, i32 52428)
  %half = shufflevector <16 x i32> %pair, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %result = call <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}

define <8 x i32> @stack_table(ptr %param) {
  %local = alloca [32 x i8]
  %local_int = ptrtoint ptr %local to i32
  %param_int = ptrtoint ptr %param to i32
  %a = insertelement <16 x i32> poison, i32 %local_int, i32 0
  %aa = shufflevector <16 x i32> %a, <16 x i32> poison, <16 x i32> zeroinitializer
  %b = insertelement <16 x i32> poison, i32 %param_int, i32 0
  %bb = shufflevector <16 x i32> %b, <16 x i32> poison, <16 x i32> zeroinitializer
  %pair = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %aa, <16 x i32> %bb, i32 52428)
  %half = shufflevector <16 x i32> %pair, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %result = call <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}

define <8 x i32> @aie2_gather() {
  %a = call <16 x i32> @llvm.aie2.vbroadcast32.I512(i32 ptrtoint (ptr @table_a to i32))
  %b = call <16 x i32> @llvm.aie2.vbroadcast32.I512(i32 ptrtoint (ptr @table_b to i32))
  %pair = call <16 x i32> @llvm.aie2.vsel32(<16 x i32> %a, <16 x i32> %b, i32 52428)
  %half = call <8 x i32> @llvm.aie2.ext.I256.I512(<16 x i32> %pair, i32 0)
  %result = call <8 x i32> @llvm.aie2.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}

define <8 x i32> @partially_resolved_select(i1 %condition, <16 x i32> %unknown) {
  %a = call <16 x i32> @llvm.aie2.vbroadcast32.I512(i32 ptrtoint (ptr @table_a to i32))
  %b = call <16 x i32> @llvm.aie2.vbroadcast32.I512(i32 ptrtoint (ptr @table_b to i32))
  %pair = call <16 x i32> @llvm.aie2.vsel32(<16 x i32> %a, <16 x i32> %b, i32 52428)
  %addr = select i1 %condition, <16 x i32> %pair, <16 x i32> %unknown
  %half = call <8 x i32> @llvm.aie2.ext.I256.I512(<16 x i32> %addr, i32 0)
  %result = call <8 x i32> @llvm.aie2.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}

define <8 x i32> @aie2_gather32() {
  %a = call <16 x i32> @llvm.aie2.vbroadcast32.I512(i32 ptrtoint (ptr @table_a to i32))
  %b = call <16 x i32> @llvm.aie2.vbroadcast32.I512(i32 ptrtoint (ptr @table_b to i32))
  %pair = call <16 x i32> @llvm.aie2.vsel32(<16 x i32> %a, <16 x i32> %b, i32 52428)
  %half = call <8 x i32> @llvm.aie2.ext.I256.I512(<16 x i32> %pair, i32 0)
  %lo = call <8 x i32> @llvm.aie2.load.4x32.lo(<8 x i32> %half)
  %hi = call <8 x i32> @llvm.aie2.load.4x32.hi(<8 x i32> %half)
  %result = add <8 x i32> %lo, %hi
  ret <8 x i32> %result
}

define <8 x i32> @aie2p_gather64() {
  %a = insertelement <16 x i32> poison, i32 ptrtoint (ptr @table_a to i32), i32 0
  %aa = shufflevector <16 x i32> %a, <16 x i32> poison, <16 x i32> zeroinitializer
  %b = insertelement <16 x i32> poison, i32 ptrtoint (ptr @table_b to i32), i32 0
  %bb = shufflevector <16 x i32> %b, <16 x i32> poison, <16 x i32> zeroinitializer
  %pair = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %aa, <16 x i32> %bb, i32 52428)
  %half = shufflevector <16 x i32> %pair, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %lo = call <8 x i32> @llvm.aie2p.load.4x64.lo(<8 x i32> %half)
  %hi = call <8 x i32> @llvm.aie2p.load.4x64.hi(<8 x i32> %half)
  %result = add <8 x i32> %lo, %hi
  ret <8 x i32> %result
}

define <8 x i32> @unresolved_vsel_gather(<16 x i32> %a, <16 x i32> %b) {
  %pair = call <16 x i32> @llvm.aie2p.vsel32(<16 x i32> %a, <16 x i32> %b, i32 52428)
  %half = shufflevector <16 x i32> %pair, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %result = call <8 x i32> @llvm.aie2p.load.4x16.lo(<8 x i32> %half)
  ret <8 x i32> %result
}
