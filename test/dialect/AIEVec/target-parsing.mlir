//===- target-parsing.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The aie-target option of the vector-to-AIEVec passes.

// An unknown target is an error in each pass and in the pipeline.
// RUN: not aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie3" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not aie-opt %s --lower-vector-to-aievec="aie-target=aie3" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not aie-opt %s --optimize-aievec="aie-target=aie3" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not aie-opt %s --convert-vector-to-aievec="aie-target=aie3" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// UNKNOWN: error: unknown AIE target 'aie3'

// AIE2PS is a known target without an AIEVec lowering.
// RUN: not aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie2ps" 2>&1 | FileCheck %s --check-prefix=AIE2PS
// RUN: not aie-opt %s --lower-vector-to-aievec="aie-target=aie2ps" 2>&1 | FileCheck %s --check-prefix=AIE2PS
// RUN: not aie-opt %s --optimize-aievec="aie-target=aie2ps" 2>&1 | FileCheck %s --check-prefix=AIE2PS
// RUN: not aie-opt %s --convert-vector-to-aievec="aie-target=aie2ps" 2>&1 | FileCheck %s --check-prefix=AIE2PS
// AIE2PS: error: AIEVec has no vector lowering for target 'aie2ps'

// "aieml" is "aie2", and an empty value is "aie" (the default).
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P
// RUN: aie-opt %s --canonicalize-vector-for-aievec | FileCheck %s --check-prefix=CANON-AIE
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=" | FileCheck %s --check-prefix=CANON-AIE
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie2" | FileCheck %s --check-prefix=CANON-AIE2

// math.exp becomes aievec.exp only on AIE2P.
// AIE2-LABEL: func.func @exp
// AIE2-NOT: aievec.exp
// AIE2: call @getExpBf16
// AIE2P-LABEL: func.func @exp
// AIE2P: aievec.exp
func.func @exp(%a: vector<16xbf16>) -> vector<16xbf16> {
  %0 = math.exp %a : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// A read at a 128-bit but not 256-bit aligned offset is aligned on AIE1, so
// it is kept, and it is split on AIE2.
// CANON-AIE-LABEL: func.func @read
// CANON-AIE-NOT: vector.extract_strided_slice
// CANON-AIE: return
// CANON-AIE2-LABEL: func.func @read
// CANON-AIE2: vector.extract_strided_slice
func.func @read(%m: memref<64xi32>) -> vector<8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %m[%c4], %c0_i32 : memref<64xi32>, vector<8xi32>
  return %0 : vector<8xi32>
}
