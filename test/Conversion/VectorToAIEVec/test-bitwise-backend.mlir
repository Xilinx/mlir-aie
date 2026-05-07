// On the CPP (chess) backend, 512-bit integer vector arith.{andi,ori,xori}
// must be lowered to aievec.{band,bor,bxor}, which the chess emitter knows
// how to print. On the LLVMIR (Peano) backend, AIEVecToLLVM has no lowering
// for those aievec ops -- the standard arith→llvm route handles vector AND/OR/XOR
// natively -- so the arith ops must remain illegal-targeted only on CPP and
// pass through unchanged on LLVMIR.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=cpp" \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// AIE2P/LLVMIR is the path the original failure was reported on -- exercise
// it explicitly so the legalization regression cannot be silently
// reintroduced.
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=NO-BITWISE-AIEVEC

// CPP-LABEL: func @vec_andi_512(
// CPP-SAME:    %[[LHS:.*]]: vector<16xi32>,
// CPP-SAME:    %[[RHS:.*]]: vector<16xi32>)
// LLVM-LABEL: func @vec_andi_512(
func.func @vec_andi_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CPP:  %[[R:.*]] = aievec.band %[[LHS]], %[[RHS]] : vector<16xi32>
  // CPP:  return %[[R]]
  // LLVM-NOT: aievec.band
  // LLVM:  %[[R:.*]] = arith.andi %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM:  return %[[R]]
  %r = arith.andi %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// CPP-LABEL: func @vec_ori_512(
// LLVM-LABEL: func @vec_ori_512(
func.func @vec_ori_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CPP:  aievec.bor %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM-NOT: aievec.bor
  // LLVM:  arith.ori %{{.*}}, %{{.*}} : vector<16xi32>
  %r = arith.ori %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// CPP-LABEL: func @vec_xori_512(
// LLVM-LABEL: func @vec_xori_512(
func.func @vec_xori_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CPP:  aievec.bxor %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM-NOT: aievec.bxor
  // LLVM:  arith.xori %{{.*}}, %{{.*}} : vector<16xi32>
  %r = arith.xori %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// Non-512-bit vectors stay legal on both backends (no aievec promotion).

// CPP-LABEL: func @vec_andi_256(
// LLVM-LABEL: func @vec_andi_256(
func.func @vec_andi_256(%a: vector<8xi32>, %b: vector<8xi32>) -> vector<8xi32> {
  // CPP-NOT: aievec.band
  // LLVM-NOT: aievec.band
  // CPP:  arith.andi %{{.*}}, %{{.*}} : vector<8xi32>
  // LLVM: arith.andi %{{.*}}, %{{.*}} : vector<8xi32>
  %r = arith.andi %a, %b : vector<8xi32>
  return %r : vector<8xi32>
}

// Scalar bitwise ops are unaffected on both backends.

// CPP-LABEL: func @scalar_andi(
// LLVM-LABEL: func @scalar_andi(
func.func @scalar_andi(%a: i32, %b: i32) -> i32 {
  // CPP-NOT: aievec.band
  // LLVM-NOT: aievec.band
  // CPP:  arith.andi %{{.*}}, %{{.*}} : i32
  // LLVM: arith.andi %{{.*}}, %{{.*}} : i32
  %r = arith.andi %a, %b : i32
  return %r : i32
}

// Belt-and-braces global check for the AIE2P/LLVMIR run: no aievec.{band,bor,
// bxor,bneg} ops anywhere in the output. Anchored by the file-level CHECK so
// FileCheck won't claim vacuous success.
// NO-BITWISE-AIEVEC: module
// NO-BITWISE-AIEVEC-NOT: aievec.band
// NO-BITWISE-AIEVEC-NOT: aievec.bor
// NO-BITWISE-AIEVEC-NOT: aievec.bxor
// NO-BITWISE-AIEVEC-NOT: aievec.bneg
