// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @invalidElementType(%A : vector<4x8xf16>, %B : vector<8x4xf16>,
                              %C : vector<4x4xf32>) -> vector<4x4xf32> {
  // expected-error @+1 {{op operand #0 must be a vector compatible with a lhs operand of matrix-multiply and accumulate, but got 'vector<4x8xf16>'}}
  %0 = aievec.matmul %A, %B, %C : vector<4x8xf16>, vector<8x4xf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

func.func @invalidShape(%A : vector<4x4xbf16>, %B : vector<4x4xbf16>,
                        %C : vector<4x4xf32>) -> vector<4x4xf32> {
  // expected-error @+1 {{op operand #0 must be a vector compatible with a lhs operand of matrix-multiply and accumulate, but got 'vector<4x4xbf16>'}}
  %0 = aievec.matmul %A, %B, %C : vector<4x4xbf16>, vector<4x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

func.func @invalidContraction(%A : vector<2x4xi16>, %B : vector<2x8xi16>,
                              %C : vector<4x8xi32>) -> vector<4x8xi32> {
  // expected-error @+1 {{op failed to verify that [lhs x rhs = acc] is a valid contraction}}
  %0 = aievec.matmul %A, %B, %C : vector<2x4xi16>, vector<2x8xi16>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

func.func @invalidAccumulatorType(%A : vector<2x4xi16>, %B : vector<4x8xi16>,
                                  %C : vector<2x8xi32>) -> vector<2x8xi32> {
  // expected-error @+1 {{op operand #2 must be a vector compatible with an accumulator of matrix-multiply and accumulate, but got 'vector<2x8xi32>'}}
  %0 = aievec.matmul %A, %B, %C : vector<2x4xi16>, vector<4x8xi16>
                                  into vector<2x8xi32>
  return %0 : vector<2x8xi32>
}

// -----

func.func @invalidShuffleModeElementType(%v : vector<32xi16>)
            -> vector<32xi16> {
  // expected-error @+1 {{shuffle mode 't32_4x4' requires vectors of 32-bit elements}}
  %r = aievec.shuffle %v [t32_4x4] : vector<32xi16>
  return %r : vector<32xi16>
}

// -----

func.func @invalidShuffleModeExtraOperand(%v : vector<32xi16>)
            -> vector<32xi16> {
  // expected-error @+1 {{shuffle mode 't16_4x8' does not admit a second operand}}
  %r = aievec.shuffle %v, %v [t16_4x8] : vector<32xi16>
  return %r : vector<32xi16>
}

// -----

func.func @invalidShuffleModeMissingOperand(%v : vector<32xi16>)
            -> vector<32xi16> {
  // expected-error @+1 {{shuffle mode 't16_16x4_lo' requires a second operand}}
  %r = aievec.shuffle %v [t16_16x4_lo] : vector<32xi16>
  return %r : vector<32xi16>
}

// -----

func.func @invalidElementTypeMulElem(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<32xi64> {
  // expected-error @+1 {{'aievec.mul_elem' op failed to verify that result type is not a valid accumulator type for the lhs x rhs operands type.}}
  %t11 = aievec.mul_elem %arg0, %arg1 : vector<32xi8>, vector<32xi8>, vector<32xi64>
  return %t11 : vector<32xi64>
}