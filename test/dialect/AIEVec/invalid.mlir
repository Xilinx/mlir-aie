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
