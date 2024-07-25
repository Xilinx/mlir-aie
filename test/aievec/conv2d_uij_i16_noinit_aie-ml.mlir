// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=10 zero-offset=4" -aieml=true -canonicalize -split-input-file | FileCheck %s

func.func @conv2d (%A: memref<18x288xi16>, %B: memref<12xi16>, %C: memref<16x256xi16>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<18x288xi16>
            %b11 = affine.load %B[0] : memref<12xi16>
            %p11 = arith.muli %a11, %b11 : i16

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<18x288xi16>
            %b12 = affine.load %B[1] : memref<12xi16>
            %p12 = arith.muli %a12, %b12 : i16
            %c12 = arith.addi %p11, %p12 : i16

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<18x288xi16>
            %b13 = affine.load %B[2] : memref<12xi16>
            %p13 = arith.muli %a13, %b13 : i16
            %c13 = arith.addi %c12, %p13 : i16

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<18x288xi16>
            %b21 = affine.load %B[4] : memref<12xi16>
            %p21 = arith.muli %a21, %b21 : i16
            %c21 = arith.addi %c13, %p21 : i16

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<18x288xi16>
            %b22 = affine.load %B[5] : memref<12xi16>
            %p22 = arith.muli %a22, %b22 : i16
            %c22 = arith.addi %c21, %p22 : i16

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<18x288xi16>
            %b23 = affine.load %B[6] : memref<12xi16>
            %p23 = arith.muli %a23, %b23 : i16
            %c23 = arith.addi %c22, %p23 : i16

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<18x288xi16>
            %b31 = affine.load %B[8] : memref<12xi16>
            %p31 = arith.muli %a31, %b31 : i16
            %c31 = arith.addi %c23, %p31 : i16

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<18x288xi16>
            %b32 = affine.load %B[9] : memref<12xi16>
            %p32 = arith.muli %a32, %b32 : i16
            %c32 = arith.addi %c31, %p32 : i16

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<18x288xi16>
            %b33 = affine.load %B[10] : memref<12xi16>
            %p33 = arith.muli %a33, %b33 : i16
            %c33 = arith.addi %c32, %p33 : i16

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<16x256xi16>
        }
    }
    return
}

//CHECK-LABEL: @conv2d
// CHECK-SAME: %[[A0:[0-9a-zA-Z]*]]: memref<18x288xi16>
// CHECK-SAME: %[[A1:[0-9a-zA-Z]*]]: memref<12xi16>
// CHECK-SAME: %[[A2:[0-9a-zA-Z]*]]: memref<16x256xi16>
//      CHECK:    %[[C256:.*]] = arith.constant 256 : index
//      CHECK:    %[[C2:.*]] = arith.constant 2 : index
//      CHECK:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK:    %[[C16:.*]] = arith.constant 16 : index
//      CHECK:    %[[C10:.*]] = arith.constant 10 : i32
//      CHECK:    %[[C16_i32:.*]] = arith.constant 16 : i32
//      CHECK:    %[[C8:.*]] = arith.constant 8 : i32
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[T0:.*]] = aievec.upd %[[A1]][%[[C0:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<12xi16>, vector<16xi16>
//      CHECK:    %[[T1:.*]] = aievec.concat %[[T0:.*]], %[[T0:.*]] : vector<16xi16>, vector<32xi16>
//      CHECK:    %[[T2:.*]] = aievec.shift %[[T1:.*]], %[[T1:.*]], %[[C8]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
//      CHECK:    %[[T3:.*]] = aievec.shift %[[T1:.*]], %[[T1:.*]], %[[C16_i32]] {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
//      CHECK:    scf.for %[[A3:.*]] = %[[C0:.*]] to %[[C16:.*]] step %[[C1:.*]] {
//      CHECK:      %[[T4:.*]] = arith.addi %[[A3]], %[[C1:.*]] : index
//      CHECK:      %[[T5:.*]] = arith.addi %[[A3]], %[[C2:.*]] : index
//      CHECK:      scf.for %[[A4:.*]] = %[[C0:.*]] to %[[C256:.*]] step %[[C16:.*]] {
//      CHECK:        %[[T6:.*]] = aievec.upd %[[A0]][%[[A3]], %[[A4]]] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi16>, vector<32xi16>
//      CHECK:        %[[T7:.*]] = aievec.mul_conv %[[T6:.*]], %[[T1:.*]] {M = 16 : i32, N = 4 : i32} : vector<32xi16>, vector<32xi16>, vector<16xi64>
//      CHECK:        %[[T8:.*]] = aievec.upd %[[A0]][%[[T4:.*]], %[[A4]]] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi16>, vector<32xi16>
//      CHECK:        %[[T9:.*]] = aievec.fma_conv %[[T8:.*]], %[[T2:.*]], %[[T7:.*]] {M = 16 : i32, N = 4 : i32} : vector<32xi16>, vector<32xi16>, vector<16xi64>
//      CHECK:        %[[T10:.*]] = aievec.upd %[[A0]][%[[T5:.*]], %[[A4]]] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi16>, vector<32xi16>
//      CHECK:        %[[T11:.*]] = aievec.fma_conv %[[T10:.*]], %[[T3:.*]], %[[T9:.*]] {M = 16 : i32, N = 4 : i32} : vector<32xi16>, vector<32xi16>, vector<16xi64>
//      CHECK:        %[[T12:.*]] = aievec.srs %[[T11:.*]], %[[C10]] : vector<16xi64>, i32, vector<16xi16>
//      CHECK:        vector.transfer_write %[[T12:.*]], %[[A2]][%[[A3]], %[[A4]]] : vector<16xi16>, memref<16x256xi16>
