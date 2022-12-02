// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" -aieml=true --aie-vectorize="shift=0 dup-factor=2" | FileCheck %s

// CHECK-LABEL: func.func @conv2d(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
func.func @conv2d (%A: memref<18x288xi8>, %B: memref<48xi8>, %C: memref<16x256xi8>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //First row
            //first point
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<18x288xi8>
            %b11 = affine.load %B[0] : memref<48xi8>
            %p11 = arith.muli %a11, %b11 : i8

            //second point
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<18x288xi8>
            %b12 = affine.load %B[2] : memref<48xi8>
            %p12 = arith.muli %a12, %b12 : i8
            %c12 = arith.addi %p11, %p12 : i8

            //third point
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<18x288xi8>
            %b13 = affine.load %B[4] : memref<48xi8>
            %p13 = arith.muli %a13, %b13 : i8
            %c13 = arith.addi %c12, %p13 : i8

            //Second row
            //first point
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<18x288xi8>
            %b21 = affine.load %B[16] : memref<48xi8>
            %p21 = arith.muli %a21, %b21 : i8
            %c21 = arith.addi %c13, %p21 : i8

            //second point
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<18x288xi8>
            %b22 = affine.load %B[18] : memref<48xi8>
            %p22 = arith.muli %a22, %b22 : i8
            %c22 = arith.addi %c21, %p22 : i8

            //third point
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<18x288xi8>
            %b23 = affine.load %B[20] : memref<48xi8>
            %p23 = arith.muli %a23, %b23 : i8
            %c23 = arith.addi %c22, %p23 : i8

            //Third row
            //first point
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<18x288xi8>
            %b31 = affine.load %B[32] : memref<48xi8>
            %p31 = arith.muli %a31, %b31 : i8
            %c31 = arith.addi %c23, %p31 : i8

            //second point
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<18x288xi8>
            %b32 = affine.load %B[34] : memref<48xi8>
            %p32 = arith.muli %a32, %b32 : i8
            %c32 = arith.addi %c31, %p32 : i8

            //third point
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<18x288xi8>
            %b33 = affine.load %B[36] : memref<48xi8>
            %p33 = arith.muli %a33, %b33 : i8
            %c33 = arith.addi %c32, %p33 : i8

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<16x256xi8>
        }
    }
    return
}

//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %[[T0]] = aievec.upd %[[A1]][%c0] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
//CHECK-NEXT:    %[[T1]] = aievec.shuffle %[[T0]] {mode = 0 : i32} : vector<64xi8>, vector<64xi8>
//CHECK-NEXT:    %[[T2]] = aievec.shift %[[T1]] {shift = 8 : i32} : vector<64xi8>, vector<64xi8>
//CHECK-NEXT:    %[[T3]] = aievec.shift %[[T1]] {shift = 16 : i32} : vector<64xi8>, vector<64xi8>
//CHECK-NEXT:    %[[C0]] = arith.constant 0 : index
//CHECK-NEXT:    %[[C16]] = arith.constant 16 : index
//CHECK-NEXT:    %[[C1]] = arith.constant 1 : index
//CHECK-NEXT:    scf.for %[[A3]] = %[[C0]] to %[[C16]] step %[[C1]] {
//CHECK-NEXT:      %[[C2]] = arith.constant 1 : index
//CHECK-NEXT:      %[[T4]] = arith.addi %[[A3]], %[[C2]] : index
//CHECK-NEXT:      %[[C3]] = arith.constant 2 : index
//CHECK-NEXT:      %[[T5]] = arith.addi %[[A3]], %[[C3]] : index
//CHECK-NEXT:      %[[C4]] = arith.constant 0 : index
//CHECK-NEXT:      %[[C5]] = arith.constant 256 : index
//CHECK-NEXT:      %[[C6]] = arith.constant 32 : index
//CHECK-NEXT:      scf.for %[[A4]] = %[[C4]] to %[[C5]] step %[[C6]] {
//CHECK-NEXT:        %[[T6]] = aievec.upd %[[A0]][%[[A3]], %[[A4]]] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//CHECK-NEXT:        %[[T7]] = aievec.mul_conv %[[T6]], %[[T1]] {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//CHECK-NEXT:        %[[T8]] = aievec.upd %[[A0]][[[T4]], %[[A4]]] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//CHECK-NEXT:        %[[T9]] = aievec.fma_conv %[[T8]], %[[T2]], %[[T7]] {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//CHECK-NEXT:        %[[T10]] = aievec.upd %[[A0]][[[T5]], %[[A4]]] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//CHECK-NEXT:        %[[T11]] = aievec.fma_conv %[[T10]], %[[T3]], %[[T9]] {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//CHECK-NEXT:        %[[T12]] = aievec.srs %[[T11]] {shift = 0 : i8} : vector<32xi32>, vector<32xi8>
//CHECK-NEXT:        vector.transfer_write %[[T12]], %[[A2]][%[[A3]], %[[A4]]] {in_bounds = [true]} : vector<32xi8>, memref<16x256xi8>
