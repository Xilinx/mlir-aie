// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d_0
func.func @conv2d_0 (%A: memref<?x?xi32>, %B: memref<?xi32>, %C: memref<?x?xi32>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x?xi32>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x?xi32>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x?xi32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x?xi32>
            %b11 = affine.load %B[0] : memref<?xi32>
            %p11 = arith.muli %a11, %b11 : i32
            %c11 = arith.addi %ci, %p11 : i32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x?xi32>
            %b12 = affine.load %B[1] : memref<?xi32>
            %p12 = arith.muli %a12, %b12 : i32
            %c12 = arith.addi %c11, %p12 : i32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x?xi32>
            %b13 = affine.load %B[2] : memref<?xi32>
            %p13 = arith.muli %a13, %b13 : i32
            %c13 = arith.addi %c12, %p13 : i32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x?xi32>
            %b21 = affine.load %B[3] : memref<?xi32>
            %p21 = arith.muli %a21, %b21 : i32
            %c21 = arith.addi %c13, %p21 : i32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x?xi32>
            %b22 = affine.load %B[4] : memref<?xi32>
            %p22 = arith.muli %a22, %b22 : i32
            %c22 = arith.addi %c21, %p22 : i32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x?xi32>
            %b23 = affine.load %B[5] : memref<?xi32>
            %p23 = arith.muli %a23, %b23 : i32
            %c23 = arith.addi %c22, %p23 : i32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x?xi32>
            %b31 = affine.load %B[6] : memref<?xi32>
            %p31 = arith.muli %a31, %b31 : i32
            %c31 = arith.addi %c23, %p31 : i32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x?xi32>
            %b32 = affine.load %B[7] : memref<?xi32>
            %p32 = arith.muli %a32, %b32 : i32
            %c32 = arith.addi %c31, %p32 : i32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x?xi32>
            %b33 = affine.load %B[8] : memref<?xi32>
            %p33 = arith.muli %a33, %b33 : i32
            %c33 = arith.addi %c32, %p33 : i32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x?xi32>
        }
    }
    return
}

// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?x?xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xi32>,
// CHECK-SAME:                        %[[VAL_2:.*]]: memref<?x?xi32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_5]] : memref<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_5]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
// CHECK:           %[[VAL_9:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_12:.*]] = %[[VAL_10]] to %[[VAL_6]] step %[[VAL_11]] {
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_12]], %[[VAL_15]] : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[VAL_19:.*]] = %[[VAL_17]] to %[[VAL_7]] step %[[VAL_18]] {
// CHECK:               %[[VAL_20:.*]] = aievec.upd %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_19]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<8xi32>
// CHECK:               %[[VAL_21:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_19]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
// CHECK:               %[[VAL_22:.*]] = aievec.ups %[[VAL_20]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_23:.*]] = aievec_aie1.mac %[[VAL_21]], %[[VAL_8]], %[[VAL_22]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_25:.*]] = arith.addi %[[VAL_19]], %[[VAL_24]] : index
// CHECK:               %[[VAL_26:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_25]]], %[[VAL_21]] {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
// CHECK:               %[[VAL_27:.*]] = aievec_aie1.mac %[[VAL_26]], %[[VAL_8]], %[[VAL_23]] {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_28:.*]] = aievec_aie1.mac %[[VAL_26]], %[[VAL_8]], %[[VAL_27]] {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_29:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_19]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
// CHECK:               %[[VAL_30:.*]] = aievec_aie1.mac %[[VAL_29]], %[[VAL_8]], %[[VAL_28]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_31:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_25]]], %[[VAL_29]] {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
// CHECK:               %[[VAL_32:.*]] = aievec_aie1.mac %[[VAL_31]], %[[VAL_8]], %[[VAL_30]] {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_33:.*]] = aievec_aie1.mac %[[VAL_31]], %[[VAL_8]], %[[VAL_32]] {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_34:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_16]], %[[VAL_19]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi32>, vector<16xi32>
// CHECK:               %[[VAL_35:.*]] = aievec_aie1.mac %[[VAL_34]], %[[VAL_8]], %[[VAL_33]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_36:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_16]], %[[VAL_25]]], %[[VAL_34]] {index = 1 : i8, offset = 224 : i32} : memref<?x?xi32>, vector<16xi32>
// CHECK:               %[[VAL_37:.*]] = aievec_aie1.mac %[[VAL_36]], %[[VAL_8]], %[[VAL_35]] {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_38:.*]] = aievec_aie1.mac %[[VAL_36]], %[[VAL_9]], %[[VAL_37]] {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_39:.*]] = aievec.srs %[[VAL_38]], %[[C0]] : vector<8xi80>, i32, vector<8xi32>
// CHECK:               vector.transfer_write %[[VAL_39]], %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_19]]] : vector<8xi32>, memref<?x?xi32>

// CHECK-LABEL: func.func @conv2d_1
func.func @conv2d_1 (%A: memref<?x256xi32>, %B: memref<?xi32>, %C: memref<?x256xi32>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x256xi32>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x256xi32>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x256xi32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x256xi32>
            %b11 = affine.load %B[0] : memref<?xi32>
            %p11 = arith.muli %a11, %b11 : i32
            %c11 = arith.addi %ci, %p11 : i32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x256xi32>
            %b12 = affine.load %B[1] : memref<?xi32>
            %p12 = arith.muli %a12, %b12 : i32
            %c12 = arith.addi %c11, %p12 : i32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x256xi32>
            %b13 = affine.load %B[2] : memref<?xi32>
            %p13 = arith.muli %a13, %b13 : i32
            %c13 = arith.addi %c12, %p13 : i32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x256xi32>
            %b21 = affine.load %B[3] : memref<?xi32>
            %p21 = arith.muli %a21, %b21 : i32
            %c21 = arith.addi %c13, %p21 : i32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x256xi32>
            %b22 = affine.load %B[4] : memref<?xi32>
            %p22 = arith.muli %a22, %b22 : i32
            %c22 = arith.addi %c21, %p22 : i32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x256xi32>
            %b23 = affine.load %B[5] : memref<?xi32>
            %p23 = arith.muli %a23, %b23 : i32
            %c23 = arith.addi %c22, %p23 : i32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x256xi32>
            %b31 = affine.load %B[6] : memref<?xi32>
            %p31 = arith.muli %a31, %b31 : i32
            %c31 = arith.addi %c23, %p31 : i32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x256xi32>
            %b32 = affine.load %B[7] : memref<?xi32>
            %p32 = arith.muli %a32, %b32 : i32
            %c32 = arith.addi %c31, %p32 : i32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x256xi32>
            %b33 = affine.load %B[8] : memref<?xi32>
            %p33 = arith.muli %a33, %b33 : i32
            %c33 = arith.addi %c32, %p33 : i32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x256xi32>
        }
    }
    return
}

// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?x256xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xi32>,
// CHECK-SAME:                        %[[VAL_2:.*]]: memref<?x256xi32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?x256xi32>
// CHECK:           %[[VAL_6:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_4]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
// CHECK:           %[[VAL_7:.*]] = aievec.upd %[[VAL_1]]{{\[}}%[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi32>, vector<8xi32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_5]] step %[[VAL_9]] {
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_13]] : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 256 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_17]] {
// CHECK:               %[[VAL_19:.*]] = aievec.upd %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_18]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<8xi32>
// CHECK:               %[[VAL_20:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_18]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<16xi32>
// CHECK:               %[[VAL_21:.*]] = aievec.ups %[[VAL_19]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_22:.*]] = aievec_aie1.mac %[[VAL_20]], %[[VAL_6]], %[[VAL_21]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_23:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_18]], %[[VAL_23]] : index
// CHECK:               %[[VAL_25:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_24]]], %[[VAL_20]] {index = 1 : i8, offset = 224 : i32} : memref<?x256xi32>, vector<16xi32>
// CHECK:               %[[VAL_26:.*]] = aievec_aie1.mac %[[VAL_25]], %[[VAL_6]], %[[VAL_22]] {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_27:.*]] = aievec_aie1.mac %[[VAL_25]], %[[VAL_6]], %[[VAL_26]] {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_28:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_18]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<16xi32>
// CHECK:               %[[VAL_29:.*]] = aievec_aie1.mac %[[VAL_28]], %[[VAL_6]], %[[VAL_27]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_30:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_24]]], %[[VAL_28]] {index = 1 : i8, offset = 224 : i32} : memref<?x256xi32>, vector<16xi32>
// CHECK:               %[[VAL_31:.*]] = aievec_aie1.mac %[[VAL_30]], %[[VAL_6]], %[[VAL_29]] {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_32:.*]] = aievec_aie1.mac %[[VAL_30]], %[[VAL_6]], %[[VAL_31]] {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_33:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_18]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi32>, vector<16xi32>
// CHECK:               %[[VAL_34:.*]] = aievec_aie1.mac %[[VAL_33]], %[[VAL_6]], %[[VAL_32]] {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_35:.*]] = aievec.upd %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_24]]], %[[VAL_33]] {index = 1 : i8, offset = 224 : i32} : memref<?x256xi32>, vector<16xi32>
// CHECK:               %[[VAL_36:.*]] = aievec_aie1.mac %[[VAL_35]], %[[VAL_6]], %[[VAL_34]] {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_37:.*]] = aievec_aie1.mac %[[VAL_35]], %[[VAL_7]], %[[VAL_36]] {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
// CHECK:               %[[VAL_38:.*]] = aievec.srs %[[VAL_37]], %[[C0]] : vector<8xi80>, i32, vector<8xi32>
// CHECK:               vector.transfer_write %[[VAL_38]], %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_18]]] : vector<8xi32>, memref<?x256xi32>
