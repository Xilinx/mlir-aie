// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=0 zero-offset=4" -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d_0
func.func @conv2d_0 (%A: memref<?x?xi16>, %B: memref<?xi16>, %C: memref<?x?xi16>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x?xi16>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x?xi16>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x?xi16>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x?xi16>
            %b11 = affine.load %B[0] : memref<?xi16>
            %p11 = arith.muli %a11, %b11 : i16
            %c11 = arith.addi %ci, %p11 : i16

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x?xi16>
            %b12 = affine.load %B[1] : memref<?xi16>
            %p12 = arith.muli %a12, %b12 : i16
            %c12 = arith.addi %c11, %p12 : i16

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x?xi16>
            %b13 = affine.load %B[2] : memref<?xi16>
            %p13 = arith.muli %a13, %b13 : i16
            %c13 = arith.addi %c12, %p13 : i16

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x?xi16>
            %b21 = affine.load %B[4] : memref<?xi16>
            %p21 = arith.muli %a21, %b21 : i16
            %c21 = arith.addi %c13, %p21 : i16

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x?xi16>
            %b22 = affine.load %B[5] : memref<?xi16>
            %p22 = arith.muli %a22, %b22 : i16
            %c22 = arith.addi %c21, %p22 : i16

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x?xi16>
            %b23 = affine.load %B[6] : memref<?xi16>
            %p23 = arith.muli %a23, %b23 : i16
            %c23 = arith.addi %c22, %p23 : i16

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x?xi16>
            %b31 = affine.load %B[8] : memref<?xi16>
            %p31 = arith.muli %a31, %b31 : i16
            %c31 = arith.addi %c23, %p31 : i16

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x?xi16>
            %b32 = affine.load %B[9] : memref<?xi16>
            %p32 = arith.muli %a32, %b32 : i16
            %c32 = arith.addi %c31, %p32 : i16

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x?xi16>
            %b33 = affine.load %B[10] : memref<?xi16>
            %p33 = arith.muli %a33, %b33 : i16
            %c33 = arith.addi %c32, %p33 : i16

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x?xi16>
        }
    }
    return
}

// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?x?xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xi16>,
// CHECK-SAME:                        %[[VAL_2:.*]]: memref<?x?xi16>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?x?xi16>
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : memref<?x?xi16>
// CHECK:           %[[VAL_7:.*]] = aievec_aie1.upd %[[VAL_1]]{{\[}}%[[VAL_4]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_5]] step %[[VAL_9]] {
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_13]] : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 16 : index
// CHECK:             scf.for %[[VAL_17:.*]] = %[[VAL_15]] to %[[VAL_6]] step %[[VAL_16]] {
// CHECK:               %[[VAL_18:.*]] = aievec_aie1.upd %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_17]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<16xi16>
// CHECK:               %[[VAL_19:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_17]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
// CHECK:               %[[VAL_20:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_17]]], %[[VAL_19]] {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
// CHECK:               %[[VAL_21:.*]] = aievec.ups %[[VAL_18]] {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_22:.*]] = aievec_aie1.mac %[[VAL_20]], %[[VAL_7]], %[[VAL_21]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_23:.*]] = aievec_aie1.mac %[[VAL_20]], %[[VAL_7]], %[[VAL_22]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_24:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_17]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
// CHECK:               %[[VAL_25:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_17]]], %[[VAL_24]] {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
// CHECK:               %[[VAL_26:.*]] = aievec_aie1.mac %[[VAL_25]], %[[VAL_7]], %[[VAL_23]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_27:.*]] = aievec_aie1.mac %[[VAL_25]], %[[VAL_7]], %[[VAL_26]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_28:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_17]]] {index = 0 : i8, offset = 0 : i32} : memref<?x?xi16>, vector<32xi16>
// CHECK:               %[[VAL_29:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_17]]], %[[VAL_28]] {index = 1 : i8, offset = 256 : i32} : memref<?x?xi16>, vector<32xi16>
// CHECK:               %[[VAL_30:.*]] = aievec_aie1.mac %[[VAL_29]], %[[VAL_7]], %[[VAL_27]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_31:.*]] = aievec_aie1.mac %[[VAL_29]], %[[VAL_7]], %[[VAL_30]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_32:.*]] = aievec.srs %[[VAL_31]], %[[C0]] : vector<16xi48>, i32, vector<16xi16>
// CHECK:               vector.transfer_write %[[VAL_32]], %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_17]]] : vector<16xi16>, memref<?x?xi16>

//CHECK-LABEL: func.func @conv2d_1
func.func @conv2d_1 (%A: memref<?x256xi16>, %B: memref<?xi16>, %C: memref<?x256xi16>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x256xi16>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to 256 {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x256xi16>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x256xi16>
            %b11 = affine.load %B[0] : memref<?xi16>
            %p11 = arith.muli %a11, %b11 : i16
            %c11 = arith.addi %ci, %p11 : i16

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x256xi16>
            %b12 = affine.load %B[1] : memref<?xi16>
            %p12 = arith.muli %a12, %b12 : i16
            %c12 = arith.addi %c11, %p12 : i16

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x256xi16>
            %b13 = affine.load %B[2] : memref<?xi16>
            %p13 = arith.muli %a13, %b13 : i16
            %c13 = arith.addi %c12, %p13 : i16

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x256xi16>
            %b21 = affine.load %B[4] : memref<?xi16>
            %p21 = arith.muli %a21, %b21 : i16
            %c21 = arith.addi %c13, %p21 : i16

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x256xi16>
            %b22 = affine.load %B[5] : memref<?xi16>
            %p22 = arith.muli %a22, %b22 : i16
            %c22 = arith.addi %c21, %p22 : i16

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x256xi16>
            %b23 = affine.load %B[6] : memref<?xi16>
            %p23 = arith.muli %a23, %b23 : i16
            %c23 = arith.addi %c22, %p23 : i16

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x256xi16>
            %b31 = affine.load %B[8] : memref<?xi16>
            %p31 = arith.muli %a31, %b31 : i16
            %c31 = arith.addi %c23, %p31 : i16

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x256xi16>
            %b32 = affine.load %B[9] : memref<?xi16>
            %p32 = arith.muli %a32, %b32 : i16
            %c32 = arith.addi %c31, %p32 : i16

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x256xi16>
            %b33 = affine.load %B[10] : memref<?xi16>
            %p33 = arith.muli %a33, %b33 : i16
            %c33 = arith.addi %c32, %p33 : i16

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x256xi16>
        }
    }
    return
}

// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?x256xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xi16>,
// CHECK-SAME:                        %[[VAL_2:.*]]: memref<?x256xi16>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : memref<?x256xi16>
// CHECK:           %[[VAL_5:.*]] = aievec_aie1.upd %[[VAL_1]]{{\[}}%[[VAL_3]]] {index = 0 : i8, offset = 0 : i32} : memref<?xi16>, vector<16xi16>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_7]] {
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_8]], %[[VAL_11]] : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 256 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 16 : index
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_15]] {
// CHECK:               %[[VAL_17:.*]] = aievec_aie1.upd %[[VAL_2]]{{\[}}%[[VAL_8]], %[[VAL_16]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<16xi16>
// CHECK:               %[[VAL_18:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_8]], %[[VAL_16]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
// CHECK:               %[[VAL_19:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_8]], %[[VAL_16]]], %[[VAL_18]] {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
// CHECK:               %[[VAL_20:.*]] = aievec.ups %[[VAL_17]] {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_21:.*]] = aievec_aie1.mac %[[VAL_19]], %[[VAL_5]], %[[VAL_20]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_22:.*]] = aievec_aie1.mac %[[VAL_19]], %[[VAL_5]], %[[VAL_21]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_23:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_16]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
// CHECK:               %[[VAL_24:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_16]]], %[[VAL_23]] {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
// CHECK:               %[[VAL_25:.*]] = aievec_aie1.mac %[[VAL_24]], %[[VAL_5]], %[[VAL_22]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_26:.*]] = aievec_aie1.mac %[[VAL_24]], %[[VAL_5]], %[[VAL_25]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_27:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_16]]] {index = 0 : i8, offset = 0 : i32} : memref<?x256xi16>, vector<32xi16>
// CHECK:               %[[VAL_28:.*]] = aievec_aie1.upd %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_16]]], %[[VAL_27]] {index = 1 : i8, offset = 256 : i32} : memref<?x256xi16>, vector<32xi16>
// CHECK:               %[[VAL_29:.*]] = aievec_aie1.mac %[[VAL_28]], %[[VAL_5]], %[[VAL_26]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_30:.*]] = aievec_aie1.mac %[[VAL_28]], %[[VAL_5]], %[[VAL_29]] {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
// CHECK:               %[[VAL_31:.*]] = aievec.srs %[[VAL_30]], %[[C0]] : vector<16xi48>, i32, vector<16xi16>
// CHECK:               vector.transfer_write %[[VAL_31]], %[[VAL_2]]{{\[}}%[[VAL_8]], %[[VAL_16]]] : vector<16xi16>, memref<?x256xi16>
