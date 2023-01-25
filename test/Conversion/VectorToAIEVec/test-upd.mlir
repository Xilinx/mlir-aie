// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s --check-prefix=CHECK-V2

func.func @veccopy_i8(%arg0: memref<256xi8>, %arg1: memref<256xi8>) {
  %c0_i8 = arith.constant 0 : i8
  affine.for %arg2 = 0 to 256 step 16 {
    // CHECK: %[[LD:.*]] = aievec.upd {{.*}} {index = 0 : i8, offset = 0 : si32} : memref<256xi8>, vector<16xi8>
    %0 = vector.transfer_read %arg0[%arg2], %c0_i8 : memref<256xi8>, vector<16xi8>
    // CHECK: vector.transfer_write %[[LD]], {{.*}}
    vector.transfer_write %0, %arg1[%arg2] : vector<16xi8>, memref<256xi8>
  }
  return
}

func.func @veccopy_i16(%arg0: memref<256xi16>, %arg1: memref<256xi16>) {
  %c0_i16 = arith.constant 0 : i16
  affine.for %arg2 = 0 to 256 step 16 {
    // CHECK: %[[LD:.*]] = aievec.upd {{.*}} {index = 0 : i8, offset = 0 : si32} : memref<256xi16>, vector<16xi16>
    %0 = vector.transfer_read %arg0[%arg2], %c0_i16 : memref<256xi16>, vector<16xi16>
    // CHECK: vector.transfer_write %[[LD]], {{.*}}
    vector.transfer_write %0, %arg1[%arg2] : vector<16xi16>, memref<256xi16>
  }
  return
}

func.func @veccopy_i32(%arg0: memref<256xi32>, %arg1: memref<256xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %arg2 = 0 to 256 step 16 {
    // CHECK: %[[LD:.*]] = aievec.upd {{.*}} {index = 0 : i8, offset = 0 : si32} : memref<256xi32>, vector<16xi32>
    %0 = vector.transfer_read %arg0[%arg2], %c0_i32 : memref<256xi32>, vector<16xi32>
    // CHECK: vector.transfer_write %[[LD]], {{.*}}
    vector.transfer_write %0, %arg1[%arg2] : vector<16xi32>, memref<256xi32>
  }
  return
}

func.func @veccopy_long_i32(%arg0: memref<256xi32>, %arg1: memref<256xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %arg2 = 0 to 256 step 32 {
    // CHECK: %[[LD0:.*]] = aievec.upd {{.*}} {index = 0 : i8, offset = 0 : si32} : memref<256xi32>, vector<32xi32>
    // CHECK-NEXT: %[[LD1:.*]] = aievec.upd {{.*}}, %[[LD0]] {index = 1 : i8, offset = 512 : si32} : memref<256xi32>, vector<32xi32>
    // CHECK-V2: %[[LD:.*]] = aievec.upd {{.*}} {index = 0 : i8, offset = 0 : si32} : memref<256xi32>, vector<16xi32>
    %0 = vector.transfer_read %arg0[%arg2], %c0_i32 : memref<256xi32>, vector<32xi32>
    // CHECK: vector.transfer_write %[[LD1]], {{.*}}
    // CHECK-V2: vector.transfer_write %[[LD]], {{.*}}
    vector.transfer_write %0, %arg1[%arg2] : vector<32xi32>, memref<256xi32>
  }
  return
}
